"""Deterministically regenerate motion plans from a PET trace.

This module deliberately uses a virtual clock: timestamps come from the trace,
and no desktop process, wall-clock wait, or network connection is involved.
It accepts either an episode directory containing trace v1 shards or a JSONL
export containing protocol ``world_state`` messages.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import gzip
import hashlib
import json
import math
from pathlib import Path
import re
import sys
from typing import Any, Iterator, Mapping, Sequence, TextIO


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
GENERATOR_ROOT = WORKSPACE_ROOT / "services" / "generator"
PROTOCOL_ROOT = WORKSPACE_ROOT / "packages" / "protocol" / "python"
for _package_root in (GENERATOR_ROOT, PROTOCOL_ROOT):
    _package_text = str(_package_root)
    if _package_text not in sys.path:
        sys.path.insert(0, _package_text)

from pet_generator.backend import MotionBackend  # noqa: E402
from pet_generator.planner import AutoregressiveMotionBackend  # noqa: E402
from pet_generator.protocol import Envelope, PROTOCOL_NAME, PROTOCOL_VERSION  # noqa: E402
from pet_generator.state import WorldState, parse_world_state  # noqa: E402


TRACE_SCHEMA = "pet-trace"
TRACE_VERSION = 1
OUTPUT_SCHEMA = "pet-regeneration"
OUTPUT_VERSION = 1
_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")


class RegenerationError(ValueError):
    """An input error whose message is safe to show without a local path."""


@dataclass(frozen=True, slots=True)
class TraceEvent:
    kind: str
    payload: Mapping[str, Any]
    record_seq: int
    wall_time_ms: int
    elapsed_us: int


@dataclass(frozen=True, slots=True)
class SourceSpec:
    files: tuple[Path, ...]
    manifest: Mapping[str, Any]
    episode_root: Path | None


@dataclass(frozen=True, slots=True)
class CheckpointStatus:
    declared: bool
    deterministic: bool
    status: str
    expected_sha256: str | None = None
    actual_sha256: str | None = None

    def public_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "declared": self.declared,
            "status": self.status,
        }
        if self.expected_sha256 is not None:
            payload["expected_sha256"] = self.expected_sha256
        if self.actual_sha256 is not None:
            payload["actual_sha256"] = self.actual_sha256
        return payload


@dataclass(slots=True)
class RegenerationSummary:
    backend: str
    deterministic: bool
    nondeterministic_reasons: list[str] = field(default_factory=list)
    worlds_seen: int = 0
    worlds_regenerated: int = 0
    worlds_without_plan: int = 0
    plans_recorded: int = 0
    plans_compared: int = 0
    exact_matches: int = 0
    mismatches: int = 0
    cancellations_seen: int = 0
    cancellations_applied: int = 0
    result_sha256: str = ""

    @property
    def exact_match(self) -> bool | None:
        if self.plans_compared == 0:
            return None
        return self.mismatches == 0 and self.exact_matches == self.plans_compared

    def to_payload(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "cancellations_applied": self.cancellations_applied,
            "cancellations_seen": self.cancellations_seen,
            "deterministic": self.deterministic,
            "exact_match": self.exact_match,
            "exact_matches": self.exact_matches,
            "mismatches": self.mismatches,
            "nondeterministic_reasons": sorted(set(self.nondeterministic_reasons)),
            "plans_compared": self.plans_compared,
            "plans_recorded": self.plans_recorded,
            "result_sha256": self.result_sha256,
            "worlds_regenerated": self.worlds_regenerated,
            "worlds_seen": self.worlds_seen,
            "worlds_without_plan": self.worlds_without_plan,
        }


def _normalise_json(value: Any) -> Any:
    """Return a JSON value with JavaScript/Python number spelling aligned."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise RegenerationError("canonical JSON cannot contain a non-finite number")
        if value == 0:
            return 0
        if value.is_integer():
            return int(value)
        return value
    if isinstance(value, Mapping):
        return {str(key): _normalise_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalise_json(item) for item in value]
    raise RegenerationError(f"canonical JSON does not support {type(value).__name__}")


def canonical_json(value: Any) -> str:
    """Encode a stable, whitespace-free JSON representation."""

    return json.dumps(
        _normalise_json(value),
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _read_json_file(path: Path) -> Mapping[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RegenerationError("episode manifest is unreadable or invalid JSON") from exc
    if not isinstance(raw, dict):
        raise RegenerationError("episode manifest must contain a JSON object")
    return raw


def _manifest_shards(root: Path, manifest: Mapping[str, Any]) -> list[Path]:
    declared = manifest.get("shards", manifest.get("parts", manifest.get("chunks")))
    result: list[Path] = []
    if isinstance(declared, list):
        for item in declared:
            name: object = item
            if isinstance(item, dict):
                name = item.get("file", item.get("path", item.get("name")))
            if not isinstance(name, str) or not name:
                continue
            candidate = root / name
            if candidate.is_file():
                result.append(candidate)
    return result


def resolve_source(input_path: str | Path) -> SourceSpec:
    path = Path(input_path).expanduser()
    if path.is_file():
        return SourceSpec((path,), {}, path.parent)
    if not path.is_dir():
        raise RegenerationError("input does not exist or is not a file or episode directory")

    manifest_path = path / "manifest.json"
    manifest = _read_json_file(manifest_path) if manifest_path.is_file() else {}
    files = _manifest_shards(path, manifest)
    if not files:
        patterns = (
            "trace*.ndjson.gz",
            "trace*.jsonl.gz",
            "trace*.ndjson",
            "trace*.jsonl",
            "trace*.ndjson.partial",
        )
        discovered: set[Path] = set()
        for pattern in patterns:
            discovered.update(candidate for candidate in path.glob(pattern) if candidate.is_file())
        files = sorted(discovered, key=lambda item: item.name)
    else:
        # A live or crashed episode can contain finalized manifest chunks plus
        # one recoverable, uncompressed hot chunk not listed in the manifest.
        hot = sorted(path.glob("trace*.ndjson.partial"), key=lambda item: item.name)
        known = set(files)
        files.extend(item for item in hot if item.is_file() and item not in known)
    if not files:
        raise RegenerationError("episode contains no readable trace shards")
    return SourceSpec(tuple(files), manifest, path)


def _open_text(path: Path) -> TextIO:
    try:
        with path.open("rb") as probe:
            magic = probe.read(2)
    except OSError as exc:
        raise RegenerationError("a trace shard could not be opened") from exc
    try:
        if magic == b"\x1f\x8b" or path.name.endswith(".gz"):
            return gzip.open(path, mode="rt", encoding="utf-8", newline="")
        return path.open(mode="rt", encoding="utf-8", newline="")
    except OSError as exc:
        raise RegenerationError("a trace shard could not be opened") from exc


def _require_mapping(value: object, field_name: str, part: int, line: int) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise RegenerationError(f"part {part} line {line}: {field_name} must be an object")
    return value


def _event_from_raw(raw: Mapping[str, Any], ordinal: int, part: int, line: int) -> TraceEvent:
    if raw.get("schema") == TRACE_SCHEMA:
        if raw.get("version") != TRACE_VERSION:
            raise RegenerationError(f"part {part} line {line}: unsupported trace version")
        kind = raw.get("kind")
        if not isinstance(kind, str) or not kind:
            raise RegenerationError(f"part {part} line {line}: trace kind must be text")
        payload = _require_mapping(raw.get("payload"), "trace payload", part, line)
        return TraceEvent(
            kind=kind,
            payload=payload,
            record_seq=_safe_nonnegative_int(raw.get("record_seq"), ordinal),
            wall_time_ms=_safe_nonnegative_int(raw.get("wall_time_ms"), 0),
            elapsed_us=_safe_nonnegative_int(raw.get("elapsed_us"), ordinal),
        )

    # Compatibility reader for raw protocol captures.
    if raw.get("protocol") == PROTOCOL_NAME and raw.get("version") == PROTOCOL_VERSION:
        message_type = raw.get("type")
        payload = _require_mapping(raw.get("payload"), "protocol payload", part, line)
        seq = _safe_nonnegative_int(raw.get("seq"), ordinal)
        timestamp = _safe_nonnegative_int(raw.get("timestamp_ms"), 0)
        if message_type == "world_state":
            return TraceEvent(
                "world_state",
                {"seq": seq, "timestamp_ms": timestamp, "state": payload},
                ordinal,
                timestamp,
                ordinal,
            )
        if message_type == "horizon_plan":
            return TraceEvent(
                "plan_received",
                {"plan": payload, "received_at_ms": timestamp},
                ordinal,
                timestamp,
                ordinal,
            )
        if message_type == "cancel":
            return TraceEvent("cancel", payload, ordinal, timestamp, ordinal)
        return TraceEvent(str(message_type), payload, ordinal, timestamp, ordinal)

    # Compatibility reader for exported entries and bare WorldStatePayload.
    nested = raw.get("world_state", raw.get("state"))
    if isinstance(nested, dict):
        if nested.get("protocol") == PROTOCOL_NAME:
            return _event_from_raw(nested, ordinal, part, line)
        seq = _safe_nonnegative_int(raw.get("seq"), ordinal)
        timestamp = _safe_nonnegative_int(raw.get("timestamp_ms"), 0)
        return TraceEvent(
            "world_state",
            {"seq": seq, "timestamp_ms": timestamp, "state": nested},
            ordinal,
            timestamp,
            ordinal,
        )
    if "session_id" in raw and "pet" in raw and "surfaces" in raw:
        seq = _safe_nonnegative_int(raw.get("seq"), ordinal)
        timestamp = _safe_nonnegative_int(raw.get("timestamp_ms"), 0)
        state = dict(raw)
        state.pop("seq", None)
        state.pop("timestamp_ms", None)
        return TraceEvent(
            "world_state",
            {"seq": seq, "timestamp_ms": timestamp, "state": state},
            ordinal,
            timestamp,
            ordinal,
        )
    raise RegenerationError(f"part {part} line {line}: unsupported JSONL record")


def _safe_nonnegative_int(value: object, default: int) -> int:
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return default


def iter_events(source: SourceSpec) -> Iterator[TraceEvent]:
    ordinal = 0
    for part, path in enumerate(source.files, start=1):
        try:
            stream = _open_text(path)
            with stream:
                for line_number, line_text in enumerate(stream, start=1):
                    if not line_text.strip():
                        continue
                    try:
                        raw = json.loads(line_text)
                    except (json.JSONDecodeError, ValueError, RecursionError) as exc:
                        raise RegenerationError(
                            f"part {part} line {line_number}: invalid JSON"
                        ) from exc
                    if not isinstance(raw, dict):
                        raise RegenerationError(
                            f"part {part} line {line_number}: JSONL record must be an object"
                        )
                    yield _event_from_raw(raw, ordinal, part, line_number)
                    ordinal += 1
        except (OSError, UnicodeError, EOFError) as exc:
            raise RegenerationError(f"part {part}: trace shard is unreadable") from exc


def _determinism_metadata(manifest: Mapping[str, Any]) -> Mapping[str, Any]:
    metadata = manifest.get("metadata")
    if not isinstance(metadata, dict):
        return {}
    determinism = metadata.get("determinism")
    return determinism if isinstance(determinism, dict) else {}


def _checkpoint_declaration(manifest: Mapping[str, Any]) -> Mapping[str, Any] | None:
    determinism = _determinism_metadata(manifest)
    checkpoint = determinism.get("checkpoint")
    if checkpoint is None:
        checkpoint = manifest.get("checkpoint")
    if isinstance(checkpoint, str):
        return {"path": checkpoint}
    return checkpoint if isinstance(checkpoint, dict) else None


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            while True:
                block = stream.read(1024 * 1024)
                if not block:
                    break
                digest.update(block)
    except OSError as exc:
        raise RegenerationError("declared model checkpoint could not be read") from exc
    return digest.hexdigest()


def validate_checkpoint(source: SourceSpec) -> CheckpointStatus:
    declaration = _checkpoint_declaration(source.manifest)
    if declaration is None:
        return CheckpointStatus(False, True, "not_declared")

    sha_value = declaration.get("sha256", declaration.get("hash"))
    expected = sha_value.lower() if isinstance(sha_value, str) and _SHA256_RE.fullmatch(sha_value) else None
    path_value = declaration.get("path", declaration.get("file", declaration.get("relative_path")))
    if not isinstance(path_value, str) or not path_value:
        return CheckpointStatus(True, False, "checkpoint_path_missing", expected)
    if source.episode_root is None:
        return CheckpointStatus(True, False, "checkpoint_missing", expected)
    checkpoint_path = Path(path_value)
    if not checkpoint_path.is_absolute():
        checkpoint_path = source.episode_root / checkpoint_path
    if not checkpoint_path.is_file():
        return CheckpointStatus(True, False, "checkpoint_missing", expected)
    actual = _hash_file(checkpoint_path)
    if expected is None:
        return CheckpointStatus(True, False, "checkpoint_sha256_missing", None, actual)
    if actual != expected:
        return CheckpointStatus(True, False, "checkpoint_sha256_mismatch", expected, actual)
    return CheckpointStatus(True, True, "verified", expected, actual)


def _backend_checkpoint_sha256(backend: MotionBackend) -> str | None:
    """Read an explicit checkpoint identity from a pre-loaded future backend."""

    for attribute in ("checkpoint_sha256", "model_sha256"):
        value = getattr(backend, attribute, None)
        if isinstance(value, str) and _SHA256_RE.fullmatch(value):
            return value.lower()
    return None


def _world_from_event(event: TraceEvent) -> tuple[WorldState, Mapping[str, Any]]:
    payload = event.payload
    state_value = payload.get("state", payload.get("world_state", payload.get("message")))
    if not isinstance(state_value, dict):
        raise RegenerationError(f"record {event.record_seq}: world_state payload has no state object")
    if state_value.get("protocol") == PROTOCOL_NAME:
        envelope_payload = state_value.get("payload")
        if not isinstance(envelope_payload, dict):
            raise RegenerationError(f"record {event.record_seq}: world_state envelope is invalid")
        seq = _safe_nonnegative_int(state_value.get("seq"), event.record_seq)
        timestamp = _safe_nonnegative_int(state_value.get("timestamp_ms"), event.wall_time_ms)
        state = envelope_payload
    else:
        seq = _safe_nonnegative_int(payload.get("seq"), event.record_seq)
        timestamp = _safe_nonnegative_int(payload.get("timestamp_ms"), event.wall_time_ms)
        state = state_value
    envelope = Envelope(
        protocol=PROTOCOL_NAME,
        version=PROTOCOL_VERSION,
        type="world_state",
        seq=seq,
        timestamp_ms=timestamp,
        payload=state,
    )
    try:
        return parse_world_state(envelope), state
    except Exception as exc:
        raise RegenerationError(f"record {event.record_seq}: invalid world_state") from exc


def _plan_from_event(event: TraceEvent) -> Mapping[str, Any]:
    plan = event.payload.get("plan", event.payload.get("horizon_plan", event.payload.get("message")))
    if isinstance(plan, dict) and plan.get("protocol") == PROTOCOL_NAME:
        plan = plan.get("payload")
    if not isinstance(plan, dict):
        raise RegenerationError(f"record {event.record_seq}: plan_received payload has no plan object")
    return plan


def _plan_based_on_seq(plan: Mapping[str, Any], event: TraceEvent) -> int:
    value = plan.get("based_on_seq", event.payload.get("based_on_seq"))
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise RegenerationError(f"record {event.record_seq}: plan has invalid based_on_seq")
    return value


def _uint32(value: object) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool) and 0 <= value <= 0xFFFF_FFFF:
        return value
    return None


def _manifest_session_seed(manifest: Mapping[str, Any]) -> int | None:
    determinism = _determinism_metadata(manifest)
    value = determinism.get("session_seed")
    return _uint32(value)


def _derive_seed(session_seed: int, session_id: str, seq: int) -> int:
    material = f"{session_seed}:{session_id}:{seq}".encode("utf-8")
    return int.from_bytes(hashlib.blake2s(material, digest_size=4).digest(), "big")


def _select_seed(
    world: WorldState,
    recorded_plan: Mapping[str, Any] | None,
    session_seed: int | None,
) -> tuple[int, str]:
    if recorded_plan is not None:
        recorded = _uint32(recorded_plan.get("seed"))
        if recorded is not None:
            return recorded, "recorded_plan"
    if world.requested_seed is not None:
        return world.requested_seed, "world_state"
    if session_seed is not None:
        return _derive_seed(session_seed, world.session_id, world.seq), "session_seed"
    raise RegenerationError(
        f"world seq {world.seq}: no recorded plan seed, requested seed, or session seed is available"
    )


def _select_generated_at(
    world: WorldState,
    world_event: TraceEvent,
    recorded_plan: Mapping[str, Any] | None,
) -> tuple[int, str]:
    if recorded_plan is not None:
        value = recorded_plan.get("generated_at_ms")
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            return value, "recorded_plan"
    if world_event.wall_time_ms > 0:
        return world_event.wall_time_ms, "trace_virtual_clock"
    return world.timestamp_ms, "world_state_virtual_clock"


def _timing_from_manifest(manifest: Mapping[str, Any]) -> tuple[int, int] | None:
    determinism = _determinism_metadata(manifest)
    timing = determinism.get("timing")
    if not isinstance(timing, dict):
        return None
    horizon = timing.get("plan_horizon_ms")
    dt = timing.get("plan_dt_ms")
    if (
        isinstance(horizon, int)
        and not isinstance(horizon, bool)
        and isinstance(dt, int)
        and not isinstance(dt, bool)
        and horizon >= 50
        and 8 <= dt <= 250
    ):
        return horizon, dt
    return None


def _scan_source(source: SourceSpec) -> tuple[bool, tuple[int, int] | None, int]:
    has_plans = False
    inferred_timing: tuple[int, int] | None = None
    plan_count = 0
    for event in iter_events(source):
        if event.kind != "plan_received":
            continue
        has_plans = True
        plan_count += 1
        if inferred_timing is not None:
            continue
        plan = _plan_from_event(event)
        dt = plan.get("dt_ms")
        points = plan.get("points")
        if isinstance(dt, int) and not isinstance(dt, bool) and isinstance(points, list) and len(points) >= 2:
            inferred_timing = (dt * len(points), dt)
    return has_plans, inferred_timing, plan_count


class _OutputWriter:
    def __init__(self, stream: TextIO):
        self.stream = stream
        self._digest = hashlib.sha256()

    def write(self, value: Mapping[str, Any], *, include_in_digest: bool = True) -> None:
        encoded = canonical_json(value)
        self.stream.write(encoded + "\n")
        if include_in_digest:
            self._digest.update(encoded.encode("utf-8"))
            self._digest.update(b"\n")

    def hexdigest(self) -> str:
        return self._digest.hexdigest()


def _output_record(
    world: WorldState,
    plan: Mapping[str, Any],
    recorded_plan: Mapping[str, Any] | None,
    *,
    seed_source: str,
    generated_at_source: str,
    compare_only: bool,
) -> dict[str, Any]:
    generated_hash = canonical_sha256(plan)
    comparison: dict[str, Any] | None = None
    if recorded_plan is not None:
        recorded_hash = canonical_sha256(recorded_plan)
        comparison = {
            "exact_match": generated_hash == recorded_hash,
            "recorded_sha256": recorded_hash,
        }
    record: dict[str, Any] = {
        "based_on_seq": world.seq,
        "generated_at_source": generated_at_source,
        "generated_sha256": generated_hash,
        "schema": OUTPUT_SCHEMA,
        "seed_source": seed_source,
        "type": "plan",
        "version": OUTPUT_VERSION,
    }
    if comparison is not None:
        record["comparison"] = comparison
    if not compare_only:
        record["plan"] = plan
    return record


def _record_plan_stats(summary: RegenerationSummary, record: Mapping[str, Any]) -> None:
    comparison = record.get("comparison")
    if not isinstance(comparison, dict):
        return
    summary.plans_compared += 1
    if comparison.get("exact_match") is True:
        summary.exact_matches += 1
    else:
        summary.mismatches += 1


def _regenerate_one(
    backend: MotionBackend,
    event: TraceEvent,
    recorded_plan: Mapping[str, Any] | None,
    session_seed: int | None,
    compare_only: bool,
) -> tuple[dict[str, Any], WorldState, str | None, str | None]:
    world, _ = _world_from_event(event)
    seed, seed_source = _select_seed(world, recorded_plan, session_seed)
    generated_at, generated_at_source = _select_generated_at(world, event, recorded_plan)
    generated = backend.generate(world, seed, generated_at).to_payload()
    generated_plan_id = generated.get("plan_id") if isinstance(generated.get("plan_id"), str) else None
    recorded_plan_id = None
    if recorded_plan is not None and isinstance(recorded_plan.get("plan_id"), str):
        # Trace plan ids are episode-anonymized (plan-0, plan-1, ...), while
        # the backend necessarily creates its internal id before the recorder
        # sees it. Compare in the recorded identity frame and retain a reverse
        # map for subsequent cancel events.
        recorded_plan_id = str(recorded_plan["plan_id"])
        generated["plan_id"] = recorded_plan_id
    return (
        _output_record(
            world,
            generated,
            recorded_plan,
            seed_source=seed_source,
            generated_at_source=generated_at_source,
            compare_only=compare_only,
        ),
        world,
        generated_plan_id,
        recorded_plan_id,
    )


def regenerate(
    input_path: str | Path,
    output: TextIO,
    *,
    backend: MotionBackend | None = None,
    compare_only: bool = False,
    session_seed: int | None = None,
) -> RegenerationSummary:
    """Regenerate an episode into canonical JSONL and return its summary.

    ``backend`` is injectable so a future model can be evaluated against a
    recorded procedural baseline without changing the trace reader.
    """

    source = resolve_source(input_path)
    checkpoint = validate_checkpoint(source)
    selected_backend = backend or AutoregressiveMotionBackend()
    has_recorded_plans, inferred_timing, plan_count = _scan_source(source)
    timing = _timing_from_manifest(source.manifest) or inferred_timing
    if timing is not None:
        try:
            selected_backend.configure_timing(timing[0], timing[1])
        except (NotImplementedError, ValueError) as exc:
            raise RegenerationError("backend does not support the recorded timing") from exc

    effective_session_seed = session_seed if session_seed is not None else _manifest_session_seed(source.manifest)
    deterministic = checkpoint.deterministic
    reasons: list[str] = []
    if not checkpoint.deterministic:
        reasons.append(checkpoint.status)
    elif checkpoint.declared:
        loaded_checkpoint = _backend_checkpoint_sha256(selected_backend)
        if loaded_checkpoint != checkpoint.expected_sha256:
            # Validating a file is not the same as loading it. The procedural
            # backend has no checkpoint loader, and must never be presented as
            # a deterministic substitute for a recorded learned model.
            deterministic = False
            reasons.append("checkpoint_not_loaded_by_backend")
    recorded_backend = _determinism_metadata(source.manifest).get("backend")
    if isinstance(recorded_backend, str) and recorded_backend and recorded_backend != selected_backend.name:
        deterministic = False
        reasons.append("backend_mismatch")

    summary = RegenerationSummary(
        backend=selected_backend.name,
        deterministic=deterministic,
        nondeterministic_reasons=reasons,
        plans_recorded=plan_count,
    )
    writer = _OutputWriter(output)
    writer.write(
        {
            "backend": selected_backend.name,
            "checkpoint": checkpoint.public_payload(),
            "deterministic": deterministic,
            "nondeterministic_reasons": sorted(set(reasons)),
            "schema": OUTPUT_SCHEMA,
            "type": "metadata",
            "version": OUTPUT_VERSION,
        }
    )

    pending: dict[int, TraceEvent] = {}
    recorded_to_backend_plan_id: dict[str, str] = {}
    last_plan_seq = -1
    try:
        for event in iter_events(source):
            if event.kind == "world_state":
                summary.worlds_seen += 1
                if has_recorded_plans:
                    world, _ = _world_from_event(event)
                    pending[world.seq] = event
                else:
                    record, _, _, _ = _regenerate_one(
                        selected_backend,
                        event,
                        None,
                        effective_session_seed,
                        compare_only,
                    )
                    writer.write(record)
                    summary.worlds_regenerated += 1
                    _record_plan_stats(summary, record)
                continue

            if event.kind == "plan_received" and has_recorded_plans:
                recorded = _plan_from_event(event)
                based_on_seq = _plan_based_on_seq(recorded, event)
                world_event = pending.pop(based_on_seq, None)
                stale = [seq for seq in pending if last_plan_seq < seq < based_on_seq]
                for seq in stale:
                    pending.pop(seq, None)
                    summary.worlds_without_plan += 1
                last_plan_seq = max(last_plan_seq, based_on_seq)
                if world_event is None:
                    summary.worlds_without_plan += 1
                    continue
                record, _, backend_plan_id, recorded_plan_id = _regenerate_one(
                    selected_backend,
                    world_event,
                    recorded,
                    effective_session_seed,
                    compare_only,
                )
                writer.write(record)
                if backend_plan_id is not None and recorded_plan_id is not None:
                    recorded_to_backend_plan_id[recorded_plan_id] = backend_plan_id
                summary.worlds_regenerated += 1
                _record_plan_stats(summary, record)
                continue

            if event.kind == "cancel":
                summary.cancellations_seen += 1
                plan_id = event.payload.get("plan_id")
                selected_id = plan_id if isinstance(plan_id, str) and plan_id else None
                if selected_id is not None:
                    selected_id = recorded_to_backend_plan_id.get(selected_id, selected_id)
                if selected_backend.cancel(selected_id):
                    summary.cancellations_applied += 1

        if has_recorded_plans:
            summary.worlds_without_plan += len(pending)
        summary.result_sha256 = writer.hexdigest()
        writer.write(
            {
                "schema": OUTPUT_SCHEMA,
                "summary": summary.to_payload(),
                "type": "summary",
                "version": OUTPUT_VERSION,
            },
            include_in_digest=False,
        )
        output.flush()
        return summary
    finally:
        selected_backend.close()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Regenerate PET motion plans using trace time; never starts Electron or sleeps.",
    )
    parser.add_argument("input", help="episode directory, trace shard, or exported world-state JSONL")
    parser.add_argument("--output", help="write canonical JSONL to this file instead of stdout")
    parser.add_argument(
        "--compare-only",
        action="store_true",
        help="emit plan hashes and exact-match results without embedding generated plan payloads",
    )
    parser.add_argument(
        "--session-seed",
        type=lambda text: int(text, 0),
        help="fallback uint32 session seed when neither plans nor world states contain a seed",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.session_seed is not None and not 0 <= args.session_seed <= 0xFFFF_FFFF:
        parser.error("--session-seed must be a uint32")

    stream: TextIO | None = None
    try:
        if args.output:
            stream = Path(args.output).open("w", encoding="utf-8", newline="\n")
        else:
            stream = sys.stdout
        regenerate(
            args.input,
            stream,
            compare_only=args.compare_only,
            session_seed=args.session_seed,
        )
        return 0
    except RegenerationError as exc:
        print(f"regeneration failed: {exc}", file=sys.stderr)
        return 2
    except OSError:
        # Avoid echoing a potentially identifying absolute output path.
        print("regeneration failed: output could not be written", file=sys.stderr)
        return 2
    finally:
        if stream is not None and stream is not sys.stdout:
            stream.close()


if __name__ == "__main__":
    raise SystemExit(main())
