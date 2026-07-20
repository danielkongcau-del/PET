"""Long-running stdio NDJSON service and latest-state inbox."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import platform
import threading
import time
from typing import Any, Mapping, TextIO

try:
    import psutil
except ImportError:  # Optional outside the pinned pet-core environment.
    psutil = None  # type: ignore[assignment]

from . import __version__
from .backend import MotionBackend
from .protocol import Envelope, PROTOCOL_VERSION, ProtocolError, ProtocolWriter, decode_line, unix_time_ms
from .state import parse_world_state


LOGGER = logging.getLogger("pet.generator")


@dataclass(slots=True)
class ServiceMetrics:
    started_monotonic: float = field(default_factory=time.monotonic)
    lines_received: int = 0
    messages_received: int = 0
    parse_errors: int = 0
    world_states_received: int = 0
    world_states_dropped: int = 0
    stale_world_states: int = 0
    plans_generated: int = 0
    plans_cancelled: int = 0
    pings_received: int = 0
    backend_errors: int = 0
    last_world_seq: int = -1
    last_plan_latency_ms: float = 0.0
    max_plan_latency_ms: float = 0.0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def increment(self, name: str, amount: int = 1) -> None:
        with self._lock:
            setattr(self, name, int(getattr(self, name)) + amount)

    def observe_plan(self, seq: int, latency_ms: float) -> None:
        with self._lock:
            self.plans_generated += 1
            self.last_world_seq = seq
            self.last_plan_latency_ms = latency_ms
            self.max_plan_latency_ms = max(self.max_plan_latency_ms, latency_ms)

    def snapshot(self) -> tuple[dict[str, float], dict[str, float]]:
        with self._lock:
            gauges = {
                "uptime_ms": max(0.0, (time.monotonic() - self.started_monotonic) * 1000.0),
                "last_plan_latency_ms": self.last_plan_latency_ms,
                "max_plan_latency_ms": self.max_plan_latency_ms,
                "last_world_seq": float(max(self.last_world_seq, 0)),
            }
            counters = {
                "lines_received": float(self.lines_received),
                "messages_received": float(self.messages_received),
                "parse_errors": float(self.parse_errors),
                "world_states_received": float(self.world_states_received),
                "world_states_dropped": float(self.world_states_dropped),
                "stale_world_states": float(self.stale_world_states),
                "plans_generated": float(self.plans_generated),
                "plans_cancelled": float(self.plans_cancelled),
                "pings_received": float(self.pings_received),
                "backend_errors": float(self.backend_errors),
            }
        return gauges, counters


@dataclass(frozen=True, slots=True)
class _InboundError:
    error: ProtocolError


class _LatestStateInbox:
    """Ordered inbox with capacity one specifically for pending world states.

    Control messages retain order.  A newer pending world state replaces the
    older state in-place, so messages after that state keep their relative
    ordering.  Edge-triggered click events are merged during replacement.
    """

    def __init__(self, metrics: ServiceMetrics):
        self._items: deque[Envelope | _InboundError] = deque()
        self._condition = threading.Condition()
        self._closed = False
        self._metrics = metrics

    def put(self, item: Envelope | _InboundError) -> None:
        with self._condition:
            if isinstance(item, Envelope) and item.type == "world_state":
                new_session = item.payload.get("session_id")
                # Search backwards so a hello or a different-session world
                # state is a hard ordering barrier.  Non-session controls such
                # as ping retain the existing latest-state behavior.
                for index in range(len(self._items) - 1, -1, -1):
                    existing = self._items[index]
                    if not isinstance(existing, Envelope):
                        continue
                    if existing.type == "hello":
                        break
                    if existing.type != "world_state":
                        continue
                    if existing.payload.get("session_id") != new_session:
                        break
                    self._items[index] = self._merge_world_states(existing, item)
                    self._metrics.increment("world_states_dropped")
                    self._condition.notify()
                    return
            self._items.append(item)
            self._condition.notify()

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._condition.notify_all()

    def get(self) -> Envelope | _InboundError | None:
        with self._condition:
            while not self._items and not self._closed:
                self._condition.wait()
            if self._items:
                return self._items.popleft()
            return None

    @staticmethod
    def _merge_world_states(older: Envelope, newer: Envelope) -> Envelope:
        old_clicks = older.payload.get("clicks", [])
        new_clicks = newer.payload.get("clicks", [])
        if not isinstance(old_clicks, list) or not isinstance(new_clicks, list):
            return newer
        combined: list[object] = []
        seen: set[str] = set()
        for click in [*old_clicks, *new_clicks]:
            click_id = click.get("id") if isinstance(click, dict) else None
            key = click_id if isinstance(click_id, str) else repr(click)
            if key in seen:
                continue
            seen.add(key)
            combined.append(click)
        payload = dict(newer.payload)
        payload["clicks"] = combined[-32:]
        return Envelope(
            protocol=newer.protocol,
            version=newer.version,
            type=newer.type,
            seq=newer.seq,
            timestamp_ms=newer.timestamp_ms,
            payload=payload,
        )


class GeneratorService:
    def __init__(
        self,
        backend: MotionBackend,
        *,
        session_seed: int,
        metrics_interval_ms: int = 5_000,
    ):
        self.backend = backend
        self.session_seed = session_seed
        self.metrics_interval_ms = max(0, metrics_interval_ms)
        self.metrics = ServiceMetrics()
        self._session_id: str | None = None
        self._ready = False
        self._last_metrics_monotonic = time.monotonic()
        self._process_metrics = psutil.Process(os.getpid()) if psutil is not None else None
        if self._process_metrics is not None:
            self._process_metrics.cpu_percent(interval=None)

    def run(self, input_stream: TextIO, output_stream: TextIO) -> int:
        writer = ProtocolWriter(output_stream)
        inbox = _LatestStateInbox(self.metrics)
        reader = threading.Thread(
            target=self._read_input,
            args=(input_stream, inbox),
            name="pet-generator-stdin",
            daemon=True,
        )
        reader.start()
        LOGGER.info("generator started backend=%s", self.backend.name)
        try:
            while True:
                item = inbox.get()
                if item is None:
                    break
                if isinstance(item, _InboundError):
                    self._send_error(writer, item.error)
                    continue
                should_continue = self._handle(item, writer)
                if not should_continue:
                    break
        except BrokenPipeError:
            LOGGER.info("stdout pipe closed by host")
        finally:
            self.backend.close()
            LOGGER.info("generator stopped")
        return 0

    def _read_input(self, stream: TextIO, inbox: _LatestStateInbox) -> None:
        try:
            for line in stream:
                self.metrics.increment("lines_received")
                try:
                    envelope = decode_line(line)
                except ProtocolError as exc:
                    self.metrics.increment("parse_errors")
                    LOGGER.warning("rejected input code=%s related_seq=%s", exc.code, exc.request_seq)
                    inbox.put(_InboundError(exc))
                    continue
                self.metrics.increment("messages_received")
                if envelope.type == "world_state":
                    self.metrics.increment("world_states_received")
                inbox.put(envelope)
        except (OSError, UnicodeError) as exc:
            LOGGER.error("stdin reader failed: %s", exc)
            inbox.put(_InboundError(ProtocolError("stdin_error", "generator input stream failed")))
        except MemoryError:
            raise
        except Exception:
            # Unexpected decoder failures must close the stream cleanly and
            # produce a protocol-safe error.  SystemExit/KeyboardInterrupt are
            # BaseException subclasses and deliberately remain uncaught.
            LOGGER.exception("unexpected stdin reader failure")
            inbox.put(_InboundError(ProtocolError("stdin_error", "generator input stream failed")))
        finally:
            inbox.close()

    def _handle(self, envelope: Envelope, writer: ProtocolWriter) -> bool:
        if envelope.type == "hello":
            self._handle_hello(envelope, writer)
            return True
        if not self._ready:
            self._send_error(
                writer,
                ProtocolError("handshake_required", "send hello before other messages", request_seq=envelope.seq),
            )
            return True
        if envelope.type == "world_state":
            self._handle_world_state(envelope, writer)
        elif envelope.type == "cancel":
            self._handle_cancel(envelope)
        elif envelope.type == "ping":
            self._handle_ping(envelope, writer)
        elif envelope.type == "metrics":
            # Host metrics are telemetry input, not a request/response command.
            LOGGER.debug("host metrics received seq=%s", envelope.seq)
        else:
            self._send_error(
                writer,
                ProtocolError(
                    "unexpected_message",
                    f"host must not send {envelope.type!r} to the generator",
                    request_seq=envelope.seq,
                ),
            )
        return True

    def _handle_hello(self, envelope: Envelope, writer: ProtocolWriter) -> None:
        try:
            session_id, horizon_ms, dt_ms = self._validate_hello(envelope)
            self.backend.configure_timing(horizon_ms, dt_ms)
            self.backend.prepare()
        except (NotImplementedError, ValueError) as exc:
            self._send_error(
                writer,
                ProtocolError("unsupported_config", str(exc), request_seq=envelope.seq),
            )
            return
        except ProtocolError as exc:
            self._send_error(writer, exc)
            return
        if self._session_id is not None and session_id != self._session_id:
            self.backend.cancel()
        self._session_id = session_id
        self._ready = True

        # Detect skeletal capability from host hello (prefer 3D over 2D)
        host_caps = envelope.payload.get("capabilities", [])
        if not isinstance(host_caps, list):
            host_caps = []
        skeletal_enabled = "skeletal_motion_3d_local_quat" in host_caps or "skeletal_motion" in host_caps
        skeletal_3d = "skeletal_motion_3d_local_quat" in host_caps
        # set 3D flag first: _load_skeletal_metadata reads _skeletal_3d to pick cat-skeleton-3d.json
        self.backend.set_skeletal_3d(skeletal_3d)
        self.backend.set_skeletal_enabled(skeletal_enabled)
        if not skeletal_enabled:
            LOGGER.info("Host does not advertise skeletal_motion; bone rotations will not be generated.")

        ready_at = unix_time_ms()
        skeleton_sha256 = self._compute_skeleton_sha256()

        writer.send(
            "ready",
            {
                "session_id": session_id,
                "generator": {
                    "name": "pet-python-generator",
                    "version": __version__,
                    "pid": os.getpid(),
                    "python_version": platform.python_version(),
                    "device": "cpu",
                    **({"skeleton_sha256": skeleton_sha256} if skeleton_sha256 else {}),
                },
                "accepted_version": PROTOCOL_VERSION,
                "capabilities": [
                    "autoregressive_motion",
                    "cancel_v1",
                    "click_reaction",
                    "deterministic_seed",
                    "latest_state",
                    "metrics",
                    "metrics_v1",
                    "skeletal_motion",
                    "skeletal_motion_3d_local_quat",
                    "window_top_locomotion",
                    "world_state_v1",
                ],
                "ready_at_ms": ready_at,
            },
        )
        LOGGER.info("handshake ready session=%s skeletal=%s", session_id, skeletal_enabled)

    @staticmethod
    def _validate_hello(envelope: Envelope) -> tuple[str, int, int]:
        payload = envelope.payload
        session_id = payload.get("session_id")
        if not isinstance(session_id, str) or not session_id or len(session_id) > 128:
            raise ProtocolError("invalid_hello", "payload.session_id must be a non-empty string", request_seq=envelope.seq)
        if payload.get("requested_version") != PROTOCOL_VERSION:
            raise ProtocolError("unsupported_version", "hello requested_version must be 1", request_seq=envelope.seq)
        config = payload.get("config")
        if not isinstance(config, dict):
            raise ProtocolError("invalid_hello", "payload.config must be an object", request_seq=envelope.seq)
        dt_ms = config.get("plan_dt_ms")
        horizon_ms = config.get("plan_horizon_ms")
        if not isinstance(dt_ms, int) or isinstance(dt_ms, bool) or not 8 <= dt_ms <= 250:
            raise ProtocolError("invalid_hello", "payload.config.plan_dt_ms is invalid", request_seq=envelope.seq)
        if not isinstance(horizon_ms, int) or isinstance(horizon_ms, bool) or not 50 <= horizon_ms <= 5_000:
            raise ProtocolError("invalid_hello", "payload.config.plan_horizon_ms is invalid", request_seq=envelope.seq)
        return session_id, horizon_ms, dt_ms

    def _handle_world_state(self, envelope: Envelope, writer: ProtocolWriter) -> None:
        if envelope.seq <= self.metrics.last_world_seq:
            self.metrics.increment("stale_world_states")
            self._send_error(
                writer,
                ProtocolError(
                    "stale_world_state",
                    "world_state seq must increase monotonically",
                    request_seq=envelope.seq,
                ),
            )
            return
        started = time.perf_counter()
        try:
            world = parse_world_state(envelope)
            if world.session_id != self._session_id:
                raise ProtocolError(
                    "session_mismatch",
                    "world_state session_id does not match hello",
                    request_seq=envelope.seq,
                )
            seed = self._seed_for(world.session_id, world.seq, world.requested_seed)
            generated_at = unix_time_ms()
            plan = self.backend.generate(world, seed, generated_at)
            writer.send("horizon_plan", plan.to_payload())
        except ProtocolError as exc:
            self._send_error(writer, exc)
            return
        except Exception:
            self.metrics.increment("backend_errors")
            LOGGER.exception("backend failed seq=%s", envelope.seq)
            self._send_error(
                writer,
                ProtocolError("backend_error", "motion backend failed", request_seq=envelope.seq),
            )
            return
        latency_ms = (time.perf_counter() - started) * 1000.0
        self.metrics.observe_plan(envelope.seq, latency_ms)
        LOGGER.debug("plan generated seq=%s latency_ms=%.3f", envelope.seq, latency_ms)
        self._maybe_send_metrics(writer)

    def _seed_for(self, session_id: str, seq: int, requested: int | None) -> int:
        if requested is not None:
            return requested & 0xFFFF_FFFF
        material = f"{self.session_seed}:{session_id}:{seq}".encode("utf-8")
        return int.from_bytes(hashlib.blake2s(material, digest_size=4).digest(), "big")

    def _handle_cancel(self, envelope: Envelope) -> None:
        payload = envelope.payload
        plan_id_value = payload.get("plan_id")
        plan_id = plan_id_value if isinstance(plan_id_value, str) and plan_id_value else None
        if self.backend.cancel(plan_id):
            self.metrics.increment("plans_cancelled")
        LOGGER.debug("cancel received seq=%s reason=%s", envelope.seq, payload.get("reason"))

    def _handle_ping(self, envelope: Envelope, writer: ProtocolWriter) -> None:
        nonce = envelope.payload.get("nonce")
        sent_at = envelope.payload.get("sent_at_ms")
        if not isinstance(nonce, str) or not nonce or len(nonce) > 128:
            self._send_error(
                writer,
                ProtocolError("invalid_ping", "payload.nonce must be a non-empty string", request_seq=envelope.seq),
            )
            return
        if not isinstance(sent_at, int) or isinstance(sent_at, bool) or sent_at < 0:
            self._send_error(
                writer,
                ProtocolError("invalid_ping", "payload.sent_at_ms must be a non-negative integer", request_seq=envelope.seq),
            )
            return
        received_at = unix_time_ms()
        self.metrics.increment("pings_received")
        writer.send(
            "pong",
            {"nonce": nonce, "ping_sent_at_ms": sent_at, "received_at_ms": received_at},
        )

    def _maybe_send_metrics(self, writer: ProtocolWriter) -> None:
        if self.metrics_interval_ms <= 0:
            return
        now = time.monotonic()
        if (now - self._last_metrics_monotonic) * 1000.0 < self.metrics_interval_ms:
            return
        gauges, counters = self.metrics.snapshot()
        if self._process_metrics is not None:
            try:
                memory = self._process_metrics.memory_info()
                gauges["process_rss_bytes"] = float(memory.rss)
                gauges["process_cpu_percent"] = float(self._process_metrics.cpu_percent(interval=None))
                gauges["process_threads"] = float(self._process_metrics.num_threads())
            except (OSError, psutil.Error):
                pass
        backend_metrics = self.backend.metrics()
        if isinstance(backend_metrics.get("active_jump"), bool):
            gauges["backend_active_jump"] = 1.0 if backend_metrics["active_jump"] else 0.0
        for key, value in backend_metrics.items():
            if key == "active_jump" or isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            numeric = float(value)
            if math.isfinite(numeric):
                gauges[f"backend_{key}"] = numeric
        writer.send(
            "metrics",
            {
                "source": "generator",
                "window_ms": self.metrics_interval_ms,
                "gauges": gauges,
                "counters": counters,
                "labels": {"backend": self.backend.name},
            },
        )
        self._last_metrics_monotonic = now

    @staticmethod
    def _compute_skeleton_sha256() -> str | None:
        """Compute SHA-256 of the canonical 3D skeleton definition for capability negotiation."""
        try:
            # Prefer 3D skeleton; fall back to legacy 2D.
            for name in ("cat-skeleton-3d.json", "cat-skeleton.json"):
                skeleton_path = Path(__file__).resolve().parents[3] / "assets" / "pet" / "runtime" / name
                if skeleton_path.is_file():
                    raw = skeleton_path.read_bytes()
                    return hashlib.sha256(raw).hexdigest()
            return None
        except Exception:
            LOGGER.warning("Failed to compute skeleton SHA-256", exc_info=True)
            return None

    @staticmethod
    def _send_error(writer: ProtocolWriter, error: ProtocolError) -> None:
        payload: dict[str, Any] = {
            "code": error.code,
            "message": str(error)[:1024],
            "recoverable": True,
        }
        if error.request_seq is not None:
            payload["related_seq"] = error.request_seq
        writer.send("error", payload)
