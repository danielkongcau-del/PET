"""Versioned NDJSON protocol primitives.

Only this module serializes messages written to stdout.  Keeping the writer in
one place makes it much harder for a diagnostic print to corrupt the stream.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import sys
import threading
import time
from typing import Any, Mapping, TextIO


def _load_shared_protocol_validator():
    """Load the canonical workspace validator without a pip installation."""

    try:
        from pet_protocol import ProtocolValidationError, validate_message

        return ProtocolValidationError, validate_message
    except ImportError:
        workspace_package = Path(__file__).resolve().parents[3] / "packages" / "protocol" / "python"
        if workspace_package.is_dir():
            package_path = str(workspace_package)
            if package_path not in sys.path:
                sys.path.insert(0, package_path)
        try:
            from pet_protocol import ProtocolValidationError, validate_message

            return ProtocolValidationError, validate_message
        except ImportError as exc:
            raise RuntimeError(
                "canonical pet_protocol validator is unavailable; run from the PET workspace"
            ) from exc


SharedProtocolValidationError, validate_shared_message = _load_shared_protocol_validator()


PROTOCOL_NAME = "pet-motion"
PROTOCOL_VERSION = 1
MAX_LINE_BYTES = 1_048_576
MESSAGE_TYPES = frozenset(
    {"hello", "ready", "world_state", "horizon_plan", "cancel", "ping", "pong", "metrics", "error"}
)


class ProtocolError(ValueError):
    """A recoverable malformed-message error."""

    def __init__(self, code: str, message: str, *, request_seq: int | None = None):
        super().__init__(message)
        self.code = code
        self.request_seq = request_seq


@dataclass(frozen=True, slots=True)
class Envelope:
    protocol: str
    version: int
    type: str
    seq: int
    timestamp_ms: int
    payload: Mapping[str, Any]


def unix_time_ms() -> int:
    return time.time_ns() // 1_000_000


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _require_int(value: object, field: str, *, minimum: int | None = None) -> int:
    if not _is_int(value):
        raise ProtocolError("invalid_envelope", f"{field} must be an integer")
    result = int(value)
    if minimum is not None and result < minimum:
        raise ProtocolError("invalid_envelope", f"{field} must be >= {minimum}")
    return result


def decode_line(line: str) -> Envelope:
    if not isinstance(line, str):
        raise ProtocolError("invalid_json", "input line must be text")
    if len(line.encode("utf-8", errors="replace")) > MAX_LINE_BYTES:
        raise ProtocolError("message_too_large", f"message exceeds {MAX_LINE_BYTES} bytes")
    if not line.strip():
        raise ProtocolError("empty_message", "empty input line")

    try:
        raw = json.loads(line)
    except json.JSONDecodeError as exc:
        raise ProtocolError("invalid_json", f"invalid JSON at column {exc.colno}") from exc
    except (ValueError, OverflowError, RecursionError) as exc:
        raise ProtocolError("invalid_json", "invalid JSON value or nesting") from exc
    if not isinstance(raw, dict):
        raise ProtocolError("invalid_envelope", "message must be a JSON object")

    request_seq = raw.get("seq") if _is_int(raw.get("seq")) else None
    if raw.get("protocol") != PROTOCOL_NAME:
        raise ProtocolError(
            "unsupported_protocol",
            f"protocol must be {PROTOCOL_NAME!r}",
            request_seq=request_seq,
        )
    if raw.get("version") != PROTOCOL_VERSION:
        raise ProtocolError(
            "unsupported_version",
            f"only protocol version {PROTOCOL_VERSION} is supported",
            request_seq=request_seq,
        )

    message_type = raw.get("type")
    if not isinstance(message_type, str) or not message_type or len(message_type) > 64:
        raise ProtocolError("invalid_envelope", "type must be a non-empty string", request_seq=request_seq)
    if message_type not in MESSAGE_TYPES:
        raise ProtocolError("unsupported_message_type", f"unsupported message type {message_type!r}", request_seq=request_seq)
    seq = _require_int(raw.get("seq"), "seq", minimum=0)
    timestamp_ms = _require_int(raw.get("timestamp_ms"), "timestamp_ms", minimum=0)
    payload = raw.get("payload")
    if not isinstance(payload, dict):
        raise ProtocolError("invalid_envelope", "payload must be an object", request_seq=seq)

    try:
        validate_shared_message(raw)
    except SharedProtocolValidationError as exc:
        # The canonical validator reports JSON paths and rule names, never the
        # rejected values, so returning this message cannot echo screen data.
        raise ProtocolError("invalid_message", str(exc), request_seq=seq) from exc

    return Envelope(
        protocol=PROTOCOL_NAME,
        version=PROTOCOL_VERSION,
        type=message_type,
        seq=seq,
        timestamp_ms=timestamp_ms,
        payload=payload,
    )


class ProtocolWriter:
    """Thread-safe, sequence-owning NDJSON writer."""

    def __init__(self, stream: TextIO):
        self._stream = stream
        self._seq = 0
        self._lock = threading.Lock()

    @property
    def next_seq(self) -> int:
        return self._seq

    def send(self, message_type: str, payload: Mapping[str, Any]) -> int:
        with self._lock:
            seq = self._seq
            envelope = {
                "protocol": PROTOCOL_NAME,
                "version": PROTOCOL_VERSION,
                "type": message_type,
                "seq": seq,
                "timestamp_ms": unix_time_ms(),
                "payload": dict(payload),
            }
            validate_shared_message(envelope)
            # allow_nan=False prevents non-standard JSON from leaking into IPC.
            encoded = json.dumps(
                envelope,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
            )
            self._stream.write(encoded + "\n")
            self._stream.flush()
            self._seq += 1
            return seq


def finite_number(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProtocolError("invalid_world_state", f"{field} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ProtocolError("invalid_world_state", f"{field} must be finite")
    return result


def optional_finite_number(value: object, default: float, field: str) -> float:
    if value is None:
        return default
    return finite_number(value, field)
