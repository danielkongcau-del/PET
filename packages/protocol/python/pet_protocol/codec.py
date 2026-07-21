"""Dependency-free wire codec and strict boundary checks for protocol v1."""

from __future__ import annotations

import json
import math
import time
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Optional, Sequence, Set, Tuple, cast

from .v1 import MessageType, PetMotionMessage, PROTOCOL_NAME, PROTOCOL_VERSION

MESSAGE_TYPES = {
    "hello",
    "ready",
    "world_state",
    "horizon_plan",
    "cancel",
    "ping",
    "pong",
    "metrics",
    "error",
}
BEHAVIORS = {"idle", "walk", "jump", "click_reaction", "landing", "falling", "hidden", "fallback"}
MAX_SAFE_INTEGER = 9_007_199_254_740_991


class ProtocolValidationError(ValueError):
    """Raised when an untrusted NDJSON message violates protocol v1."""


def _fail(path: str, message: str) -> None:
    raise ProtocolValidationError(f"{path}: {message}")


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(path, "expected object")
    return cast(Mapping[str, Any], value)


def _keys(
    value: Mapping[str, Any],
    path: str,
    required: Set[str],
    optional: Optional[Set[str]] = None,
) -> None:
    optional = optional or set()
    missing = required - value.keys()
    if missing:
        _fail(path, f"missing properties {sorted(missing)}")
    extra = value.keys() - required - optional
    if extra:
        _fail(path, f"unexpected properties {sorted(extra)}")


def _str(value: Any, path: str, *, maximum: int = 128) -> str:
    if not isinstance(value, str) or not value or len(value) > maximum:
        _fail(path, f"expected non-empty string of at most {maximum} characters")
    return value


def _bool(value: Any, path: str) -> bool:
    if not isinstance(value, bool):
        _fail(path, "expected boolean")
    return value


def _int(
    value: Any,
    path: str,
    *,
    minimum: int = 0,
    maximum: Optional[int] = MAX_SAFE_INTEGER,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        _fail(path, "expected integer")
    if value < minimum or (maximum is not None and value > maximum):
        _fail(path, "integer out of range")
    return value


def _number(
    value: Any,
    path: str,
    *,
    minimum: Optional[float] = None,
    maximum: Optional[float] = None,
    exclusive_minimum: Optional[float] = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(path, "expected finite number")
    try:
        number = float(value)
    except (OverflowError, ValueError):
        _fail(path, "number is too large")
    if not math.isfinite(number):
        _fail(path, "expected finite number")
    if minimum is not None and number < minimum:
        _fail(path, "number below minimum")
    if maximum is not None and number > maximum:
        _fail(path, "number above maximum")
    if exclusive_minimum is not None and number <= exclusive_minimum:
        _fail(path, "number below exclusive minimum")
    return number


def _enum(value: Any, path: str, choices: Set[Any]) -> Any:
    try:
        valid = value in choices
    except TypeError:
        valid = False
    if not valid:
        _fail(path, f"expected one of {sorted(choices, key=str)}")
    return value


def _array(value: Any, path: str, *, maximum: Optional[int] = None) -> Sequence[Any]:
    if not isinstance(value, list):
        _fail(path, "expected array")
    if maximum is not None and len(value) > maximum:
        _fail(path, f"array exceeds {maximum} items")
    return value


def _vec3(value: Any, path: str) -> None:
    arr = _array(value, path, maximum=3)
    if len(arr) != 3:
        _fail(path, "expected exactly 3 elements")
    for i, item in enumerate(arr):
        _number(item, f"{path}[{i}]")


def _quat(value: Any, path: str) -> None:
    arr = _array(value, path, maximum=4)
    if len(arr) != 4:
        _fail(path, "expected exactly 4 elements")
    for i, item in enumerate(arr):
        _number(item, f"{path}[{i}]", minimum=-1, maximum=1)
    norm = math.hypot(float(arr[0]), float(arr[1]), float(arr[2]), float(arr[3]))
    if norm < 0.001 or abs(norm - 1.0) >= 0.001:
        _fail(path, f"expected unit quaternion, got norm {norm:.6f}")


def _rect(value: Any, path: str) -> None:
    item = _mapping(value, path)
    _keys(item, path, {"x", "y", "width", "height"})
    _number(item["x"], f"{path}.x")
    _number(item["y"], f"{path}.y")
    _number(item["width"], f"{path}.width", minimum=0)
    _number(item["height"], f"{path}.height", minimum=0)


def _runtime(value: Any, path: str) -> None:
    item = _mapping(value, path)
    _keys(item, path, {"name", "version", "pid"}, {"python_version", "torch_version", "device", "skeleton_sha256"})
    _str(item["name"], f"{path}.name")
    _str(item["version"], f"{path}.version", maximum=64)
    _int(item["pid"], f"{path}.pid")
    for key in ("python_version", "torch_version", "device"):
        if key in item:
            _str(item[key], f"{path}.{key}", maximum=128)


def _string_array(value: Any, path: str) -> None:
    items = _array(value, path)
    found = set()
    for index, item in enumerate(items):
        text = _str(item, f"{path}[{index}]")
        if text in found:
            _fail(path, "items must be unique")
        found.add(text)


def _validate_hello(payload: Mapping[str, Any]) -> None:
    _keys(payload, "payload", {"session_id", "host", "requested_version", "capabilities", "config"})
    _str(payload["session_id"], "payload.session_id")
    _runtime(payload["host"], "payload.host")
    if payload["requested_version"] != 1:
        _fail("payload.requested_version", "expected 1")
    _string_array(payload["capabilities"], "payload.capabilities")
    config = _mapping(payload["config"], "payload.config")
    _keys(
        config,
        "payload.config",
        {"world_state_hz", "plan_horizon_ms", "plan_dt_ms", "pet_width", "pet_height", "privacy"},
    )
    _number(config["world_state_hz"], "payload.config.world_state_hz", exclusive_minimum=0, maximum=240)
    _int(config["plan_horizon_ms"], "payload.config.plan_horizon_ms", minimum=50, maximum=5000)
    _int(config["plan_dt_ms"], "payload.config.plan_dt_ms", minimum=8, maximum=250)
    _number(config["pet_width"], "payload.config.pet_width", exclusive_minimum=0)
    _number(config["pet_height"], "payload.config.pet_height", exclusive_minimum=0)
    privacy = _mapping(config["privacy"], "payload.config.privacy")
    _keys(privacy, "payload.config.privacy", {"screen_capture_enabled", "keyboard_enabled", "recording_enabled"})
    for key in privacy:
        _bool(privacy[key], f"payload.config.privacy.{key}")


def _validate_ready(payload: Mapping[str, Any]) -> None:
    _keys(payload, "payload", {"session_id", "generator", "accepted_version", "capabilities", "ready_at_ms"})
    _str(payload["session_id"], "payload.session_id")
    _runtime(payload["generator"], "payload.generator")
    if payload["accepted_version"] != 1:
        _fail("payload.accepted_version", "expected 1")
    _string_array(payload["capabilities"], "payload.capabilities")
    _int(payload["ready_at_ms"], "payload.ready_at_ms")


def _validate_world(payload: Mapping[str, Any]) -> None:
    _keys(
        payload,
        "payload",
        {"session_id", "coordinate_space", "displays", "windows", "surfaces", "pet", "cursor", "clicks", "scene"},
        {"seed"},
    )
    _str(payload["session_id"], "payload.session_id")
    if payload["coordinate_space"] != "physical_px":
        _fail("payload.coordinate_space", "expected physical_px")
    if "seed" in payload:
        _int(payload["seed"], "payload.seed", maximum=4294967295)

    displays = _array(payload["displays"], "payload.displays")
    if not displays:
        _fail("payload.displays", "expected at least one display")
    for index, raw in enumerate(displays):
        path = f"payload.displays[{index}]"
        item = _mapping(raw, path)
        _keys(item, path, {"id", "bounds", "work_area", "scale_factor", "is_primary"})
        _str(item["id"], f"{path}.id")
        _rect(item["bounds"], f"{path}.bounds")
        _rect(item["work_area"], f"{path}.work_area")
        _number(item["scale_factor"], f"{path}.scale_factor", exclusive_minimum=0)
        _bool(item["is_primary"], f"{path}.is_primary")

    for index, raw in enumerate(_array(payload["windows"], "payload.windows")):
        path = f"payload.windows[{index}]"
        item = _mapping(raw, path)
        _keys(
            item,
            path,
            {
                "id",
                "display_id",
                "bounds",
                "z_order",
                "visible",
                "minimized",
                "maximized",
                "fullscreen",
                "active",
                "occluded",
                "eligible",
            },
        )
        _str(item["id"], f"{path}.id")
        _str(item["display_id"], f"{path}.display_id")
        _rect(item["bounds"], f"{path}.bounds")
        _int(item["z_order"], f"{path}.z_order")
        for key in ("visible", "minimized", "maximized", "fullscreen", "active", "occluded", "eligible"):
            _bool(item[key], f"{path}.{key}")

    for index, raw in enumerate(_array(payload["surfaces"], "payload.surfaces")):
        path = f"payload.surfaces[{index}]"
        item = _mapping(raw, path)
        _keys(
            item,
            path,
            {"id", "kind", "display_id", "x1", "x2", "y", "enabled", "occluded"},
            {"window_id", "vx", "vy"},
        )
        _str(item["id"], f"{path}.id")
        _enum(item["kind"], f"{path}.kind", {"window_top", "work_area_floor"})
        _str(item["display_id"], f"{path}.display_id")
        if "window_id" in item:
            _str(item["window_id"], f"{path}.window_id")
        x1 = _number(item["x1"], f"{path}.x1")
        x2 = _number(item["x2"], f"{path}.x2")
        if x2 < x1:
            _fail(path, "x2 must be greater than or equal to x1")
        _number(item["y"], f"{path}.y")
        _bool(item["enabled"], f"{path}.enabled")
        _bool(item["occluded"], f"{path}.occluded")
        for key in ("vx", "vy"):
            if key in item:
                _number(item[key], f"{path}.{key}")

    pet = _mapping(payload["pet"], "payload.pet")
    _keys(
        pet,
        "payload.pet",
        {"x", "y", "width", "height", "foot_x", "foot_y", "vx", "vy", "facing", "behavior", "visible", "user_dragging"},
        {"surface_id"},
    )
    for key in ("x", "y", "foot_x", "foot_y", "vx", "vy"):
        _number(pet[key], f"payload.pet.{key}")
    for key in ("width", "height"):
        _number(pet[key], f"payload.pet.{key}", exclusive_minimum=0)
    _enum(pet["facing"], "payload.pet.facing", {-1, 1})
    _enum(pet["behavior"], "payload.pet.behavior", BEHAVIORS)
    _bool(pet["visible"], "payload.pet.visible")
    _bool(pet["user_dragging"], "payload.pet.user_dragging")
    if "surface_id" in pet:
        _str(pet["surface_id"], "payload.pet.surface_id")

    cursor = _mapping(payload["cursor"], "payload.cursor")
    _keys(cursor, "payload.cursor", {"x", "y", "left_down", "right_down", "middle_down", "over_pet"})
    _number(cursor["x"], "payload.cursor.x")
    _number(cursor["y"], "payload.cursor.y")
    for key in ("left_down", "right_down", "middle_down", "over_pet"):
        _bool(cursor[key], f"payload.cursor.{key}")

    for index, raw in enumerate(_array(payload["clicks"], "payload.clicks", maximum=32)):
        path = f"payload.clicks[{index}]"
        item = _mapping(raw, path)
        _keys(item, path, {"id", "button", "x", "y", "target", "timestamp_ms"})
        _str(item["id"], f"{path}.id")
        _enum(item["button"], f"{path}.button", {"left", "right", "middle"})
        _number(item["x"], f"{path}.x")
        _number(item["y"], f"{path}.y")
        _enum(item["target"], f"{path}.target", {"pet", "desktop", "window"})
        _int(item["timestamp_ms"], f"{path}.timestamp_ms")

    scene = _mapping(payload["scene"], "payload.scene")
    _keys(scene, "payload.scene", {"fullscreen_active", "pet_allowed"}, {"suspend_reason"})
    _bool(scene["fullscreen_active"], "payload.scene.fullscreen_active")
    _bool(scene["pet_allowed"], "payload.scene.pet_allowed")
    if "suspend_reason" in scene:
        _enum(scene["suspend_reason"], "payload.scene.suspend_reason", {"fullscreen", "system_ui", "secure_desktop", "user_paused"})


def _validate_plan(payload: Mapping[str, Any]) -> None:
    _keys(
        payload,
        "payload",
        {"plan_id", "based_on_seq", "behavior", "generated_at_ms", "valid_until_ms", "dt_ms", "confidence", "seed", "points"},
        {"target"},
    )
    _str(payload["plan_id"], "payload.plan_id")
    _int(payload["based_on_seq"], "payload.based_on_seq")
    _enum(payload["behavior"], "payload.behavior", BEHAVIORS)
    generated = _int(payload["generated_at_ms"], "payload.generated_at_ms")
    valid_until = _int(payload["valid_until_ms"], "payload.valid_until_ms")
    if valid_until <= generated:
        _fail("payload.valid_until_ms", "must be later than generated_at_ms")
    dt_ms = _int(payload["dt_ms"], "payload.dt_ms", minimum=8, maximum=250)
    _number(payload["confidence"], "payload.confidence", minimum=0, maximum=1)
    _int(payload["seed"], "payload.seed", maximum=4294967295)
    if "target" in payload:
        target = _mapping(payload["target"], "payload.target")
        _keys(target, "payload.target", {"surface_id", "foot_x", "foot_y"})
        _str(target["surface_id"], "payload.target.surface_id")
        _number(target["foot_x"], "payload.target.foot_x")
        _number(target["foot_y"], "payload.target.foot_y")

    points = _array(payload["points"], "payload.points", maximum=128)
    if not points:
        _fail("payload.points", "expected at least one point")
    previous_t = -1
    for index, raw in enumerate(points):
        path = f"payload.points[{index}]"
        point = _mapping(raw, path)
        _keys(point, path, {"t_ms", "dx", "dy", "vx", "vy", "facing", "lean", "squash", "bob", "expression"},
              {"bone_rotations", "facial_params", "root_translation", "root_rotation", "local_rotation_deltas"})
        t_ms = _int(point["t_ms"], f"{path}.t_ms")
        if index == 0 and t_ms != 0:
            _fail(f"{path}.t_ms", "first point must start at zero")
        if t_ms <= previous_t:
            _fail(f"{path}.t_ms", "point times must increase strictly")
        if index and t_ms - previous_t != dt_ms:
            _fail(f"{path}.t_ms", "point spacing must equal dt_ms")
        previous_t = t_ms
        for key in ("dx", "dy"):
            _number(point[key], f"{path}.{key}", minimum=-16384, maximum=16384)
        for key in ("vx", "vy"):
            _number(point[key], f"{path}.{key}", minimum=-20000, maximum=20000)
        _enum(point["facing"], f"{path}.facing", {-1, 1})
        _number(point["lean"], f"{path}.lean", minimum=-1, maximum=1)
        _number(point["squash"], f"{path}.squash", minimum=0.5, maximum=1.5)
        _number(point["bob"], f"{path}.bob", minimum=-48, maximum=48)
        _str(point["expression"], f"{path}.expression", maximum=32)
        if "bone_rotations" in point:
            rotations = _array(point["bone_rotations"], f"{path}.bone_rotations", maximum=32)
            if not rotations:
                _fail(f"{path}.bone_rotations", "expected at least one entry")
            for ri, value in enumerate(rotations):
                _number(value, f"{path}.bone_rotations[{ri}]", minimum=-3.1416, maximum=3.1416)
        if "facial_params" in point:
            fp = _mapping(point["facial_params"], f"{path}.facial_params")
            _keys(fp, f"{path}.facial_params", set(), {"eye_scale", "eye_squint", "mouth_open", "ear_angle", "brow_tilt"})
            for key, lo, hi in (("eye_scale", 0.5, 1.5), ("eye_squint", 0, 1), ("mouth_open", 0, 1),
                                 ("ear_angle", -0.5, 0.5), ("brow_tilt", -1, 1)):
                if key in fp:
                    _number(fp[key], f"{path}.facial_params.{key}", minimum=lo, maximum=hi)
        has_legacy = "bone_rotations" in point
        has_3d = "root_translation" in point or "root_rotation" in point or "local_rotation_deltas" in point
        if has_legacy and has_3d:
            _fail(path, "bone_rotations and 3D skeletal fields are mutually exclusive")
        if has_3d:
            if "root_translation" not in point or "root_rotation" not in point or "local_rotation_deltas" not in point:
                _fail(path, "3D skeletal fields (root_translation, root_rotation, local_rotation_deltas) must be all-or-none")
        if "root_translation" in point:
            _vec3(point["root_translation"], f"{path}.root_translation")
        if "root_rotation" in point:
            _quat(point["root_rotation"], f"{path}.root_rotation")
        if "local_rotation_deltas" in point:
            deltas = _array(point["local_rotation_deltas"], f"{path}.local_rotation_deltas", maximum=512)
            if not deltas:
                _fail(f"{path}.local_rotation_deltas", "expected at least one entry")
            for di, q in enumerate(deltas):
                _quat(q, f"{path}.local_rotation_deltas[{di}]")


def _validate_cancel(payload: Mapping[str, Any]) -> None:
    _keys(payload, "payload", {"reason", "requested_at_ms"}, {"plan_id", "based_on_seq"})
    if "plan_id" in payload:
        _str(payload["plan_id"], "payload.plan_id")
    if "based_on_seq" in payload:
        _int(payload["based_on_seq"], "payload.based_on_seq")
    _enum(payload["reason"], "payload.reason", {"user_drag", "topology_change", "safety", "newer_state", "shutdown"})
    _int(payload["requested_at_ms"], "payload.requested_at_ms")


def _validate_ping(payload: Mapping[str, Any]) -> None:
    _keys(payload, "payload", {"nonce", "sent_at_ms"})
    _str(payload["nonce"], "payload.nonce")
    _int(payload["sent_at_ms"], "payload.sent_at_ms")


def _validate_pong(payload: Mapping[str, Any]) -> None:
    _keys(payload, "payload", {"nonce", "ping_sent_at_ms", "received_at_ms"})
    _str(payload["nonce"], "payload.nonce")
    _int(payload["ping_sent_at_ms"], "payload.ping_sent_at_ms")
    _int(payload["received_at_ms"], "payload.received_at_ms")


def _number_map(value: Any, path: str) -> None:
    item = _mapping(value, path)
    for key, raw in item.items():
        _str(key, f"{path}.<key>")
        _number(raw, f"{path}.{key}")


def _validate_metrics(payload: Mapping[str, Any]) -> None:
    _keys(payload, "payload", {"source", "window_ms", "gauges", "counters"}, {"labels"})
    _enum(payload["source"], "payload.source", {"host", "generator"})
    _int(payload["window_ms"], "payload.window_ms")
    _number_map(payload["gauges"], "payload.gauges")
    _number_map(payload["counters"], "payload.counters")
    if "labels" in payload:
        labels = _mapping(payload["labels"], "payload.labels")
        for key, value in labels.items():
            _str(key, "payload.labels.<key>")
            _str(value, f"payload.labels.{key}")


def _validate_error(payload: Mapping[str, Any]) -> None:
    _keys(payload, "payload", {"code", "message", "recoverable"}, {"related_seq", "plan_id", "details"})
    _str(payload["code"], "payload.code")
    _str(payload["message"], "payload.message", maximum=1024)
    _bool(payload["recoverable"], "payload.recoverable")
    if "related_seq" in payload:
        _int(payload["related_seq"], "payload.related_seq")
    if "plan_id" in payload:
        _str(payload["plan_id"], "payload.plan_id")
    if "details" in payload:
        details = _mapping(payload["details"], "payload.details")
        for key, value in details.items():
            _str(key, "payload.details.<key>")
            if value is not None and (isinstance(value, (dict, list)) or not isinstance(value, (str, int, float, bool))):
                _fail(f"payload.details.{key}", "expected scalar value")
            if isinstance(value, float) and not math.isfinite(value):
                _fail(f"payload.details.{key}", "expected finite number")


_PAYLOAD_VALIDATORS = {
    "hello": _validate_hello,
    "ready": _validate_ready,
    "world_state": _validate_world,
    "horizon_plan": _validate_plan,
    "cancel": _validate_cancel,
    "ping": _validate_ping,
    "pong": _validate_pong,
    "metrics": _validate_metrics,
    "error": _validate_error,
}


def validate_message(message: Any) -> PetMotionMessage:
    """Validate an already-decoded value and return it with the wire type."""

    envelope = _mapping(message, "message")
    _keys(envelope, "message", {"protocol", "version", "type", "seq", "timestamp_ms", "payload"})
    if envelope["protocol"] != PROTOCOL_NAME:
        _fail("message.protocol", f"expected {PROTOCOL_NAME}")
    if envelope["version"] != PROTOCOL_VERSION:
        _fail("message.version", f"expected {PROTOCOL_VERSION}")
    message_type = _enum(envelope["type"], "message.type", MESSAGE_TYPES)
    _int(envelope["seq"], "message.seq")
    _int(envelope["timestamp_ms"], "message.timestamp_ms")
    payload = _mapping(envelope["payload"], "message.payload")
    _PAYLOAD_VALIDATORS[message_type](payload)
    return cast(PetMotionMessage, message)


def _reject_constant(value: str) -> None:
    raise ProtocolValidationError(f"invalid JSON number {value}")


def decode_ndjson_line(line: str) -> PetMotionMessage:
    """Decode exactly one UTF-8 NDJSON record and validate the complete payload."""

    content = line
    if content.endswith("\n"):
        content = content[:-1]
        if content.endswith("\r"):
            content = content[:-1]
    if "\n" in content or "\r" in content:
        raise ProtocolValidationError("expected exactly one NDJSON record")
    stripped = content.strip()
    if not stripped:
        raise ProtocolValidationError("empty NDJSON record")
    try:
        value = json.loads(stripped, parse_constant=_reject_constant)
    except json.JSONDecodeError as exc:
        raise ProtocolValidationError(f"invalid JSON: {exc.msg}") from exc
    except (ValueError, OverflowError, RecursionError) as exc:
        raise ProtocolValidationError("invalid JSON value or nesting") from exc
    return validate_message(value)


def encode_ndjson(message: Mapping[str, Any]) -> str:
    """Validate and encode one compact UTF-8 NDJSON record."""

    validate_message(message)
    return json.dumps(message, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n"


def make_message(
    message_type: MessageType,
    seq: int,
    payload: Mapping[str, Any],
    *,
    timestamp_ms: Optional[int] = None,
) -> PetMotionMessage:
    """Build and validate a v1 envelope."""

    message: Dict[str, Any] = {
        "protocol": PROTOCOL_NAME,
        "version": PROTOCOL_VERSION,
        "type": message_type,
        "seq": seq,
        "timestamp_ms": int(time.time() * 1000) if timestamp_ms is None else timestamp_ms,
        "payload": dict(payload),
    }
    return validate_message(message)
