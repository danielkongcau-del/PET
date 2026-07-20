"""Validated, backend-facing world-state representation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .protocol import Envelope, ProtocolError, finite_number, optional_finite_number


@dataclass(frozen=True, slots=True)
class DisplayState:
    id: str
    x: float
    y: float
    width: float
    height: float
    work_x: float
    work_y: float
    work_width: float
    work_height: float
    scale_factor: float


@dataclass(frozen=True, slots=True)
class SurfaceState:
    id: str
    kind: str
    display_id: str
    window_id: str | None
    x1: float
    x2: float
    y: float
    enabled: bool
    occluded: bool

    @property
    def width(self) -> float:
        return self.x2 - self.x1


@dataclass(frozen=True, slots=True)
class PetState:
    x: float
    y: float
    width: float
    height: float
    foot_x: float
    foot_y: float
    vx: float
    vy: float
    facing: int
    behavior: str
    visible: bool
    user_dragging: bool
    surface_id: str | None


@dataclass(frozen=True, slots=True)
class ClickState:
    id: str
    button: str
    x: float
    y: float
    target: str
    timestamp_ms: int


@dataclass(frozen=True, slots=True)
class CursorState:
    x: float
    y: float
    left_down: bool


@dataclass(frozen=True, slots=True)
class SceneState:
    fullscreen_active: bool
    pet_allowed: bool
    suspend_reason: str | None


@dataclass(frozen=True, slots=True)
class WorldState:
    seq: int
    timestamp_ms: int
    session_id: str
    coordinate_space: str
    displays: tuple[DisplayState, ...]
    surfaces: tuple[SurfaceState, ...]
    pet: PetState
    cursor: CursorState | None
    clicks: tuple[ClickState, ...]
    scene: SceneState
    requested_seed: int | None


def _mapping(value: object, field: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ProtocolError("invalid_world_state", f"{field} must be an object")
    return value


def _sequence(value: object, field: str) -> Sequence[Any]:
    if not isinstance(value, list):
        raise ProtocolError("invalid_world_state", f"{field} must be an array")
    return value


def _text(value: object, field: str, *, default: str | None = None) -> str:
    if value is None and default is not None:
        return default
    if not isinstance(value, str) or not value or len(value) > 512:
        raise ProtocolError("invalid_world_state", f"{field} must be a non-empty string")
    return value


def _boolean(value: object, field: str, default: bool) -> bool:
    if value is None:
        return default
    if not isinstance(value, bool):
        raise ProtocolError("invalid_world_state", f"{field} must be a boolean")
    return value


def _rect(raw: Mapping[str, Any], field: str) -> tuple[float, float, float, float]:
    x = finite_number(raw.get("x"), f"{field}.x")
    y = finite_number(raw.get("y"), f"{field}.y")
    width = finite_number(raw.get("width"), f"{field}.width")
    height = finite_number(raw.get("height"), f"{field}.height")
    if width <= 0 or height <= 0:
        raise ProtocolError("invalid_world_state", f"{field} dimensions must be positive")
    return x, y, width, height


def _parse_displays(raw_displays: object) -> tuple[DisplayState, ...]:
    result: list[DisplayState] = []
    for index, item in enumerate(_sequence(raw_displays, "payload.displays")):
        raw = _mapping(item, f"payload.displays[{index}]")
        bounds_raw = raw.get("bounds", raw)
        bounds = _rect(_mapping(bounds_raw, f"payload.displays[{index}].bounds"), f"payload.displays[{index}].bounds")
        work_raw = raw.get("work_area", raw.get("workArea", bounds_raw))
        work = _rect(_mapping(work_raw, f"payload.displays[{index}].work_area"), f"payload.displays[{index}].work_area")
        scale = optional_finite_number(raw.get("scale_factor", raw.get("scaleFactor")), 1.0, f"payload.displays[{index}].scale_factor")
        if scale <= 0:
            raise ProtocolError("invalid_world_state", f"payload.displays[{index}].scale_factor must be positive")
        result.append(
            DisplayState(
                id=_text(raw.get("id"), f"payload.displays[{index}].id"),
                x=bounds[0],
                y=bounds[1],
                width=bounds[2],
                height=bounds[3],
                work_x=work[0],
                work_y=work[1],
                work_width=work[2],
                work_height=work[3],
                scale_factor=scale,
            )
        )
    if not result:
        raise ProtocolError("invalid_world_state", "payload.displays must contain at least one display")
    return tuple(result)


def _parse_surfaces(raw_surfaces: object) -> tuple[SurfaceState, ...]:
    result: list[SurfaceState] = []
    seen: set[str] = set()
    for index, item in enumerate(_sequence(raw_surfaces, "payload.surfaces")):
        raw = _mapping(item, f"payload.surfaces[{index}]")
        prefix = f"payload.surfaces[{index}]"
        surface_id = _text(raw.get("id"), f"{prefix}.id")
        if surface_id in seen:
            raise ProtocolError("invalid_world_state", f"duplicate surface id {surface_id!r}")
        seen.add(surface_id)
        kind = _text(raw.get("kind"), f"{prefix}.kind")
        if kind not in {"window_top", "work_area_floor"}:
            # Future protocol surface kinds are safe to ignore until a backend
            # explicitly learns how to stand on them.
            continue
        x1 = finite_number(raw.get("x1"), f"{prefix}.x1")
        x2 = finite_number(raw.get("x2"), f"{prefix}.x2")
        if x2 < x1:
            x1, x2 = x2, x1
        if x2 - x1 < 1:
            continue
        window_id_value = raw.get("window_id")
        window_id = None if window_id_value is None else _text(window_id_value, f"{prefix}.window_id")
        result.append(
            SurfaceState(
                id=surface_id,
                kind=kind,
                display_id=_text(raw.get("display_id"), f"{prefix}.display_id"),
                window_id=window_id,
                x1=x1,
                x2=x2,
                y=finite_number(raw.get("y"), f"{prefix}.y"),
                enabled=_boolean(raw.get("enabled"), f"{prefix}.enabled", True),
                occluded=_boolean(raw.get("occluded"), f"{prefix}.occluded", False),
            )
        )
    return tuple(result)


def _parse_pet(value: object) -> PetState:
    raw = _mapping(value, "payload.pet")
    x = finite_number(raw.get("x"), "payload.pet.x")
    y = finite_number(raw.get("y"), "payload.pet.y")
    width = finite_number(raw.get("width"), "payload.pet.width")
    height = finite_number(raw.get("height"), "payload.pet.height")
    if width <= 0 or height <= 0:
        raise ProtocolError("invalid_world_state", "payload.pet dimensions must be positive")
    facing_raw = raw.get("facing", 1)
    if facing_raw not in (-1, 1):
        raise ProtocolError("invalid_world_state", "payload.pet.facing must be -1 or 1")
    surface_value = raw.get("surface_id")
    return PetState(
        x=x,
        y=y,
        width=width,
        height=height,
        foot_x=optional_finite_number(raw.get("foot_x"), x + width / 2, "payload.pet.foot_x"),
        foot_y=optional_finite_number(raw.get("foot_y"), y + height, "payload.pet.foot_y"),
        vx=optional_finite_number(raw.get("vx"), 0.0, "payload.pet.vx"),
        vy=optional_finite_number(raw.get("vy"), 0.0, "payload.pet.vy"),
        facing=int(facing_raw),
        behavior=_text(raw.get("behavior"), "payload.pet.behavior", default="idle"),
        visible=_boolean(raw.get("visible"), "payload.pet.visible", True),
        user_dragging=_boolean(raw.get("user_dragging"), "payload.pet.user_dragging", False),
        surface_id=None if surface_value is None else _text(surface_value, "payload.pet.surface_id"),
    )


def _parse_cursor(value: object) -> CursorState | None:
    if value is None:
        return None
    raw = _mapping(value, "payload.cursor")
    buttons = raw.get("buttons")
    left_down = False
    if isinstance(buttons, dict):
        left_down = bool(buttons.get("left", False))
    elif isinstance(raw.get("left_down"), bool):
        left_down = bool(raw["left_down"])
    return CursorState(
        x=finite_number(raw.get("x"), "payload.cursor.x"),
        y=finite_number(raw.get("y"), "payload.cursor.y"),
        left_down=left_down,
    )


def _parse_scene(value: object) -> SceneState:
    raw = _mapping(value, "payload.scene")
    reason_value = raw.get("suspend_reason")
    reason = None if reason_value is None else _text(reason_value, "payload.scene.suspend_reason")
    return SceneState(
        fullscreen_active=_boolean(raw.get("fullscreen_active"), "payload.scene.fullscreen_active", False),
        pet_allowed=_boolean(raw.get("pet_allowed"), "payload.scene.pet_allowed", True),
        suspend_reason=reason,
    )


def _parse_clicks(value: object) -> tuple[ClickState, ...]:
    result: list[ClickState] = []
    for index, item in enumerate(_sequence(value, "payload.clicks")):
        raw = _mapping(item, f"payload.clicks[{index}]")
        prefix = f"payload.clicks[{index}]"
        button = _text(raw.get("button"), f"{prefix}.button")
        if button not in {"left", "right", "middle"}:
            raise ProtocolError("invalid_world_state", f"{prefix}.button is unsupported")
        timestamp = raw.get("timestamp_ms")
        if not isinstance(timestamp, int) or isinstance(timestamp, bool) or timestamp < 0:
            raise ProtocolError("invalid_world_state", f"{prefix}.timestamp_ms must be a non-negative integer")
        result.append(
            ClickState(
                id=_text(raw.get("id"), f"{prefix}.id"),
                button=button,
                x=finite_number(raw.get("x"), f"{prefix}.x"),
                y=finite_number(raw.get("y"), f"{prefix}.y"),
                target=_text(raw.get("target"), f"{prefix}.target"),
                timestamp_ms=timestamp,
            )
        )
    return tuple(result)


def parse_world_state(envelope: Envelope) -> WorldState:
    if envelope.type != "world_state":
        raise ProtocolError("invalid_message_type", "expected a world_state message", request_seq=envelope.seq)
    raw = envelope.payload
    coordinate_space = _text(raw.get("coordinate_space"), "payload.coordinate_space")
    if coordinate_space != "physical_px":
        raise ProtocolError(
            "unsupported_coordinate_space",
            "only physical_px coordinates are supported by protocol v1",
            request_seq=envelope.seq,
        )
    requested_seed_raw = raw.get("seed")
    if requested_seed_raw is not None and (
        not isinstance(requested_seed_raw, int)
        or isinstance(requested_seed_raw, bool)
        or not 0 <= requested_seed_raw <= 0xFFFF_FFFF
    ):
        raise ProtocolError("invalid_world_state", "payload.seed must be a uint32", request_seq=envelope.seq)
    try:
        return WorldState(
            seq=envelope.seq,
            timestamp_ms=envelope.timestamp_ms,
            session_id=_text(raw.get("session_id"), "payload.session_id"),
            coordinate_space=coordinate_space,
            displays=_parse_displays(raw.get("displays")),
            surfaces=_parse_surfaces(raw.get("surfaces")),
            pet=_parse_pet(raw.get("pet")),
            cursor=_parse_cursor(raw.get("cursor")),
            clicks=_parse_clicks(raw.get("clicks", [])),
            scene=_parse_scene(raw.get("scene", {"fullscreen_active": False, "pet_allowed": True})),
            requested_seed=None if requested_seed_raw is None else int(requested_seed_raw),
        )
    except ProtocolError as exc:
        if exc.request_seq is None:
            exc.request_seq = envelope.seq
        raise
