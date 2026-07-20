"""Static Python mirror for PET Motion Protocol v1.

The protocol deliberately uses dictionaries on the wire.  TypedDict keeps the
Python generator honest without adding a runtime dependency such as pydantic.
Use :mod:`pet_protocol.codec` at the untrusted process boundary.
"""

from __future__ import annotations

from typing import Dict, List, Literal, Tuple, TypedDict, Union

PROTOCOL_NAME: Literal["pet-motion"] = "pet-motion"
PROTOCOL_VERSION: Literal[1] = 1

MessageType = Literal[
    "hello",
    "ready",
    "world_state",
    "horizon_plan",
    "cancel",
    "ping",
    "pong",
    "metrics",
    "error",
]
Behavior = Literal[
    "idle",
    "walk",
    "jump",
    "click_reaction",
    "landing",
    "falling",
    "hidden",
    "fallback",
]


class Rect(TypedDict):
    x: float
    y: float
    width: float
    height: float


class RuntimeInfoRequired(TypedDict):
    name: str
    version: str
    pid: int


class RuntimeInfo(RuntimeInfoRequired, total=False):
    python_version: str
    torch_version: str
    device: str
    skeleton_sha256: str


class PrivacyConfig(TypedDict):
    screen_capture_enabled: bool
    keyboard_enabled: bool
    recording_enabled: bool


class HelloConfig(TypedDict):
    world_state_hz: float
    plan_horizon_ms: int
    plan_dt_ms: int
    pet_width: float
    pet_height: float
    privacy: PrivacyConfig


class HelloPayload(TypedDict):
    session_id: str
    host: RuntimeInfo
    requested_version: Literal[1]
    capabilities: List[str]
    config: HelloConfig


class ReadyPayload(TypedDict):
    session_id: str
    generator: RuntimeInfo
    accepted_version: Literal[1]
    capabilities: List[str]
    ready_at_ms: int


class DisplayState(TypedDict):
    id: str
    bounds: Rect
    work_area: Rect
    scale_factor: float
    is_primary: bool


class WindowState(TypedDict):
    id: str
    display_id: str
    bounds: Rect
    z_order: int
    visible: bool
    minimized: bool
    maximized: bool
    fullscreen: bool
    active: bool
    occluded: bool
    eligible: bool


class SurfaceStateRequired(TypedDict):
    id: str
    kind: Literal["window_top", "work_area_floor"]
    display_id: str
    x1: float
    x2: float
    y: float
    enabled: bool
    occluded: bool


class SurfaceState(SurfaceStateRequired, total=False):
    window_id: str
    vx: float
    vy: float


class PetStateRequired(TypedDict):
    x: float
    y: float
    width: float
    height: float
    foot_x: float
    foot_y: float
    vx: float
    vy: float
    facing: Literal[-1, 1]
    behavior: Behavior
    visible: bool
    user_dragging: bool


class PetState(PetStateRequired, total=False):
    surface_id: str


class CursorState(TypedDict):
    x: float
    y: float
    left_down: bool
    right_down: bool
    middle_down: bool
    over_pet: bool


class ClickEvent(TypedDict):
    id: str
    button: Literal["left", "right", "middle"]
    x: float
    y: float
    target: Literal["pet", "desktop", "window"]
    timestamp_ms: int


class SceneStateRequired(TypedDict):
    fullscreen_active: bool
    pet_allowed: bool


class SceneState(SceneStateRequired, total=False):
    suspend_reason: Literal["fullscreen", "system_ui", "secure_desktop", "user_paused"]


class WorldStatePayloadRequired(TypedDict):
    session_id: str
    coordinate_space: Literal["physical_px"]
    displays: List[DisplayState]
    windows: List[WindowState]
    surfaces: List[SurfaceState]
    pet: PetState
    cursor: CursorState
    clicks: List[ClickEvent]
    scene: SceneState


class WorldStatePayload(WorldStatePayloadRequired, total=False):
    seed: int


class PlanTarget(TypedDict):
    surface_id: str
    foot_x: float
    foot_y: float


# Legacy planar skeletal capability
CAP_SKELETAL_MOTION: Literal["skeletal_motion"] = "skeletal_motion"
# 3D skeletal capability with local quaternion rotations
CAP_SKELETAL_MOTION_3D: Literal["skeletal_motion_3d_local_quat"] = "skeletal_motion_3d_local_quat"

Vec3 = Tuple[float, float, float]
Quat = Tuple[float, float, float, float]


class FacialParams(TypedDict, total=False):
    eye_scale: float
    eye_squint: float
    mouth_open: float
    ear_angle: float
    brow_tilt: float


class PlanPointRequired(TypedDict):
    """Plan point fields that are always required."""
    t_ms: int
    dx: float
    dy: float
    vx: float
    vy: float
    facing: Literal[-1, 1]
    lean: float
    squash: float
    bob: float
    expression: str


class PlanPoint(PlanPointRequired, total=False):
    """Full plan point including optional skeletal pose fields."""
    bone_rotations: List[float]
    facial_params: FacialParams
    # ── 3D skeletal pose (v2) ──
    root_translation: Vec3
    root_rotation: Quat
    local_rotation_deltas: List[Quat]


class HorizonPlanPayloadRequired(TypedDict):
    plan_id: str
    based_on_seq: int
    behavior: Behavior
    generated_at_ms: int
    valid_until_ms: int
    dt_ms: int
    confidence: float
    seed: int
    points: List[PlanPoint]


class HorizonPlanPayload(HorizonPlanPayloadRequired, total=False):
    target: PlanTarget


class CancelPayloadRequired(TypedDict):
    reason: Literal["user_drag", "topology_change", "safety", "newer_state", "shutdown"]
    requested_at_ms: int


class CancelPayload(CancelPayloadRequired, total=False):
    plan_id: str
    based_on_seq: int


class PingPayload(TypedDict):
    nonce: str
    sent_at_ms: int


class PongPayload(TypedDict):
    nonce: str
    ping_sent_at_ms: int
    received_at_ms: int


class MetricsPayloadRequired(TypedDict):
    source: Literal["host", "generator"]
    window_ms: int
    gauges: Dict[str, float]
    counters: Dict[str, float]


class MetricsPayload(MetricsPayloadRequired, total=False):
    labels: Dict[str, str]


ScalarDetail = Union[str, int, float, bool, None]


class ErrorPayloadRequired(TypedDict):
    code: str
    message: str
    recoverable: bool


class ErrorPayload(ErrorPayloadRequired, total=False):
    related_seq: int
    plan_id: str
    details: Dict[str, ScalarDetail]


class EnvelopeBase(TypedDict):
    protocol: Literal["pet-motion"]
    version: Literal[1]
    seq: int
    timestamp_ms: int


class Envelope(EnvelopeBase):
    """Loosely discriminated envelope for generic queue/codec code."""

    type: MessageType
    payload: object


class HelloMessage(EnvelopeBase):
    type: Literal["hello"]
    payload: HelloPayload


class ReadyMessage(EnvelopeBase):
    type: Literal["ready"]
    payload: ReadyPayload


class WorldStateMessage(EnvelopeBase):
    type: Literal["world_state"]
    payload: WorldStatePayload


class HorizonPlanMessage(EnvelopeBase):
    type: Literal["horizon_plan"]
    payload: HorizonPlanPayload


class CancelMessage(EnvelopeBase):
    type: Literal["cancel"]
    payload: CancelPayload


class PingMessage(EnvelopeBase):
    type: Literal["ping"]
    payload: PingPayload


class PongMessage(EnvelopeBase):
    type: Literal["pong"]
    payload: PongPayload


class MetricsMessage(EnvelopeBase):
    type: Literal["metrics"]
    payload: MetricsPayload


class ErrorMessage(EnvelopeBase):
    type: Literal["error"]
    payload: ErrorPayload
PetMotionMessage = Union[
    HelloMessage,
    ReadyMessage,
    WorldStateMessage,
    HorizonPlanMessage,
    CancelMessage,
    PingMessage,
    PongMessage,
    MetricsMessage,
    ErrorMessage,
]
