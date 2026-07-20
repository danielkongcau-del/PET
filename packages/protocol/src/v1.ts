export const PROTOCOL_NAME = "pet-motion" as const;
export const PROTOCOL_VERSION = 1 as const;

/** Capability string for skeletal motion rendering with per-bone rotations. */
export const CAP_SKELETAL_MOTION = "skeletal_motion" as const;

/** Capability string for 3D skeletal motion with local quaternion rotations. */
export const CAP_SKELETAL_MOTION_3D = "skeletal_motion_3d_local_quat" as const;

/** 3-component vector [x, y, z]. */
export type Vec3 = [number, number, number];

/** Normalized quaternion [x, y, z, w]. */
export type Quat = [number, number, number, number];

export type MessageType =
  | "hello"
  | "ready"
  | "world_state"
  | "horizon_plan"
  | "cancel"
  | "ping"
  | "pong"
  | "metrics"
  | "error";

export interface Envelope<TType extends MessageType = MessageType, TPayload = unknown> {
  protocol: typeof PROTOCOL_NAME;
  version: typeof PROTOCOL_VERSION;
  type: TType;
  /** Monotonic within one sender. Host and generator own separate sequences. */
  seq: number;
  /** Unix epoch milliseconds. */
  timestamp_ms: number;
  payload: TPayload;
}

export interface Rect {
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface RuntimeInfo {
  name: string;
  version: string;
  pid: number;
  python_version?: string;
  torch_version?: string;
  device?: string;
  /** SHA-256 of the canonical skeleton definition (cat-skeleton.json) this generator was trained with. */
  skeleton_sha256?: string;
}

export interface PrivacyConfig {
  screen_capture_enabled: boolean;
  keyboard_enabled: boolean;
  recording_enabled: boolean;
}

export interface HelloPayload {
  session_id: string;
  host: RuntimeInfo;
  requested_version: 1;
  capabilities: string[];
  config: {
    world_state_hz: number;
    plan_horizon_ms: number;
    plan_dt_ms: number;
    pet_width: number;
    pet_height: number;
    privacy: PrivacyConfig;
  };
}

export interface ReadyPayload {
  session_id: string;
  generator: RuntimeInfo;
  accepted_version: 1;
  capabilities: string[];
  ready_at_ms: number;
}

export interface DisplayState {
  id: string;
  bounds: Rect;
  work_area: Rect;
  scale_factor: number;
  is_primary: boolean;
}

export interface WindowState {
  id: string;
  display_id: string;
  bounds: Rect;
  z_order: number;
  visible: boolean;
  minimized: boolean;
  maximized: boolean;
  fullscreen: boolean;
  active: boolean;
  occluded: boolean;
  eligible: boolean;
}

export type SurfaceKind = "window_top" | "work_area_floor";

export interface SurfaceState {
  id: string;
  kind: SurfaceKind;
  display_id: string;
  window_id?: string;
  x1: number;
  x2: number;
  y: number;
  enabled: boolean;
  occluded: boolean;
  vx?: number;
  vy?: number;
}

export type Behavior =
  | "idle"
  | "walk"
  | "jump"
  | "click_reaction"
  | "landing"
  | "falling"
  | "hidden"
  | "fallback";

export interface PetState {
  /** Top-left of the 96x96 logical pet window in physical screen pixels. */
  x: number;
  y: number;
  width: number;
  height: number;
  /** Host-authoritative collision anchor. */
  foot_x: number;
  foot_y: number;
  vx: number;
  vy: number;
  facing: -1 | 1;
  behavior: Behavior;
  visible: boolean;
  user_dragging: boolean;
  surface_id?: string;
}

export interface CursorState {
  x: number;
  y: number;
  left_down: boolean;
  right_down: boolean;
  middle_down: boolean;
  over_pet: boolean;
}

export interface ClickEvent {
  id: string;
  button: "left" | "right" | "middle";
  x: number;
  y: number;
  target: "pet" | "desktop" | "window";
  timestamp_ms: number;
}

export interface SceneState {
  fullscreen_active: boolean;
  pet_allowed: boolean;
  suspend_reason?: "fullscreen" | "system_ui" | "secure_desktop" | "user_paused";
}

export interface WorldStatePayload {
  session_id: string;
  /** Coordinates crossing the process boundary are always physical screen pixels. */
  coordinate_space: "physical_px";
  displays: DisplayState[];
  windows: WindowState[];
  surfaces: SurfaceState[];
  pet: PetState;
  cursor: CursorState;
  /** Edge-triggered events since the preceding world_state. */
  clicks: ClickEvent[];
  scene: SceneState;
  /** Optional deterministic override for replay and integration tests. */
  seed?: number;
}

export interface PlanTarget {
  surface_id: string;
  foot_x: number;
  foot_y: number;
}

export interface PlanPoint {
  /** Offset from the start of this horizon. */
  t_ms: number;
  /** Position offset from the based_on_seq foot anchor, not an absolute root position. */
  dx: number;
  dy: number;
  vx: number;
  vy: number;
  facing: -1 | 1;
  lean: number;
  squash: number;
  bob: number;
  expression: string;
  /** Optional: per-bone rotation angles in radians (v1 planar, legacy).
   *  Mutually exclusive with root_translation / root_rotation / local_rotation_deltas. */
  bone_rotations?: number[];
  /** Optional: continuous facial parameters. */
  facial_params?: FacialParams;

  // ── 3D skeletal pose (v2) ──
  /** Root translation in plan-frame coordinates (replaces dx/dy for 3D). */
  root_translation?: Vec3;
  /** Root orientation as normalized quaternion [x,y,z,w]. */
  root_rotation?: Quat;
  /** Per-joint local rotation deltas (quaternions), in skeleton joint order.
   *  Excludes __motion_root__. Mutually exclusive with bone_rotations. */
  local_rotation_deltas?: Quat[];
}

export interface FacialParams {
  eye_scale: number;    // [0.5, 1.5] — surprised=large, sleepy=small
  eye_squint: number;   // [0, 1]    — annoyed/curious squint
  mouth_open: number;   // [0, 1]    — surprised/happy open mouth
  ear_angle: number;    // [-0.5, 0.5] — scared=flat, curious=perked
  brow_tilt: number;    // [-1, 1]   — annoyed=down, sad=up
}

/** Per-bone rotation angles. Index mapping is defined by the active skeleton. */
export type BoneRotations = number[];

export interface HorizonPlanPayload {
  plan_id: string;
  /** seq of the world_state whose foot anchor defines point dx/dy. */
  based_on_seq: number;
  behavior: Behavior;
  generated_at_ms: number;
  valid_until_ms: number;
  dt_ms: number;
  confidence: number;
  seed: number;
  target?: PlanTarget;
  points: PlanPoint[];
}

export interface CancelPayload {
  /** Omit to cancel every outstanding plan for this session. */
  plan_id?: string;
  based_on_seq?: number;
  reason: "user_drag" | "topology_change" | "safety" | "newer_state" | "shutdown";
  requested_at_ms: number;
}

export interface PingPayload {
  nonce: string;
  sent_at_ms: number;
}

export interface PongPayload {
  nonce: string;
  ping_sent_at_ms: number;
  received_at_ms: number;
}

export interface MetricsPayload {
  source: "host" | "generator";
  window_ms: number;
  gauges: Record<string, number>;
  counters: Record<string, number>;
  labels?: Record<string, string>;
}

export type ScalarDetail = string | number | boolean | null;

export interface ErrorPayload {
  code: string;
  message: string;
  recoverable: boolean;
  related_seq?: number;
  plan_id?: string;
  details?: Record<string, ScalarDetail>;
}

export type HelloMessage = Envelope<"hello", HelloPayload>;
export type ReadyMessage = Envelope<"ready", ReadyPayload>;
export type WorldStateMessage = Envelope<"world_state", WorldStatePayload>;
export type HorizonPlanMessage = Envelope<"horizon_plan", HorizonPlanPayload>;
export type CancelMessage = Envelope<"cancel", CancelPayload>;
export type PingMessage = Envelope<"ping", PingPayload>;
export type PongMessage = Envelope<"pong", PongPayload>;
export type MetricsMessage = Envelope<"metrics", MetricsPayload>;
export type ErrorMessage = Envelope<"error", ErrorPayload>;

export type PetMotionMessage =
  | HelloMessage
  | ReadyMessage
  | WorldStateMessage
  | HorizonPlanMessage
  | CancelMessage
  | PingMessage
  | PongMessage
  | MetricsMessage
  | ErrorMessage;

/** Header guard only. Validate untrusted payloads with the bundled JSON Schema. */
export function isProtocolEnvelope(value: unknown): value is PetMotionMessage {
  if (typeof value !== "object" || value === null || Array.isArray(value)) return false;
  const candidate = value as Record<string, unknown>;
  return (
    candidate.protocol === PROTOCOL_NAME &&
    candidate.version === PROTOCOL_VERSION &&
    typeof candidate.type === "string" &&
    MESSAGE_TYPES.has(candidate.type as MessageType) &&
    typeof candidate.seq === "number" &&
    Number.isSafeInteger(candidate.seq) &&
    candidate.seq >= 0 &&
    typeof candidate.timestamp_ms === "number" &&
    Number.isSafeInteger(candidate.timestamp_ms) &&
    candidate.timestamp_ms >= 0 &&
    typeof candidate.payload === "object" &&
    candidate.payload !== null &&
    !Array.isArray(candidate.payload)
  );
}

export function encodeNdjson<TType extends MessageType, TPayload>(
  message: Envelope<TType, TPayload>,
): string {
  return `${JSON.stringify(message)}\n`;
}

export function decodeNdjsonLine(line: string): PetMotionMessage {
  const value: unknown = JSON.parse(line);
  if (!isProtocolEnvelope(value)) throw new Error("Invalid PET motion protocol envelope");
  return value;
}

const MESSAGE_TYPES = new Set<MessageType>([
  "hello",
  "ready",
  "world_state",
  "horizon_plan",
  "cancel",
  "ping",
  "pong",
  "metrics",
  "error",
]);
