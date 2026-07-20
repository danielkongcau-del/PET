/**
 * Desktop trust-boundary adapter. Wire types and envelope encoding come from
 * the single authoritative @pet/protocol package; only strict runtime payload
 * checks and the 1 MiB stream limit live here.
 */
import {
  PROTOCOL_NAME,
  PROTOCOL_VERSION,
  CAP_SKELETAL_MOTION,
  CAP_SKELETAL_MOTION_3D,
  decodeNdjsonLine,
  encodeNdjson,
  type Behavior,
  type BoneRotations,
  type ClickEvent,
  type DisplayState,
  type Envelope as SharedEnvelope,
  type FacialParams,
  type HelloPayload,
  type HorizonPlanPayload,
  type MessageType,
  type MetricsPayload,
  type PetState,
  type PlanPoint,
  type PlanTarget,
  type Quat,
  type ReadyPayload,
  type Rect,
  type SceneState,
  type SurfaceState,
  type Vec3,
  type WindowState,
  type WorldStatePayload,
} from "@pet/protocol";

export { PROTOCOL_NAME, PROTOCOL_VERSION, CAP_SKELETAL_MOTION, CAP_SKELETAL_MOTION_3D };
export type {
  Behavior,
  BoneRotations,
  ClickEvent,
  DisplayState,
  FacialParams,
  HelloPayload,
  HorizonPlanPayload,
  MessageType,
  MetricsPayload,
  PetState,
  PlanPoint,
  PlanTarget,
  Quat,
  ReadyPayload,
  Rect,
  SceneState,
  SurfaceState,
  Vec3,
  WindowState,
  WorldStatePayload,
};

export type Envelope<T extends MessageType = MessageType, P = unknown> = SharedEnvelope<T, P>;
export const MAX_NDJSON_LINE_BYTES = 1_048_576;

const allowedBehaviors = new Set<Behavior>([
  "idle", "walk", "jump", "click_reaction", "landing", "falling", "hidden", "fallback",
]);

export function createEnvelope<T extends MessageType, P>(type: T, seq: number, payload: P, now = Date.now()): Envelope<T, P> {
  if (!Number.isSafeInteger(seq) || seq < 0) throw new Error("Protocol sequence must be a non-negative safe integer.");
  return { protocol: PROTOCOL_NAME, version: PROTOCOL_VERSION, type, seq, timestamp_ms: now, payload };
}

export function encodeEnvelope(envelope: Envelope<MessageType, unknown>): string {
  const encoded = encodeNdjson(envelope);
  if (Buffer.byteLength(encoded, "utf8") > MAX_NDJSON_LINE_BYTES) throw new Error("message_too_large");
  return encoded;
}

export function decodeEnvelope(line: string): Envelope<MessageType, unknown> {
  if (Buffer.byteLength(line, "utf8") > MAX_NDJSON_LINE_BYTES) throw new Error("message_too_large");
  const message = decodeNdjsonLine(line) as unknown as Envelope<MessageType, unknown>;
  if (!hasOnlyKeys(message as unknown as Record<string, unknown>, ["protocol", "version", "type", "seq", "timestamp_ms", "payload"])) {
    throw new Error("invalid_envelope");
  }
  return message;
}

export function parseReady(envelope: Envelope<MessageType, unknown>): ReadyPayload | null {
  if (envelope.type !== "ready" || !isRecord(envelope.payload)) return null;
  const p = envelope.payload;
  if (
    !hasOnlyKeys(p, ["session_id", "generator", "accepted_version", "capabilities", "ready_at_ms"]) ||
    typeof p.session_id !== "string" ||
    !isRecord(p.generator) ||
    !hasOnlyKeys(p.generator, ["name", "version", "pid", "python_version", "torch_version", "device", "skeleton_sha256"]) ||
    typeof p.generator.name !== "string" ||
    typeof p.generator.version !== "string" ||
    !isNonNegativeInteger(p.generator.pid) ||
    (p.generator.skeleton_sha256 !== undefined && typeof p.generator.skeleton_sha256 !== "string") ||
    p.accepted_version !== PROTOCOL_VERSION ||
    !Array.isArray(p.capabilities) || !p.capabilities.every((entry) => typeof entry === "string") ||
    !isNonNegativeInteger(p.ready_at_ms)
  ) return null;
  return p as unknown as ReadyPayload;
}

export function parseHorizonPlan(envelope: Envelope<MessageType, unknown>, now = Date.now()): HorizonPlanPayload | null {
  if (envelope.type !== "horizon_plan" || !isRecord(envelope.payload)) return null;
  const p = envelope.payload;
  if (
    !hasOnlyKeys(p, ["plan_id", "based_on_seq", "behavior", "generated_at_ms", "valid_until_ms", "dt_ms", "confidence", "seed", "target", "points"]) ||
    typeof p.plan_id !== "string" || p.plan_id.length < 1 || p.plan_id.length > 128 ||
    !isNonNegativeInteger(p.based_on_seq) ||
    typeof p.behavior !== "string" || !allowedBehaviors.has(p.behavior as Behavior) ||
    !isNonNegativeInteger(p.generated_at_ms) ||
    !isNonNegativeInteger(p.valid_until_ms) || p.valid_until_ms <= now || p.valid_until_ms > now + 5_000 ||
    !isNonNegativeInteger(p.dt_ms) || p.dt_ms < 8 || p.dt_ms > 250 ||
    !isFiniteNumber(p.confidence) || p.confidence < 0 || p.confidence > 1 ||
    !isNonNegativeInteger(p.seed) || p.seed > 0xffff_ffff ||
    !Array.isArray(p.points) || p.points.length < 1 || p.points.length > 128
  ) return null;
  if (p.target !== undefined && !isPlanTarget(p.target)) return null;
  let previousT = -1;
  for (let index = 0; index < p.points.length; index += 1) {
    const point = p.points[index];
    if (!isPlanPoint(point) || point.t_ms <= previousT) return null;
    if (index === 0 && point.t_ms !== 0) return null;
    if (index > 0 && point.t_ms - previousT !== p.dt_ms) return null;
    previousT = point.t_ms;
  }
  return p as unknown as HorizonPlanPayload;
}

function isPlanTarget(value: unknown): value is PlanTarget {
  return isRecord(value) && hasOnlyKeys(value, ["surface_id", "foot_x", "foot_y"]) && typeof value.surface_id === "string" && value.surface_id.length > 0 &&
    value.surface_id.length <= 128 && isFiniteNumber(value.foot_x) && isFiniteNumber(value.foot_y);
}

const ALLOWED_PLAN_POINT_KEYS = [
  "t_ms", "dx", "dy", "vx", "vy", "facing", "lean", "squash", "bob", "expression",
  "bone_rotations", "facial_params",
  "root_translation", "root_rotation", "local_rotation_deltas",
];

function isPlanPoint(value: unknown): value is PlanPoint {
  if (!isRecord(value)) return false;
  if (!hasOnlyKeys(value, ALLOWED_PLAN_POINT_KEYS)) return false;
  if (!isNonNegativeInteger(value.t_ms)) return false;
  if (!isFiniteNumber(value.dx) || Math.abs(value.dx) > 16_384) return false;
  if (!isFiniteNumber(value.dy) || Math.abs(value.dy) > 16_384) return false;
  if (!isFiniteNumber(value.vx) || Math.abs(value.vx) > 20_000) return false;
  if (!isFiniteNumber(value.vy) || Math.abs(value.vy) > 20_000) return false;
  if (value.facing !== -1 && value.facing !== 1) return false;
  if (!isFiniteNumber(value.lean) || value.lean < -1 || value.lean > 1) return false;
  if (!isFiniteNumber(value.squash) || value.squash < 0.5 || value.squash > 1.5) return false;
  if (!isFiniteNumber(value.bob) || value.bob < -48 || value.bob > 48) return false;
  if (typeof value.expression !== "string" || value.expression.length < 1 || value.expression.length > 32) return false;
  if (value.bone_rotations !== undefined && !isBoneRotations(value.bone_rotations)) return false;
  if (value.facial_params !== undefined && !isFacialParams(value.facial_params)) return false;
  // 3D fields — mutually exclusive with legacy bone_rotations
  if (value.root_translation !== undefined && !isVec3(value.root_translation)) return false;
  if (value.root_rotation !== undefined && !isQuat(value.root_rotation)) return false;
  if (value.local_rotation_deltas !== undefined && !isQuatArray(value.local_rotation_deltas)) return false;
  if (value.bone_rotations !== undefined && (value.root_translation !== undefined || value.local_rotation_deltas !== undefined)) {
    return false; // mutual exclusion
  }
  // 3D pose fields must be all-or-none
  const has3D = value.root_translation !== undefined || value.root_rotation !== undefined || value.local_rotation_deltas !== undefined;
  if (has3D && (value.root_translation === undefined || value.root_rotation === undefined || value.local_rotation_deltas === undefined)) {
    return false; // partial 3D data
  }
  return true;
}

function isBoneRotations(value: unknown): value is number[] {
  // Safety ceiling: no realistic skeleton exceeds 32 model-driven bones.
  // validateBoneRotationsLength (below) provides the skeleton-driven exact check.
  if (!Array.isArray(value) || value.length < 1 || value.length > 32) return false;
  return value.every((item) => isFiniteNumber(item) && item >= -3.1416 && item <= 3.1416);
}

/**
 * Strict validation: bone_rotations must have exactly the expected length.
 * Call after the skeleton config has been loaded.
 */
export function validateBoneRotationsLength(
  boneRotations: number[] | undefined,
  expectedCount: number,
): boolean {
  if (boneRotations === undefined) return true; // absence is fine
  return boneRotations.length === expectedCount;
}

function isVec3(value: unknown): value is Vec3 {
  return Array.isArray(value) && value.length === 3 &&
    value.every((item) => isFiniteNumber(item));
}

function isQuat(value: unknown): value is Quat {
  if (!(Array.isArray(value) && value.length === 4)) return false;
  if (!value.every((item) => isFiniteNumber(item) && item >= -1 && item <= 1)) return false;
  // Reject degenerate quaternions (zero vector, or norm far from 1).
  const norm = Math.hypot(value[0], value[1], value[2], value[3]);
  return norm > 0.001 && Math.abs(norm - 1) < 0.001;
}

function isQuatArray(value: unknown): value is Quat[] {
  return Array.isArray(value) && value.length >= 1 && value.length <= 128 &&
    value.every((item) => isQuat(item));
}

function isFacialParams(value: unknown): boolean {
  if (!isRecord(value)) return false;
  if (!hasOnlyKeys(value, ["eye_scale", "eye_squint", "mouth_open", "ear_angle", "brow_tilt"])) return false;
  return isFiniteNumber(value.eye_scale) && value.eye_scale >= 0.5 && value.eye_scale <= 1.5 &&
    isFiniteNumber(value.eye_squint) && value.eye_squint >= 0 && value.eye_squint <= 1 &&
    isFiniteNumber(value.mouth_open) && value.mouth_open >= 0 && value.mouth_open <= 1 &&
    isFiniteNumber(value.ear_angle) && value.ear_angle >= -0.5 && value.ear_angle <= 0.5 &&
    isFiniteNumber(value.brow_tilt) && value.brow_tilt >= -1 && value.brow_tilt <= 1;
}

export function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

export function parsePong(envelope: Envelope<MessageType, unknown>): { readonly nonce: string; readonly ping_sent_at_ms: number; readonly received_at_ms: number } | null {
  if (envelope.type !== "pong" || !isRecord(envelope.payload)) return null;
  const p = envelope.payload;
  if (!hasOnlyKeys(p, ["nonce", "ping_sent_at_ms", "received_at_ms"]) ||
    typeof p.nonce !== "string" || p.nonce.length < 1 || p.nonce.length > 128 ||
    !isNonNegativeInteger(p.ping_sent_at_ms) || !isNonNegativeInteger(p.received_at_ms)) return null;
  return { nonce: p.nonce, ping_sent_at_ms: p.ping_sent_at_ms, received_at_ms: p.received_at_ms };
}

export function parseMetrics(envelope: Envelope<MessageType, unknown>): MetricsPayload | null {
  if (envelope.type !== "metrics" || !isRecord(envelope.payload)) return null;
  const p = envelope.payload;
  if (!hasOnlyKeys(p, ["source", "window_ms", "gauges", "counters", "labels"]) ||
    (p.source !== "host" && p.source !== "generator") ||
    !isNonNegativeInteger(p.window_ms) ||
    !isFiniteNumberMap(p.gauges) || !isFiniteNumberMap(p.counters) ||
    (p.labels !== undefined && !isStringMap(p.labels))) return null;
  return p as unknown as MetricsPayload;
}

function hasOnlyKeys(value: Record<string, unknown>, allowed: readonly string[]): boolean {
  const allowedSet = new Set(allowed);
  return Object.keys(value).every((key) => allowedSet.has(key));
}

function isFiniteNumber(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

function isFiniteNumberMap(value: unknown): value is Record<string, number> {
  return isRecord(value) && Object.entries(value).every(([key, item]) =>
    key.length > 0 && key.length <= 128 && isFiniteNumber(item));
}

function isStringMap(value: unknown): value is Record<string, string> {
  return isRecord(value) && Object.entries(value).every(([key, item]) =>
    key.length > 0 && key.length <= 128 && typeof item === "string" && item.length <= 128);
}

function isNonNegativeInteger(value: unknown): value is number {
  return Number.isSafeInteger(value) && (value as number) >= 0;
}
