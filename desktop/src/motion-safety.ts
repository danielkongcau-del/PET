/**
 * Host-authoritative plan acceptance and interpolation. The generator proposes
 * short horizons; this module decides whether they are fresh and safe enough
 * to reach the sole BrowserWindow position writer.
 */
import { clamp } from "./geometry.js";
import type { FacialParams, HorizonPlanPayload, PlanPoint, PlanPointBase, Quat, SurfaceState, Vec3 } from "./protocol.js";

export interface FootAnchor {
  readonly x: number;
  readonly y: number;
}

export interface SampledPlan {
  readonly planId: string;
  readonly behavior: HorizonPlanPayload["behavior"];
  readonly targetSurfaceId?: string;
  readonly point: PlanPoint;
  readonly foot: FootAnchor;
}

export type PlanRejection =
  | "accepted"
  | "future_state"
  | "stale_state"
  | "missing_anchor"
  | "expired"
  | "future_generation"
  | "invalid_target"
  | "carrier_changed"
  | "state_conflict"
  | "pose_encoding_mismatch"
  | "low_confidence";

export interface CarrierAnchor {
  readonly windowId: string;
  readonly surfaceId: string;
}

interface StoredCarrierAnchor {
  readonly windowId: string;
  readonly capturedSurfaceId: string;
  readonly currentSurfaceId: string;
  readonly offset: FootAnchor;
}

interface StoredAnchor {
  readonly foot: FootAnchor;
  readonly carrier?: StoredCarrierAnchor;
}

interface AdjustedTarget {
  readonly surface_id: string;
  readonly foot_x: number;
  readonly foot_y: number;
}

interface ActivePlan {
  readonly plan: HorizonPlanPayload;
  readonly origin: FootAnchor;
  readonly target?: AdjustedTarget;
  readonly targetFollowsCarrier: boolean;
  readonly carrier?: StoredCarrierAnchor;
}

export interface CarrierRebaseResult {
  readonly active: "none" | "rebased" | "cancelled";
}

export class SafePlanExecutor {
  #anchors = new Map<number, StoredAnchor>();
  #latestWorldSeq = -1;
  #active: ActivePlan | null = null;

  get activePlanId(): string | null {
    return this.#active?.plan.plan_id ?? null;
  }

  get basedOnSeq(): number | null {
    return this.#active?.plan.based_on_seq ?? null;
  }

  rememberWorldState(seq: number, anchor: FootAnchor, carrier?: CarrierAnchor): void {
    this.#latestWorldSeq = Math.max(this.#latestWorldSeq, seq);
    this.#anchors.set(seq, {
      foot: { ...anchor },
      ...(carrier ? {
        carrier: {
          windowId: carrier.windowId,
          capturedSurfaceId: carrier.surfaceId,
          currentSurfaceId: carrier.surfaceId,
          offset: { x: 0, y: 0 },
        },
      } : {}),
    });
    while (this.#anchors.size > 64) {
      const oldest = this.#anchors.keys().next().value as number | undefined;
      if (oldest === undefined) break;
      this.#anchors.delete(oldest);
    }
  }

  offer(
    plan: HorizonPlanPayload,
    surfaces: readonly SurfaceState[],
    now = Date.now(),
    continuityFoot?: FootAnchor,
  ): PlanRejection {
    if (plan.based_on_seq > this.#latestWorldSeq) return "future_state";
    if (this.#latestWorldSeq - plan.based_on_seq > 20) return "stale_state";
    const anchor = this.#anchors.get(plan.based_on_seq);
    if (!anchor) return "missing_anchor";
    if (plan.valid_until_ms <= now) return "expired";
    if (plan.generated_at_ms > now + 100) return "future_generation";
    if (plan.confidence < 0.05) return "low_confidence";
    const offerPoint = pointAtTime(plan, now);
    if (!offerPoint) return "expired";
    const targetFollowsCarrier = Boolean(plan.target && anchor.carrier &&
      plan.target.surface_id === anchor.carrier.capturedSurfaceId);
    if (anchor.carrier && (anchor.carrier.offset.x !== 0 || anchor.carrier.offset.y !== 0) &&
      (plan.behavior === "jump" || plan.behavior === "falling") && !targetFollowsCarrier) return "carrier_changed";
    const adjustedTarget = plan.target ? {
      surface_id: targetFollowsCarrier ? anchor.carrier!.currentSurfaceId : plan.target.surface_id,
      foot_x: plan.target.foot_x + (targetFollowsCarrier ? anchor.carrier!.offset.x : 0),
      foot_y: plan.target.foot_y + (targetFollowsCarrier ? anchor.carrier!.offset.y : 0),
    } : undefined;
    if (adjustedTarget) {
      const target = surfaces.find((surface) => surface.id === adjustedTarget.surface_id);
      if (!target || !target.enabled || target.occluded ||
        adjustedTarget.foot_x < target.x1 || adjustedTarget.foot_x > target.x2 ||
        Math.abs(adjustedTarget.foot_y - target.y) > 3) return "invalid_target";
    }
    let origin = { ...anchor.foot };
    if (plan.behavior === "falling" && continuityFoot) {
      // A rolling reply is anchored to the slightly older world state. Keep
      // its future deltas, but translate the horizon so replacing the
      // previous fall cannot pull the pet back to that old anchor.
      origin = {
        x: continuityFoot.x - offerPoint.dx,
        y: continuityFoot.y - offerPoint.dy,
      };
    }
    this.#active = {
      plan,
      origin,
      targetFollowsCarrier,
      ...(adjustedTarget ? { target: adjustedTarget } : {}),
      ...(anchor.carrier ? { carrier: { ...anchor.carrier, offset: { ...anchor.carrier.offset } } } : {}),
    };
    return "accepted";
  }

  sample(now = Date.now()): SampledPlan | null {
    const active = this.#active;
    if (!active) return null;
    if (now >= active.plan.valid_until_ms) {
      this.#active = null;
      return null;
    }
    const point = pointAtTime(active.plan, now);
    if (!point) {
      this.#active = null;
      return null;
    }
    const targetSurfaceId = active.target?.surface_id;
    return {
      planId: active.plan.plan_id,
      behavior: active.plan.behavior,
      ...(targetSurfaceId ? { targetSurfaceId } : {}),
      point,
      foot: { x: active.origin.x + point.dx, y: active.origin.y + point.dy },
    };
  }

  cancel(): void {
    this.#active = null;
  }

  /**
   * Establishes a topology barrier. Plans already in transit may still refer
   * to an old absolute-coordinate anchor after their carrier window moved;
   * clearing anchors makes those late replies fail with missing_anchor while
   * preserving sequence monotonicity for the next world state.
   */
  invalidateAnchors(): void {
    this.#active = null;
    this.#anchors.clear();
  }

  /** Rebase grounded motion into a carrier window's translated frame. */
  rebaseCarrier(windowId: string, delta: FootAnchor, currentSurfaceId: string): CarrierRebaseResult {
    for (const [seq, anchor] of this.#anchors) {
      if (anchor.carrier?.windowId !== windowId) continue;
      this.#anchors.set(seq, {
        foot: { x: anchor.foot.x + delta.x, y: anchor.foot.y + delta.y },
        carrier: {
          ...anchor.carrier,
          currentSurfaceId,
          offset: {
            x: anchor.carrier.offset.x + delta.x,
            y: anchor.carrier.offset.y + delta.y,
          },
        },
      });
    }

    const active = this.#active;
    if (active?.carrier?.windowId !== windowId) return { active: "none" };
    const canFollowCarrier = active.targetFollowsCarrier ||
      (!active.target && active.plan.behavior !== "jump" && active.plan.behavior !== "falling");
    if (!canFollowCarrier) {
      this.#active = null;
      return { active: "cancelled" };
    }
    const target = active.target && active.targetFollowsCarrier ? {
      surface_id: currentSurfaceId,
      foot_x: active.target.foot_x + delta.x,
      foot_y: active.target.foot_y + delta.y,
    } : active.target;
    this.#active = {
      ...active,
      origin: { x: active.origin.x + delta.x, y: active.origin.y + delta.y },
      ...(target ? { target } : {}),
      carrier: {
        ...active.carrier,
        currentSurfaceId,
        offset: {
          x: active.carrier.offset.x + delta.x,
          y: active.carrier.offset.y + delta.y,
        },
      },
    };
    return { active: "rebased" };
  }

  cancelIfTargetInvalid(surfaces: readonly SurfaceState[]): boolean {
    const plannedTarget = this.#active?.target;
    if (!plannedTarget) return false;
    const target = surfaces.find((surface) => surface.id === plannedTarget.surface_id);
    if (target?.enabled && !target.occluded && plannedTarget.foot_x >= target.x1 &&
      plannedTarget.foot_x <= target.x2 && Math.abs(plannedTarget.foot_y - target.y) <= 6) return false;
    this.#active = null;
    return true;
  }

  getDebugPath(): { readonly planId: string; readonly basedOnSeq: number; readonly points: readonly FootAnchor[]; readonly targetSurfaceId?: string } | null {
    if (!this.#active) return null;
    const targetSurfaceId = this.#active.target?.surface_id;
    return {
      planId: this.#active.plan.plan_id,
      basedOnSeq: this.#active.plan.based_on_seq,
      points: this.#active.plan.points.map((point) => ({ x: this.#active!.origin.x + point.dx, y: this.#active!.origin.y + point.dy })),
      ...(targetSurfaceId ? { targetSurfaceId } : {}),
    };
  }
}

function pointAtTime(plan: HorizonPlanPayload, now: number): PlanPoint | null {
  const elapsed = Math.max(0, now - plan.generated_at_ms);
  const points = plan.points;
  const last = points[points.length - 1];
  if (!last || elapsed > last.t_ms + plan.dt_ms) return null;
  let left = points[0]!;
  let right = left;
  for (let index = 1; index < points.length; index += 1) {
    const candidate = points[index]!;
    if (candidate.t_ms >= elapsed) {
      right = candidate;
      break;
    }
    left = candidate;
    right = candidate;
  }
  const span = Math.max(1, right.t_ms - left.t_ms);
  const alpha = clamp((elapsed - left.t_ms) / span, 0, 1);
  return interpolatePoint(left, right, alpha, Math.round(elapsed));
}

function interpolatePoint(left: PlanPoint, right: PlanPoint, alpha: number, tMs: number): PlanPoint {
  const lerp = (a: number, b: number): number => a + (b - a) * alpha;
  const src = alpha < 0.5 ? left : right;
  const base: PlanPointBase = {
    t_ms: tMs,
    dx: lerp(left.dx, right.dx),
    dy: lerp(left.dy, right.dy),
    vx: lerp(left.vx, right.vx),
    vy: lerp(left.vy, right.vy),
    facing: alpha < 0.5 ? left.facing : right.facing,
    lean: lerp(left.lean, right.lean),
    squash: lerp(left.squash, right.squash),
    bob: lerp(left.bob, right.bob),
    expression: alpha < 0.5 ? left.expression : right.expression,
  };

  const leftLegacy = left.bone_rotations;
  const rightLegacy = right.bone_rotations;
  const canInterpolateLegacy = leftLegacy !== undefined && rightLegacy !== undefined &&
    leftLegacy.length === rightLegacy.length;
  const left3D = complete3DPose(left);
  const right3D = complete3DPose(right);
  const canInterpolate3D = left3D !== null && right3D !== null &&
    left3D.localRotationDeltas.length === right3D.localRotationDeltas.length;

  let point: PlanPoint;
  if (canInterpolateLegacy) {
    point = {
      ...base,
      bone_rotations: leftLegacy.map((angle, index) =>
        interpolateAngle(angle, rightLegacy[index]!, alpha)),
    };
  } else if (canInterpolate3D) {
    point = {
      ...base,
      root_translation: interpolateVec3(
        left3D.rootTranslation,
        right3D.rootTranslation,
        alpha,
      ),
      root_rotation: slerpQuat(left3D.rootRotation, right3D.rootRotation, alpha),
      local_rotation_deltas: left3D.localRotationDeltas.map((quaternion, index) =>
        slerpQuat(quaternion, right3D.localRotationDeltas[index]!, alpha)),
    };
  } else if (src.bone_rotations !== undefined) {
    point = { ...base, bone_rotations: [...src.bone_rotations] };
  } else {
    const pose = complete3DPose(src);
    point = pose
      ? {
          ...base,
          root_translation: [...pose.rootTranslation] as Vec3,
          root_rotation: [...pose.rootRotation] as Quat,
          local_rotation_deltas: pose.localRotationDeltas.map((quaternion) =>
            [...quaternion] as Quat),
        }
      : base;
  }

  const leftFace = left.facial_params;
  const rightFace = right.facial_params;
  if (leftFace !== undefined && rightFace !== undefined) {
    return { ...point, facial_params: interpolateFacialParams(leftFace, rightFace, alpha) };
  } else if (src.facial_params !== undefined) {
    return { ...point, facial_params: { ...src.facial_params } };
  }
  return point;
}

interface Complete3DPose {
  readonly rootTranslation: Vec3;
  readonly rootRotation: Quat;
  readonly localRotationDeltas: readonly Quat[];
}

function complete3DPose(point: PlanPoint): Complete3DPose | null {
  if (point.root_translation === undefined || point.root_rotation === undefined ||
      point.local_rotation_deltas === undefined) return null;
  return {
    rootTranslation: point.root_translation,
    rootRotation: point.root_rotation,
    localRotationDeltas: point.local_rotation_deltas,
  };
}

function interpolateVec3(left: Vec3, right: Vec3, alpha: number): Vec3 {
  return [
    left[0] + (right[0] - left[0]) * alpha,
    left[1] + (right[1] - left[1]) * alpha,
    left[2] + (right[2] - left[2]) * alpha,
  ];
}

function wrapAngle(angle: number): number {
  let wrapped = angle;
  while (wrapped > Math.PI) wrapped -= Math.PI * 2;
  while (wrapped < -Math.PI) wrapped += Math.PI * 2;
  return wrapped;
}

function interpolateAngle(left: number, right: number, alpha: number): number {
  const delta = wrapAngle(right - left);
  return wrapAngle(left + delta * alpha);
}

function normalizeQuaternion(quaternion: Quat): Quat {
  const norm = Math.hypot(...quaternion);
  if (!Number.isFinite(norm) || norm <= 1e-12) return [0, 0, 0, 1];
  return quaternion.map((component) => component / norm) as Quat;
}

function slerpQuat(leftInput: Quat, rightInput: Quat, alpha: number): Quat {
  const left = normalizeQuaternion(leftInput);
  let right = normalizeQuaternion(rightInput);
  let dot = left[0] * right[0] + left[1] * right[1] +
    left[2] * right[2] + left[3] * right[3];
  if (dot < 0) {
    right = right.map((component) => -component) as Quat;
    dot = -dot;
  }
  dot = clamp(dot, -1, 1);

  if (dot > 0.9995) {
    return normalizeQuaternion(left.map((component, index) =>
      component + (right[index]! - component) * alpha) as Quat);
  }

  const theta = Math.acos(dot);
  const sinTheta = Math.sin(theta);
  if (Math.abs(sinTheta) <= 1e-12) return left;
  const leftWeight = Math.sin((1 - alpha) * theta) / sinTheta;
  const rightWeight = Math.sin(alpha * theta) / sinTheta;
  return normalizeQuaternion(left.map((component, index) =>
    component * leftWeight + right[index]! * rightWeight) as Quat);
}

function interpolateFacialParams(
  left: FacialParams,
  right: FacialParams,
  alpha: number,
): FacialParams {
  const result: FacialParams = {};
  const keys = [
    "eye_scale", "eye_squint", "mouth_open", "ear_angle", "brow_tilt",
  ] as const;
  for (const key of keys) {
    const leftValue = left[key];
    const rightValue = right[key];
    if (leftValue !== undefined && rightValue !== undefined) {
      result[key] = leftValue + (rightValue - leftValue) * alpha;
      continue;
    }
    // A sparse channel appears and disappears at the same nearest-keyframe
    // boundary used by expression/facing, rather than entering NaN arithmetic.
    const sourceValue = alpha < 0.5 ? leftValue : rightValue;
    if (sourceValue !== undefined) result[key] = sourceValue;
  }
  return result;
}
