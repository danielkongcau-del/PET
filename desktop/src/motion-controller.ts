/**
 * Single-writer motion loop, following the invariant established by OpenPets
 * apps/desktop/src/pet-motion-engine.ts: one shared ticker owns every
 * continuous BrowserWindow position write. The generator only proposes plans.
 */
import type { Point } from "electron";

import { physicalPointToDip } from "./coordinates.js";
import { FacingStabilizer } from "./facing-stabilizer.js";
import { clamp, findCrossedSurface, nearestSupportingSurface, rectContainsPoint, surfaceSupportsPoint } from "./geometry.js";
import { debug, info, warn } from "./logger.js";
import { SafePlanExecutor, type PlanRejection } from "./motion-safety.js";
import { PET_WINDOW_DIP, PetWindow } from "./pet-window.js";
import { resolveSurfaceAttachment, surfaceIdForMotionSample } from "./surface-attachment.js";
import type { GeneratorStatus, SkeletalConfig } from "./generator-bridge.js";
import type { Behavior, DisplayState, FacialParams, HorizonPlanPayload, PetState, SurfaceState, WindowState } from "./protocol.js";
import { validateBoneRotationsLength } from "./protocol.js";
import type { SurfaceSnapshot } from "./surface-tracker.js";

export interface MotionControllerOptions {
  readonly petWindow: PetWindow;
  readonly initialFootPhysical: Point;
  readonly onPlanCancelled: (reason: "topology_change" | "safety") => void;
  readonly onMotionSample?: (sample: MotionSampleTelemetry) => void;
}

export interface MotionSampleTelemetry {
  readonly timestampMs: number;
  readonly dtMs: number;
  readonly foot: Point;
  readonly velocity: Point;
  readonly behavior: Behavior;
  readonly source: "plan" | "fallback" | "hidden";
  readonly surfaceId?: string;
  readonly planId?: string;
  readonly basedOnSeq?: number;
}

export interface MotionDebugState {
  readonly foot: Point;
  readonly surfaces: readonly SurfaceState[];
  readonly displays: readonly DisplayState[];
  readonly plan: ReturnType<SafePlanExecutor["getDebugPath"]>;
  readonly behavior: Behavior;
  readonly generatorStatus: GeneratorStatus;
  readonly lastPlanRejection: PlanRejection | null;
}

const TICK_MS = 16;
// Keep fallback physics identical to the procedural generator so a temporary
// plan gap during an involuntary fall cannot change acceleration mid-air.
const GRAVITY_PHYSICAL_PER_SECOND_SQUARED = 980;
const MAX_ROOT_SPEED_PHYSICAL_PER_SECOND = 2_200;

export class MotionController {
  readonly #petWindow: PetWindow;
  readonly #onPlanCancelled: MotionControllerOptions["onPlanCancelled"];
  readonly #onMotionSample?: MotionControllerOptions["onMotionSample"];
  readonly #executor = new SafePlanExecutor();
  readonly #facing = new FacingStabilizer();
  #foot: Point;
  #velocity: Point = { x: 0, y: 0 };
  #behavior: Behavior = "fallback";
  #surfaceId: string | null = null;
  #surfaces: readonly SurfaceState[] = [];
  #windows: readonly WindowState[] = [];
  #displays: readonly DisplayState[] = [];
  #sceneAllowed = true;
  #paused = false;
  #pointerOpaque = false;
  #generatorStatus: GeneratorStatus = "starting";
  #debug = false;
  #timer: NodeJS.Timeout | null = null;
  #lastTickMs = Date.now();
  #lastPlanRejection: PlanRejection | null = null;
  #clickFeedbackStartedAt = 0;
  #modelDrivenCount = 0;

  constructor(options: MotionControllerOptions) {
    this.#petWindow = options.petWindow;
    this.#onPlanCancelled = options.onPlanCancelled;
    this.#onMotionSample = options.onMotionSample;
    this.#foot = options.initialFootPhysical;
  }

  start(): void {
    if (this.#timer) return;
    this.#lastTickMs = Date.now();
    this.#timer = setInterval(() => this.#tick(), TICK_MS);
  }

  dispose(): void {
    if (this.#timer) clearInterval(this.#timer);
    this.#timer = null;
    this.#executor.cancel();
  }

  setSurfaceSnapshot(snapshot: SurfaceSnapshot): void {
    const previousSurfaces = this.#surfaces;
    const previousWindows = this.#windows;
    const attachedSurfaceId = this.#surfaceId;
    const previousAttachedSurface = attachedSurfaceId
      ? previousSurfaces.find((surface) => surface.id === attachedSurfaceId)
      : undefined;
    const attachment = attachedSurfaceId && previousAttachedSurface?.kind === "window_top"
      ? resolveSurfaceAttachment({
        foot: this.#foot,
        surfaceId: attachedSurfaceId,
        previousSurfaces,
        nextSurfaces: snapshot.surfaces,
        previousWindows,
        nextWindows: snapshot.windows,
        nextDisplays: snapshot.displays,
      })
      : null;
    this.#surfaces = snapshot.surfaces;
    this.#windows = snapshot.windows;
    this.#displays = snapshot.displays;
    this.#sceneAllowed = snapshot.scene.pet_allowed;
    this.#petWindow.setSceneAllowed(snapshot.scene.pet_allowed);
    let cancellationReported = false;
    const reportCancellation = (): void => {
      if (cancellationReported) return;
      this.#onPlanCancelled("topology_change");
      cancellationReported = true;
    };
    const cancelInvalidTarget = (): void => {
      if (!this.#executor.cancelIfTargetInvalid(snapshot.surfaces)) return;
      warn("motion", "active plan cancelled after target surface changed");
      reportCancellation();
    };

    if (previousAttachedSurface?.kind === "window_top") {
      if (attachment) {
        if (attachment.changed) {
          const delta = { x: attachment.foot.x - this.#foot.x, y: attachment.foot.y - this.#foot.y };
          const previousCarrier = previousWindows.find((window) => window.id === previousAttachedSurface.window_id);
          const nextCarrier = snapshot.windows.find((window) => window.id === previousAttachedSurface.window_id);
          const carrierResized = Boolean(previousCarrier && nextCarrier && (
            previousCarrier.bounds.width !== nextCarrier.bounds.width ||
            previousCarrier.bounds.height !== nextCarrier.bounds.height
          ));
          let activeCarrierPlan: "none" | "rebased" | "cancelled" = "none";
          if (carrierResized) {
            // A translation preserves every local trajectory delta. A resize
            // does not, so keep the physical attachment but request a fresh
            // plan instead of applying a misleading uniform translation.
            const hadActivePlan = this.#executor.activePlanId !== null;
            this.#executor.invalidateAnchors();
            if (hadActivePlan) activeCarrierPlan = "cancelled";
          } else {
            activeCarrierPlan = this.#executor.rebaseCarrier(
              previousAttachedSurface.window_id!,
              delta,
              attachment.surface.id,
            ).active;
          }
          this.#foot = attachment.foot;
          this.#surfaceId = attachment.surface.id;
          this.#placePetWindow();
          if (activeCarrierPlan === "cancelled") reportCancellation();
          debug("motion", "pet attachment followed carrier window", {
            surface_id: attachment.surface.id,
            window_id: attachment.surface.window_id,
            carrier_resized: carrierResized,
          });
        }
        cancelInvalidTarget();
        return;
      }

      const lostSurfaceId = previousAttachedSurface.id;
      const hadActivePlan = this.#executor.activePlanId !== null;
      this.#executor.invalidateAnchors();
      if (hadActivePlan) reportCancellation();
      this.#surfaceId = null;
      this.#velocity = { x: 0, y: Math.max(0, this.#velocity.y) };
      this.#behavior = "falling";
      warn("motion", "carrier window no longer has a usable top segment", { surface_id: lostSurfaceId });
      return;
    }

    cancelInvalidTarget();

    const supportingSurface = this.#surfaceId
      ? snapshot.surfaces.find((surface) => surface.id === this.#surfaceId)
      : undefined;
    if (this.#surfaceId && (!supportingSurface || !surfaceSupportsPoint(this.#foot, supportingSurface))) {
      const lostSurfaceId = this.#surfaceId;
      this.#surfaceId = null;
      // Any absolute-coordinate horizon becomes unsafe after its supporting
      // surface disappears. The carrier-window path above handles movement
      // without treating it as a disappearance.
      const hadActivePlan = this.#executor.activePlanId !== null;
      this.#executor.invalidateAnchors();
      this.#velocity = { x: 0, y: Math.max(0, this.#velocity.y) };
      this.#behavior = "falling";
      if (hadActivePlan) reportCancellation();
      warn("motion", "support surface no longer intersects pet foot", { surface_id: lostSurfaceId });
    }
  }

  setGeneratorStatus(status: GeneratorStatus): void {
    this.#generatorStatus = status;
    if (status !== "ready") {
      this.#executor.cancel();
      this.#facing.resetPending();
      this.#behavior = "fallback";
    }
  }

  setPaused(paused: boolean): void {
    if (this.#paused === paused) return;
    this.#paused = paused;
    this.#executor.cancel();
    this.#facing.resetPending();
    this.#velocity = { x: 0, y: 0 };
    this.#behavior = "fallback";
    info("motion", paused ? "paused" : "resumed");
  }

  setPointerOpaque(opaque: boolean): void {
    this.#pointerOpaque = opaque;
  }

  setDebug(enabled: boolean): void {
    this.#debug = enabled;
  }

  /** Must be called after skeletal negotiation to enable bone rotation length validation. */
  setSkeletalConfig(config: SkeletalConfig): void {
    this.#modelDrivenCount = config.modelDrivenCount;
  }

  registerClickFeedback(): void {
    this.#clickFeedbackStartedAt = Date.now();
    this.#behavior = "click_reaction";
  }

  rememberWorldState(seq: number): void {
    const surface = this.#surfaceId ? this.#surfaces.find((candidate) => candidate.id === this.#surfaceId) : undefined;
    const carrier = surface?.kind === "window_top" && surface.window_id
      ? { windowId: surface.window_id, surfaceId: surface.id }
      : undefined;
    this.#executor.rememberWorldState(seq, this.#foot, carrier);
  }

  offerPlan(plan: HorizonPlanPayload): PlanRejection {
    if (plan.behavior === "falling" && this.#surfaceId !== null) {
      this.#lastPlanRejection = "state_conflict";
      debug("motion", "late falling plan ignored after landing", { based_on_seq: plan.based_on_seq });
      return "state_conflict";
    }
    const now = Date.now();
    const continuityFoot = plan.behavior === "falling" ? this.#foot : undefined;
    const result = this.#executor.offer(plan, this.#surfaces, now, continuityFoot);
    this.#lastPlanRejection = result === "accepted" ? null : result;
    if (result === "missing_anchor" || result === "carrier_changed" || result === "state_conflict") {
      debug("motion", "stale carrier-frame plan ignored", { reason: result, based_on_seq: plan.based_on_seq });
    } else if (result !== "accepted") warn("motion", "horizon plan rejected", { reason: result, based_on_seq: plan.based_on_seq });
    else debug("motion", "horizon plan accepted", { behavior: plan.behavior, points: plan.points.length });
    return result;
  }

  getPetState(): PetState {
    const display = this.#displayForPoint(this.#foot);
    const scale = display?.scale_factor ?? 1;
    const width = PET_WINDOW_DIP * scale;
    const height = PET_WINDOW_DIP * scale;
    const anchor = this.#petWindow.footAnchorDip;
    return {
      x: this.#foot.x - anchor.x * scale,
      y: this.#foot.y - anchor.y * scale,
      width,
      height,
      foot_x: this.#foot.x,
      foot_y: this.#foot.y,
      vx: this.#velocity.x,
      vy: this.#velocity.y,
      facing: this.#facing.value,
      behavior: !this.#sceneAllowed ? "hidden" : this.#behavior,
      visible: this.#petWindow.visible,
      user_dragging: false,
      ...(this.#surfaceId ? { surface_id: this.#surfaceId } : {}),
    };
  }

  getDebugState(): MotionDebugState {
    return {
      foot: this.#foot,
      surfaces: this.#surfaces,
      displays: this.#displays,
      plan: this.#executor.getDebugPath(),
      behavior: this.#behavior,
      generatorStatus: this.#generatorStatus,
      lastPlanRejection: this.#lastPlanRejection,
    };
  }

  get pointerOpaque(): boolean {
    return this.#pointerOpaque;
  }

  #tick(): void {
    const now = Date.now();
    const dt = clamp((now - this.#lastTickMs) / 1_000, 0.001, 0.05);
    this.#lastTickMs = now;
    if (!this.#sceneAllowed) {
      this.#facing.resetPending();
      this.#behavior = "hidden";
      this.#emitMotionSample(now, dt, "hidden");
      return;
    }

    let lean = 0;
    let squash = 1;
    let bob = 0;
    let expression = "neutral";
    let boneRotations: number[] | undefined;
    let facialParams: FacialParams | undefined;
    let rootTranslation: import("@pet/protocol").Vec3 | undefined;
    let rootRotation: import("@pet/protocol").Quat | undefined;
    let localRotationDeltas: import("@pet/protocol").Quat[] | undefined;
    const previous = this.#foot;
    const sample = !this.#paused && this.#generatorStatus === "ready" ? this.#executor.sample(now) : null;
    if (sample) {
      const limited = limitStep(previous, sample.foot, MAX_ROOT_SPEED_PHYSICAL_PER_SECOND * dt);
      const safe = this.#clampToWorkArea(limited);
      const landed = findCrossedSurface(previous, safe, this.#surfaces);
      if (landed) {
        this.#foot = { x: clamp(safe.x, landed.x1, landed.x2), y: landed.y };
        this.#surfaceId = landed.id;
        this.#behavior = "landing";
        if (sample.targetSurfaceId && landed.id !== sample.targetSurfaceId) {
          this.#executor.cancel();
          this.#onPlanCancelled("safety");
          debug("motion", "jump intercepted by an intermediate surface", {
            landed_surface_id: landed.id,
            target_surface_id: sample.targetSurfaceId,
          });
        }
      } else {
        this.#foot = safe;
        this.#surfaceId = surfaceIdForMotionSample(this.#surfaceId, safe, sample.behavior, this.#surfaces);
        this.#behavior = sample.behavior;
      }
      this.#facing.update(sample.point.facing, now);
      lean = sample.point.lean;
      squash = sample.point.squash;
      bob = sample.point.bob;
      expression = sample.point.expression;
      boneRotations = sample.point.bone_rotations;
      if (boneRotations && !validateBoneRotationsLength(boneRotations, this.#modelDrivenCount)) {
        warn("motion", "bone_rotations length mismatch; ignoring skeletal pose", {
          received: boneRotations.length,
          expected: this.#modelDrivenCount,
        });
        boneRotations = undefined;
      }
      // 3D pose takes precedence over legacy 2D when both the plan and the
      // negotiated mode provide it.  The protocol layer already enforces
      // mutual exclusion — a point cannot carry both encodings.
      rootTranslation = sample.point.root_translation;
      rootRotation = sample.point.root_rotation;
      localRotationDeltas = sample.point.local_rotation_deltas;
      if (localRotationDeltas && localRotationDeltas.length !== this.#modelDrivenCount) {
        warn("motion", "local_rotation_deltas length mismatch; ignoring 3D skeletal pose", {
          received: localRotationDeltas.length,
          expected: this.#modelDrivenCount,
        });
        rootTranslation = undefined;
        rootRotation = undefined;
        localRotationDeltas = undefined;
      }
      facialParams = sample.point.facial_params;
    } else {
      this.#facing.resetPending();
      this.#applyFallback(dt);
      bob = this.#surfaceId ? Math.sin(now / 420) * 0.35 : 0;
      expression = this.#generatorStatus === "ready" ? "neutral" : "sleepy";
      boneRotations = undefined;
      facialParams = undefined;
      // During fallback, hold the last skeletal pose so the FK renderer
      // doesn't flicker back to whole-sprite mode.  rootTranslation stays
      // at identity because dx/dy carries the world motion.
      if (this.#modelDrivenCount > 0) {
        rootTranslation = [0, 0, 0];
        rootRotation = [0, 0, 0, 1];
        localRotationDeltas = new Array<[number,number,number,number]>(this.#modelDrivenCount).fill([0, 0, 0, 1]);
      } else {
        rootTranslation = undefined;
        rootRotation = undefined;
        localRotationDeltas = undefined;
      }
    }

    const clickAge = now - this.#clickFeedbackStartedAt;
    if (clickAge >= 0 && clickAge < 180) {
      const phase = clickAge / 180;
      squash *= phase < 0.45 ? 0.72 + phase * 0.35 : 0.88 + (phase - 0.45) * 0.22;
      bob -= Math.sin(phase * Math.PI) * 2.5;
      expression = "surprised";
      this.#behavior = "click_reaction";
    }

    this.#velocity = { x: (this.#foot.x - previous.x) / dt, y: (this.#foot.y - previous.y) / dt };
    this.#placePetWindow();
    this.#petWindow.sendVisualState({
      facing: this.#facing.value,
      lean,
      squash,
      bob,
      expression,
      behavior: this.#behavior,
      generatorStatus: this.#generatorStatus,
      debug: this.#debug,
      ...(boneRotations ? { boneRotations } : {}),
      ...(facialParams ? { facialParams } : {}),
      ...(rootTranslation ? { rootTranslation } : {}),
      ...(rootRotation ? { rootRotation } : {}),
      ...(localRotationDeltas ? { localRotationDeltas } : {}),
    });
    this.#emitMotionSample(now, dt, sample ? "plan" : "fallback", sample?.planId, this.#executor.basedOnSeq ?? undefined);
  }

  #emitMotionSample(
    timestampMs: number,
    dtSeconds: number,
    source: MotionSampleTelemetry["source"],
    planId?: string,
    basedOnSeq?: number,
  ): void {
    if (!this.#onMotionSample) return;
    this.#onMotionSample({
      timestampMs,
      dtMs: dtSeconds * 1_000,
      foot: { ...this.#foot },
      velocity: { ...this.#velocity },
      behavior: this.#behavior,
      source,
      ...(this.#surfaceId ? { surfaceId: this.#surfaceId } : {}),
      ...(planId ? { planId } : {}),
      ...(basedOnSeq !== undefined ? { basedOnSeq } : {}),
    });
  }

  #applyFallback(dt: number): void {
    const currentSurface = this.#surfaceId ? this.#surfaces.find((surface) => surface.id === this.#surfaceId && surface.enabled && !surface.occluded) : undefined;
    if (currentSurface && surfaceSupportsPoint(this.#foot, currentSurface)) {
      // Surface snapshots reattach the pet in the carrier window's coordinate
      // frame. Integrating the last observed window velocity here would count
      // the same drag twice at the next snapshot.
      const proposed = { x: this.#foot.x, y: currentSurface.y };
      this.#foot = limitStep(this.#foot, proposed, MAX_ROOT_SPEED_PHYSICAL_PER_SECOND * dt);
      this.#velocity = { x: 0, y: 0 };
      this.#behavior = "fallback";
      return;
    }
    const support = nearestSupportingSurface(this.#foot, this.#surfaces, 8);
    if (support) {
      this.#surfaceId = support.id;
      this.#foot = { x: this.#foot.x, y: support.y };
      this.#velocity = { x: 0, y: 0 };
      this.#behavior = "fallback";
      return;
    }

    const nextVelocityY = this.#velocity.y + GRAVITY_PHYSICAL_PER_SECOND_SQUARED * dt;
    let next = this.#clampToWorkArea({ x: this.#foot.x, y: this.#foot.y + nextVelocityY * dt });
    const landed = findCrossedSurface(this.#foot, next, this.#surfaces);
    if (landed) {
      next = { x: clamp(next.x, landed.x1, landed.x2), y: landed.y };
      this.#surfaceId = landed.id;
      this.#velocity = { x: 0, y: 0 };
      this.#behavior = "landing";
    } else {
      this.#surfaceId = null;
      this.#velocity = { x: 0, y: nextVelocityY };
      this.#behavior = "falling";
    }
    this.#foot = next;
  }

  #clampToWorkArea(point: Point): Point {
    const display = this.#displayForPoint(point) ?? this.#displays[0];
    if (!display) return point;
    const anchor = this.#petWindow.footAnchorDip;
    const leftExtent = anchor.x * display.scale_factor;
    const rightExtent = (PET_WINDOW_DIP - anchor.x) * display.scale_factor;
    const topExtent = anchor.y * display.scale_factor;
    return {
      x: clamp(point.x, display.work_area.x + leftExtent, display.work_area.x + display.work_area.width - rightExtent),
      y: clamp(point.y, display.work_area.y + topExtent, display.work_area.y + display.work_area.height),
    };
  }

  #placePetWindow(): void {
    const renderDisplay = this.#displayForPoint(this.#foot);
    if (renderDisplay) this.#petWindow.setFootPhysical(this.#foot, renderDisplay.id);
    else this.#petWindow.setFootDip(physicalPointToDip(this.#foot));
  }

  #displayForPoint(point: Point): DisplayState | undefined {
    const containing = this.#displays.find((display) => rectContainsPoint(display.work_area, point));
    if (containing) return containing;
    let best = this.#displays[0];
    let bestDistance = Number.POSITIVE_INFINITY;
    for (const display of this.#displays) {
      const x = clamp(point.x, display.work_area.x, display.work_area.x + display.work_area.width);
      const y = clamp(point.y, display.work_area.y, display.work_area.y + display.work_area.height);
      const distance = Math.hypot(point.x - x, point.y - y);
      if (distance < bestDistance) {
        best = display;
        bestDistance = distance;
      }
    }
    return best;
  }
}

function limitStep(from: Point, to: Point, maxDistance: number): Point {
  const dx = to.x - from.x;
  const dy = to.y - from.y;
  const distance = Math.hypot(dx, dy);
  if (distance <= maxDistance || distance === 0) return to;
  const scale = maxDistance / distance;
  return { x: from.x + dx * scale, y: from.y + dy * scale };
}
