import assert from "node:assert/strict";
import test from "node:test";

import { FacingStabilizer } from "../src/facing-stabilizer.js";
import { findCrossedSurface, surfaceSupportsPoint } from "../src/geometry.js";
import { SafePlanExecutor } from "../src/motion-safety.js";
import type { HorizonPlanPayload, PlanPoint, PlanPointBase, SurfaceState } from "../src/protocol.js";

const surface: SurfaceState = {
  id: "window-1:top:0",
  kind: "window_top",
  display_id: "display-1",
  window_id: "window-1",
  x1: 100,
  x2: 800,
  y: 400,
  enabled: true,
  occluded: false,
};

function plan(now: number): HorizonPlanPayload {
  return {
    plan_id: "plan-a",
    based_on_seq: 10,
    behavior: "walk",
    generated_at_ms: now,
    valid_until_ms: now + 1_000,
    dt_ms: 100,
    confidence: 1,
    seed: 1,
    target: { surface_id: surface.id, foot_x: 300, foot_y: 400 },
    points: [
      { t_ms: 0, dx: 0, dy: 0, vx: 0, vy: 0, facing: 1, lean: 0, squash: 1, bob: 0, expression: "neutral" },
      { t_ms: 100, dx: 20, dy: 0, vx: 200, vy: 0, facing: 1, lean: 0.2, squash: 1, bob: 1, expression: "happy" },
    ],
  };
}

function planPointBase(point: PlanPoint): PlanPointBase {
  return {
    t_ms: point.t_ms,
    dx: point.dx,
    dy: point.dy,
    vx: point.vx,
    vy: point.vy,
    facing: point.facing,
    lean: point.lean,
    squash: point.squash,
    bob: point.bob,
    expression: point.expression,
    ...(point.facial_params ? { facial_params: { ...point.facial_params } } : {}),
  };
}

test("accepted plan is anchored to its exact world-state foot", () => {
  const now = 1_800_000_000_000;
  const executor = new SafePlanExecutor();
  executor.rememberWorldState(10, { x: 280, y: 400 });
  assert.equal(executor.offer(plan(now), [surface], now), "accepted");
  const midpoint = executor.sample(now + 50);
  assert.ok(midpoint);
  assert.equal(midpoint.foot.x, 290);
  assert.equal(midpoint.foot.y, 400);
  assert.equal(executor.getDebugPath()?.basedOnSeq, 10);
});

test("future, stale and moved-target plans are rejected or cancelled", () => {
  const now = 1_800_000_000_000;
  const executor = new SafePlanExecutor();
  executor.rememberWorldState(10, { x: 280, y: 400 });
  assert.equal(executor.offer({ ...plan(now), based_on_seq: 11 }, [surface], now), "future_state");
  for (let seq = 11; seq <= 31; seq += 1) executor.rememberWorldState(seq, { x: 280, y: 400 });
  assert.equal(executor.offer(plan(now), [surface], now), "stale_state");

  const current = { ...plan(now), based_on_seq: 31 };
  executor.rememberWorldState(31, { x: 280, y: 400 });
  assert.equal(executor.offer(current, [surface], now), "accepted");
  assert.equal(executor.cancelIfTargetInvalid([{ ...surface, y: 420 }]), true);
  assert.equal(executor.activePlanId, null);
});

test("topology invalidation rejects plans still in transit from an old anchor", () => {
  const now = 1_800_000_000_000;
  const executor = new SafePlanExecutor();
  executor.rememberWorldState(10, { x: 280, y: 400 });
  assert.equal(executor.offer(plan(now), [surface], now), "accepted");

  executor.invalidateAnchors();
  assert.equal(executor.activePlanId, null);
  assert.equal(executor.offer(plan(now), [surface], now), "missing_anchor");

  executor.rememberWorldState(11, { x: 380, y: 440 });
  const fresh = { ...plan(now), based_on_seq: 11 };
  assert.equal(executor.offer(fresh, [surface], now), "accepted");
  assert.deepEqual(executor.sample(now)?.foot, { x: 380, y: 440 });
});

test("grounded plans and in-flight anchors follow a translated carrier frame", () => {
  const now = 1_800_000_000_000;
  const executor = new SafePlanExecutor();
  executor.rememberWorldState(10, { x: 280, y: 400 }, { windowId: "window-1", surfaceId: surface.id });
  assert.equal(executor.offer(plan(now), [surface], now), "accepted");

  const shifted: SurfaceState = { ...surface, id: "window-1:top:1", x1: 200, x2: 900, y: 440 };
  assert.deepEqual(executor.rebaseCarrier("window-1", { x: 100, y: 40 }, shifted.id), { active: "rebased" });
  assert.deepEqual(executor.sample(now)?.foot, { x: 380, y: 440 });
  assert.equal(executor.getDebugPath()?.targetSurfaceId, shifted.id);
  assert.equal(executor.cancelIfTargetInvalid([shifted]), false);

  // A rolling walk reply generated just before the drag is translated into
  // the current carrier frame instead of being rejected or pulling backward.
  assert.equal(executor.offer(plan(now), [shifted], now), "accepted");
  assert.deepEqual(executor.sample(now)?.foot, { x: 380, y: 440 });
});

test("carrier translation preserves in-progress walking velocity", () => {
  const now = 1_800_000_000_000;
  const executor = new SafePlanExecutor();
  executor.rememberWorldState(10, { x: 280, y: 400 }, { windowId: "window-1", surfaceId: surface.id });
  assert.equal(executor.offer(plan(now), [surface], now), "accepted");
  assert.deepEqual(executor.sample(now + 50)?.foot, { x: 290, y: 400 });

  const shifted = { ...surface, x1: 200, x2: 900 };
  executor.rebaseCarrier("window-1", { x: 100, y: 0 }, shifted.id);
  const continued = executor.sample(now + 50);
  assert.deepEqual(continued?.foot, { x: 390, y: 400 });
  assert.equal(continued?.point.vx, 100);
});

test("consecutive carrier translations are accumulated exactly once for late plans", () => {
  const now = 1_800_000_000_000;
  const executor = new SafePlanExecutor();
  executor.rememberWorldState(10, { x: 280, y: 400 }, { windowId: "window-1", surfaceId: surface.id });
  executor.rebaseCarrier("window-1", { x: 100, y: 0 }, surface.id);
  executor.rebaseCarrier("window-1", { x: 20, y: 0 }, surface.id);
  const shifted = { ...surface, x1: 220, x2: 920 };

  assert.equal(executor.offer(plan(now), [shifted], now), "accepted");
  assert.deepEqual(executor.sample(now)?.foot, { x: 400, y: 400 });
  assert.deepEqual(executor.sample(now + 100)?.foot, { x: 420, y: 400 });
  assert.equal(executor.cancelIfTargetInvalid([shifted]), false);
});

test("an in-flight cross-surface jump is regenerated after its source carrier moves", () => {
  const now = 1_800_000_000_000;
  const executor = new SafePlanExecutor();
  executor.rememberWorldState(10, { x: 280, y: 400 }, { windowId: "window-1", surfaceId: surface.id });
  executor.rebaseCarrier("window-1", { x: 100, y: 0 }, surface.id);
  const floor: SurfaceState = {
    id: "display-1:floor", kind: "work_area_floor", display_id: "display-1",
    x1: 0, x2: 1_920, y: 1_040, enabled: true, occluded: false,
  };
  const jump = {
    ...plan(now),
    behavior: "jump" as const,
    target: { surface_id: floor.id, foot_x: 380, foot_y: floor.y },
  };
  assert.equal(executor.offer(jump, [surface, floor], now), "carrier_changed");
});

test("a voluntary downward jump may target the work-area floor", () => {
  const now = 1_800_000_000_000;
  const executor = new SafePlanExecutor();
  const floor: SurfaceState = {
    id: "display-1:floor", kind: "work_area_floor", display_id: "display-1",
    x1: 0, x2: 1_920, y: 1_040, enabled: true, occluded: false,
  };
  executor.rememberWorldState(10, { x: 280, y: 400 }, { windowId: "window-1", surfaceId: surface.id });
  const drop = {
    ...plan(now),
    behavior: "jump" as const,
    target: { surface_id: floor.id, foot_x: 360, foot_y: floor.y },
  };

  assert.equal(executor.offer(drop, [surface, floor], now), "accepted");
  assert.equal(executor.getDebugPath()?.targetSurfaceId, floor.id);
});

test("rolling falling horizons hand off without pulling back to an older anchor", () => {
  const now = 1_800_000_000_000;
  const floor: SurfaceState = {
    id: "display-1:floor", kind: "work_area_floor", display_id: "display-1",
    x1: 0, x2: 1_920, y: 1_040, enabled: true, occluded: false,
  };
  const falling = (basedOnSeq: number, generatedAt: number, planId: string): HorizonPlanPayload => ({
    ...plan(generatedAt),
    plan_id: planId,
    based_on_seq: basedOnSeq,
    behavior: "falling",
    target: { surface_id: floor.id, foot_x: 280, foot_y: floor.y },
    points: [
      { t_ms: 0, dx: 0, dy: 0, vx: 0, vy: 400, facing: 1, lean: 0, squash: 0.96, bob: 0, expression: "surprised" },
      { t_ms: 100, dx: 0, dy: 50, vx: 0, vy: 600, facing: 1, lean: 0, squash: 0.96, bob: 0, expression: "surprised" },
    ],
  });
  const executor = new SafePlanExecutor();
  executor.rememberWorldState(10, { x: 280, y: 400 });
  assert.equal(executor.offer(falling(10, now, "fall-a"), [floor], now + 50, { x: 280, y: 430 }), "accepted");
  assert.deepEqual(executor.sample(now + 50)?.foot, { x: 280, y: 430 });
  assert.deepEqual(executor.sample(now + 100)?.foot, { x: 280, y: 455 });

  executor.rememberWorldState(11, { x: 280, y: 430 });
  assert.equal(executor.offer(falling(11, now + 50, "fall-b"), [floor], now + 75, { x: 280, y: 442.5 }), "accepted");
  assert.deepEqual(executor.sample(now + 75)?.foot, { x: 280, y: 442.5 });
  assert.deepEqual(executor.sample(now + 100)?.foot, { x: 280, y: 455 });
});

test("a plan whose sampled horizon is already exhausted is rejected", () => {
  const now = 1_800_000_000_000;
  const executor = new SafePlanExecutor();
  executor.rememberWorldState(10, { x: 280, y: 400 });
  const exhausted = { ...plan(now), valid_until_ms: now + 1_000 };

  assert.equal(executor.offer(exhausted, [surface], now + 201), "expired");
  assert.equal(executor.activePlanId, null);
});

test("3D pose sampling lerps root motion and uses normalized shortest-arc SLERP", () => {
  const now = 1_800_000_000_000;
  const executor = new SafePlanExecutor();
  executor.rememberWorldState(10, { x: 280, y: 400 });
  const half = Math.sqrt(0.5);
  const skeletal = {
    ...plan(now),
    points: [
      {
        ...planPointBase(plan(now).points[0]!),
        root_translation: [0, 2, 4] as [number, number, number],
        root_rotation: [0, 0, 0, 1] as [number, number, number, number],
        local_rotation_deltas: [[0, 0, 0, 1] as [number, number, number, number]],
        facial_params: {
          eye_scale: 0.5, eye_squint: 0, mouth_open: 0,
          ear_angle: -0.5, brow_tilt: -1,
        },
      },
      {
        ...planPointBase(plan(now).points[1]!),
        root_translation: [10, 4, 8] as [number, number, number],
        // q and -q describe the same orientation; interpolation must not spin.
        root_rotation: [0, 0, 0, -1] as [number, number, number, number],
        local_rotation_deltas: [[0, 0, 1, 0] as [number, number, number, number]],
        facial_params: {
          eye_scale: 1.5, eye_squint: 1, mouth_open: 1,
          ear_angle: 0.5, brow_tilt: 1,
        },
      },
    ],
  } satisfies HorizonPlanPayload;

  assert.equal(executor.offer(skeletal, [surface], now), "accepted");
  const midpoint = executor.sample(now + 50)?.point;
  assert.ok(midpoint);
  assert.deepEqual(midpoint.root_translation, [5, 3, 6]);
  assert.deepEqual(midpoint.root_rotation, [0, 0, 0, 1]);
  assert.ok(Math.abs(midpoint.local_rotation_deltas![0]![2] - half) < 1e-12);
  assert.ok(Math.abs(midpoint.local_rotation_deltas![0]![3] - half) < 1e-12);
  assert.deepEqual(midpoint.facial_params, {
    eye_scale: 1, eye_squint: 0.5, mouth_open: 0.5,
    ear_angle: 0, brow_tilt: 0,
  });
});

test("sparse facial channels interpolate without manufacturing NaN fields", () => {
  const now = 1_800_000_000_000;
  const executor = new SafePlanExecutor();
  executor.rememberWorldState(10, { x: 280, y: 400 });
  const sparse = {
    ...plan(now),
    points: [
      {
        ...planPointBase(plan(now).points[0]!),
        facial_params: { eye_scale: 0.5, mouth_open: 0 },
      },
      {
        ...planPointBase(plan(now).points[1]!),
        facial_params: { mouth_open: 1, brow_tilt: 0.5 },
      },
    ],
  } satisfies HorizonPlanPayload;

  assert.equal(executor.offer(sparse, [surface], now), "accepted");
  assert.deepEqual(executor.sample(now + 25)?.point.facial_params, {
    eye_scale: 0.5,
    mouth_open: 0.25,
  });
  assert.deepEqual(executor.sample(now + 50)?.point.facial_params, {
    mouth_open: 0.5,
    brow_tilt: 0.5,
  });
  assert.ok(Object.values(executor.sample(now + 50)?.point.facial_params ?? {}).every(Number.isFinite));
});

test("legacy pose sampling follows the shortest angular arc across pi", () => {
  const now = 1_800_000_000_000;
  const executor = new SafePlanExecutor();
  executor.rememberWorldState(10, { x: 280, y: 400 });
  const legacy = {
    ...plan(now),
    points: [
      { ...planPointBase(plan(now).points[0]!), bone_rotations: [3.13] },
      { ...planPointBase(plan(now).points[1]!), bone_rotations: [-3.13] },
    ],
  } satisfies HorizonPlanPayload;

  assert.equal(executor.offer(legacy, [surface], now), "accepted");
  const midpoint = executor.sample(now + 50)?.point;
  assert.ok(midpoint?.bone_rotations);
  assert.ok(Math.abs(Math.abs(midpoint.bone_rotations[0]!) - Math.PI) < 1e-12,
    "the midpoint stays near pi instead of rotating through zero");
});

test("a pose encoding transition copies exactly one atomic branch", () => {
  const now = 1_800_000_000_000;
  const executor = new SafePlanExecutor();
  executor.rememberWorldState(10, { x: 280, y: 400 });
  const changingEncoding = {
    ...plan(now),
    points: [
      { ...planPointBase(plan(now).points[0]!), bone_rotations: [0.25] },
      {
        ...planPointBase(plan(now).points[1]!),
        root_translation: [1, 2, 3] as [number, number, number],
        root_rotation: [0, 0, 0, 1] as [number, number, number, number],
        local_rotation_deltas: [[0, 0, 0, 1] as [number, number, number, number]],
      },
    ],
  } satisfies HorizonPlanPayload;

  assert.equal(executor.offer(changingEncoding, [surface], now), "accepted");
  const before = executor.sample(now + 49)?.point;
  assert.deepEqual(before?.bone_rotations, [0.25]);
  assert.equal(before?.root_translation, undefined);
  const after = executor.sample(now + 50)?.point;
  assert.equal(after?.bone_rotations, undefined);
  assert.deepEqual(after?.root_translation, [1, 2, 3]);
  assert.deepEqual(after?.local_rotation_deltas, [[0, 0, 0, 1]]);
});

test("horizontal walking on a top edge is not classified as a landing", () => {
  assert.equal(findCrossedSurface({ x: 280, y: 400 }, { x: 300, y: 400 }, [surface]), null);
  assert.equal(findCrossedSurface({ x: 300, y: 400 }, { x: 300, y: 410 }, [surface]), null);
});

test("a stable surface id is not support after its geometry moves away", () => {
  assert.equal(surfaceSupportsPoint({ x: 280, y: 400 }, surface), true);
  assert.equal(surfaceSupportsPoint({ x: 280, y: 400 }, { ...surface, y: 402 }), false);
  assert.equal(surfaceSupportsPoint({ x: 280, y: 400 }, { ...surface, x1: 300 }), false);
  assert.equal(surfaceSupportsPoint({ x: 280, y: 400 }, { ...surface, occluded: true }), false);
});

test("one-frame opposite directions cannot mirror the sprite", () => {
  const facing = new FacingStabilizer(1, 96);
  assert.equal(facing.update(-1, 1_000), 1);
  assert.equal(facing.update(1, 1_016), 1);
  assert.equal(facing.update(-1, 1_032), 1);
  assert.equal(facing.update(1, 1_048), 1);
  assert.equal(facing.value, 1);
});

test("a sustained direction change still turns the sprite promptly", () => {
  const facing = new FacingStabilizer(1, 96);
  assert.equal(facing.update(-1, 2_000), 1);
  assert.equal(facing.update(-1, 2_095), 1);
  assert.equal(facing.update(-1, 2_096), -1);
  assert.equal(facing.value, -1);
});
