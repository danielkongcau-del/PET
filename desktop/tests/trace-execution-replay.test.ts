import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import type { HorizonPlanPayload, SurfaceState, WorldStatePayload } from "../src/protocol.js";
import { replayTraceExecution } from "../src/trace/execution-replay.js";
import type { JsonObject, TraceKind, TraceRecord } from "../src/trace/format.js";

const epoch = 1_800_000_000_000;
const surface: SurfaceState = {
  id: "surface-0",
  kind: "window_top",
  display_id: "display-0",
  window_id: "window-0",
  x1: 100,
  x2: 500,
  y: 400,
  enabled: true,
  occluded: false,
};

function record(seq: number, kind: TraceKind, payload: JsonObject, wallOffsetMs: number): TraceRecord {
  return {
    schema: "pet-trace",
    version: 1,
    record_seq: seq,
    wall_time_ms: epoch + wallOffsetMs,
    elapsed_us: wallOffsetMs * 1_000,
    kind,
    payload,
  };
}

function worldState(): WorldStatePayload {
  return {
    session_id: "episode",
    coordinate_space: "physical_px",
    displays: [],
    windows: [],
    surfaces: [surface],
    pet: {
      x: 152, y: 304, width: 96, height: 96,
      foot_x: 200, foot_y: 400, vx: 0, vy: 0,
      facing: 1, behavior: "walk", visible: true, user_dragging: false, surface_id: surface.id,
    },
    cursor: { x: 0, y: 0, left_down: false, right_down: false, middle_down: false, over_pet: false },
    clicks: [],
    scene: { fullscreen_active: false, pet_allowed: true },
  };
}

function plan(planId: string, basedOnSeq: number): HorizonPlanPayload {
  return {
    plan_id: planId,
    based_on_seq: basedOnSeq,
    behavior: "walk",
    generated_at_ms: epoch + 20,
    valid_until_ms: epoch + 1_020,
    dt_ms: 100,
    confidence: 1,
    seed: 123,
    target: { surface_id: surface.id, foot_x: 220, foot_y: 400 },
    points: [
      { t_ms: 0, dx: 0, dy: 0, vx: 200, vy: 0, facing: 1, lean: 0, squash: 1, bob: 0, expression: "neutral" },
      { t_ms: 100, dx: 20, dy: 0, vx: 200, vy: 0, facing: 1, lean: 0, squash: 1, bob: 0, expression: "neutral" },
    ],
  };
}

function replayFixture(): TraceRecord[] {
  return [
    record(0, "surface_snapshot", { captured_at_ms: epoch, displays: [], windows: [], surfaces: [surface] as unknown as JsonObject["surfaces"], scene: { fullscreen_active: false, pet_allowed: true } }, 0),
    record(1, "world_state", { seq: 10, timestamp_ms: epoch + 10, state: worldState() as unknown as JsonObject }, 10),
    record(2, "plan_received", { plan: plan("plan-accepted", 10) as unknown as JsonObject, received_at_ms: epoch + 20 }, 20),
    record(3, "plan_result", { plan_id: "plan-accepted", based_on_seq: 10, result: "accepted" }, 21),
    // At t=50 the interpolated foot is (210, 400); the recorded sample is two pixels away.
    record(4, "motion_sample", { timestamp_ms: epoch + 70, dt_ms: 16, foot: { x: 212, y: 400 }, velocity: { x: 200, y: 0 }, behavior: "walk", surface_id: surface.id, plan_id: "plan-accepted", source: "generator" }, 70),
    // No world state 11 exists, so this is deterministically rejected as future_state.
    record(5, "plan_received", { plan: plan("plan-rejected", 11) as unknown as JsonObject, received_at_ms: epoch + 80 }, 80),
    record(6, "plan_result", { plan_id: "plan-rejected", based_on_seq: 11, result: "future_state" }, 81),
  ];
}

test("execution replay uses recorded virtual time for accept/reject matching and interpolation error", () => {
  const originalDateNow = Date.now;
  Date.now = () => { throw new Error("replay touched the real clock"); };
  try {
    const replay = replayTraceExecution(replayFixture());
    assert.equal(replay.worldStates, 1);
    assert.equal(replay.plansOffered, 2);
    assert.equal(replay.recordedPlanResults, 2);
    assert.equal(replay.decisionMatches, 2);
    assert.equal(replay.decisionMismatches, 0);
    assert.equal(replay.sampledPlanPoints, 1);
    assert.equal(replay.samplesCompared, 1);
    assert.deepEqual(replay.samplePositionErrorPx, { count: 1, min: 2, max: 2, mean: 2, p50: 2, p95: 2, p99: 2 });
    assert.deepEqual(replay.issues, []);
  } finally {
    Date.now = originalDateNow;
  }
});

test("execution replay digest and events are identical for identical logical input order", () => {
  const records = replayFixture();
  const first = replayTraceExecution(records);
  const second = replayTraceExecution(records);
  const shuffledStorageOrder = replayTraceExecution([...records].reverse());
  assert.equal(first.eventDigestSha256, second.eventDigestSha256);
  assert.equal(first.eventDigestSha256, shuffledStorageOrder.eventDigestSha256);
  assert.equal(first.eventCount, second.eventCount);
  assert.match(first.eventDigestSha256, /^[a-f0-9]{64}$/);
});

test("execution replay core has no Electron, window enumeration, timers, or live desktop imports", () => {
  const source = readFileSync(new URL("../src/trace/execution-replay.js", import.meta.url), "utf8");
  assert.doesNotMatch(source, /from\s+["']electron["']/);
  assert.doesNotMatch(source, /surface-tracker|pet-window|generator-bridge|get-windows/);
  assert.doesNotMatch(source, /Date\.now\(|setTimeout\(|setInterval\(/);
  assert.doesNotThrow(() => replayTraceExecution(replayFixture()));
});
