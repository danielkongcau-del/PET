import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import { fileURLToPath } from "node:url";
import { resolve } from "node:path";

import { createEnvelope, decodeEnvelope, encodeEnvelope, parseHorizonPlan, parseMetrics } from "../src/protocol.js";
import type { PlanPoint, Quat, Vec3 } from "../src/protocol.js";

const TYPE_POINT_BASE = {
  t_ms: 0,
  dx: 0,
  dy: 0,
  vx: 0,
  vy: 0,
  facing: 1 as const,
  lean: 0,
  squash: 1,
  bob: 0,
  expression: "neutral",
};
const TYPE_VEC3 = [0, 0, 0] as Vec3;
const TYPE_QUAT = [0, 0, 0, 1] as Quat;

const VALID_3D_TYPE_POINT: PlanPoint = {
  ...TYPE_POINT_BASE,
  root_translation: TYPE_VEC3,
  root_rotation: TYPE_QUAT,
  local_rotation_deltas: [TYPE_QUAT],
};
// @ts-expect-error A partial 3D branch must fail at compile time.
const INVALID_PARTIAL_3D_TYPE_POINT: PlanPoint = {
  ...TYPE_POINT_BASE,
  root_translation: TYPE_VEC3,
};
// @ts-expect-error Legacy and 3D branches are mutually exclusive at compile time.
const INVALID_MIXED_TYPE_POINT: PlanPoint = {
  ...VALID_3D_TYPE_POINT,
  bone_rotations: [0],
};
void INVALID_PARTIAL_3D_TYPE_POINT;
void INVALID_MIXED_TYPE_POINT;

function validPlan(now: number) {
  return {
    plan_id: "plan-1",
    based_on_seq: 4,
    behavior: "jump" as const,
    generated_at_ms: now,
    valid_until_ms: now + 1_000,
    dt_ms: 33,
    confidence: 0.9,
    seed: 42,
    target: { surface_id: "window-1:top:0", foot_x: 500, foot_y: 300 },
    points: [
      { t_ms: 0, dx: 0, dy: 0, vx: 0, vy: -100, facing: 1 as const, lean: 0, squash: 1, bob: 0, expression: "neutral" },
      { t_ms: 33, dx: 5, dy: -4, vx: 150, vy: -80, facing: 1 as const, lean: 0.1, squash: 0.9, bob: -1, expression: "focused" },
    ],
  };
}

test("NDJSON envelope round-trips through the shared protocol codec", () => {
  const message = createEnvelope("ping", 7, { nonce: "n", sent_at_ms: 100 }, 100);
  const decoded = decodeEnvelope(encodeEnvelope(message).trimEnd());
  assert.equal(decoded.protocol, "pet-motion");
  assert.equal(decoded.version, 1);
  assert.equal(decoded.seq, 7);
});

test("plan boundary accepts exact dt sequence and rejects timing drift", () => {
  const now = 1_800_000_000_000;
  const plan = validPlan(now);
  assert.ok(parseHorizonPlan(createEnvelope("horizon_plan", 2, plan, now), now));
  const invalid = { ...plan, points: [plan.points[0], { ...plan.points[1], t_ms: 34 }] };
  assert.equal(parseHorizonPlan(createEnvelope("horizon_plan", 3, invalid, now), now), null);
});

test("facial parameters accept bounded sparse overrides and reject invalid channels", () => {
  const now = 1_800_000_000_000;
  const plan = validPlan(now);
  const withFacial = (facialParams: Record<string, unknown>) => ({
    ...plan,
    points: [
      { ...plan.points[0], facial_params: facialParams },
      plan.points[1],
    ],
  });

  assert.ok(parseHorizonPlan(
    createEnvelope("horizon_plan", 20, withFacial({ mouth_open: 0.5 }), now),
    now,
  ));
  assert.equal(parseHorizonPlan(
    createEnvelope("horizon_plan", 21, withFacial({ unknown: 0.5 }), now),
    now,
  ), null);
  assert.equal(parseHorizonPlan(
    createEnvelope("horizon_plan", 22, withFacial({ mouth_open: Number.NaN }), now),
    now,
  ), null);
  assert.equal(parseHorizonPlan(
    createEnvelope("horizon_plan", 23, withFacial({ mouth_open: 1.1 }), now),
    now,
  ), null);
});

test("expired plans never cross the host authority boundary", () => {
  const now = 1_800_000_000_000;
  const plan = { ...validPlan(now), valid_until_ms: now };
  assert.equal(parseHorizonPlan(createEnvelope("horizon_plan", 2, plan, now), now), null);
});

test("3D pose fields are an atomic branch and cannot mix with legacy rotations", () => {
  const now = 1_800_000_000_000;
  const plan = validPlan(now);
  const points = plan.points.map((point) => ({
    ...point,
    root_translation: [0, 0, 0] as [number, number, number],
    root_rotation: [0, 0, 0, 1] as [number, number, number, number],
    local_rotation_deltas: [[0, 0, 0, 1] as [number, number, number, number]],
  }));
  const complete = { ...plan, points };
  assert.ok(parseHorizonPlan(createEnvelope("horizon_plan", 20, complete, now), now));

  for (const field of [
    "root_translation",
    "root_rotation",
    "local_rotation_deltas",
  ] as const) {
    const first = { ...points[0] };
    delete first[field];
    const invalid = { ...complete, points: [first, ...points.slice(1)] };
    assert.equal(
      parseHorizonPlan(createEnvelope("horizon_plan", 21, invalid, now), now),
      null,
      `partial 3D group unexpectedly accepted without ${field}`,
    );
  }

  const mixed = {
    ...complete,
    points: [{ ...points[0], bone_rotations: [0] }, ...points.slice(1)],
  };
  assert.equal(
    parseHorizonPlan(createEnvelope("horizon_plan", 22, mixed, now), now),
    null,
  );
});

test("generator metrics are finite, strict and available to the debug boundary", () => {
  const valid = createEnvelope("metrics", 8, {
    source: "generator",
    window_ms: 5_000,
    gauges: { last_plan_latency_ms: 2.5 },
    counters: { world_states_dropped: 3 },
    labels: { backend: "autoregressive-v0" },
  });
  assert.deepEqual(parseMetrics(valid)?.gauges, { last_plan_latency_ms: 2.5 });
  assert.equal(parseMetrics(createEnvelope("metrics", 9, {
    ...valid.payload,
    gauges: { last_plan_latency_ms: Number.POSITIVE_INFINITY },
  })), null);
  assert.equal(parseMetrics(createEnvelope("metrics", 10, {
    ...valid.payload,
    unexpected: 1,
  })), null);
});

test("desktop decodes the authoritative cross-language v1 fixture", async () => {
  const workspaceRoot = resolve(fileURLToPath(new URL("../../..", import.meta.url)));
  const fixture = await readFile(resolve(workspaceRoot, "packages", "protocol", "fixtures", "v1", "session.ndjson"), "utf8");
  const messages = fixture.trim().split(/\r?\n/).map(decodeEnvelope);
  assert.deepEqual(new Set(messages.map((message) => message.type)), new Set([
    "hello", "ready", "world_state", "horizon_plan", "cancel", "ping", "pong", "metrics", "error",
  ]));
  const plan = messages.find((message) => message.type === "horizon_plan");
  assert.ok(plan);
  assert.ok(parseHorizonPlan(plan, 1_784_491_200_040));
});
