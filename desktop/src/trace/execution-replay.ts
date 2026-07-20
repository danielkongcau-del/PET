import { createHash } from "node:crypto";

import { SafePlanExecutor, type PlanRejection } from "../motion-safety.js";
import type { HorizonPlanPayload, SurfaceState, WorldStatePayload } from "../protocol.js";
import type { JsonObject, TraceRecord } from "./format.js";

export interface ReplayDistribution {
  readonly count: number;
  readonly min: number | null;
  readonly max: number | null;
  readonly mean: number | null;
  readonly p50: number | null;
  readonly p95: number | null;
  readonly p99: number | null;
}

export interface ExecutionReplayResult {
  readonly schema: "pet-execution-replay";
  readonly version: 1;
  readonly deterministic: true;
  readonly inputRecords: number;
  readonly worldStates: number;
  readonly plansOffered: number;
  readonly recordedPlanResults: number;
  readonly decisionMatches: number;
  readonly decisionMismatches: number;
  readonly sampledPlanPoints: number;
  readonly samplesCompared: number;
  readonly samplePositionErrorPx: ReplayDistribution;
  readonly issues: readonly string[];
  readonly eventCount: number;
  readonly eventDigestSha256: string;
}

/**
 * Replays host plan acceptance and interpolation against a virtual clock.
 * It never enumerates or moves live windows; the recorded surface topology is
 * the only world visible to the safety core.
 */
export function replayTraceExecution(records: readonly TraceRecord[]): ExecutionReplayResult {
  const executor = new SafePlanExecutor();
  const sorted = [...records].sort((left, right) => left.record_seq - right.record_seq);
  const decisions = new Map<string, PlanRejection>();
  const issues: string[] = [];
  const errors: number[] = [];
  const digest = createHash("sha256");
  let surfaces: readonly SurfaceState[] = [];
  let lastFoot: { x: number; y: number } | undefined;
  let worldStates = 0;
  let plansOffered = 0;
  let recordedPlanResults = 0;
  let decisionMatches = 0;
  let decisionMismatches = 0;
  let sampledPlanPoints = 0;
  let samplesCompared = 0;
  let eventCount = 0;

  const event = (value: JsonObject): void => {
    digest.update(stableStringify(value)).update("\n");
    eventCount += 1;
  };

  for (const record of sorted) {
    const payload = asObject(record.payload);
    if (!payload) continue;
    if (record.kind === "surface_snapshot") {
      surfaces = asSurfaces(payload.surfaces) ?? surfaces;
      continue;
    }
    if (record.kind === "world_state") {
      const state = asObject(payload.state) as unknown as WorldStatePayload | null;
      const seq = asSafeInteger(payload.seq);
      const pet = state ? asObject(state.pet) : null;
      const x = pet ? asFiniteNumber(pet.foot_x) : undefined;
      const y = pet ? asFiniteNumber(pet.foot_y) : undefined;
      if (state && Array.isArray(state.surfaces)) surfaces = state.surfaces;
      if (seq === undefined || x === undefined || y === undefined) {
        issues.push(`invalid_world_state:${record.record_seq}`);
        continue;
      }
      const surfaceId = pet && typeof pet.surface_id === "string" ? pet.surface_id : undefined;
      const surface = surfaceId ? surfaces.find((candidate) => candidate.id === surfaceId) : undefined;
      const carrier = surface?.kind === "window_top" && surface.window_id
        ? { windowId: surface.window_id, surfaceId: surface.id }
        : undefined;
      executor.rememberWorldState(seq, { x, y }, carrier);
      lastFoot = { x, y };
      worldStates += 1;
      event({ type: "world", seq, x, y });
      continue;
    }
    if (record.kind === "plan_received") {
      const plan = asPlan(payload.plan);
      if (!plan) {
        issues.push(`invalid_plan:${record.record_seq}`);
        continue;
      }
      const offeredAt = asFiniteNumber(payload.received_at_ms) ?? record.wall_time_ms;
      const continuity = plan.behavior === "falling" ? lastFoot : undefined;
      const result = executor.offer(plan, surfaces, offeredAt, continuity);
      decisions.set(plan.plan_id, result);
      plansOffered += 1;
      event({ type: "offer", plan_id: plan.plan_id, based_on_seq: plan.based_on_seq, result });
      continue;
    }
    if (record.kind === "plan_result") {
      const planId = typeof payload.plan_id === "string" ? payload.plan_id : undefined;
      const recorded = typeof payload.result === "string" ? payload.result : undefined;
      if (!planId || !recorded) {
        issues.push(`invalid_plan_result:${record.record_seq}`);
        continue;
      }
      const replayed = decisions.get(planId);
      recordedPlanResults += 1;
      if (replayed === recorded) decisionMatches += 1;
      else {
        decisionMismatches += 1;
        issues.push(`decision_mismatch:${planId}:${recorded}:${replayed ?? "missing"}`);
      }
      event({ type: "decision", plan_id: planId, recorded, replayed: replayed ?? null });
      continue;
    }
    if (record.kind === "cancel") {
      executor.cancel();
      event({ type: "cancel", reason: typeof payload.reason === "string" ? payload.reason : "unknown" });
      continue;
    }
    if (record.kind !== "motion_sample") continue;
    const at = asFiniteNumber(payload.timestamp_ms) ?? record.wall_time_ms;
    const sample = executor.sample(at);
    if (!sample) continue;
    sampledPlanPoints += 1;
    const foot = asObject(payload.foot);
    const actualX = foot ? asFiniteNumber(foot.x) : undefined;
    const actualY = foot ? asFiniteNumber(foot.y) : undefined;
    const recordedPlanId = typeof payload.plan_id === "string" ? payload.plan_id : undefined;
    if (actualX !== undefined && actualY !== undefined && (!recordedPlanId || recordedPlanId === sample.planId)) {
      const error = Math.hypot(sample.foot.x - actualX, sample.foot.y - actualY);
      errors.push(error);
      samplesCompared += 1;
      lastFoot = { x: actualX, y: actualY };
      event({
        type: "sample",
        at,
        plan_id: sample.planId,
        x: round(sample.foot.x),
        y: round(sample.foot.y),
        error: round(error),
      });
    }
  }

  return {
    schema: "pet-execution-replay",
    version: 1,
    deterministic: true,
    inputRecords: sorted.length,
    worldStates,
    plansOffered,
    recordedPlanResults,
    decisionMatches,
    decisionMismatches,
    sampledPlanPoints,
    samplesCompared,
    samplePositionErrorPx: distribution(errors),
    issues,
    eventCount,
    eventDigestSha256: digest.digest("hex"),
  };
}

function asPlan(value: unknown): HorizonPlanPayload | null {
  const plan = asObject(value);
  if (!plan || typeof plan.plan_id !== "string" || asSafeInteger(plan.based_on_seq) === undefined ||
    typeof plan.behavior !== "string" || !Array.isArray(plan.points) ||
    asFiniteNumber(plan.generated_at_ms) === undefined || asFiniteNumber(plan.valid_until_ms) === undefined) return null;
  return plan as unknown as HorizonPlanPayload;
}

function asSurfaces(value: unknown): readonly SurfaceState[] | null {
  return Array.isArray(value) ? value as readonly SurfaceState[] : null;
}

function asObject(value: unknown): Record<string, unknown> | null {
  return typeof value === "object" && value !== null && !Array.isArray(value)
    ? value as Record<string, unknown>
    : null;
}

function asSafeInteger(value: unknown): number | undefined {
  return typeof value === "number" && Number.isSafeInteger(value) && value >= 0 ? value : undefined;
}

function asFiniteNumber(value: unknown): number | undefined {
  return typeof value === "number" && Number.isFinite(value) ? value : undefined;
}

function distribution(values: readonly number[]): ReplayDistribution {
  const sorted = values.filter(Number.isFinite).slice().sort((left, right) => left - right);
  if (sorted.length === 0) return { count: 0, min: null, max: null, mean: null, p50: null, p95: null, p99: null };
  const percentile = (q: number): number => {
    const index = (sorted.length - 1) * q;
    const lower = Math.floor(index);
    const upper = Math.ceil(index);
    const alpha = index - lower;
    return sorted[lower]! + (sorted[upper]! - sorted[lower]!) * alpha;
  };
  return {
    count: sorted.length,
    min: sorted[0]!,
    max: sorted.at(-1)!,
    mean: sorted.reduce((sum, value) => sum + value, 0) / sorted.length,
    p50: percentile(0.5),
    p95: percentile(0.95),
    p99: percentile(0.99),
  };
}

function stableStringify(value: unknown): string {
  if (Array.isArray(value)) return `[${value.map(stableStringify).join(",")}]`;
  if (value && typeof value === "object") {
    const object = value as Record<string, unknown>;
    return `{${Object.keys(object).sort().map((key) => `${JSON.stringify(key)}:${stableStringify(object[key])}`).join(",")}}`;
  }
  return JSON.stringify(value);
}

function round(value: number): number {
  return Math.round(value * 1_000_000) / 1_000_000;
}
