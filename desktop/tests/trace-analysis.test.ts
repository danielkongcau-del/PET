import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import { gzipSync } from "node:zlib";

import { analyzeTraceEpisode, loadTraceEpisode, type TraceRecord } from "../src/trace/analysis.js";
import { renderTraceReportHtml, writeTraceReport } from "../src/trace/report.js";

function record(seq: number, kind: TraceRecord["kind"], payload: TraceRecord["payload"], elapsedMs = seq * 10): TraceRecord {
  return { schema: "pet-trace", version: 1, record_seq: seq, wall_time_ms: 1_800_000_000_000 + elapsedMs, elapsed_us: elapsedMs * 1_000, kind, payload };
}

function fixture(): TraceRecord[] {
  return [
    record(0, "surface_snapshot", { captured_at_ms: 1, displays: [], windows: [], scene: {}, surfaces: [{ id: "s", x1: 0, x2: 500, y: 100 }] }),
    record(1, "world_state", { seq: 1, timestamp_ms: 1_800_000_000_010, state: { clicks: [{ id: "click-1", timestamp_ms: 1_800_000_000_010 }] } }),
    record(2, "plan_received", { received_at_ms: 1_800_000_000_020, plan: { generated_at_ms: 1_800_000_000_000 } }),
    record(3, "plan_received", { latency_ms: 40, plan: {} }),
    record(4, "plan_result", { plan_id: "a", result: "accepted" }),
    record(5, "plan_result", { plan_id: "b", result: "missing_anchor" }),
    record(6, "cancel", { reason: "topology_change" }),
    record(7, "motion_sample", { timestamp_ms: 1_800_000_000_070, dt_ms: 16, foot: { x: 10, y: 100 }, velocity: { x: 0, y: 0 }, behavior: "click_reaction", surface_id: "s" }),
    record(8, "motion_sample", { timestamp_ms: 1_800_000_000_086, dt_ms: 16, foot: { x: 10, y: 100 }, velocity: { x: 0, y: 0 }, behavior: "jump" }),
    record(9, "motion_sample", { timestamp_ms: 1_800_000_000_102, dt_ms: 16, foot: { x: 10, y: 100 }, velocity: { x: 0, y: 0 }, behavior: "landing", surface_id: "s" }),
    record(10, "generator_status", { status: "ready", restart_count: 2 }),
    record(11, "process_metrics", { host_cpu_percent: 2.5, host_rss_bytes: 1024 }),
    record(12, "recording_gap", { dropped_motion_samples: 3 }),
  ];
}

test("trace analysis computes deterministic plan, motion, click, process and integrity metrics", () => {
  const records = fixture();
  const a = analyzeTraceEpisode(records);
  const b = analyzeTraceEpisode(records);
  assert.equal(a.deterministicDigest, b.deterministicDigest);
  assert.deepEqual({ ...a.plans.latencyMs, p99: Number(a.plans.latencyMs.p99?.toFixed(3)) }, { count: 2, min: 20, max: 40, mean: 30, p50: 30, p95: 39, p99: 39.8 });
  assert.equal(a.plans.accepted, 1);
  assert.equal(a.plans.rejected, 1);
  assert.equal(a.plans.rejectionReasons.missing_anchor, 1);
  assert.equal(a.plans.normalTopologyCancellations, 1);
  assert.equal(a.runtime.generatorRestarts, 2);
  assert.equal(a.runtime.recordingGaps, 1);
  assert.equal(a.runtime.droppedMotionSamples, 3);
  assert.equal(a.interaction.clickResponseMs.p50, 60);
  assert.equal(a.motion.landingAttempts, 1);
  assert.equal(a.motion.landingSuccessRate, 1);
  assert.equal(a.process.hostCpuPercent.p50, 2.5);
  assert.ok(a.series.motionX.length <= 5_000);
});

test("thresholds and baseline regression produce explicit FAIL verdicts", () => {
  const baseline = analyzeTraceEpisode(fixture(), { thresholds: { recordingGapsMax: 2 } });
  const changed = fixture().map((entry) => entry.kind === "plan_received" ? record(entry.record_seq, "plan_received", { latency_ms: 200, plan: {} }, entry.elapsed_us / 1_000) : entry);
  const report = analyzeTraceEpisode(changed, { thresholds: { recordingGapsMax: 2 }, baseline });
  assert.equal(report.checks.find((entry) => entry.name === "plan_latency_p95_ms")?.status, "FAIL");
  assert.equal(report.regressions.find((entry) => entry.name === "regression_plan_latency_p95_ms")?.status, "FAIL");
  assert.equal(report.verdict, "FAIL");
});

test("screen escape and falling through a known surface are hard failures while generator process gauges are recognized", () => {
  const records = [
    record(0, "surface_snapshot", { displays: [{ id: "d", bounds: { x: 0, y: 0, width: 200, height: 200 } }], windows: [], scene: {}, surfaces: [{ id: "s", x1: 0, x2: 150, y: 100, enabled: true, occluded: false }] }),
    record(1, "motion_sample", { timestamp_ms: 1010, dt_ms: 16, foot: { x: 20, y: 90 }, velocity: { x: 0, y: 100 }, behavior: "falling" }),
    record(2, "motion_sample", { timestamp_ms: 1026, dt_ms: 16, foot: { x: 20, y: 110 }, velocity: { x: 0, y: 100 }, behavior: "falling" }),
    record(3, "motion_sample", { timestamp_ms: 1042, dt_ms: 16, foot: { x: 220, y: 110 }, velocity: { x: 0, y: 0 }, behavior: "falling" }),
    record(4, "generator_metrics", { metrics: { gauges: { process_cpu_percent: 3.25, process_rss_bytes: 4096 } } }),
  ];
  const report = analyzeTraceEpisode(records);
  assert.equal(report.runtime.screenBoundsViolations, 1);
  assert.equal(report.runtime.surfacePenetrations, 1);
  assert.equal(report.process.generatorCpuPercent.p50, 3.25);
  assert.equal(report.process.generatorRssBytes.p50, 4096);
  assert.equal(report.checks.find((entry) => entry.name === "screen_bounds_violations")?.status, "FAIL");
  assert.equal(report.checks.find((entry) => entry.name === "surface_penetrations")?.status, "FAIL");
});

test("loader validates gzip hashes and report is an offline escaped single file", () => {
  const directory = mkdtempSync(join(tmpdir(), "pet-trace-analysis-"));
  const compressed = gzipSync(`${fixture().map((entry) => JSON.stringify(entry)).join("\n")}\n`);
  const sha256 = createHash("sha256").update(compressed).digest("hex");
  writeFileSync(join(directory, "trace-000001.ndjson.gz"), compressed);
  writeFileSync(join(directory, "manifest.json"), JSON.stringify({ chunks: [{ file: "trace-000001.ndjson.gz", sha256 }], incomplete: false }));
  const loaded = loadTraceEpisode(directory);
  assert.equal(loaded.records.length, fixture().length);
  assert.deepEqual(loaded.issues, []);
  const original = analyzeTraceEpisode(loaded);
  const malicious = { ...original, plans: { ...original.plans, rejectionReasons: { "<img src=x onerror=alert(1)>": 1 } } };
  const html = renderTraceReportHtml(malicious);
  assert.ok(!html.includes("<img src=x"));
  assert.ok(!html.includes("https://"));
  assert.ok(html.includes("\\u003cimg") || html.includes("&lt;img"));
  const output = writeTraceReport(join(directory, "report"), original);
  assert.equal(JSON.parse(readFileSync(output.jsonPath, "utf8")).deterministicDigest, original.deterministicDigest);
  assert.ok(readFileSync(output.htmlPath, "utf8").includes("PET trace report"));
});
