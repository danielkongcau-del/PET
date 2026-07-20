import { createHash } from "node:crypto";
import { existsSync, readFileSync, readdirSync } from "node:fs";
import { gunzipSync } from "node:zlib";
import { basename, join } from "node:path";

import { TRACE_SCHEMA, TRACE_VERSION, type TraceRecord as FormatTraceRecord } from "./format.js";

export { TRACE_SCHEMA, TRACE_VERSION };
export type TraceRecord = FormatTraceRecord;

export interface LoadedTraceFile {
  readonly name: string;
  readonly sha256: string;
  readonly recordCount: number;
  readonly partial: boolean;
}

export interface LoadedTraceEpisode {
  readonly manifest: Record<string, unknown> | null;
  readonly records: readonly TraceRecord[];
  readonly files: readonly LoadedTraceFile[];
  readonly issues: readonly string[];
}

export type Verdict = "PASS" | "WARN" | "FAIL";

export interface TraceThresholds {
  readonly version: 1;
  readonly planLatencyP95Ms: number;
  readonly clickResponseP95Ms: number;
  readonly carrierErrorP95Px: number;
  readonly positionJumpMaxPx: number;
  readonly landingSuccessRateMin: number;
  readonly fallbackRatioMax: number;
  readonly recordingGapsMax: number;
  readonly invalidValuesMax: number;
  readonly screenBoundsViolationsMax: number;
  readonly surfacePenetrationsMax: number;
  readonly regressionRatioMax: number;
}

export const DEFAULT_TRACE_THRESHOLDS: TraceThresholds = Object.freeze({
  version: 1,
  planLatencyP95Ms: 100,
  clickResponseP95Ms: 100,
  carrierErrorP95Px: 2,
  positionJumpMaxPx: 4,
  landingSuccessRateMin: 0.98,
  fallbackRatioMax: 0.05,
  recordingGapsMax: 0,
  invalidValuesMax: 0,
  screenBoundsViolationsMax: 0,
  surfacePenetrationsMax: 0,
  regressionRatioMax: 1.2,
});

export interface MetricDistribution {
  readonly count: number;
  readonly min: number | null;
  readonly max: number | null;
  readonly mean: number | null;
  readonly p50: number | null;
  readonly p95: number | null;
  readonly p99: number | null;
}

export interface MetricCheck {
  readonly name: string;
  readonly value: number | null;
  readonly threshold: number;
  readonly operator: "<=" | ">=";
  readonly status: Verdict;
  readonly reason?: string;
}

export interface ReportSeriesPoint {
  readonly elapsed_ms: number;
  readonly value: number;
}

export interface TraceAnalysisReport {
  readonly schema: "pet-trace-report";
  readonly version: 1;
  readonly source: {
    readonly recordCount: number;
    readonly firstRecordSeq: number | null;
    readonly lastRecordSeq: number | null;
    readonly durationMs: number;
    readonly incomplete: boolean;
    readonly integrityOk: boolean;
    readonly issues: readonly string[];
    readonly files: readonly LoadedTraceFile[];
  };
  readonly counts: Readonly<Record<string, number>>;
  readonly plans: {
    readonly received: number;
    readonly accepted: number;
    readonly rejected: number;
    readonly cancelled: number;
    readonly rejectionReasons: Readonly<Record<string, number>>;
    readonly cancellationReasons: Readonly<Record<string, number>>;
    readonly normalTopologyCancellations: number;
    readonly abnormalCancellations: number;
    readonly latencyMs: MetricDistribution;
  };
  readonly runtime: {
    readonly fallbackRatio: number;
    readonly generatorRestarts: number;
    readonly recordingGaps: number;
    readonly droppedMotionSamples: number;
    readonly invalidValues: number;
    readonly screenBoundsViolations: number;
    readonly surfacePenetrations: number;
  };
  readonly motion: {
    readonly positionJumpPx: MetricDistribution;
    readonly velocitySeamPxPerS: MetricDistribution;
    readonly accelerationPxPerS2: MetricDistribution;
    readonly jerkPxPerS3: MetricDistribution;
    readonly carrierErrorPx: MetricDistribution;
    readonly landingAttempts: number;
    readonly landingSuccesses: number;
    readonly landingSuccessRate: number | null;
  };
  readonly interaction: {
    readonly clicks: number;
    readonly clickResponses: number;
    readonly clickResponseMs: MetricDistribution;
  };
  readonly process: {
    readonly hostCpuPercent: MetricDistribution;
    readonly hostRssBytes: MetricDistribution;
    readonly generatorCpuPercent: MetricDistribution;
    readonly generatorRssBytes: MetricDistribution;
  };
  readonly checks: readonly MetricCheck[];
  readonly regressions: readonly MetricCheck[];
  readonly verdict: Verdict;
  readonly series: {
    readonly motionX: readonly ReportSeriesPoint[];
    readonly motionY: readonly ReportSeriesPoint[];
    readonly carrierError: readonly ReportSeriesPoint[];
    readonly planLatency: readonly ReportSeriesPoint[];
    readonly hostCpu: readonly ReportSeriesPoint[];
    readonly hostRss: readonly ReportSeriesPoint[];
  };
  readonly deterministicDigest: string;
}

interface MotionSample {
  readonly timeMs: number;
  readonly elapsedMs: number;
  readonly dtMs: number;
  readonly x: number;
  readonly y: number;
  readonly vx: number;
  readonly vy: number;
  readonly behavior: string;
  readonly surfaceId?: string;
}

interface SurfaceGeometry { readonly x1: number; readonly x2: number; readonly y: number }
interface DisplayGeometry { readonly x: number; readonly y: number; readonly width: number; readonly height: number }

const AIRBORNE_BEHAVIORS = new Set(["jump", "falling"]);
const NORMAL_TOPOLOGY_TOKENS = [
  "topology", "surface_changed", "surface_missing", "target_missing", "target_closed",
  "window_closed", "window_moved", "window_resize", "maximiz", "attachment_lost", "carrier_changed", "display_change",
];

export function loadTraceEpisode(directory: string): LoadedTraceEpisode {
  const issues: string[] = [];
  let manifest: Record<string, unknown> | null = null;
  const manifestPath = join(directory, "manifest.json");
  if (existsSync(manifestPath)) {
    try {
      const parsed = JSON.parse(readFileSync(manifestPath, "utf8")) as unknown;
      if (isRecord(parsed)) manifest = parsed;
      else issues.push("manifest_not_object");
    } catch (error) {
      issues.push(`manifest_invalid:${errorMessage(error)}`);
    }
  } else {
    issues.push("manifest_missing");
  }

  const names = readdirSync(directory)
    .filter((name) => /^trace-.*\.ndjson\.gz$/i.test(name) || /trace-.*\.partial$/i.test(name))
    .sort((a, b) => a.localeCompare(b, "en"));
  if (names.length === 0) issues.push("trace_files_missing");

  const records: TraceRecord[] = [];
  const files: LoadedTraceFile[] = [];
  for (const name of names) {
    const path = join(directory, name);
    let raw: Buffer;
    try { raw = readFileSync(path); }
    catch (error) { issues.push(`file_unreadable:${name}:${errorMessage(error)}`); continue; }
    const sha256 = createHash("sha256").update(raw).digest("hex");
    verifyManifestHash(manifest, name, sha256, issues);
    const partial = name.toLowerCase().endsWith(".partial");
    let text: string;
    try {
      const gzipEncoded = name.toLowerCase().endsWith(".gz") || name.toLowerCase().endsWith(".gz.partial") || (raw[0] === 0x1f && raw[1] === 0x8b);
      text = gzipEncoded ? gunzipSync(raw).toString("utf8") : raw.toString("utf8");
    }
    catch (error) { issues.push(`file_decode_failed:${name}:${errorMessage(error)}`); continue; }
    const before = records.length;
    const lines = text.split(/\r?\n/);
    for (let index = 0; index < lines.length; index += 1) {
      const line = lines[index];
      if (!line?.trim()) continue;
      try {
        const parsed = JSON.parse(line) as unknown;
        const validation = validateTraceRecord(parsed);
        if (typeof validation === "string") issues.push(`record_invalid:${name}:${index + 1}:${validation}`);
        else records.push(validation);
      } catch (error) {
        // A process crash may leave only the final partial line truncated.
        if (!(partial && index === lines.length - 1)) issues.push(`record_json_invalid:${name}:${index + 1}:${errorMessage(error)}`);
      }
    }
    files.push({ name: basename(name), sha256, recordCount: records.length - before, partial });
  }
  validateOrdering(records, issues);
  return { manifest, records, files, issues };
}

export function analyzeTraceEpisode(
  episode: Pick<LoadedTraceEpisode, "records" | "files" | "issues" | "manifest"> | readonly TraceRecord[],
  options: { readonly thresholds?: Partial<TraceThresholds>; readonly baseline?: TraceAnalysisReport; readonly maxSeriesPoints?: number } = {},
): TraceAnalysisReport {
  const loaded = Array.isArray(episode)
    ? { records: episode as readonly TraceRecord[], files: [] as readonly LoadedTraceFile[], issues: [] as readonly string[], manifest: null }
    : episode as Pick<LoadedTraceEpisode, "records" | "files" | "issues" | "manifest">;
  const records = [...loaded.records].sort((a, b) => a.record_seq - b.record_seq);
  const thresholds = { ...DEFAULT_TRACE_THRESHOLDS, ...options.thresholds };
  const counts: Record<string, number> = {};
  const rejectionReasons: Record<string, number> = {};
  const cancellationReasons: Record<string, number> = {};
  const planLatencies: number[] = [];
  const positionJumps: number[] = [];
  const velocitySeams: number[] = [];
  const accelerations: number[] = [];
  const jerks: number[] = [];
  const carrierErrors: number[] = [];
  const clickResponses: number[] = [];
  const hostCpu: number[] = [];
  const hostRss: number[] = [];
  const generatorCpu: number[] = [];
  const generatorRss: number[] = [];
  const motionSamples: MotionSample[] = [];
  const pendingClicks = new Map<string, number>();
  const seenClicks = new Set<string>();
  const surfaceGeometry = new Map<string, SurfaceGeometry>();
  let displayGeometry: readonly DisplayGeometry[] = [];
  const series = {
    motionX: [] as ReportSeriesPoint[], motionY: [] as ReportSeriesPoint[], carrierError: [] as ReportSeriesPoint[],
    planLatency: [] as ReportSeriesPoint[], hostCpu: [] as ReportSeriesPoint[], hostRss: [] as ReportSeriesPoint[],
  };

  let accepted = 0;
  let rejected = 0;
  let cancelled = 0;
  let normalTopologyCancellations = 0;
  let restartCount = 0;
  let recordingGaps = 0;
  let droppedMotionSamples = 0;
  let invalidValues = 0;
  let screenBoundsViolations = 0;
  let surfacePenetrations = 0;
  let fallbackMs = 0;
  let sampledMs = 0;
  let clicks = 0;
  let airborne = false;
  let landingAttempts = 0;
  let landingSuccesses = 0;
  let previousMotion: MotionSample | undefined;
  let previousAcceleration: { x: number; y: number } | undefined;

  for (const record of records) {
    counts[record.kind] = (counts[record.kind] ?? 0) + 1;
    if (!allFinite(record.payload)) invalidValues += 1;
    const payload = isRecord(record.payload) ? record.payload : {};
    if (!isRecord(record.payload)) invalidValues += 1;
    if (record.kind === "surface_snapshot") {
      updateSurfaces(surfaceGeometry, payload);
      displayGeometry = parseDisplays(payload);
    }

    if (record.kind === "world_state") {
      const state = recordObject(payload, "state") ?? payload;
      const clickArray = arrayValue(state.clicks);
      for (const rawClick of clickArray) {
        if (!isRecord(rawClick)) continue;
        const id = stringValue(rawClick.id) ?? `seq-${record.record_seq}-${clicks}`;
        if (seenClicks.has(id)) continue;
        const time = numberValue(rawClick.timestamp_ms) ?? record.wall_time_ms;
        seenClicks.add(id);
        pendingClicks.set(id, time);
        clicks += 1;
      }
    } else if (record.kind === "plan_received") {
      const plan = recordObject(payload, "plan") ?? payload;
      const receivedAt = numberValue(payload.received_at_ms) ?? record.wall_time_ms;
      const generatedAt = numberValue(plan.generated_at_ms, payload.generated_at_ms);
      const direct = numberValue(payload.latency_ms, payload.generation_latency_ms, payload.plan_latency_ms);
      const latency = direct ?? (generatedAt === undefined ? undefined : receivedAt - generatedAt);
      if (latency !== undefined && latency >= 0 && Number.isFinite(latency)) {
        planLatencies.push(latency);
        series.planLatency.push({ elapsed_ms: record.elapsed_us / 1_000, value: latency });
      }
    } else if (record.kind === "plan_result") {
      const result = (stringValue(payload.result, payload.status) ?? "unknown").toLowerCase();
      const reason = (stringValue(payload.reason) ?? (result === "accepted" ? "accepted" : result)).toLowerCase();
      if (result === "accepted") accepted += 1;
      else if (result === "cancel" || result === "cancelled" || result === "canceled") {
        cancelled += 1;
        increment(cancellationReasons, reason);
        if (isNormalTopologyReason(reason)) normalTopologyCancellations += 1;
      } else {
        rejected += 1;
        increment(rejectionReasons, reason);
      }
    } else if (record.kind === "cancel") {
      const reason = (stringValue(payload.reason) ?? "unknown").toLowerCase();
      cancelled += 1;
      increment(cancellationReasons, reason);
      if (isNormalTopologyReason(reason)) normalTopologyCancellations += 1;
    } else if (record.kind === "generator_status") {
      restartCount = Math.max(restartCount, Math.max(0, Math.trunc(numberValue(payload.restart_count) ?? 0)));
    } else if (record.kind === "recording_gap") {
      recordingGaps += 1;
      droppedMotionSamples += Math.max(0, Math.trunc(numberValue(payload.dropped_motion_samples, payload.dropped) ?? 0));
    } else if (record.kind === "process_metrics") {
      collectMetric(payload, ["host_cpu_percent"], hostCpu, series.hostCpu, record);
      collectMetric(payload, ["host_rss_bytes"], hostRss, series.hostRss, record);
      collectMetric(payload, ["generator_cpu_percent"], generatorCpu);
      collectMetric(payload, ["generator_rss_bytes"], generatorRss);
    } else if (record.kind === "generator_metrics") {
      const metrics = recordObject(payload, "metrics") ?? payload;
      const gauges = recordObject(metrics, "gauges") ?? metrics;
      collectMetric(gauges, ["process_cpu_percent", "cpu_percent", "generator_cpu_percent"], generatorCpu);
      collectMetric(gauges, ["process_rss_bytes", "rss_bytes", "generator_rss_bytes"], generatorRss);
    } else if (record.kind === "motion_sample") {
      const sample = parseMotionSample(record);
      if (!sample) { invalidValues += 1; continue; }
      motionSamples.push(sample);
      if (displayGeometry.length > 0 && !displayGeometry.some((display) => pointInDisplay(sample.x, sample.y, display))) {
        screenBoundsViolations += 1;
      }
      series.motionX.push({ elapsed_ms: sample.elapsedMs, value: sample.x });
      series.motionY.push({ elapsed_ms: sample.elapsedMs, value: sample.y });
      const sampleDt = sample.dtMs > 0 ? sample.dtMs : (previousMotion ? Math.max(0, sample.timeMs - previousMotion.timeMs) : 0);
      sampledMs += sampleDt;
      if (sample.behavior === "fallback") fallbackMs += sampleDt;

      if (pendingClicks.size > 0 && sample.behavior === "click_reaction") {
        for (const [id, clickTime] of pendingClicks) {
          const response = sample.timeMs - clickTime;
          if (response >= 0) clickResponses.push(response);
          pendingClicks.delete(id);
        }
      }

      const nowAirborne = AIRBORNE_BEHAVIORS.has(sample.behavior) && !sample.surfaceId;
      if (!airborne && nowAirborne) { landingAttempts += 1; airborne = true; }
      if (airborne && !nowAirborne && sample.surfaceId) { landingSuccesses += 1; airborne = false; }

      if (sample.surfaceId) {
        const surface = surfaceGeometry.get(sample.surfaceId);
        const directError = numberValue(payload.carrier_error_px);
        const carrierError = directError ?? (surface ? pointToSurfaceError(sample.x, sample.y, surface) : undefined);
        if (carrierError !== undefined && Number.isFinite(carrierError)) {
          carrierErrors.push(carrierError);
          series.carrierError.push({ elapsed_ms: sample.elapsedMs, value: carrierError });
        }
      }

      if (previousMotion) {
        const dtSec = Math.max(0.001, sampleDt / 1_000);
        const expectedDx = (previousMotion.vx + sample.vx) * 0.5 * dtSec;
        const expectedDy = (previousMotion.vy + sample.vy) * 0.5 * dtSec;
        positionJumps.push(Math.hypot(sample.x - previousMotion.x - expectedDx, sample.y - previousMotion.y - expectedDy));
        velocitySeams.push(Math.hypot(sample.vx - previousMotion.vx, sample.vy - previousMotion.vy));
        const acceleration = { x: (sample.vx - previousMotion.vx) / dtSec, y: (sample.vy - previousMotion.vy) / dtSec };
        accelerations.push(Math.hypot(acceleration.x, acceleration.y));
        if (previousAcceleration) jerks.push(Math.hypot(acceleration.x - previousAcceleration.x, acceleration.y - previousAcceleration.y) / dtSec);
        previousAcceleration = acceleration;
        if (crossesSurfaceWithoutLanding(previousMotion, sample, surfaceGeometry)) surfacePenetrations += 1;
      }
      previousMotion = sample;
    }
  }
  if (airborne) airborne = false; // An unfinished final attempt remains an unsuccessful attempt.

  const latencyDistribution = distribution(planLatencies);
  const carrierDistribution = distribution(carrierErrors);
  const clickDistribution = distribution(clickResponses);
  const jumpDistribution = distribution(positionJumps);
  const landingRate = landingAttempts === 0 ? null : landingSuccesses / landingAttempts;
  const fallbackRatio = sampledMs > 0 ? fallbackMs / sampledMs : 0;
  const checks: MetricCheck[] = [
    maxCheck("plan_latency_p95_ms", latencyDistribution.p95, thresholds.planLatencyP95Ms),
    maxCheck("click_response_p95_ms", clickDistribution.p95, thresholds.clickResponseP95Ms),
    maxCheck("carrier_error_p95_px", carrierDistribution.p95, thresholds.carrierErrorP95Px),
    maxCheck("position_jump_max_px", jumpDistribution.max, thresholds.positionJumpMaxPx),
    minCheck("landing_success_rate", landingRate, thresholds.landingSuccessRateMin),
    maxCheck("fallback_ratio", fallbackRatio, thresholds.fallbackRatioMax),
    maxCheck("recording_gaps", recordingGaps, thresholds.recordingGapsMax),
    maxCheck("invalid_values", invalidValues, thresholds.invalidValuesMax),
    maxCheck("screen_bounds_violations", screenBoundsViolations, thresholds.screenBoundsViolationsMax),
    maxCheck("surface_penetrations", surfacePenetrations, thresholds.surfacePenetrationsMax),
  ];

  const regressions = options.baseline ? compareBaseline(
    { planLatencyP95: latencyDistribution.p95, clickResponseP95: clickDistribution.p95, carrierErrorP95: carrierDistribution.p95 },
    options.baseline,
    thresholds.regressionRatioMax,
  ) : [];
  const issueVerdict: Verdict = loaded.issues.length > 0 ? (loaded.issues.some((issue) => issue.includes("sha256_mismatch") || issue.startsWith("record_invalid")) ? "FAIL" : "WARN") : "PASS";
  const verdict = worstVerdict(issueVerdict, ...checks.map((check) => check.status), ...regressions.map((check) => check.status));
  const maxPoints = Math.max(1, Math.min(5_000, Math.trunc(options.maxSeriesPoints ?? 5_000)));
  const source = {
    recordCount: records.length,
    firstRecordSeq: records[0]?.record_seq ?? null,
    lastRecordSeq: records.at(-1)?.record_seq ?? null,
    durationMs: records.length > 1 ? Math.max(0, (records.at(-1)?.elapsed_us ?? 0) - (records[0]?.elapsed_us ?? 0)) / 1_000 : 0,
    incomplete: loaded.files.some((file) => file.partial) || loaded.manifest?.incomplete === true,
    integrityOk: !loaded.issues.some((issue) => issue.includes("sha256_mismatch") || issue.startsWith("record_invalid") || issue.startsWith("record_json_invalid")),
    issues: [...loaded.issues], files: [...loaded.files],
  };
  const reportWithoutDigest = {
    schema: "pet-trace-report" as const, version: 1 as const, source, counts: sortNumericRecord(counts),
    plans: {
      received: counts.plan_received ?? 0, accepted, rejected, cancelled,
      rejectionReasons: sortNumericRecord(rejectionReasons), cancellationReasons: sortNumericRecord(cancellationReasons),
      normalTopologyCancellations, abnormalCancellations: Math.max(0, cancelled - normalTopologyCancellations), latencyMs: latencyDistribution,
    },
    runtime: { fallbackRatio, generatorRestarts: restartCount, recordingGaps, droppedMotionSamples, invalidValues, screenBoundsViolations, surfacePenetrations },
    motion: {
      positionJumpPx: jumpDistribution, velocitySeamPxPerS: distribution(velocitySeams),
      accelerationPxPerS2: distribution(accelerations), jerkPxPerS3: distribution(jerks), carrierErrorPx: carrierDistribution,
      landingAttempts, landingSuccesses, landingSuccessRate: landingRate,
    },
    interaction: { clicks, clickResponses: clickResponses.length, clickResponseMs: clickDistribution },
    process: { hostCpuPercent: distribution(hostCpu), hostRssBytes: distribution(hostRss), generatorCpuPercent: distribution(generatorCpu), generatorRssBytes: distribution(generatorRss) },
    checks, regressions, verdict,
    series: {
      motionX: downsampleSeries(series.motionX, maxPoints), motionY: downsampleSeries(series.motionY, maxPoints),
      carrierError: downsampleSeries(series.carrierError, maxPoints), planLatency: downsampleSeries(series.planLatency, maxPoints),
      hostCpu: downsampleSeries(series.hostCpu, maxPoints), hostRss: downsampleSeries(series.hostRss, maxPoints),
    },
  };
  const deterministicDigest = createHash("sha256").update(stableStringify(reportWithoutDigest)).digest("hex");
  return { ...reportWithoutDigest, deterministicDigest };
}

export function distribution(values: readonly number[]): MetricDistribution {
  const sorted = values.filter(Number.isFinite).slice().sort((a, b) => a - b);
  if (sorted.length === 0) return { count: 0, min: null, max: null, mean: null, p50: null, p95: null, p99: null };
  const sum = sorted.reduce((total, value) => total + value, 0);
  return {
    count: sorted.length, min: sorted[0] ?? null, max: sorted.at(-1) ?? null, mean: sum / sorted.length,
    p50: percentile(sorted, 0.5), p95: percentile(sorted, 0.95), p99: percentile(sorted, 0.99),
  };
}

export function downsampleSeries(points: readonly ReportSeriesPoint[], maximum = 5_000): readonly ReportSeriesPoint[] {
  if (points.length <= maximum) return [...points];
  // Deterministic min/max bucket sampling retains short spikes better than stride sampling.
  const output: ReportSeriesPoint[] = [points[0]!];
  const interiorBudget = Math.max(0, maximum - 2);
  const bucketCount = Math.max(1, Math.floor(interiorBudget / 2));
  const interior = points.slice(1, -1);
  for (let bucket = 0; bucket < bucketCount && output.length < maximum - 1; bucket += 1) {
    const start = Math.floor(bucket * interior.length / bucketCount);
    const end = Math.max(start + 1, Math.floor((bucket + 1) * interior.length / bucketCount));
    const slice = interior.slice(start, end);
    if (slice.length === 0) continue;
    let min = slice[0]!;
    let max = slice[0]!;
    for (const point of slice) {
      if (point.value < min.value) min = point;
      if (point.value > max.value) max = point;
    }
    const ordered = min.elapsed_ms <= max.elapsed_ms ? [min, max] : [max, min];
    for (const point of ordered) if (output.length < maximum - 1 && output.at(-1) !== point) output.push(point);
  }
  output.push(points.at(-1)!);
  return output.slice(0, maximum);
}

function validateTraceRecord(value: unknown): TraceRecord | string {
  if (!isRecord(value)) return "not_object";
  if (value.schema !== TRACE_SCHEMA || value.version !== TRACE_VERSION) return "schema_or_version";
  if (!Number.isSafeInteger(value.record_seq) || (value.record_seq as number) < 0) return "record_seq";
  if (!finite(value.wall_time_ms) || value.wall_time_ms < 0) return "wall_time_ms";
  if (!finite(value.elapsed_us) || value.elapsed_us < 0) return "elapsed_us";
  if (typeof value.kind !== "string" || value.kind.length < 1 || value.kind.length > 128) return "kind";
  if (!isRecord(value.payload)) return "payload";
  return value as unknown as TraceRecord;
}

function validateOrdering(records: readonly TraceRecord[], issues: string[]): void {
  let priorSeq = -1;
  let priorWall = -1;
  let priorElapsed = -1;
  for (const record of records) {
    if (record.record_seq <= priorSeq) issues.push(`record_seq_non_monotonic:${record.record_seq}`);
    if (record.wall_time_ms < priorWall) issues.push(`wall_time_non_monotonic:${record.record_seq}`);
    if (record.elapsed_us < priorElapsed) issues.push(`elapsed_time_non_monotonic:${record.record_seq}`);
    priorSeq = record.record_seq; priorWall = record.wall_time_ms; priorElapsed = record.elapsed_us;
  }
}

function verifyManifestHash(manifest: Record<string, unknown> | null, name: string, actual: string, issues: string[]): void {
  if (!manifest) return;
  const collections = [manifest.chunks, manifest.files, manifest.parts, manifest.segments];
  for (const collection of collections) {
    if (!Array.isArray(collection)) continue;
    const match = collection.find((entry) => isRecord(entry) && (entry.name === name || entry.path === name || entry.file === name));
    if (!isRecord(match)) continue;
    const expected = stringValue(match.sha256, match.sha_256, match.hash);
    if (expected && expected.toLowerCase() !== actual) issues.push(`sha256_mismatch:${name}`);
    return;
  }
}

function updateSurfaces(target: Map<string, SurfaceGeometry>, payload: Record<string, unknown>): void {
  const surfaces = arrayValue(payload.surfaces);
  target.clear();
  for (const item of surfaces) {
    if (!isRecord(item)) continue;
    const id = stringValue(item.id);
    const x1 = numberValue(item.x1); const x2 = numberValue(item.x2); const y = numberValue(item.y);
    if (item.enabled !== false && item.occluded !== true && id && x1 !== undefined && x2 !== undefined && y !== undefined) target.set(id, { x1, x2, y });
  }
}

function parseDisplays(payload: Record<string, unknown>): readonly DisplayGeometry[] {
  const result: DisplayGeometry[] = [];
  for (const item of arrayValue(payload.displays)) {
    if (!isRecord(item)) continue;
    const bounds = isRecord(item.bounds) ? item.bounds : item;
    const x = numberValue(bounds.x); const y = numberValue(bounds.y);
    const width = numberValue(bounds.width); const height = numberValue(bounds.height);
    if (x !== undefined && y !== undefined && width !== undefined && height !== undefined && width > 0 && height > 0) {
      result.push({ x, y, width, height });
    }
  }
  return result;
}

function pointInDisplay(x: number, y: number, display: DisplayGeometry): boolean {
  // A foot exactly on the lower/right work-area boundary is still considered visible.
  return x >= display.x && x <= display.x + display.width && y >= display.y && y <= display.y + display.height;
}

function crossesSurfaceWithoutLanding(previous: MotionSample, current: MotionSample, surfaces: ReadonlyMap<string, SurfaceGeometry>): boolean {
  if (current.surfaceId) return false;
  if (current.y <= previous.y) return false;
  for (const [id, surface] of surfaces) {
    if (id === previous.surfaceId) continue;
    if (!(previous.y < surface.y - 0.5 && current.y > surface.y + 0.5)) continue;
    const ratio = (surface.y - previous.y) / (current.y - previous.y);
    const crossingX = previous.x + (current.x - previous.x) * ratio;
    if (crossingX >= surface.x1 && crossingX <= surface.x2) return true;
  }
  return false;
}

function parseMotionSample(record: TraceRecord): MotionSample | null {
  if (!isRecord(record.payload)) return null;
  const p = record.payload;
  const foot = recordObject(p, "foot");
  const velocity = recordObject(p, "velocity");
  const x = numberValue(foot?.x, p.foot_x, p.x);
  const y = numberValue(foot?.y, p.foot_y, p.y);
  const vx = numberValue(velocity?.x, p.vx) ?? 0;
  const vy = numberValue(velocity?.y, p.vy) ?? 0;
  if (x === undefined || y === undefined || ![x, y, vx, vy].every(Number.isFinite)) return null;
  const surfaceId = stringValue(p.surface_id);
  return {
    timeMs: numberValue(p.timestamp_ms) ?? record.wall_time_ms,
    elapsedMs: record.elapsed_us / 1_000,
    dtMs: Math.max(0, numberValue(p.dt_ms) ?? 0), x, y, vx, vy,
    behavior: stringValue(p.behavior) ?? "unknown",
    ...(surfaceId ? { surfaceId } : {}),
  };
}

function collectMetric(payload: Record<string, unknown>, keys: readonly string[], values: number[], series?: ReportSeriesPoint[], record?: TraceRecord): void {
  const value = numberValue(...keys.map((key) => payload[key]));
  if (value === undefined || !Number.isFinite(value)) return;
  values.push(value);
  if (series && record) series.push({ elapsed_ms: record.elapsed_us / 1_000, value });
}

function pointToSurfaceError(x: number, y: number, surface: SurfaceGeometry): number {
  const horizontal = x < surface.x1 ? surface.x1 - x : x > surface.x2 ? x - surface.x2 : 0;
  return Math.hypot(horizontal, y - surface.y);
}

function percentile(sorted: readonly number[], q: number): number {
  if (sorted.length === 1) return sorted[0]!;
  const index = (sorted.length - 1) * q;
  const lower = Math.floor(index); const upper = Math.ceil(index); const weight = index - lower;
  return sorted[lower]! * (1 - weight) + sorted[upper]! * weight;
}

function maxCheck(name: string, value: number | null, threshold: number): MetricCheck {
  if (value === null) return { name, value, threshold, operator: "<=", status: "WARN", reason: "no_samples" };
  const fail = value > threshold;
  const warn = !fail && threshold > 0 && value > threshold * 0.8;
  return { name, value, threshold, operator: "<=", status: fail ? "FAIL" : warn ? "WARN" : "PASS" };
}

function minCheck(name: string, value: number | null, threshold: number): MetricCheck {
  if (value === null) return { name, value, threshold, operator: ">=", status: "WARN", reason: "no_samples" };
  const fail = value < threshold;
  const warn = !fail && value < threshold + (1 - threshold) * 0.5;
  return { name, value, threshold, operator: ">=", status: fail ? "FAIL" : warn ? "WARN" : "PASS" };
}

function compareBaseline(current: { planLatencyP95: number | null; clickResponseP95: number | null; carrierErrorP95: number | null }, baseline: TraceAnalysisReport, ratio: number): MetricCheck[] {
  const pairs: readonly [string, number | null, number | null][] = [
    ["regression_plan_latency_p95_ms", current.planLatencyP95, baseline.plans.latencyMs.p95],
    ["regression_click_response_p95_ms", current.clickResponseP95, baseline.interaction.clickResponseMs.p95],
    ["regression_carrier_error_p95_px", current.carrierErrorP95, baseline.motion.carrierErrorPx.p95],
  ];
  return pairs.map(([name, value, old]) => {
    if (value === null || old === null || old <= 0) return { name, value, threshold: old === null ? 0 : old * ratio, operator: "<=", status: "PASS" as const, reason: "not_comparable" };
    const threshold = old * ratio;
    return { name, value, threshold, operator: "<=", status: value > threshold ? "FAIL" as const : "PASS" as const, ...(value > threshold ? { reason: "over_20_percent_regression" } : {}) };
  });
}

function isNormalTopologyReason(reason: string): boolean { return NORMAL_TOPOLOGY_TOKENS.some((token) => reason.includes(token)); }
function worstVerdict(...values: readonly Verdict[]): Verdict { return values.includes("FAIL") ? "FAIL" : values.includes("WARN") ? "WARN" : "PASS"; }
function increment(target: Record<string, number>, key: string): void { target[key] = (target[key] ?? 0) + 1; }
function sortNumericRecord(value: Record<string, number>): Readonly<Record<string, number>> { return Object.fromEntries(Object.entries(value).sort(([a], [b]) => a.localeCompare(b, "en"))); }
function finite(value: unknown): value is number { return typeof value === "number" && Number.isFinite(value); }
function isRecord(value: unknown): value is Record<string, unknown> { return typeof value === "object" && value !== null && !Array.isArray(value); }
function recordObject(parent: Record<string, unknown>, key: string): Record<string, unknown> | undefined { const value = parent[key]; return isRecord(value) ? value : undefined; }
function arrayValue(value: unknown): readonly unknown[] { return Array.isArray(value) ? value : []; }
function stringValue(...values: readonly unknown[]): string | undefined { return values.find((value): value is string => typeof value === "string"); }
function numberValue(...values: readonly unknown[]): number | undefined { return values.find((value): value is number => finite(value)); }
function allFinite(value: unknown): boolean {
  if (typeof value === "number") return Number.isFinite(value);
  if (Array.isArray(value)) return value.every(allFinite);
  if (isRecord(value)) return Object.values(value).every(allFinite);
  return true;
}
function errorMessage(error: unknown): string { return error instanceof Error ? error.message.replace(/[\r\n:]+/g, "_") : "unknown"; }
function stableStringify(value: unknown): string {
  if (Array.isArray(value)) return `[${value.map(stableStringify).join(",")}]`;
  if (isRecord(value)) return `{${Object.keys(value).sort().map((key) => `${JSON.stringify(key)}:${stableStringify(value[key])}`).join(",")}}`;
  return JSON.stringify(value) ?? "null";
}
