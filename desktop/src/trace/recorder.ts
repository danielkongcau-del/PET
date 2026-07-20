import { createHash, randomUUID } from "node:crypto";
import { createReadStream, createWriteStream } from "node:fs";
import {
  mkdir,
  open,
  readFile,
  readdir,
  rename,
  stat,
  statfs,
  unlink,
  writeFile,
  type FileHandle,
} from "node:fs/promises";
import path from "node:path";
import { pipeline } from "node:stream/promises";
import { createGzip } from "node:zlib";

import type { HorizonPlanPayload, SurfaceState, WorldStatePayload } from "../protocol.js";
import {
  TRACE_MANIFEST_SCHEMA,
  TRACE_SCHEMA,
  TRACE_VERSION,
  type JsonObject,
  type JsonValue,
  type RecordResult,
  type SanitizedSurfaceSnapshot,
  type TraceChunkManifest,
  type TraceKind,
  type TraceManifest,
  type TracePriority,
  type TraceRecord,
  type TraceRecordingStatus,
  type TraceSanitizer,
  type TraceStartOptions,
  type TraceSurfaceSnapshotInput,
} from "./format.js";

const DEFAULT_CHUNK_DURATION_MS = 5 * 60 * 1_000;
const DEFAULT_CHUNK_BYTES = 64 * 1_024 * 1_024;
const DEFAULT_MINIMUM_FREE_BYTES = 1 * 1_024 * 1_024 * 1_024;
const DEFAULT_MAX_BUFFERED_BYTES = 2 * 1_024 * 1_024;
const DEFAULT_DISK_CHECK_INTERVAL_MS = 5_000;

export interface TraceRecorderOptions {
  readonly rootDir: string;
  readonly maxChunkDurationMs?: number;
  readonly maxChunkBytes?: number;
  readonly minimumFreeBytes?: number;
  readonly maxBufferedBytes?: number;
  readonly diskCheckIntervalMs?: number;
  readonly nowMs?: () => number;
  readonly monotonicUs?: () => number;
  readonly getFreeBytes?: (directory: string) => Promise<number>;
  readonly onStateChange?: (status: TraceRecordingStatus) => void;
  readonly onError?: (error: Error) => void;
}

export class InsufficientDiskSpaceError extends Error {
  readonly freeBytes: number;
  readonly requiredBytes: number;

  constructor(freeBytes: number, requiredBytes: number) {
    super(`Trace recording requires at least ${requiredBytes} free bytes; only ${freeBytes} are available.`);
    this.name = "InsufficientDiskSpaceError";
    this.freeBytes = freeBytes;
    this.requiredBytes = requiredBytes;
  }
}

interface NormalizedOptions {
  readonly rootDir: string;
  readonly maxChunkDurationUs: number;
  readonly maxChunkBytes: number;
  readonly minimumFreeBytes: number;
  readonly maxBufferedBytes: number;
  readonly diskCheckIntervalMs: number;
  readonly nowMs: () => number;
  readonly monotonicUs: () => number;
  readonly getFreeBytes: (directory: string) => Promise<number>;
  readonly onStateChange?: (status: TraceRecordingStatus) => void;
  readonly onError?: (error: Error) => void;
}

interface HotChunk {
  readonly index: number;
  readonly partialPath: string;
  readonly startedElapsedUs: number;
  handle: FileHandle;
  bytes: number;
  records: number;
  firstRecordSeq: number | null;
  lastRecordSeq: number | null;
  endedElapsedUs: number;
}

interface PendingGap {
  count: number;
  fromElapsedUs: number;
  toElapsedUs: number;
}

interface ActiveEpisode {
  readonly id: string;
  readonly directory: string;
  readonly startedAtMs: number;
  readonly startedMonotonicUs: number;
  readonly anonymizer: EpisodeAnonymizer;
  manifest: TraceManifest;
  chunk: HotChunk;
  queue: Promise<void>;
  accepting: boolean;
  stopPromise: Promise<void> | null;
  diskTimer: NodeJS.Timeout | null;
  diskPollRunning: boolean;
  nextRecordSeq: number;
  pendingBytes: number;
  totalBytesWritten: number;
  droppedSamples: number;
  pendingGap: PendingGap | null;
  fatalError: Error | null;
}

interface QueuedRecord {
  readonly record: TraceRecord;
  readonly line: string;
  readonly bytes: number;
}

export class TraceRecorder implements TraceSanitizer {
  readonly #options: NormalizedOptions;
  #active: ActiveEpisode | null = null;
  #recoveryPerformed = false;

  constructor(options: TraceRecorderOptions) {
    if (options.rootDir.trim().length === 0) throw new Error("Trace rootDir must not be empty.");
    this.#options = {
      rootDir: path.resolve(options.rootDir),
      maxChunkDurationUs: positiveFinite(options.maxChunkDurationMs ?? DEFAULT_CHUNK_DURATION_MS, "maxChunkDurationMs") * 1_000,
      maxChunkBytes: positiveFinite(options.maxChunkBytes ?? DEFAULT_CHUNK_BYTES, "maxChunkBytes"),
      minimumFreeBytes: nonNegativeFinite(options.minimumFreeBytes ?? DEFAULT_MINIMUM_FREE_BYTES, "minimumFreeBytes"),
      maxBufferedBytes: positiveFinite(options.maxBufferedBytes ?? DEFAULT_MAX_BUFFERED_BYTES, "maxBufferedBytes"),
      diskCheckIntervalMs: positiveFinite(options.diskCheckIntervalMs ?? DEFAULT_DISK_CHECK_INTERVAL_MS, "diskCheckIntervalMs"),
      nowMs: options.nowMs ?? Date.now,
      monotonicUs: options.monotonicUs ?? (() => Number(process.hrtime.bigint() / 1_000n)),
      getFreeBytes: options.getFreeBytes ?? defaultGetFreeBytes,
      ...(options.onStateChange === undefined ? {} : { onStateChange: options.onStateChange }),
      ...(options.onError === undefined ? {} : { onError: options.onError }),
    };
  }

  get rootDir(): string {
    return this.#options.rootDir;
  }

  get status(): TraceRecordingStatus {
    const active = this.#active;
    if (active === null) return { active: false, elapsedMs: 0, bytes: 0, episodeDir: null, droppedSamples: 0 };
    return {
      active: active.accepting,
      elapsedMs: Math.max(0, this.#elapsedUs(active) / 1_000),
      bytes: active.totalBytesWritten + active.pendingBytes,
      episodeDir: active.directory,
      droppedSamples: active.droppedSamples,
    };
  }

  async start(options: TraceStartOptions = {}): Promise<string> {
    if (this.#active !== null) throw new Error("A trace episode is already active.");
    await mkdir(this.#options.rootDir, { recursive: true });
    if (!this.#recoveryPerformed) await this.recoverIncompleteEpisodes();
    await this.#assertDiskSpace();

    const startedAtMs = this.#options.nowMs();
    const id = randomUUID();
    const label = sanitizeLabel(options.label);
    const timestamp = new Date(startedAtMs).toISOString().replaceAll(":", "-").replace(".", "-");
    const directoryName = `${timestamp}-${id.slice(0, 8)}${label === undefined ? "" : `-${label}`}`;
    const directory = path.join(this.#options.rootDir, directoryName);
    await mkdir(directory, { recursive: false });

    const anonymizer = new EpisodeAnonymizer();
    const metadata = toSafeJson(options.metadata ?? {}, anonymizer) as JsonObject;
    let manifest = createManifest(id, startedAtMs, metadata, label);
    await writeManifestAtomic(directory, manifest);
    const startedMonotonicUs = this.#options.monotonicUs();
    const chunk = await openHotChunk(directory, 0, 0);
    const active: ActiveEpisode = {
      id,
      directory,
      startedAtMs,
      startedMonotonicUs,
      anonymizer,
      manifest,
      chunk,
      queue: Promise.resolve(),
      accepting: true,
      stopPromise: null,
      diskTimer: null,
      diskPollRunning: false,
      nextRecordSeq: 0,
      pendingBytes: 0,
      totalBytesWritten: 0,
      droppedSamples: 0,
      pendingGap: null,
      fatalError: null,
    };
    this.#active = active;
    this.#queueRecord(active, "session_start", {
      episode_id: id,
      ...(label === undefined ? {} : { label }),
      metadata,
    }, "critical", startedAtMs, 0);
    active.diskTimer = setInterval(() => void this.#pollDisk(active), this.#options.diskCheckIntervalMs);
    active.diskTimer.unref();
    this.#emitState();
    return directory;
  }

  record(kind: TraceKind, payload: unknown, priority: TracePriority = kind === "motion_sample" ? "sample" : "critical"): RecordResult {
    const active = this.#active;
    if (active === null || !active.accepting) return "inactive";
    const wallTimeMs = this.#options.nowMs();
    const elapsedUs = this.#elapsedUs(active);
    const sanitized = this.#sanitizePayload(active.anonymizer, kind, payload);
    const estimatedBytes = Buffer.byteLength(JSON.stringify(sanitized), "utf8") + 192;
    if (kind === "motion_sample" && priority === "sample" && active.pendingBytes + estimatedBytes > this.#options.maxBufferedBytes) {
      active.droppedSamples += 1;
      if (active.pendingGap === null) {
        active.pendingGap = { count: 1, fromElapsedUs: elapsedUs, toElapsedUs: elapsedUs };
      } else {
        active.pendingGap.count += 1;
        active.pendingGap.toElapsedUs = elapsedUs;
      }
      this.#emitState();
      return "dropped";
    }
    this.#flushPendingGap(active, wallTimeMs, elapsedUs);
    this.#queueRecord(active, kind, sanitized, priority, wallTimeMs, elapsedUs, true);
    return "accepted";
  }

  async stop(reason = "stopped"): Promise<void> {
    const active = this.#active;
    if (active === null) return;
    if (active.stopPromise !== null) return active.stopPromise;
    active.accepting = false;
    if (active.diskTimer !== null) {
      clearInterval(active.diskTimer);
      active.diskTimer = null;
    }
    active.stopPromise = this.#finishStop(active, reason);
    this.#emitState();
    return active.stopPromise;
  }

  async flush(): Promise<void> {
    const active = this.#active;
    if (active === null) return;
    this.#flushPendingGap(active, this.#options.nowMs(), this.#elapsedUs(active));
    await active.queue;
  }

  /** Recover hot chunks left by a process crash. Safe to call more than once. */
  async recoverIncompleteEpisodes(): Promise<readonly string[]> {
    if (this.#active !== null) throw new Error("Cannot recover traces while recording.");
    await mkdir(this.#options.rootDir, { recursive: true });
    const recovered: string[] = [];
    const entries = await readdir(this.#options.rootDir, { withFileTypes: true });
    for (const entry of entries) {
      if (!entry.isDirectory()) continue;
      const directory = path.join(this.#options.rootDir, entry.name);
      const names = await readdir(directory);
      const partials = names.filter((name) => /^trace-\d{5}\.ndjson\.partial$/.test(name)).sort();
      if (partials.length === 0) continue;
      await recoverEpisode(directory, partials);
      recovered.push(directory);
    }
    this.#recoveryPerformed = true;
    return recovered;
  }

  sanitizeWorldState(world: WorldStatePayload): WorldStatePayload {
    return (this.#active?.anonymizer ?? new EpisodeAnonymizer()).worldState(world);
  }

  sanitizeSurfaceSnapshot(snapshot: TraceSurfaceSnapshotInput): SanitizedSurfaceSnapshot {
    return (this.#active?.anonymizer ?? new EpisodeAnonymizer()).surfaceSnapshot(snapshot);
  }

  sanitizePlan(plan: HorizonPlanPayload): HorizonPlanPayload {
    return (this.#active?.anonymizer ?? new EpisodeAnonymizer()).plan(plan);
  }

  sanitizePayload(kind: TraceKind, payload: unknown): JsonValue {
    return this.#sanitizePayload(this.#active?.anonymizer ?? new EpisodeAnonymizer(), kind, payload);
  }

  #sanitizePayload(anonymizer: EpisodeAnonymizer, kind: TraceKind, payload: unknown): JsonValue {
    if (!isObject(payload)) return toSafeJson(payload, anonymizer);
    let specialized: unknown = payload;
    let identifiersMapped = false;
    if (kind === "world_state") {
      if (isObject(payload.state)) {
        specialized = { ...payload, state: anonymizer.worldState(payload.state as unknown as WorldStatePayload) };
        identifiersMapped = true;
      } else if (Array.isArray(payload.displays) && Array.isArray(payload.windows)) {
        specialized = anonymizer.worldState(payload as unknown as WorldStatePayload);
        identifiersMapped = true;
      }
    } else if (kind === "surface_snapshot" && Array.isArray(payload.displays) && Array.isArray(payload.windows)) {
      specialized = anonymizer.surfaceSnapshot(payload as unknown as TraceSurfaceSnapshotInput);
      identifiersMapped = true;
    } else if (kind === "plan_received") {
      if (isObject(payload.plan)) {
        specialized = { ...payload, plan: anonymizer.plan(payload.plan as unknown as HorizonPlanPayload) };
        identifiersMapped = true;
      } else if (Array.isArray(payload.points)) {
        specialized = anonymizer.plan(payload as unknown as HorizonPlanPayload);
        identifiersMapped = true;
      }
    } else if (kind === "plan_result" || kind === "cancel") {
      specialized = {
        ...payload,
        ...(typeof payload.plan_id === "string" ? { plan_id: anonymizer.planId(payload.plan_id) } : {}),
      };
      identifiersMapped = true;
    } else if (kind === "motion_sample") {
      specialized = {
        ...payload,
        ...(typeof payload.surface_id === "string" ? { surface_id: anonymizer.surfaceId(payload.surface_id) } : {}),
        ...(typeof payload.plan_id === "string" ? { plan_id: anonymizer.planId(payload.plan_id) } : {}),
      };
      identifiersMapped = true;
    }
    return toSafeJson(specialized, anonymizer, new Set<object>(), !identifiersMapped);
  }

  #elapsedUs(active: ActiveEpisode): number {
    return Math.max(0, Math.round(this.#options.monotonicUs() - active.startedMonotonicUs));
  }

  #flushPendingGap(active: ActiveEpisode, wallTimeMs: number, elapsedUs: number): void {
    const gap = active.pendingGap;
    if (gap === null) return;
    active.pendingGap = null;
    this.#queueRecord(active, "recording_gap", {
      stream: "motion_sample",
      reason: "backpressure",
      dropped: gap.count,
      from_elapsed_us: gap.fromElapsedUs,
      to_elapsed_us: gap.toElapsedUs,
    }, "critical", wallTimeMs, elapsedUs);
  }

  #queueRecord(
    active: ActiveEpisode,
    kind: TraceKind,
    payload: unknown,
    _priority: TracePriority,
    wallTimeMs: number,
    elapsedUs: number,
    alreadySanitized = false,
  ): void {
    const record: TraceRecord = {
      schema: TRACE_SCHEMA,
      version: TRACE_VERSION,
      record_seq: active.nextRecordSeq,
      wall_time_ms: Math.round(wallTimeMs),
      elapsed_us: Math.round(elapsedUs),
      kind,
      payload: alreadySanitized ? payload as JsonValue : this.#sanitizePayload(active.anonymizer, kind, payload),
    };
    active.nextRecordSeq += 1;
    const line = `${JSON.stringify(record)}\n`;
    const queued: QueuedRecord = { record, line, bytes: Buffer.byteLength(line, "utf8") };
    active.pendingBytes += queued.bytes;
    active.queue = active.queue
      .then(async () => {
        try {
          if (active.fatalError === null) await this.#writeQueuedRecord(active, queued);
        } finally {
          active.pendingBytes = Math.max(0, active.pendingBytes - queued.bytes);
          this.#emitState();
        }
      })
      .catch((error: unknown) => this.#handleFatalError(active, asError(error)));
  }

  async #writeQueuedRecord(active: ActiveEpisode, queued: QueuedRecord): Promise<void> {
    const chunk = active.chunk;
    const durationExceeded = queued.record.elapsed_us - chunk.startedElapsedUs >= this.#options.maxChunkDurationUs;
    const bytesExceeded = chunk.bytes + queued.bytes > this.#options.maxChunkBytes;
    if (chunk.records > 0 && (durationExceeded || bytesExceeded)) {
      await this.#finalizeCurrentChunk(active);
      active.chunk = await openHotChunk(active.directory, active.chunk.index + 1, queued.record.elapsed_us);
    }
    await active.chunk.handle.write(queued.line);
    active.chunk.bytes += queued.bytes;
    active.chunk.records += 1;
    active.chunk.firstRecordSeq ??= queued.record.record_seq;
    active.chunk.lastRecordSeq = queued.record.record_seq;
    active.chunk.endedElapsedUs = queued.record.elapsed_us;
    active.totalBytesWritten += queued.bytes;
  }

  async #finalizeCurrentChunk(active: ActiveEpisode): Promise<void> {
    const chunk = active.chunk;
    await chunk.handle.sync();
    await chunk.handle.close();
    if (chunk.records === 0 || chunk.firstRecordSeq === null || chunk.lastRecordSeq === null) {
      await safeUnlink(chunk.partialPath);
      return;
    }
    const finalName = chunkFileName(chunk.index);
    const finalPath = path.join(active.directory, finalName);
    const gzipPartial = `${finalPath}.partial`;
    await safeUnlink(gzipPartial);
    await gzipFile(chunk.partialPath, gzipPartial);
    await rename(gzipPartial, finalPath);
    const compressed = await stat(finalPath);
    const descriptor: TraceChunkManifest = {
      index: chunk.index,
      file: finalName,
      first_record_seq: chunk.firstRecordSeq,
      last_record_seq: chunk.lastRecordSeq,
      record_count: chunk.records,
      started_elapsed_us: chunk.startedElapsedUs,
      ended_elapsed_us: chunk.endedElapsedUs,
      uncompressed_bytes: chunk.bytes,
      compressed_bytes: compressed.size,
      sha256: await sha256File(finalPath),
    };
    await safeUnlink(chunk.partialPath);
    active.manifest = withChunk(active.manifest, descriptor, active.droppedSamples);
    await writeManifestAtomic(active.directory, active.manifest);
  }

  async #finishStop(active: ActiveEpisode, reason: string): Promise<void> {
    const endedAtMs = this.#options.nowMs();
    const durationUs = this.#elapsedUs(active);
    try {
      this.#flushPendingGap(active, endedAtMs, durationUs);
      if (active.fatalError === null) {
        this.#queueRecord(active, "session_end", { reason }, "critical", endedAtMs, durationUs);
      }
      await active.queue;
      if (active.fatalError === null) await this.#finalizeCurrentChunk(active);
      active.manifest = {
        ...active.manifest,
        ended_at_ms: Math.round(endedAtMs),
        duration_us: durationUs,
        incomplete: active.fatalError !== null,
        end_reason: active.fatalError === null ? reason : "write_error",
        dropped_motion_samples: active.droppedSamples,
      };
      await writeManifestAtomic(active.directory, active.manifest);
    } catch (error: unknown) {
      const failure = asError(error);
      this.#notifyError(failure);
      try {
        active.manifest = {
          ...active.manifest,
          ended_at_ms: Math.round(endedAtMs),
          duration_us: durationUs,
          incomplete: true,
          end_reason: "write_error",
          dropped_motion_samples: active.droppedSamples,
        };
        await writeManifestAtomic(active.directory, active.manifest);
      } catch {
        // The hot .ndjson.partial remains recoverable on the next launch.
      }
      throw failure;
    } finally {
      if (this.#active === active) this.#active = null;
      this.#emitState();
    }
  }

  async #pollDisk(active: ActiveEpisode): Promise<void> {
    if (this.#active !== active || !active.accepting || active.diskPollRunning) return;
    active.diskPollRunning = true;
    try {
      const freeBytes = await this.#options.getFreeBytes(this.#options.rootDir);
      if (freeBytes < this.#options.minimumFreeBytes) {
        this.#notifyError(new InsufficientDiskSpaceError(freeBytes, this.#options.minimumFreeBytes));
        await this.stop("low_disk");
      }
    } catch (error: unknown) {
      this.#notifyError(asError(error));
    } finally {
      active.diskPollRunning = false;
    }
  }

  async #assertDiskSpace(): Promise<void> {
    const freeBytes = await this.#options.getFreeBytes(this.#options.rootDir);
    if (freeBytes < this.#options.minimumFreeBytes) {
      throw new InsufficientDiskSpaceError(freeBytes, this.#options.minimumFreeBytes);
    }
  }

  #handleFatalError(active: ActiveEpisode, error: Error): void {
    if (active.fatalError !== null) return;
    active.fatalError = error;
    active.accepting = false;
    this.#notifyError(error);
    queueMicrotask(() => void this.stop("write_error").catch((stopError: unknown) => this.#notifyError(asError(stopError))));
  }

  #notifyError(error: Error): void {
    try {
      this.#options.onError?.(error);
    } catch {
      // Recorder diagnostics must never affect pet motion.
    }
  }

  #emitState(): void {
    try {
      this.#options.onStateChange?.(this.status);
    } catch {
      // UI callbacks are outside the persistence trust boundary.
    }
  }
}

class EpisodeAnonymizer {
  readonly #displayIds = new StableIdMap("display");
  readonly #windowIds = new StableIdMap("window");
  readonly #surfaceIds = new StableIdMap("surface");
  readonly #planIds = new StableIdMap("plan");
  readonly #clickIds = new StableIdMap("click");

  displayId(value: string): string {
    return this.#displayIds.get(value);
  }

  windowId(value: string): string {
    return this.#windowIds.get(value);
  }

  surfaceId(value: string): string {
    return this.#surfaceIds.get(value);
  }

  planId(value: string): string {
    return this.#planIds.get(value);
  }

  surface(surface: SurfaceState): SurfaceState {
    return {
      ...surface,
      id: this.surfaceId(surface.id),
      display_id: this.displayId(surface.display_id),
      ...(surface.window_id === undefined ? {} : { window_id: this.windowId(surface.window_id) }),
    };
  }

  worldState(world: WorldStatePayload): WorldStatePayload {
    return {
      ...world,
      session_id: "session",
      displays: world.displays.map((display) => ({ ...display, id: this.displayId(display.id) })),
      windows: world.windows.map((window) => ({
        ...window,
        id: this.windowId(window.id),
        display_id: this.displayId(window.display_id),
      })),
      surfaces: world.surfaces.map((surface) => this.surface(surface)),
      pet: {
        ...world.pet,
        ...(world.pet.surface_id === undefined ? {} : { surface_id: this.surfaceId(world.pet.surface_id) }),
      },
      clicks: world.clicks.map((click) => ({ ...click, id: this.#clickIds.get(click.id) })),
    };
  }

  surfaceSnapshot(snapshot: TraceSurfaceSnapshotInput): SanitizedSurfaceSnapshot {
    return {
      captured_at_ms: snapshot.captured_at_ms ?? snapshot.capturedAtMs ?? 0,
      displays: snapshot.displays.map((display) => ({ ...display, id: this.displayId(display.id) })),
      windows: snapshot.windows.map((window) => ({
        ...window,
        id: this.windowId(window.id),
        display_id: this.displayId(window.display_id),
      })),
      surfaces: snapshot.surfaces.map((surface) => this.surface(surface)),
      scene: { ...snapshot.scene },
    };
  }

  plan(plan: HorizonPlanPayload): HorizonPlanPayload {
    return {
      ...plan,
      plan_id: this.planId(plan.plan_id),
      ...(plan.target === undefined ? {} : {
        target: { ...plan.target, surface_id: this.surfaceId(plan.target.surface_id) },
      }),
      points: plan.points.map((point) => ({ ...point })),
    };
  }
}

class StableIdMap {
  readonly #prefix: string;
  readonly #values = new Map<string, string>();
  readonly #outputs = new Set<string>();

  constructor(prefix: string) {
    this.#prefix = prefix;
  }

  get(value: string): string {
    // Public sanitize* helpers may be composed with record(); keep that path idempotent.
    if (this.#outputs.has(value)) return value;
    const existing = this.#values.get(value);
    if (existing !== undefined) return existing;
    const anonymized = `${this.#prefix}-${this.#values.size}`;
    this.#values.set(value, anonymized);
    this.#outputs.add(anonymized);
    return anonymized;
  }
}

async function openHotChunk(directory: string, index: number, startedElapsedUs: number): Promise<HotChunk> {
  const partialPath = path.join(directory, hotChunkFileName(index));
  return {
    index,
    partialPath,
    startedElapsedUs,
    handle: await open(partialPath, "a"),
    bytes: 0,
    records: 0,
    firstRecordSeq: null,
    lastRecordSeq: null,
    endedElapsedUs: startedElapsedUs,
  };
}

function hotChunkFileName(index: number): string {
  return `trace-${index.toString().padStart(5, "0")}.ndjson.partial`;
}

function chunkFileName(index: number): string {
  return `trace-${index.toString().padStart(5, "0")}.ndjson.gz`;
}

async function gzipFile(source: string, destination: string): Promise<void> {
  await pipeline(createReadStream(source), createGzip({ level: 9 }), createWriteStream(destination, { flags: "wx" }));
}

async function sha256File(file: string): Promise<string> {
  const hash = createHash("sha256");
  const stream = createReadStream(file);
  for await (const chunk of stream) hash.update(chunk as Buffer);
  return hash.digest("hex");
}

function createManifest(episodeId: string, startedAtMs: number, metadata: JsonObject, label: string | undefined): TraceManifest {
  return {
    schema: TRACE_MANIFEST_SCHEMA,
    version: TRACE_VERSION,
    trace_schema: TRACE_SCHEMA,
    trace_version: TRACE_VERSION,
    episode_id: episodeId,
    ...(label === undefined ? {} : { label }),
    started_at_ms: Math.round(startedAtMs),
    incomplete: true,
    metadata,
    privacy: privacyManifest(),
    chunks: [],
    total_records: 0,
    total_uncompressed_bytes: 0,
    total_compressed_bytes: 0,
    dropped_motion_samples: 0,
  };
}

function privacyManifest() {
  return {
    screen_capture: false,
    window_titles: false,
    process_names: false,
    keyboard_input: false,
    absolute_paths: false,
    episode_scoped_ids: true,
  } as const;
}

function withChunk(manifest: TraceManifest, chunk: TraceChunkManifest, droppedSamples: number): TraceManifest {
  const chunks = [...manifest.chunks.filter((item) => item.index !== chunk.index), chunk].sort((a, b) => a.index - b.index);
  return {
    ...manifest,
    chunks,
    total_records: chunks.reduce((sum, item) => sum + item.record_count, 0),
    total_uncompressed_bytes: chunks.reduce((sum, item) => sum + item.uncompressed_bytes, 0),
    total_compressed_bytes: chunks.reduce((sum, item) => sum + item.compressed_bytes, 0),
    dropped_motion_samples: droppedSamples,
  };
}

async function writeManifestAtomic(directory: string, manifest: TraceManifest): Promise<void> {
  const target = path.join(directory, "manifest.json");
  const temporary = path.join(directory, `manifest.json.${process.pid}.${randomUUID()}.partial`);
  await writeFile(temporary, `${JSON.stringify(manifest, null, 2)}\n`, { encoding: "utf8", flag: "wx" });
  await rename(temporary, target);
}

async function recoverEpisode(directory: string, partialNames: readonly string[]): Promise<void> {
  const existing = await readManifest(path.join(directory, "manifest.json"));
  let manifest = existing;
  let lastRecoveredRecord: TraceRecord | null = null;
  for (const partialName of partialNames) {
    const partialPath = path.join(directory, partialName);
    const recovered = await readRecoverableRecords(partialPath);
    const indexMatch = /^trace-(\d{5})/.exec(partialName);
    const index = Number(indexMatch?.[1] ?? 0);
    if (recovered.records.length === 0) {
      await rename(partialPath, `${partialPath}.corrupt`);
      continue;
    }
    const cleanPath = `${partialPath}.recovered`;
    await writeFile(cleanPath, recovered.text, { encoding: "utf8", flag: "wx" });
    const finalName = chunkFileName(index);
    const finalPath = path.join(directory, finalName);
    const gzipPartial = `${finalPath}.partial`;
    await safeUnlink(gzipPartial);
    await gzipFile(cleanPath, gzipPartial);
    await rename(gzipPartial, finalPath);
    await safeUnlink(cleanPath);
    await safeUnlink(partialPath);
    const first = recovered.records[0]!;
    const last = recovered.records.at(-1)!;
    lastRecoveredRecord = last;
    const compressed = await stat(finalPath);
    const descriptor: TraceChunkManifest = {
      index,
      file: finalName,
      first_record_seq: first.record_seq,
      last_record_seq: last.record_seq,
      record_count: recovered.records.length,
      started_elapsed_us: first.elapsed_us,
      ended_elapsed_us: last.elapsed_us,
      uncompressed_bytes: Buffer.byteLength(recovered.text, "utf8"),
      compressed_bytes: compressed.size,
      sha256: await sha256File(finalPath),
    };
    if (manifest === null) {
      manifest = createManifest(path.basename(directory), first.wall_time_ms, {}, undefined);
    }
    manifest = withChunk(manifest, descriptor, manifest.dropped_motion_samples);
  }
  if (manifest === null) return;
  const lastChunk = manifest.chunks.at(-1);
  manifest = {
    ...manifest,
    ...(lastRecoveredRecord === null ? {} : { ended_at_ms: lastRecoveredRecord.wall_time_ms }),
    ...(lastChunk === undefined ? {} : { duration_us: lastChunk.ended_elapsed_us }),
    incomplete: true,
    end_reason: "recovered_after_crash",
  };
  await writeManifestAtomic(directory, manifest);
}

async function readManifest(file: string): Promise<TraceManifest | null> {
  try {
    const parsed = JSON.parse(await readFile(file, "utf8")) as unknown;
    if (!isObject(parsed) || parsed.schema !== TRACE_MANIFEST_SCHEMA || parsed.version !== TRACE_VERSION || !Array.isArray(parsed.chunks)) return null;
    return parsed as unknown as TraceManifest;
  } catch (error: unknown) {
    if (isNodeError(error) && error.code === "ENOENT") return null;
    return null;
  }
}

async function readRecoverableRecords(file: string): Promise<{ readonly records: readonly TraceRecord[]; readonly text: string }> {
  const source = await readFile(file, "utf8");
  const records: TraceRecord[] = [];
  const lines: string[] = [];
  for (const line of source.split("\n")) {
    if (line.trim().length === 0) continue;
    try {
      const parsed = JSON.parse(line) as unknown;
      if (!isTraceRecord(parsed)) continue;
      records.push(parsed);
      lines.push(JSON.stringify(parsed));
    } catch {
      // A crash may leave one torn final line; all preceding complete records survive.
    }
  }
  return { records, text: lines.length === 0 ? "" : `${lines.join("\n")}\n` };
}

function isTraceRecord(value: unknown): value is TraceRecord {
  return isObject(value) && value.schema === TRACE_SCHEMA && value.version === TRACE_VERSION &&
    Number.isSafeInteger(value.record_seq) && Number.isFinite(value.wall_time_ms) && Number.isFinite(value.elapsed_us) &&
    typeof value.kind === "string" && "payload" in value;
}

async function safeUnlink(file: string): Promise<void> {
  try {
    await unlink(file);
  } catch (error: unknown) {
    if (!isNodeError(error) || error.code !== "ENOENT") throw error;
  }
}

async function defaultGetFreeBytes(directory: string): Promise<number> {
  const info = await statfs(directory);
  return info.bavail * info.bsize;
}

function sanitizeLabel(value: string | undefined): string | undefined {
  if (value === undefined) return undefined;
  const sanitized = value.trim().toLowerCase().replace(/[^a-z0-9_-]+/g, "-").replace(/^-+|-+$/g, "").slice(0, 48);
  return sanitized.length === 0 ? undefined : sanitized;
}

const PRIVATE_KEYS = new Set([
  "title", "window_title", "owner_name", "owner_path", "app_name", "process_name", "process_path",
  "executable", "command_line", "username", "user_name", "computer_name", "hostname", "path", "absolute_path",
  "keyboard", "keyboard_input", "keys", "screenshot", "screen_capture", "pixels", "image", "audio",
]);

function toSafeJson(
  value: unknown,
  anonymizer: EpisodeAnonymizer,
  seen = new Set<object>(),
  mapIdentifiers = true,
): JsonValue {
  if (value === null) return null;
  if (typeof value === "string" || typeof value === "boolean") return value;
  if (typeof value === "number") return Number.isFinite(value) ? value : null;
  if (typeof value === "bigint") return value.toString();
  if (typeof value === "undefined" || typeof value === "function" || typeof value === "symbol") return null;
  if (typeof value !== "object") return String(value);
  if (seen.has(value)) return "[Circular]";
  seen.add(value);
  if (Array.isArray(value)) {
    const result = value.map((item) => toSafeJson(item, anonymizer, seen, mapIdentifiers));
    seen.delete(value);
    return result;
  }
  const result: JsonObject = {};
  for (const [key, item] of Object.entries(value)) {
    const normalizedKey = key.toLowerCase();
    if (PRIVATE_KEYS.has(normalizedKey) || normalizedKey.endsWith("_absolute_path")) continue;
    if (item === undefined || typeof item === "function" || typeof item === "symbol") continue;
    if (mapIdentifiers && normalizedKey === "window_id" && typeof item === "string") result[key] = anonymizer.windowId(item);
    else if (mapIdentifiers && normalizedKey === "display_id" && typeof item === "string") result[key] = anonymizer.displayId(item);
    else if (mapIdentifiers && normalizedKey === "surface_id" && typeof item === "string") result[key] = anonymizer.surfaceId(item);
    else if (mapIdentifiers && normalizedKey === "plan_id" && typeof item === "string") result[key] = anonymizer.planId(item);
    else result[key] = toSafeJson(item, anonymizer, seen, mapIdentifiers);
  }
  seen.delete(value);
  return result;
}

function isObject(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function isNodeError(value: unknown): value is NodeJS.ErrnoException {
  return value instanceof Error && "code" in value;
}

function asError(value: unknown): Error {
  return value instanceof Error ? value : new Error(String(value));
}

function positiveFinite(value: number, name: string): number {
  if (!Number.isFinite(value) || value <= 0) throw new Error(`${name} must be a positive finite number.`);
  return value;
}

function nonNegativeFinite(value: number, name: string): number {
  if (!Number.isFinite(value) || value < 0) throw new Error(`${name} must be a non-negative finite number.`);
  return value;
}
