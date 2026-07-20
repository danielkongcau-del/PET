import type {
  DisplayState,
  HorizonPlanPayload,
  SceneState,
  SurfaceState,
  WindowState,
  WorldStatePayload,
} from "../protocol.js";

export const TRACE_SCHEMA = "pet-trace" as const;
export const TRACE_VERSION = 1 as const;
export const TRACE_MANIFEST_SCHEMA = "pet-trace-manifest" as const;

export type TracePriority = "critical" | "sample";

export type TraceKind =
  | "surface_snapshot"
  | "world_state"
  | "plan_received"
  | "plan_result"
  | "cancel"
  | "motion_sample"
  | "generator_status"
  | "generator_metrics"
  | "process_metrics"
  | "recording_gap"
  | "marker"
  | "session_start"
  | "session_end";

export type JsonPrimitive = string | number | boolean | null;
export type JsonValue = JsonPrimitive | JsonValue[] | { [key: string]: JsonValue };
export type JsonObject = { [key: string]: JsonValue };

export interface TraceRecord<TPayload extends JsonValue = JsonValue> {
  readonly schema: typeof TRACE_SCHEMA;
  readonly version: typeof TRACE_VERSION;
  readonly record_seq: number;
  readonly wall_time_ms: number;
  readonly elapsed_us: number;
  readonly kind: TraceKind;
  readonly payload: TPayload;
}

export interface TraceChunkManifest {
  readonly index: number;
  readonly file: string;
  readonly first_record_seq: number;
  readonly last_record_seq: number;
  readonly record_count: number;
  readonly started_elapsed_us: number;
  readonly ended_elapsed_us: number;
  readonly uncompressed_bytes: number;
  readonly compressed_bytes: number;
  /** SHA-256 of the final gzip file, encoded as lowercase hexadecimal. */
  readonly sha256: string;
}

export interface TracePrivacyManifest {
  readonly screen_capture: false;
  readonly window_titles: false;
  readonly process_names: false;
  readonly keyboard_input: false;
  readonly absolute_paths: false;
  readonly episode_scoped_ids: true;
}

export interface TraceManifest {
  readonly schema: typeof TRACE_MANIFEST_SCHEMA;
  readonly version: typeof TRACE_VERSION;
  readonly trace_schema: typeof TRACE_SCHEMA;
  readonly trace_version: typeof TRACE_VERSION;
  readonly episode_id: string;
  readonly label?: string;
  readonly started_at_ms: number;
  readonly ended_at_ms?: number;
  readonly duration_us?: number;
  readonly incomplete: boolean;
  readonly end_reason?: string;
  readonly metadata: JsonObject;
  readonly privacy: TracePrivacyManifest;
  readonly chunks: readonly TraceChunkManifest[];
  readonly total_records: number;
  readonly total_uncompressed_bytes: number;
  readonly total_compressed_bytes: number;
  readonly dropped_motion_samples: number;
}

export interface TraceSurfaceSnapshotInput {
  readonly captured_at_ms?: number;
  readonly capturedAtMs?: number;
  readonly displays: readonly DisplayState[];
  readonly windows: readonly WindowState[];
  readonly surfaces: readonly SurfaceState[];
  readonly scene: SceneState;
}

export interface SanitizedSurfaceSnapshot {
  readonly captured_at_ms: number;
  readonly displays: readonly DisplayState[];
  readonly windows: readonly WindowState[];
  readonly surfaces: readonly SurfaceState[];
  readonly scene: SceneState;
}

export interface TraceRecordingStatus {
  readonly active: boolean;
  readonly elapsedMs: number;
  readonly bytes: number;
  readonly episodeDir: string | null;
  readonly droppedSamples: number;
}

export interface TraceStartOptions {
  readonly label?: string;
  /** Versioned runtime/config/checkpoint information used by replay. */
  readonly metadata?: Readonly<Record<string, unknown>>;
}

export type RecordResult = "accepted" | "dropped" | "inactive";

export interface TraceSanitizer {
  sanitizeWorldState(world: WorldStatePayload): WorldStatePayload;
  sanitizeSurfaceSnapshot(snapshot: TraceSurfaceSnapshotInput): SanitizedSurfaceSnapshot;
  sanitizePlan(plan: HorizonPlanPayload): HorizonPlanPayload;
  sanitizePayload(kind: TraceKind, payload: unknown): JsonValue;
}
