import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { gunzipSync } from "node:zlib";
import { mkdtemp, mkdir, readFile, readdir, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import test from "node:test";

import type { HorizonPlanPayload, WorldStatePayload } from "../src/protocol.js";
import { TRACE_SCHEMA, TRACE_VERSION, type TraceManifest, type TraceRecord } from "../src/trace/format.js";
import { InsufficientDiskSpaceError, TraceRecorder } from "../src/trace/recorder.js";

const plentyOfSpace = async (): Promise<number> => Number.MAX_SAFE_INTEGER;

test("records an episode as readable gzip with manifest hashes", async () => {
  const root = await temporaryRoot();
  const recorder = new TraceRecorder({ rootDir: root, minimumFreeBytes: 0, getFreeBytes: plentyOfSpace });
  const episode = await recorder.start({ label: "Maximize Stutter", metadata: { app_version: "0.1.0", path: "C:\\private" } });
  assert.equal(recorder.status.active, true);
  assert.equal(recorder.record("marker", { name: "before-maximize" }), "accepted");
  await recorder.stop();

  const manifest = await loadManifest(episode);
  assert.equal(manifest.incomplete, false);
  assert.equal(manifest.label, "maximize-stutter");
  assert.equal(manifest.chunks.length, 1);
  assert.equal(manifest.total_records, 3);
  assert.equal("path" in manifest.metadata, false);
  const chunk = manifest.chunks[0]!;
  const compressed = await readFile(path.join(episode, chunk.file));
  assert.equal(createHash("sha256").update(compressed).digest("hex"), chunk.sha256);
  const records = decodeRecords(compressed);
  assert.deepEqual(records.map((record) => record.kind), ["session_start", "marker", "session_end"]);
  assert.deepEqual(records.map((record) => record.record_seq), [0, 1, 2]);
  assert.ok(records.every((record) => record.schema === TRACE_SCHEMA && record.version === TRACE_VERSION));
  assert.equal((await readdir(episode)).some((name) => name.startsWith("manifest.json.") && name.endsWith(".partial")), false);
  assert.equal(recorder.status.active, false);
});

test("anonymizes display, window, surface, click and plan ids consistently", async () => {
  const root = await temporaryRoot();
  const recorder = new TraceRecorder({ rootDir: root, minimumFreeBytes: 0, getFreeBytes: plentyOfSpace });
  const episode = await recorder.start();
  const world = sampleWorld();
  const plan = samplePlan();
  recorder.record("world_state", { seq: 7, timestamp_ms: 1_000, state: world });
  recorder.record("surface_snapshot", {
    captured_at_ms: 1_001,
    displays: world.displays,
    windows: world.windows,
    surfaces: world.surfaces,
    scene: world.scene,
    title: "must never be recorded",
    owner_path: "C:\\private.exe",
  });
  recorder.record("plan_received", { plan, received_at_ms: 1_002 });
  recorder.record("plan_result", { plan_id: plan.plan_id, based_on_seq: 7, result: "accepted" });
  await recorder.stop();

  const records = await loadAllRecords(episode);
  const worldRecord = records.find((record) => record.kind === "world_state")!;
  const worldPayload = worldRecord.payload as Record<string, unknown>;
  const sanitizedWorld = worldPayload.state as WorldStatePayload;
  assert.equal(sanitizedWorld.session_id, "session");
  assert.equal(sanitizedWorld.displays[0]?.id, "display-0");
  assert.equal(sanitizedWorld.windows[0]?.id, "window-0");
  assert.equal(sanitizedWorld.windows[0]?.display_id, "display-0");
  assert.equal(sanitizedWorld.surfaces[0]?.id, "surface-0");
  assert.equal(sanitizedWorld.surfaces[0]?.window_id, "window-0");
  assert.equal(sanitizedWorld.pet.surface_id, "surface-0");
  assert.equal(sanitizedWorld.clicks[0]?.id, "click-0");
  const planPayload = records.find((record) => record.kind === "plan_received")!.payload as unknown as { plan: HorizonPlanPayload };
  assert.equal(planPayload.plan.plan_id, "plan-0");
  assert.equal(planPayload.plan.target?.surface_id, "surface-0");
  const resultPayload = records.find((record) => record.kind === "plan_result")!.payload as Record<string, unknown>;
  assert.equal(resultPayload.plan_id, "plan-0");
  const snapshotPayload = records.find((record) => record.kind === "surface_snapshot")!.payload as Record<string, unknown>;
  assert.equal("title" in snapshotPayload, false);
  assert.equal("owner_path" in snapshotPayload, false);
});

test("rotates small chunks and keeps every critical record", async () => {
  const root = await temporaryRoot();
  const recorder = new TraceRecorder({
    rootDir: root,
    minimumFreeBytes: 0,
    maxChunkBytes: 420,
    getFreeBytes: plentyOfSpace,
  });
  const episode = await recorder.start();
  for (let index = 0; index < 12; index += 1) {
    assert.equal(recorder.record("marker", { index, detail: "x".repeat(120) }), "accepted");
  }
  await recorder.stop();
  const manifest = await loadManifest(episode);
  assert.ok(manifest.chunks.length > 1);
  assert.equal(manifest.total_records, 14);
  assert.deepEqual((await loadAllRecords(episode)).map((record) => record.record_seq), Array.from({ length: 14 }, (_, index) => index));
});

test("rotates when the configurable chunk duration elapses", async () => {
  const root = await temporaryRoot();
  let monotonicUs = 50_000;
  const recorder = new TraceRecorder({
    rootDir: root,
    minimumFreeBytes: 0,
    maxChunkDurationMs: 1,
    monotonicUs: () => monotonicUs,
    getFreeBytes: plentyOfSpace,
  });
  const episode = await recorder.start();
  monotonicUs += 2_000;
  recorder.record("marker", { name: "after-threshold" });
  await recorder.stop();
  const manifest = await loadManifest(episode);
  assert.equal(manifest.chunks.length, 2);
  assert.equal(manifest.chunks[1]?.started_elapsed_us, 2_000);
});

test("each start-stop pair creates a new episode and resets record identity", async () => {
  const root = await temporaryRoot();
  const recorder = new TraceRecorder({ rootDir: root, minimumFreeBytes: 0, getFreeBytes: plentyOfSpace });
  const first = await recorder.start();
  recorder.record("plan_result", { plan_id: "source-plan", based_on_seq: 1, result: "accepted" });
  await recorder.stop();
  const second = await recorder.start();
  recorder.record("plan_result", { plan_id: "source-plan", based_on_seq: 2, result: "accepted" });
  await recorder.stop();
  assert.notEqual(first, second);
  for (const episode of [first, second]) {
    const records = await loadAllRecords(episode);
    assert.equal(records[0]?.record_seq, 0);
    const result = records.find((record) => record.kind === "plan_result")!.payload as Record<string, unknown>;
    assert.equal(result.plan_id, "plan-0");
  }
});

test("drops only samples under backpressure and emits an explicit gap", async () => {
  const root = await temporaryRoot();
  const recorder = new TraceRecorder({
    rootDir: root,
    minimumFreeBytes: 0,
    maxBufferedBytes: 1,
    getFreeBytes: plentyOfSpace,
  });
  const episode = await recorder.start();
  assert.equal(recorder.record("motion_sample", { foot: { x: 1, y: 2 } }), "dropped");
  assert.equal(recorder.record("motion_sample", { foot: { x: 2, y: 3 } }), "dropped");
  assert.equal(recorder.record("marker", { name: "critical" }), "accepted");
  await recorder.stop();

  const records = await loadAllRecords(episode);
  assert.equal(records.some((record) => record.kind === "motion_sample"), false);
  const gap = records.find((record) => record.kind === "recording_gap")!;
  assert.equal((gap.payload as Record<string, unknown>).dropped, 2);
  assert.equal(records.some((record) => record.kind === "marker"), true);
  assert.equal((await loadManifest(episode)).dropped_motion_samples, 2);
});

test("recovers complete lines from a torn hot chunk and marks the episode incomplete", async () => {
  const root = await temporaryRoot();
  const episode = path.join(root, "crashed-episode");
  await mkdir(episode, { recursive: true });
  const records = [traceRecord(0, "session_start"), traceRecord(1, "marker")];
  await writeFile(
    path.join(episode, "trace-00000.ndjson.partial"),
    `${records.map((record) => JSON.stringify(record)).join("\n")}\n{"torn":`,
    "utf8",
  );

  const recorder = new TraceRecorder({ rootDir: root, minimumFreeBytes: 0, getFreeBytes: plentyOfSpace });
  assert.deepEqual(await recorder.recoverIncompleteEpisodes(), [episode]);
  const manifest = await loadManifest(episode);
  assert.equal(manifest.incomplete, true);
  assert.equal(manifest.end_reason, "recovered_after_crash");
  assert.equal(manifest.duration_us, 1_000);
  assert.equal(manifest.total_records, 2);
  assert.deepEqual((await loadAllRecords(episode)).map((record) => record.record_seq), [0, 1]);
  assert.equal((await readdir(episode)).some((name) => name.endsWith(".ndjson.partial")), false);
});

test("refuses to start when less than the configured free space remains", async () => {
  const root = await temporaryRoot();
  const recorder = new TraceRecorder({
    rootDir: root,
    minimumFreeBytes: 1_024,
    getFreeBytes: async () => 1_023,
  });
  await assert.rejects(recorder.start(), InsufficientDiskSpaceError);
  assert.equal(recorder.status.active, false);
});

test("stops safely and notifies when free space falls below the runtime floor", async () => {
  const root = await temporaryRoot();
  let freeBytes = 2_048;
  let notifyError: ((error: Error) => void) | undefined;
  const notified = new Promise<Error>((resolve) => { notifyError = resolve; });
  const recorder = new TraceRecorder({
    rootDir: root,
    minimumFreeBytes: 1_024,
    diskCheckIntervalMs: 5,
    getFreeBytes: async () => freeBytes,
    onError: (error) => notifyError?.(error),
  });
  const episode = await recorder.start();
  freeBytes = 512;
  assert.ok(await notified instanceof InsufficientDiskSpaceError);
  for (let attempt = 0; attempt < 100 && recorder.status.episodeDir !== null; attempt += 1) {
    await new Promise<void>((resolve) => setTimeout(resolve, 5));
  }
  assert.equal(recorder.status.episodeDir, null);
  const manifest = await loadManifest(episode);
  assert.equal(manifest.incomplete, false);
  assert.equal(manifest.end_reason, "low_disk");
});

function sampleWorld(): WorldStatePayload {
  return {
    session_id: "private-session-id",
    coordinate_space: "physical_px",
    displays: [{
      id: "physical-display-77",
      bounds: { x: 0, y: 0, width: 1920, height: 1080 },
      work_area: { x: 0, y: 0, width: 1920, height: 1040 },
      scale_factor: 1,
      is_primary: true,
    }],
    windows: [{
      id: "native-window-1234",
      display_id: "physical-display-77",
      bounds: { x: 100, y: 200, width: 800, height: 600 },
      z_order: 0,
      visible: true,
      minimized: false,
      maximized: false,
      fullscreen: false,
      active: true,
      occluded: false,
      eligible: true,
    }],
    surfaces: [{
      id: "native-window-1234:top:0",
      kind: "window_top",
      display_id: "physical-display-77",
      window_id: "native-window-1234",
      x1: 100,
      x2: 900,
      y: 200,
      enabled: true,
      occluded: false,
    }],
    pet: {
      x: 200,
      y: 104,
      width: 96,
      height: 96,
      foot_x: 248,
      foot_y: 200,
      vx: 0,
      vy: 0,
      facing: 1,
      behavior: "idle",
      visible: true,
      user_dragging: false,
      surface_id: "native-window-1234:top:0",
    },
    cursor: { x: 240, y: 150, left_down: false, right_down: false, middle_down: false, over_pet: true },
    clicks: [{ id: "private-click-id", button: "left", x: 240, y: 150, target: "pet", timestamp_ms: 1_000 }],
    scene: { fullscreen_active: false, pet_allowed: true },
    seed: 42,
  };
}

function samplePlan(): HorizonPlanPayload {
  return {
    plan_id: "generator-plan-private-id",
    based_on_seq: 7,
    behavior: "walk",
    generated_at_ms: 1_001,
    valid_until_ms: 2_001,
    dt_ms: 33,
    confidence: 1,
    seed: 42,
    target: { surface_id: "native-window-1234:top:0", foot_x: 300, foot_y: 200 },
    points: [{ t_ms: 0, dx: 0, dy: 0, vx: 0, vy: 0, facing: 1, lean: 0, squash: 1, bob: 0, expression: "neutral" }],
  };
}

function traceRecord(recordSeq: number, kind: "session_start" | "marker"): TraceRecord {
  return {
    schema: TRACE_SCHEMA,
    version: TRACE_VERSION,
    record_seq: recordSeq,
    wall_time_ms: 10_000 + recordSeq,
    elapsed_us: recordSeq * 1_000,
    kind,
    payload: {},
  };
}

async function temporaryRoot(): Promise<string> {
  return mkdtemp(path.join(os.tmpdir(), "pet-trace-test-"));
}

async function loadManifest(episode: string): Promise<TraceManifest> {
  return JSON.parse(await readFile(path.join(episode, "manifest.json"), "utf8")) as TraceManifest;
}

function decodeRecords(compressed: Buffer): TraceRecord[] {
  return gunzipSync(compressed).toString("utf8").trim().split("\n").filter(Boolean).map((line) => JSON.parse(line) as TraceRecord);
}

async function loadAllRecords(episode: string): Promise<TraceRecord[]> {
  const manifest = await loadManifest(episode);
  const records: TraceRecord[] = [];
  for (const chunk of manifest.chunks) records.push(...decodeRecords(await readFile(path.join(episode, chunk.file))));
  return records;
}
