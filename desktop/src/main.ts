import { app, dialog, Notification, screen } from "electron";
import { randomUUID } from "node:crypto";
import { mkdir, readdir, stat } from "node:fs/promises";
import { join, resolve } from "node:path";

import { dipPointToPhysical } from "./coordinates.js";
import { DebugOverlay } from "./debug-overlay.js";
import { GeneratorBridge, type GeneratorStatus } from "./generator-bridge.js";
import { info, initializeLogger, logError, warn } from "./logger.js";
import { MotionController } from "./motion-controller.js";
import { PetWindow, PET_WINDOW_DIP } from "./pet-window.js";
import type { ClickEvent, SceneState, WorldStatePayload } from "./protocol.js";
import { SurfaceTracker, type SurfaceSnapshot } from "./surface-tracker.js";
import { PetTray } from "./tray.js";
import type { TraceRecordingStatus } from "./trace/format.js";
import { TraceRecorder } from "./trace/recorder.js";
import { buildRuntimeTraceMetadata } from "./trace/runtime-metadata.js";

interface Runtime {
  readonly petWindow: PetWindow;
  readonly motion: MotionController;
  readonly overlay: DebugOverlay;
  readonly surfaceTracker: SurfaceTracker;
  readonly bridge: GeneratorBridge;
  readonly recorder: TraceRecorder;
  readonly tray: PetTray;
  readonly worldTimer: NodeJS.Timeout;
  readonly uiTimer: NodeJS.Timeout;
}

const singleInstance = app.requestSingleInstanceLock();
if (!singleInstance) {
  app.quit();
} else {
  app.setAppUserModelId("PET.GenerativeDesktopPet");
  void app.whenReady().then(bootstrap).catch((error) => {
    logError("app", "bootstrap failed", error);
    dialog.showErrorBox("PET 启动失败", error instanceof Error ? error.message : String(error));
    app.quit();
  });
}

let runtime: Runtime | null = null;
let quitting = false;

app.on("window-all-closed", () => {
  // Tray-first host: keep the main process alive until the explicit Exit item.
});

app.on("before-quit", (event) => {
  if (quitting || !runtime) return;
  event.preventDefault();
  quitting = true;
  const closing = runtime;
  runtime = null;
  void shutdownRuntime(closing);
});

async function shutdownRuntime(closing: Runtime): Promise<void> {
  clearInterval(closing.worldTimer);
  clearInterval(closing.uiTimer);
  closing.surfaceTracker.dispose();
  closing.motion.dispose();
  // dispose() emits the final generator shutdown/cancel telemetry while the
  // recorder still accepts critical records.
  closing.bridge.dispose();
  try {
    await closing.recorder.stop("app_quit");
  } catch (error) {
    logError("trace", "failed to finalize recording while quitting", error);
  } finally {
    closing.overlay.dispose();
    closing.petWindow.dispose();
    closing.tray.dispose();
    app.quit();
  }
}

async function bootstrap(): Promise<void> {
  if (process.platform !== "win32") {
    dialog.showErrorBox("平台不受支持", "PET 首个里程碑仅支持 Windows 11。");
    app.quit();
    return;
  }

  const projectRoot = resolve(process.env.PET_PROJECT_ROOT || resolve(app.getAppPath(), ".."));
  const logsDirectory = join(app.getPath("userData"), "logs");
  const recordingsDirectory = join(projectRoot, "data", "recordings");
  initializeLogger(join(logsDirectory, "pet-host.log"));
  info("app", "startup begin", { platform: process.platform, electron: process.versions.electron });
  await mkdir(recordingsDirectory, { recursive: true });

  let recordingStatus: TraceRecordingStatus = {
    active: false,
    elapsedMs: 0,
    bytes: 0,
    episodeDir: null,
    droppedSamples: 0,
  };
  const recorder = new TraceRecorder({
    rootDir: recordingsDirectory,
    onStateChange: (status) => { recordingStatus = status; },
    onError: (error) => {
      logError("trace", "recording stopped after a persistence error", error);
      notifyUser("PET 轨迹记录已停止", "磁盘空间不足或轨迹文件无法写入；桌宠会继续正常运行。");
    },
  });
  try {
    const recovered = await recorder.recoverIncompleteEpisodes();
    if (recovered.length > 0) {
      warn("trace", "recovered incomplete recording episodes", { count: recovered.length });
      notifyUser("PET 已恢复轨迹", `已恢复 ${recovered.length} 个意外中断的记录；这些会标记为 incomplete。`);
    }
  } catch (error) {
    logError("trace", "failed to recover incomplete recordings", error);
  }
  void warnWhenRecordingStoreIsLarge(recordingsDirectory);

  const primary = screen.getPrimaryDisplay();
  const initialFootDip = {
    x: primary.workArea.x + primary.workArea.width - PET_WINDOW_DIP,
    y: primary.workArea.y + primary.workArea.height,
  };
  const initialFootPhysical = dipPointToPhysical(initialFootDip);
  const sessionId = randomUUID();
  let currentSurfaceSnapshot: SurfaceSnapshot | null = null;
  let generatorStatus: GeneratorStatus = "starting";
  let paused = false;
  let debugEnabled = process.env.PET_DEBUG_OVERLAY === "1";
  let pointerOpaque = false;
  let clicks: ClickEvent[] = [];
  let bridge: GeneratorBridge | null = null;
  let motion: MotionController | null = null;
  let tray: PetTray | null = null;
  let latestGeneratorMetadata: Readonly<Record<string, unknown>> | undefined;
  let recordingToggleBusy = false;

  const publishWorldState = (): void => {
    if (!bridge || !motion || !currentSurfaceSnapshot) return;
    const cursor = dipPointToPhysical(screen.getCursorScreenPoint());
    const scene: SceneState = paused
      ? { fullscreen_active: currentSurfaceSnapshot.scene.fullscreen_active, pet_allowed: false, suspend_reason: "user_paused" }
      : currentSurfaceSnapshot.scene;
    const payload: WorldStatePayload = {
      session_id: sessionId,
      coordinate_space: "physical_px",
      displays: [...currentSurfaceSnapshot.displays],
      windows: [...currentSurfaceSnapshot.windows],
      surfaces: [...currentSurfaceSnapshot.surfaces],
      pet: motion.getPetState(),
      cursor: {
        x: cursor.x,
        y: cursor.y,
        left_down: false,
        right_down: false,
        middle_down: false,
        over_pet: pointerOpaque,
      },
      clicks: [...clicks],
      scene,
    };
    const envelope = bridge.sendWorldState(payload);
    motion.rememberWorldState(envelope.seq);
    recorder.record("world_state", {
      seq: envelope.seq,
      timestamp_ms: envelope.timestamp_ms,
      state: envelope.payload,
    });
    if (generatorStatus === "ready") clicks = [];
  };

  const petWindow = new PetWindow({
    initialFootDip,
    projectRoot,
    onPointerOpaque: (opaque) => {
      pointerOpaque = opaque;
      motion?.setPointerOpaque(opaque);
    },
    onClick: (button) => {
      if (!motion) return;
      const cursor = dipPointToPhysical(screen.getCursorScreenPoint());
      clicks.push({ id: randomUUID(), button, x: cursor.x, y: cursor.y, target: "pet", timestamp_ms: Date.now() });
      if (clicks.length > 32) clicks = clicks.slice(-32);
      motion.registerClickFeedback();
      publishWorldState();
    },
  });
  await petWindow.load();

  motion = new MotionController({
    petWindow,
    initialFootPhysical,
    onPlanCancelled: (reason) => bridge?.sendCancel(reason),
    onMotionSample: (sample) => {
      recorder.record("motion_sample", {
        timestamp_ms: sample.timestampMs,
        dt_ms: sample.dtMs,
        foot: sample.foot,
        velocity: sample.velocity,
        behavior: sample.behavior,
        source: sample.source,
        ...(sample.surfaceId ? { surface_id: sample.surfaceId } : {}),
        ...(sample.planId ? { plan_id: sample.planId } : {}),
        ...(sample.basedOnSeq !== undefined ? { based_on_seq: sample.basedOnSeq } : {}),
      }, "sample");
    },
  });

  const overlay = new DebugOverlay();
  await overlay.initialize();
  overlay.setEnabled(debugEnabled);
  motion.setDebug(debugEnabled);

  const recordingIndicator = () => {
    const status = recorder.status;
    return { active: status.active, elapsedMs: status.elapsedMs, bytesWritten: status.bytes };
  };
  const refreshRecordingUi = (): void => {
    const status = recordingIndicator();
    overlay.setRecording(status);
    tray?.refresh({ paused, debug: debugEnabled, generatorStatus, recording: status });
  };
  const refreshTray = (): void => refreshRecordingUi();

  bridge = new GeneratorBridge({
    projectRoot,
    sessionId,
    petWidthPhysical: PET_WINDOW_DIP * primary.scaleFactor,
    petHeightPhysical: PET_WINDOW_DIP * primary.scaleFactor,
    onStatus: (status) => {
      generatorStatus = status;
      motion?.setGeneratorStatus(status);
      recorder.record("generator_status", { status, restart_count: bridge?.restartCount ?? 0 });
      refreshTray();
    },
    onPlan: (plan) => {
      const receivedAt = Date.now();
      recorder.record("plan_received", {
        plan,
        received_at_ms: receivedAt,
        latency_ms: Math.max(0, receivedAt - plan.generated_at_ms),
      });
      const result = motion?.offerPlan(plan);
      if (result) recorder.record("plan_result", {
        plan_id: plan.plan_id,
        based_on_seq: plan.based_on_seq,
        result,
      });
      if (result && result !== "accepted") bridge?.sendCancel("safety", plan.plan_id, plan.based_on_seq);
    },
    onMetrics: (metrics) => {
      overlay.setGeneratorMetrics(metrics);
      recorder.record("generator_metrics", { metrics });
    },
    onEnvelopeSent: (envelope) => {
      if (envelope.type === "cancel") recorder.record("cancel", envelope.payload);
    },
    onEnvelopeReceived: (envelope) => {
      if (envelope.type !== "ready" || typeof envelope.payload !== "object" || envelope.payload === null) return;
      const ready = envelope.payload as Record<string, unknown>;
      if (typeof ready.generator === "object" && ready.generator !== null) {
        latestGeneratorMetadata = ready.generator as Readonly<Record<string, unknown>>;
        recorder.record("marker", { name: "generator_runtime", generator: latestGeneratorMetadata });
      }
    },
    onProtocolError: (reason) => {
      warn("protocol", "generator boundary rejected input", { reason });
      recorder.record("marker", { name: "protocol_error", reason });
    },
    onSkeletalMode: (mode) => {
      if (mode === "full") {
        const config = bridge?.skeletalConfig;
        if (config && config.modelDrivenCount > 0) {
          motion?.setSkeletalConfig(config);
          petWindow.sendSkeletalConfig(config.raw);
          info("skeleton", "FK rendering active", { modelDrivenCount: config.modelDrivenCount });
        }
      }
    },
  });

  const surfaceTracker = new SurfaceTracker({
    ownProcessId: process.pid,
    minimumSurfaceWidthDip: PET_WINDOW_DIP,
    onSnapshot: (snapshot) => {
      currentSurfaceSnapshot = snapshot;
      motion?.setSurfaceSnapshot(snapshot);
      overlay.setSceneAllowed(snapshot.scene.pet_allowed);
      recorder.record("surface_snapshot", {
        captured_at_ms: snapshot.capturedAtMs,
        displays: snapshot.displays,
        windows: snapshot.windows,
        surfaces: snapshot.surfaces,
        scene: snapshot.scene,
      });
    },
  });

  const startRecording = async (trigger: "tray" | "environment"): Promise<void> => {
    if (recorder.status.active || recordingToggleBusy) return;
    recordingToggleBusy = true;
    try {
      if (!currentSurfaceSnapshot) await surfaceTracker.refreshNow();
      const snapshot = currentSurfaceSnapshot;
      if (!snapshot) throw new Error("No desktop geometry snapshot is available yet.");
      const metadata = await buildRuntimeTraceMetadata({
        projectRoot,
        appVersion: app.getVersion(),
        displays: snapshot.displays,
        ...(latestGeneratorMetadata ? { generator: latestGeneratorMetadata } : {}),
      });
      const label = process.env.PET_RECORD_LABEL;
      const directory = await recorder.start({
        ...(label ? { label } : {}),
        metadata,
      });
      recorder.record("marker", {
        name: "initial_runtime_config",
        trigger,
        paused,
        debug_overlay: debugEnabled,
        generator_status: generatorStatus,
      });
      recorder.record("surface_snapshot", {
        captured_at_ms: snapshot.capturedAtMs,
        displays: snapshot.displays,
        windows: snapshot.windows,
        surfaces: snapshot.surfaces,
        scene: snapshot.scene,
      });
      recorder.record("generator_status", { status: generatorStatus, restart_count: bridge?.restartCount ?? 0 });
      publishWorldState();
      info("trace", "recording episode started", { trigger, directory: "data/recordings/<episode>" });
    } catch (error) {
      logError("trace", "failed to start recording", error);
      notifyUser("PET 无法开始轨迹记录", "请检查磁盘空间与 data/recordings 目录的写入权限。");
    } finally {
      recordingToggleBusy = false;
      refreshRecordingUi();
    }
  };

  const stopRecording = async (reason = "user_stopped"): Promise<void> => {
    if (!recorder.status.active || recordingToggleBusy) return;
    recordingToggleBusy = true;
    try {
      await recorder.stop(reason);
      info("trace", "recording episode finalized", { reason });
    } catch (error) {
      logError("trace", "failed to finalize recording", error);
      notifyUser("PET 轨迹保存不完整", "热记录仍保留为 .partial，下次启动时会尝试恢复。");
    } finally {
      recordingToggleBusy = false;
      refreshRecordingUi();
    }
  };

  const toggleRecording = (): void => {
    if (recorder.status.active) void stopRecording();
    else void startRecording("tray");
  };

  tray = await PetTray.create(projectRoot, logsDirectory, recordingsDirectory, {
    togglePaused: () => {
      paused = !paused;
      motion?.setPaused(paused);
      if (paused) bridge?.sendCancel("safety");
      refreshTray();
      publishWorldState();
    },
    toggleDebug: () => {
      debugEnabled = !debugEnabled;
      overlay.setEnabled(debugEnabled);
      motion?.setDebug(debugEnabled);
      refreshTray();
    },
    toggleRecording,
    restartGenerator: () => bridge?.restart(),
    quit: () => app.quit(),
  }, { paused, debug: debugEnabled, generatorStatus, recording: recordingIndicator() });

  motion.setGeneratorStatus(generatorStatus);
  motion.start();
  surfaceTracker.start();
  bridge.start();

  let previousCpu = process.cpuUsage();
  let previousCpuAt = Date.now();
  let nextProcessMetricsAt = previousCpuAt + 1_000;
  const worldTimer = setInterval(() => {
    publishWorldState();
    if (motion) {
      overlay.setGeneratorRestarts(bridge?.restartCount ?? 0);
      overlay.setRecording(recordingIndicator());
      overlay.update(motion.getDebugState());
    }
    const now = Date.now();
    if (now >= nextProcessMetricsAt) {
      const cpu = process.cpuUsage(previousCpu);
      const elapsedUs = Math.max(1, (now - previousCpuAt) * 1_000);
      recorder.record("process_metrics", {
        host_rss_bytes: process.memoryUsage().rss,
        host_cpu_percent: ((cpu.user + cpu.system) / elapsedUs) * 100,
        ...(bridge?.childPid ? { generator_pid: bridge.childPid } : {}),
      });
      previousCpu = process.cpuUsage();
      previousCpuAt = now;
      nextProcessMetricsAt = now + 1_000;
    }
  }, 50);
  const uiTimer = setInterval(refreshRecordingUi, 1_000);

  runtime = { petWindow, motion, overlay, surfaceTracker, bridge, recorder, tray, worldTimer, uiTimer };
  if (process.env.PET_RECORD_SESSION === "1") void startRecording("environment");
  info("app", "startup complete", { session: "local", debug: debugEnabled });
}

function notifyUser(title: string, body: string): void {
  try {
    if (Notification.isSupported()) new Notification({ title, body, silent: true }).show();
  } catch (error) {
    logError("app", "failed to show desktop notification", error);
  }
}

async function warnWhenRecordingStoreIsLarge(directory: string): Promise<void> {
  try {
    const bytes = await directorySize(directory);
    if (bytes <= 5 * 1_024 * 1_024 * 1_024) return;
    warn("trace", "recording store exceeds warning threshold", { bytes, threshold_bytes: 5 * 1_024 * 1_024 * 1_024 });
    notifyUser("PET 轨迹目录超过 5 GiB", "记录不会自动删除；请在方便时通过托盘打开轨迹目录并手动整理。");
  } catch (error) {
    logError("trace", "failed to inspect recording store size", error);
  }
}

async function directorySize(directory: string): Promise<number> {
  let total = 0;
  const entries = await readdir(directory, { withFileTypes: true });
  for (const entry of entries) {
    const child = join(directory, entry.name);
    if (entry.isDirectory()) total += await directorySize(child);
    else if (entry.isFile()) total += (await stat(child)).size;
  }
  return total;
}
