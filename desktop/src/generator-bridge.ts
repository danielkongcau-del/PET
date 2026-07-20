import { spawn, type ChildProcessWithoutNullStreams } from "node:child_process";
import { createHash } from "node:crypto";
import { existsSync } from "node:fs";
import { readFile } from "node:fs/promises";
import { randomUUID } from "node:crypto";
import { join, resolve } from "node:path";

import { CAP_SKELETAL_MOTION, CAP_SKELETAL_MOTION_3D, createEnvelope, decodeEnvelope, encodeEnvelope, parseHorizonPlan, parseMetrics, parsePong, parseReady, type Envelope, type HelloPayload, type HorizonPlanPayload, type MessageType, type MetricsPayload, type WorldStatePayload } from "./protocol.js";
import { debug, info, logError, warn } from "./logger.js";

export type GeneratorStatus = "disabled" | "starting" | "ready" | "degraded" | "stopped";
export type SkeletalMode = "full" | "host_only" | "generator_only" | "none";

/** Parsed from cat-skeleton.json — the single source of truth for bone structure. */
export interface SkeletalConfig {
  /** Full skeleton JSON as loaded from disk (null if unavailable). */
  readonly raw: Record<string, unknown> | null;
  /** SHA-256 of the normalized skeleton JSON. */
  readonly sha256: string | null;
  /** Bone IDs whose physics.mode is "model_driven", in definition order. */
  readonly modelDrivenBoneIds: readonly string[];
  /** Convenience: modelDrivenBoneIds.length. */
  readonly modelDrivenCount: number;
}

export interface GeneratorBridgeOptions {
  readonly projectRoot: string;
  readonly sessionId: string;
  readonly petWidthPhysical: number;
  readonly petHeightPhysical: number;
  readonly onStatus: (status: GeneratorStatus) => void;
  readonly onPlan: (plan: HorizonPlanPayload) => void;
  readonly onMetrics?: (metrics: MetricsPayload) => void;
  readonly onProtocolError?: (reason: string) => void;
  readonly onEnvelopeSent?: (envelope: Envelope) => void;
  readonly onEnvelopeReceived?: (envelope: Envelope) => void;
  readonly onProcessChanged?: (pid: number | null) => void;
  /** Called when skeletal capability negotiation finishes. */
  readonly onSkeletalMode?: (mode: SkeletalMode) => void;
}

const READY_TIMEOUT_MS = 8_000;
const HEARTBEAT_INTERVAL_MS = 2_000;
const HEARTBEAT_STALE_MS = 6_000;

const HOST_CAPABILITIES = [
  "window_top_surfaces",
  "pet_clicks",
  "safe_motion",
  "fullscreen_guard",
  CAP_SKELETAL_MOTION,
  CAP_SKELETAL_MOTION_3D,
] as const;

export class GeneratorBridge {
  readonly #options: GeneratorBridgeOptions;
  #child: ChildProcessWithoutNullStreams | null = null;
  #status: GeneratorStatus = "stopped";
  #outgoingSeq = 0;
  #lastChildSeq = -1;
  #stdoutBuffer = "";
  #stderrBuffer = "";
  #pendingWorldState: Envelope<"world_state", WorldStatePayload> | null = null;
  #stdinBlocked = false;
  #readyTimer: NodeJS.Timeout | null = null;
  #heartbeatTimer: NodeJS.Timeout | null = null;
  #restartTimer: NodeJS.Timeout | null = null;
  #lastPongAt = 0;
  #pendingPings = new Map<string, number>();
  #restartAttempt = 0;
  #generation = 0;
  #disposed = false;
  #restartCount = 0;
  #skeletalMode: SkeletalMode = "none";
  #skeletalConfig: SkeletalConfig = {
    raw: null,
    sha256: null,
    modelDrivenBoneIds: [],
    modelDrivenCount: 0,
  };
  #skeletalConfigLoaded = false;

  constructor(options: GeneratorBridgeOptions) {
    this.#options = options;
  }

  get status(): GeneratorStatus {
    return this.#status;
  }

  get restartCount(): number {
    return this.#restartCount;
  }

  get skeletalMode(): SkeletalMode {
    return this.#skeletalMode;
  }

  /** The loaded skeleton definition. Call loadSkeletalConfig() first. */
  get skeletalConfig(): SkeletalConfig {
    return this.#skeletalConfig;
  }

  get childPid(): number | null {
    return this.#child?.pid ?? null;
  }

  start(): void {
    if (this.#disposed || this.#child || this.#restartTimer) return;
    if (process.env.PET_DISABLE_GENERATOR === "1") {
      this.#setStatus("disabled");
      info("generator", "sidecar disabled by environment");
      return;
    }
    this.#spawnChild();
  }

  restart(): void {
    if (this.#disposed || this.#status === "disabled") return;
    info("generator", "manual restart requested");
    this.#restartAttempt = 0;
    this.#stopCurrentChild("restart");
    this.#scheduleRestart(0);
  }

  sendWorldState(payload: WorldStatePayload): Envelope<"world_state", WorldStatePayload> {
    const carriedClicks = this.#pendingWorldState?.payload.clicks ?? [];
    let effectivePayload = payload;
    if (carriedClicks.length > 0) {
      const byId = new Map([...carriedClicks, ...payload.clicks].map((click) => [click.id, click]));
      effectivePayload = { ...payload, clicks: [...byId.values()].slice(-32) };
    }
    const envelope = createEnvelope("world_state", this.#outgoingSeq++, effectivePayload);
    this.#pendingWorldState = envelope;
    this.#flushWorldState();
    return envelope;
  }

  sendCancel(reason: "user_drag" | "topology_change" | "safety" | "newer_state" | "shutdown", planId?: string, basedOnSeq?: number): void {
    if (!this.#child || this.#status === "stopped" || this.#status === "disabled") return;
    const payload: Record<string, unknown> = { reason, requested_at_ms: Date.now() };
    if (planId) payload.plan_id = planId;
    if (basedOnSeq !== undefined) payload.based_on_seq = basedOnSeq;
    this.#writeEnvelope(createEnvelope("cancel", this.#outgoingSeq++, payload));
  }

  dispose(): void {
    if (this.#disposed) return;
    this.#disposed = true;
    this.#clearTimers();
    this.sendCancel("shutdown");
    const child = this.#child;
    this.#child = null;
    this.#options.onProcessChanged?.(null);
    if (child && !child.killed) {
      child.stdin.end();
      const killTimer = setTimeout(() => child.kill(), 600);
      killTimer.unref?.();
      child.once("exit", () => clearTimeout(killTimer));
    }
    this.#setStatus("stopped");
  }

  #spawnChild(): void {
    const generation = ++this.#generation;
    if (this.#restartAttempt > 0) this.#restartCount += 1;
    const defaultPython = "D:\\Anaconda\\envs\\pet-core\\python.exe";
    const python = process.env.PET_PYTHON || (existsSync(defaultPython) ? defaultPython : "python");
    const entry = resolve(process.env.PET_GENERATOR_ENTRY || resolve(this.#options.projectRoot, "services", "generator", "run.py"));
    const cwd = resolve(process.env.PET_GENERATOR_CWD || this.#options.projectRoot);

    this.#setStatus("starting");
    this.#lastChildSeq = -1;
    this.#stdoutBuffer = "";
    this.#stderrBuffer = "";
    this.#stdinBlocked = false;
    this.#lastPongAt = Date.now();
    this.#pendingPings.clear();

    let child: ChildProcessWithoutNullStreams;
    try {
      child = spawn(python, [entry, "--metrics-interval-ms", "5000"], {
        cwd,
        env: { ...process.env, PYTHONUNBUFFERED: "1" },
        stdio: ["pipe", "pipe", "pipe"],
        windowsHide: true,
      });
    } catch (error) {
      logError("generator", "sidecar spawn threw", error);
      this.#setStatus("degraded");
      this.#scheduleRestart();
      return;
    }
    this.#child = child;

    child.once("spawn", () => {
      if (generation !== this.#generation || this.#disposed) return;
      this.#options.onProcessChanged?.(child.pid ?? null);
      info("generator", "sidecar spawned", { pid: child.pid ?? 0, attempt: this.#restartAttempt });
      const hello: HelloPayload = {
        session_id: this.#options.sessionId,
        host: { name: "pet-electron-host", version: "0.1.0", pid: process.pid },
        requested_version: 1,
        capabilities: [...HOST_CAPABILITIES],
        config: {
          world_state_hz: 20,
          plan_horizon_ms: 400,
          plan_dt_ms: 33,
          pet_width: this.#options.petWidthPhysical,
          pet_height: this.#options.petHeightPhysical,
          privacy: { screen_capture_enabled: false, keyboard_enabled: false, recording_enabled: false },
        },
      };
      this.#writeEnvelope(createEnvelope("hello", this.#outgoingSeq++, hello));
      this.#readyTimer = setTimeout(() => {
        warn("generator", "sidecar readiness timeout");
        this.#failGeneration(generation, "ready_timeout");
      }, READY_TIMEOUT_MS);
    });

    child.stdout.setEncoding("utf8");
    child.stdout.on("data", (chunk: string) => this.#handleStdout(generation, chunk));
    child.stderr.setEncoding("utf8");
    child.stderr.on("data", (chunk: string) => this.#handleStderr(chunk));
    child.stdin.on("drain", () => {
      this.#stdinBlocked = false;
      this.#flushWorldState();
    });
    child.once("error", (error) => {
      if (generation !== this.#generation) return;
      logError("generator", "sidecar process error", error);
      this.#failGeneration(generation, "process_error");
    });
    child.once("exit", (code, signal) => {
      if (generation !== this.#generation) return;
      this.#child = null;
      this.#options.onProcessChanged?.(null);
      this.#clearChildTimers();
      if (this.#disposed) return;
      warn("generator", "sidecar exited; safe idle active", { code: code ?? -1, signal: signal ?? "none" });
      this.#setStatus("degraded");
      this.#scheduleRestart();
    });
  }

  #handleStdout(generation: number, chunk: string): void {
    if (generation !== this.#generation) return;
    this.#stdoutBuffer += chunk;
    if (Buffer.byteLength(this.#stdoutBuffer, "utf8") > 2 * 1_048_576) {
      this.#failGeneration(generation, "stdout_buffer_overflow");
      return;
    }
    for (;;) {
      const newline = this.#stdoutBuffer.indexOf("\n");
      if (newline < 0) break;
      const line = this.#stdoutBuffer.slice(0, newline).replace(/\r$/, "");
      this.#stdoutBuffer = this.#stdoutBuffer.slice(newline + 1);
      if (!line.trim()) continue;
      let envelope: Envelope<MessageType, unknown>;
      try {
        envelope = decodeEnvelope(line);
      } catch (error) {
        warn("generator", "invalid NDJSON ignored", { reason: error instanceof Error ? error.message : "invalid" });
        this.#options.onProtocolError?.("invalid_envelope");
        continue;
      }
      if (envelope.seq <= this.#lastChildSeq) {
        warn("generator", "non-monotonic child sequence ignored", { seq: envelope.seq, previous: this.#lastChildSeq });
        continue;
      }
      this.#lastChildSeq = envelope.seq;
      this.#options.onEnvelopeReceived?.(envelope);
      this.#handleEnvelope(envelope);
    }
  }

  #handleEnvelope(envelope: Envelope): void {
    if (envelope.type === "ready") {
      const ready = parseReady(envelope);
      if (!ready || ready.session_id !== this.#options.sessionId) {
        warn("generator", "invalid ready handshake ignored");
        return;
      }
      if (this.#readyTimer) clearTimeout(this.#readyTimer);
      this.#readyTimer = null;
      this.#restartAttempt = 0;
      this.#lastPongAt = Date.now();
      this.#setStatus("ready");
      info("generator", "sidecar ready", { pid: ready.generator.pid, capabilities: ready.capabilities.length });
      this.#startHeartbeat();
      // Await negotiation so skeletal config is loaded before the first world
      // state is flushed — otherwise bone_rotations in early plans may be
      // length-checked against a zero count.
      void this.#negotiateSkeletalMode(ready).then(() => this.#flushWorldState());
      return;
    }
    if (envelope.type === "pong") {
      const pong = parsePong(envelope);
      if (!pong || this.#pendingPings.get(pong.nonce) !== pong.ping_sent_at_ms) {
        warn("generator", "unexpected pong ignored");
        return;
      }
      this.#pendingPings.delete(pong.nonce);
      this.#lastPongAt = Date.now();
      return;
    }
    if (envelope.type === "horizon_plan") {
      if (this.#status !== "ready") return;
      const plan = parseHorizonPlan(envelope);
      if (!plan) {
        warn("generator", "invalid or expired horizon plan ignored");
        this.#options.onProtocolError?.("invalid_plan");
        return;
      }
      this.#options.onPlan(plan);
      return;
    }
    if (envelope.type === "metrics") {
      const metrics = parseMetrics(envelope);
      if (!metrics || metrics.source !== "generator") {
        warn("generator", "invalid generator metrics ignored");
        this.#options.onProtocolError?.("invalid_metrics");
        return;
      }
      this.#options.onMetrics?.(metrics);
      return;
    }
    if (envelope.type === "error") {
      warn("generator", "sidecar reported a recoverable protocol error");
    }
  }

  #handleStderr(chunk: string): void {
    this.#stderrBuffer += chunk;
    for (;;) {
      const newline = this.#stderrBuffer.indexOf("\n");
      if (newline < 0) break;
      const line = this.#stderrBuffer.slice(0, newline).trim();
      this.#stderrBuffer = this.#stderrBuffer.slice(newline + 1);
      if (line) debug("generator", "sidecar diagnostic", { bytes: Buffer.byteLength(line), category: "stderr" });
    }
    if (this.#stderrBuffer.length > 8_192) this.#stderrBuffer = this.#stderrBuffer.slice(-4_096);
  }

  #startHeartbeat(): void {
    if (this.#heartbeatTimer) clearInterval(this.#heartbeatTimer);
    this.#heartbeatTimer = setInterval(() => {
      if (!this.#child || this.#status !== "ready") return;
      if (Date.now() - this.#lastPongAt > HEARTBEAT_STALE_MS) {
        warn("generator", "heartbeat stale; restarting sidecar");
        this.#failGeneration(this.#generation, "heartbeat_stale");
        return;
      }
      const nonce = randomUUID();
      const sentAt = Date.now();
      this.#pendingPings.set(nonce, sentAt);
      for (const [pendingNonce, pendingAt] of this.#pendingPings) {
        if (sentAt - pendingAt > HEARTBEAT_STALE_MS) this.#pendingPings.delete(pendingNonce);
      }
      this.#writeEnvelope(createEnvelope("ping", this.#outgoingSeq++, { nonce, sent_at_ms: sentAt }));
    }, HEARTBEAT_INTERVAL_MS);
    this.#heartbeatTimer.unref?.();
  }

  #flushWorldState(): void {
    if (!this.#pendingWorldState || !this.#child || this.#status !== "ready" || this.#stdinBlocked) return;
    const envelope = this.#pendingWorldState;
    this.#pendingWorldState = null;
    this.#writeEnvelope(envelope);
  }

  #writeEnvelope(envelope: Envelope): void {
    const child = this.#child;
    if (!child || child.stdin.destroyed || child.killed) return;
    try {
      this.#options.onEnvelopeSent?.(envelope);
      const accepted = child.stdin.write(encodeEnvelope(envelope), "utf8");
      if (!accepted) this.#stdinBlocked = true;
    } catch (error) {
      logError("generator", "failed writing to sidecar", error);
    }
  }

  #failGeneration(generation: number, reason: string): void {
    if (generation !== this.#generation || this.#disposed) return;
    this.#setStatus("degraded");
    this.#options.onProtocolError?.(reason);
    this.#stopCurrentChild(reason);
    this.#scheduleRestart();
  }

  #stopCurrentChild(reason: string): void {
    const child = this.#child;
    this.#child = null;
    this.#options.onProcessChanged?.(null);
    this.#generation += 1;
    this.#clearChildTimers();
    if (!child || child.killed) return;
    debug("generator", "stopping sidecar", { reason });
    child.stdin.end();
    child.kill();
  }

  #scheduleRestart(delayOverride?: number): void {
    if (this.#disposed || this.#restartTimer || this.#status === "disabled") return;
    const delay = delayOverride ?? Math.min(5_000, 250 * 2 ** Math.min(this.#restartAttempt, 5));
    this.#restartAttempt += 1;
    this.#restartTimer = setTimeout(() => {
      this.#restartTimer = null;
      this.#spawnChild();
    }, delay);
    this.#restartTimer.unref?.();
  }

  #clearChildTimers(): void {
    if (this.#readyTimer) clearTimeout(this.#readyTimer);
    if (this.#heartbeatTimer) clearInterval(this.#heartbeatTimer);
    this.#readyTimer = null;
    this.#heartbeatTimer = null;
    this.#pendingPings.clear();
  }

  async #negotiateSkeletalMode(ready: ReturnType<typeof parseReady>): Promise<void> {
    if (!ready) return;
    const host3D = (HOST_CAPABILITIES as readonly string[]).includes(CAP_SKELETAL_MOTION_3D);
    const gen3D = ready.capabilities.includes(CAP_SKELETAL_MOTION_3D);
    const host2D = (HOST_CAPABILITIES as readonly string[]).includes(CAP_SKELETAL_MOTION);
    const gen2D = ready.capabilities.includes(CAP_SKELETAL_MOTION);
    const genSkeletonHash = ready.generator.skeleton_sha256 ?? null;

    const config = await this.#loadSkeletalConfig();

    let mode: SkeletalMode;
    // Prefer 3D over 2D when both are available.
    if (host3D && gen3D) {
      if (!genSkeletonHash) {
        warn("skeleton", "Generator advertises skeletal_motion_3d but did not provide a skeleton_sha256; FK disabled.");
        mode = "host_only";
      } else if (!config.sha256) {
        warn("skeleton", "Generator expects 3D skeleton but host has no cat-skeleton-3d.json.");
        mode = "generator_only";
      } else if (genSkeletonHash !== config.sha256) {
        warn("skeleton", `3D skeleton hash mismatch: gen=${genSkeletonHash.slice(0, 12)}..., host=${config.sha256.slice(0, 12)}...`);
        mode = "none";
      } else {
        info("skeleton", "3D skeletal mode fully negotiated", {
          modelDrivenCount: config.modelDrivenCount,
          jointIds: [...config.modelDrivenBoneIds],
        });
        mode = "full";
      }
    } else if (host2D && gen2D) {
      warn("skeleton", "Using legacy planar skeletal_motion; upgrade generator for 3D quaternion output.");
      mode = "full";
    } else if ((host3D || host2D) && !gen3D && !gen2D) {
      warn("skeleton", "Host supports skeletal motion but generator does not; falling back to whole-sprite deformation.");
      mode = "host_only";
    } else if (!host3D && !host2D && (gen3D || gen2D)) {
      warn("skeleton", "Generator outputs skeletal data but host renderer will ignore them.");
      mode = "generator_only";
    } else {
      mode = "none";
    }

    if (mode !== this.#skeletalMode) {
      this.#skeletalMode = mode;
      this.#options.onSkeletalMode?.(mode);
      debug("skeleton", "skeletal mode negotiated", { mode, host3D, gen3D, host2D, gen2D });
    }
  }

  /** Loads and parses cat-skeleton-3d.json once, computing SHA-256 and model_driven joint metadata. */
  async #loadSkeletalConfig(): Promise<SkeletalConfig> {
    if (this.#skeletalConfigLoaded) return this.#skeletalConfig;
    try {
      const path = join(this.#options.projectRoot, "assets", "pet", "runtime", "cat-skeleton-3d.json");
      const raw = await readFile(path);
      const sha256 = createHash("sha256").update(raw).digest("hex");
      const parsed = JSON.parse(raw.toString("utf8")) as Record<string, unknown>;

      const joints = Array.isArray(parsed.joints) ? parsed.joints as Record<string, unknown>[] : [];
      const modelDrivenJointIds = joints
        .filter((j) => {
          const physics = j.physics as Record<string, unknown> | undefined;
          const poseDofs = j.poseDofs as Record<string, unknown> | undefined;
          // Exclude __motion_root__ (parent===null, deform===false) and secondary-physics joints.
          if (j.deform === false && j.parent === null) return false;
          if (physics?.mode === "secondary" || physics?.mode === "static") return false;
          return poseDofs?.rotation === true;
        })
        .map((j) => typeof j.id === "string" ? j.id : "")
        .filter((id) => id.length > 0);

      this.#skeletalConfig = {
        raw: parsed,
        sha256,
        modelDrivenBoneIds: modelDrivenJointIds,
        modelDrivenCount: modelDrivenJointIds.length,
      };
    } catch (error) {
      debug("skeleton", "cat-skeleton-3d.json not available; 3D skeletal motion disabled", {
        reason: error instanceof Error ? error.message : "unknown",
      });
    }
    this.#skeletalConfigLoaded = true;
    return this.#skeletalConfig;
  }

  #clearTimers(): void {
    this.#clearChildTimers();
    if (this.#restartTimer) clearTimeout(this.#restartTimer);
    this.#restartTimer = null;
  }

  #setStatus(status: GeneratorStatus): void {
    if (this.#status === status) return;
    this.#status = status;
    this.#options.onStatus(status);
  }
}
