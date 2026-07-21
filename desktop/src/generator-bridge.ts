import { spawn, type ChildProcessWithoutNullStreams } from "node:child_process";
import { existsSync } from "node:fs";
import { randomUUID } from "node:crypto";
import { resolve } from "node:path";

import { CAP_SKELETAL_MOTION, CAP_SKELETAL_MOTION_3D, createEnvelope, decodeEnvelope, encodeEnvelope, parseHorizonPlan, parseMetrics, parsePong, parseReady, type Envelope, type HelloPayload, type HorizonPlanPayload, type MessageType, type MetricsPayload, type WorldStatePayload } from "./protocol.js";
import { loadCharacterRigConfig, type CharacterRigConfig } from "./character-rig.js";
import { debug, info, logError, warn } from "./logger.js";

export type GeneratorStatus = "disabled" | "starting" | "ready" | "degraded" | "stopped";
/** The negotiated pose encoding is part of the state; 2D and 3D must never share cached rig state. */
export type SkeletalMode = "full_3d" | "full_2d" | "host_only" | "generator_only" | "none";

export interface SkeletalNegotiationToken {
  readonly generation: number;
  readonly serial: number;
}

/**
 * Invalidates async capability negotiations when another ready message or
 * child generation supersedes them. Exported so the race contract can be
 * tested without spawning an Electron process.
 */
export class SkeletalNegotiationGate {
  #serial = 0;

  begin(generation: number): SkeletalNegotiationToken {
    return { generation, serial: ++this.#serial };
  }

  invalidate(): void {
    this.#serial += 1;
  }

  isCurrent(token: SkeletalNegotiationToken, generation: number): boolean {
    return token.generation === generation && token.serial === this.#serial;
  }
}

export function modeUsesQuaternionRig(mode: SkeletalMode): boolean {
  return mode === "full_3d";
}

export function modeUsesCharacterRig(mode: SkeletalMode): boolean {
  return mode === "full_3d" || mode === "full_2d";
}

export interface SkeletalCapabilityState {
  readonly host3D: boolean;
  readonly generator3D: boolean;
  readonly host2D: boolean;
  readonly generator2D: boolean;
  readonly generatorSkeletonHash: string | null;
  readonly hostSkeletonHash: string | null;
}

/** Pure compatibility decision shared by production negotiation and tests. */
export function selectSkeletalMode(state: SkeletalCapabilityState): SkeletalMode {
  if (state.host3D && state.generator3D) {
    if (!state.generatorSkeletonHash) return "host_only";
    if (!state.hostSkeletonHash) return "generator_only";
    return state.generatorSkeletonHash === state.hostSkeletonHash ? "full_3d" : "none";
  }
  if (state.host2D && state.generator2D) {
    // The planar array is indexed by the selected character's driven-joint
    // order, so matching lengths are insufficient: it needs the same exact
    // rig fingerprint as the quaternion encoding.
    if (!state.generatorSkeletonHash) return "host_only";
    if (!state.hostSkeletonHash) return "generator_only";
    return state.generatorSkeletonHash === state.hostSkeletonHash ? "full_2d" : "none";
  }
  if ((state.host3D || state.host2D) && !state.generator3D && !state.generator2D) return "host_only";
  if (!state.host3D && !state.host2D && (state.generator3D || state.generator2D)) return "generator_only";
  return "none";
}

/** Parsed from the selected character manifest (or a legacy rig fallback). */
export interface SkeletalConfig {
  /** Full skeleton JSON as loaded from disk (null if unavailable). */
  readonly raw: Record<string, unknown> | null;
  /** SHA-256 of the normalized skeleton JSON. */
  readonly sha256: string | null;
  /** Bone IDs whose physics.mode is "model_driven", in definition order. */
  readonly modelDrivenBoneIds: readonly string[];
  /** Convenience: modelDrivenBoneIds.length. */
  readonly modelDrivenCount: number;
  readonly characterId: string | null;
  readonly rigId: string | null;
  readonly source: "character_manifest" | "legacy_rig" | null;
}

export interface GeneratorBridgeOptions {
  readonly projectRoot: string;
  readonly sessionId: string;
  readonly petWidthPhysical: number;
  readonly petHeightPhysical: number;
  /** Prevalidated single source of truth shared with the renderer and trace metadata. */
  readonly characterRig?: CharacterRigConfig;
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
const MAX_LEGACY_BONE_ROTATIONS = 32;

/** Advertise 2D only when the selected character fits the v1 planar ABI. */
export function advertisedHostCapabilities(modelDrivenCount: number | null): string[] {
  return HOST_CAPABILITIES.filter((capability) => (
    capability !== CAP_SKELETAL_MOTION
    || (
      modelDrivenCount !== null
      && modelDrivenCount > 0
      && modelDrivenCount <= MAX_LEGACY_BONE_ROTATIONS
    )
  ));
}

/** Shared stale-child guard for callbacks that may fire after a restart. */
export function isCurrentChildEvent(
  eventGeneration: number,
  currentGeneration: number,
  disposed: boolean,
  childMatches: boolean,
): boolean {
  return !disposed && childMatches && eventGeneration === currentGeneration;
}

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
  readonly #skeletalNegotiation = new SkeletalNegotiationGate();
  #skeletalMode: SkeletalMode = "none";
  #skeletalConfig: SkeletalConfig = {
    raw: null,
    sha256: null,
    modelDrivenBoneIds: [],
    modelDrivenCount: 0,
    characterId: null,
    rigId: null,
    source: null,
  };
  #skeletalConfigLoaded = false;

  constructor(options: GeneratorBridgeOptions) {
    this.#options = options;
    if (options.characterRig) {
      const selected = options.characterRig;
      this.#skeletalConfig = {
        raw: selected.raw,
        sha256: selected.sha256,
        modelDrivenBoneIds: selected.modelDrivenBoneIds,
        modelDrivenCount: selected.modelDrivenCount,
        characterId: selected.characterId,
        rigId: selected.rigId,
        source: selected.source,
      };
      this.#skeletalConfigLoaded = true;
    }
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
    this.#skeletalNegotiation.invalidate();
    this.#setSkeletalMode("none");
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
    this.#skeletalNegotiation.invalidate();
    this.#setSkeletalMode("none");
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
        capabilities: advertisedHostCapabilities(
          this.#skeletalConfigLoaded ? this.#skeletalConfig.modelDrivenCount : null,
        ),
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
      if (!isCurrentChildEvent(generation, this.#generation, this.#disposed, child === this.#child)) return;
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
      this.#skeletalNegotiation.invalidate();
      this.#setSkeletalMode("none");
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
      this.#handleEnvelope(generation, envelope);
    }
  }

  #handleEnvelope(generation: number, envelope: Envelope): void {
    if (generation !== this.#generation || !this.#child || this.#disposed) return;
    if (envelope.type === "ready") {
      const ready = parseReady(envelope);
      if (!ready || ready.session_id !== this.#options.sessionId) {
        warn("generator", "invalid ready handshake ignored");
        return;
      }
      if (this.#readyTimer) clearTimeout(this.#readyTimer);
      this.#readyTimer = null;
      if (this.#heartbeatTimer) clearInterval(this.#heartbeatTimer);
      this.#heartbeatTimer = null;
      this.#pendingPings.clear();
      // A repeated ready message is a fresh handshake. Stop accepting plans
      // and clear the previous encoding before any asynchronous rig load.
      this.#setStatus("starting");
      this.#setSkeletalMode("none");
      const token = this.#skeletalNegotiation.begin(generation);
      void this.#completeReadyHandshake(generation, token, ready);
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
    this.#skeletalNegotiation.invalidate();
    this.#setSkeletalMode("none");
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

  async #completeReadyHandshake(
    generation: number,
    token: SkeletalNegotiationToken,
    ready: NonNullable<ReturnType<typeof parseReady>>,
  ): Promise<void> {
    const mode = await this.#negotiateSkeletalMode(generation, token, ready);
    if (mode === null || !this.#isCurrentNegotiation(generation, token)) return;
    this.#setSkeletalMode(mode);
    this.#restartAttempt = 0;
    this.#lastPongAt = Date.now();
    this.#setStatus("ready");
    info("generator", "sidecar ready", {
      pid: ready.generator.pid,
      capabilities: ready.capabilities.length,
      skeletalMode: mode,
    });
    this.#startHeartbeat();
    this.#flushWorldState();
  }

  async #negotiateSkeletalMode(
    generation: number,
    token: SkeletalNegotiationToken,
    ready: NonNullable<ReturnType<typeof parseReady>>,
  ): Promise<SkeletalMode | null> {
    const advertised = advertisedHostCapabilities(
      this.#skeletalConfigLoaded ? this.#skeletalConfig.modelDrivenCount : null,
    );
    const host3D = advertised.includes(CAP_SKELETAL_MOTION_3D);
    const gen3D = ready.capabilities.includes(CAP_SKELETAL_MOTION_3D);
    const host2D = advertised.includes(CAP_SKELETAL_MOTION);
    const gen2D = ready.capabilities.includes(CAP_SKELETAL_MOTION);
    const genSkeletonHash = ready.generator.skeleton_sha256 ?? null;

    const config = await this.#loadSkeletalConfig();
    if (!this.#isCurrentNegotiation(generation, token)) return null;

    const mode = selectSkeletalMode({
      host3D,
      generator3D: gen3D,
      host2D,
      generator2D: gen2D,
      generatorSkeletonHash: genSkeletonHash,
      hostSkeletonHash: config.sha256,
    });
    // Prefer 3D over 2D when both are available, and explain any fail-closed result.
    if (host3D && gen3D) {
      if (!genSkeletonHash) {
        warn("skeleton", "Generator advertises skeletal_motion_3d but did not provide a skeleton_sha256; FK disabled.");
      } else if (!config.sha256) {
        warn("skeleton", "Generator expects a 3D character rig but the host could not load the selected manifest.");
      } else if (genSkeletonHash !== config.sha256) {
        warn("skeleton", `3D skeleton hash mismatch: gen=${genSkeletonHash.slice(0, 12)}..., host=${config.sha256.slice(0, 12)}...`);
      } else {
        info("skeleton", "3D skeletal mode fully negotiated", {
          modelDrivenCount: config.modelDrivenCount,
          jointIds: [...config.modelDrivenBoneIds],
        });
      }
    } else if (host2D && gen2D) {
      if (!genSkeletonHash) {
        warn("skeleton", "Generator advertises planar skeletal_motion without a skeleton_sha256; planar FK disabled.");
      } else if (!config.sha256) {
        warn("skeleton", "Generator expects a planar character rig but the host could not load the selected manifest.");
      } else if (genSkeletonHash !== config.sha256) {
        warn("skeleton", `2D skeleton hash mismatch: gen=${genSkeletonHash.slice(0, 12)}..., host=${config.sha256.slice(0, 12)}...`);
      } else {
        warn("skeleton", "Using legacy planar skeletal_motion; upgrade generator for 3D quaternion output.");
      }
    } else if ((host3D || host2D) && !gen3D && !gen2D) {
      warn("skeleton", "Host supports skeletal motion but generator does not; falling back to whole-sprite deformation.");
    } else if (!host3D && !host2D && (gen3D || gen2D)) {
      warn("skeleton", "Generator outputs skeletal data but host renderer will ignore them.");
    }

    debug("skeleton", "skeletal mode selected", { mode, host3D, gen3D, host2D, gen2D });
    return mode;
  }

  #isCurrentNegotiation(generation: number, token: SkeletalNegotiationToken): boolean {
    return !this.#disposed && this.#child !== null && generation === this.#generation &&
      this.#skeletalNegotiation.isCurrent(token, this.#generation);
  }

  #setSkeletalMode(mode: SkeletalMode): void {
    if (mode === this.#skeletalMode) return;
    this.#skeletalMode = mode;
    this.#options.onSkeletalMode?.(mode);
    debug("skeleton", "skeletal mode changed", { mode });
  }

  /** Loads and validates the selected character rig once. */
  async #loadSkeletalConfig(): Promise<SkeletalConfig> {
    if (this.#skeletalConfigLoaded) return this.#skeletalConfig;
    try {
      const selected = await loadCharacterRigConfig(this.#options.projectRoot);
      this.#skeletalConfig = {
        raw: selected.raw,
        sha256: selected.sha256,
        modelDrivenBoneIds: selected.modelDrivenBoneIds,
        modelDrivenCount: selected.modelDrivenCount,
        characterId: selected.characterId,
        rigId: selected.rigId,
        source: selected.source,
      };
      info("skeleton", "selected character rig loaded", {
        characterId: selected.characterId,
        rigId: selected.rigId,
        modelDrivenCount: selected.modelDrivenCount,
        source: selected.source,
      });
    } catch (error) {
      const message = error instanceof Error ? error.message : "unknown";
      warn("skeleton", `Rig validation failed: ${message}; 3D skeletal motion disabled`);
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
