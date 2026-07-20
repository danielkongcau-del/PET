import { app, BrowserWindow, screen, type Display } from "electron";
import { join } from "node:path";

import type { MotionDebugState } from "./motion-controller.js";
import { displayToProtocol } from "./coordinates.js";
import type { MetricsPayload } from "./protocol.js";

export interface DebugRecordingState {
  readonly active: boolean;
  readonly elapsedMs: number;
  readonly bytesWritten: number;
}

export class DebugOverlay {
  #windows = new Map<number, BrowserWindow>();
  #enabled = process.env.PET_DEBUG_OVERLAY === "1";
  #sceneAllowed = true;
  #lastState: MotionDebugState | null = null;
  #generatorMetrics: MetricsPayload | null = null;
  #generatorRestarts = 0;
  #recording: DebugRecordingState = { active: false, elapsedMs: 0, bytesWritten: 0 };

  async initialize(): Promise<void> {
    screen.on("display-added", this.#handleTopology);
    screen.on("display-removed", this.#handleTopology);
    screen.on("display-metrics-changed", this.#handleTopology);
    await this.#rebuild();
  }

  get enabled(): boolean {
    return this.#enabled;
  }

  setEnabled(enabled: boolean): void {
    this.#enabled = enabled;
    this.#syncVisibility();
    if (enabled && this.#lastState) this.update(this.#lastState);
  }

  setSceneAllowed(allowed: boolean): void {
    this.#sceneAllowed = allowed;
    this.#syncVisibility();
  }

  setGeneratorMetrics(metrics: MetricsPayload): void {
    this.#generatorMetrics = metrics;
  }

  setGeneratorRestarts(count: number): void {
    this.#generatorRestarts = Math.max(0, Math.trunc(count));
  }

  setRecording(recording: DebugRecordingState): void {
    this.#recording = recording;
    if (this.#lastState) this.update(this.#lastState);
  }

  update(state: MotionDebugState): void {
    this.#lastState = state;
    if (!this.#enabled || !this.#sceneAllowed) return;
    const primaryId = screen.getPrimaryDisplay().id;
    for (const display of screen.getAllDisplays()) {
      const window = this.#windows.get(display.id);
      if (!window || window.isDestroyed() || window.webContents.isDestroyed()) continue;
      const protocolDisplay = displayToProtocol(display, primaryId);
      const surfaces = state.surfaces.filter((surface) => surface.display_id === protocolDisplay.id);
      const plan = state.plan ? {
        ...state.plan,
        points: state.plan.points.filter((point) =>
          point.x >= protocolDisplay.bounds.x && point.x <= protocolDisplay.bounds.x + protocolDisplay.bounds.width &&
          point.y >= protocolDisplay.bounds.y && point.y <= protocolDisplay.bounds.y + protocolDisplay.bounds.height),
      } : null;
      window.webContents.send("pet:debug-state", {
        ...state,
        display: protocolDisplay,
        surfaces,
        plan,
        generatorMetrics: this.#generatorMetrics,
        generatorRestarts: this.#generatorRestarts,
        recording: this.#recording,
      });
    }
  }

  dispose(): void {
    screen.off("display-added", this.#handleTopology);
    screen.off("display-removed", this.#handleTopology);
    screen.off("display-metrics-changed", this.#handleTopology);
    for (const window of this.#windows.values()) if (!window.isDestroyed()) window.destroy();
    this.#windows.clear();
  }

  readonly #handleTopology = (): void => {
    void this.#rebuild();
  };

  async #rebuild(): Promise<void> {
    const displays = screen.getAllDisplays();
    const liveIds = new Set(displays.map((display) => display.id));
    for (const [id, window] of this.#windows) {
      if (liveIds.has(id)) continue;
      if (!window.isDestroyed()) window.destroy();
      this.#windows.delete(id);
    }
    await Promise.all(displays.map((display) => this.#ensureWindow(display)));
    this.#syncVisibility();
    if (this.#lastState) this.update(this.#lastState);
  }

  async #ensureWindow(display: Display): Promise<void> {
    const existing = this.#windows.get(display.id);
    if (existing && !existing.isDestroyed()) {
      existing.setBounds(display.bounds, false);
      return;
    }
    const window = new BrowserWindow({
      title: "PET Debug Overlay",
      ...display.bounds,
      frame: false,
      transparent: true,
      resizable: false,
      movable: false,
      focusable: false,
      skipTaskbar: true,
      alwaysOnTop: true,
      show: false,
      hasShadow: false,
      backgroundColor: "#00000000",
      webPreferences: {
        nodeIntegration: false,
        contextIsolation: true,
        sandbox: true,
        webSecurity: true,
        preload: join(app.getAppPath(), "src", "debug-preload.cjs"),
      },
    });
    window.setMenu(null);
    window.setIgnoreMouseEvents(true);
    window.setAlwaysOnTop(true, "floating");
    window.webContents.setWindowOpenHandler(() => ({ action: "deny" }));
    window.webContents.on("will-navigate", (event) => event.preventDefault());
    this.#windows.set(display.id, window);
    await window.loadFile(join(app.getAppPath(), "src", "debug", "index.html"));
  }

  #syncVisibility(): void {
    const show = this.#enabled && this.#sceneAllowed;
    for (const window of this.#windows.values()) {
      if (window.isDestroyed()) continue;
      if (show) window.showInactive();
      else window.hide();
    }
  }
}
