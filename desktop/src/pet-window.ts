/**
 * Reduced pet-window host derived from OpenPets apps/desktop/src/pet-window.ts
 * at 806659a46ef5b6833a3fbf69d21801af736acbff. The always-on-top recovery,
 * forwarded mouse passthrough and cursor-probe watchdog are direct adaptations.
 */
import electron, { type BrowserWindow as ElectronBrowserWindow, type IpcMainEvent, type Point } from "electron";
import { readFile } from "node:fs/promises";
import { join } from "node:path";

import { coordinateFrameForDisplay, lockedDipBoundsForPhysicalFoot, nearestDisplayForPhysicalPoint } from "./coordinates.js";
import type { CharacterRigConfig } from "./character-rig.js";
import type { Behavior, FacialParams, Quat, Vec3 } from "./protocol.js";
import type { GeneratorStatus } from "./generator-bridge.js";
import { debug, logError } from "./logger.js";

const { app, BrowserWindow, ipcMain, screen } = electron;

export const PET_WINDOW_DIP = 96;
const PET_CANVAS_PIXELS = 48;
export const PET_FOOT_ANCHOR_DIP: Readonly<Point> = Object.freeze({
  x: 24 * PET_WINDOW_DIP / PET_CANVAS_PIXELS,
  y: 46 * PET_WINDOW_DIP / PET_CANVAS_PIXELS,
});
const HIT_TEST_PROBE_MS = 50;
const PASSTHROUGH_REARM_MS = 1_000;
const RENDERER_RECOVERY_BASE_MS = 250;
const RENDERER_RECOVERY_MAX_MS = 8_000;
const RENDERER_STABLE_RESET_MS = 15_000;

export function rendererRecoveryDelayMs(consecutiveFailures: number): number {
  const exponent = Math.max(0, Math.min(10, Math.floor(consecutiveFailures) - 1));
  return Math.min(RENDERER_RECOVERY_MAX_MS, RENDERER_RECOVERY_BASE_MS * 2 ** exponent);
}

export interface PetVisualState {
  readonly facing: -1 | 1;
  readonly lean: number;
  readonly squash: number;
  readonly bob: number;
  readonly expression: string;
  readonly behavior: Behavior;
  readonly generatorStatus: GeneratorStatus;
  readonly debug: boolean;
  /** Per-bone rotation angles (radians). Present only in negotiated legacy full_2d mode. */
  readonly boneRotations?: readonly number[];
  /** Continuous facial parameters. Takes precedence over expression string. */
  readonly facialParams?: FacialParams;
  /** Root translation in plan-frame coordinates (3D mode). */
  readonly rootTranslation?: Vec3;
  /** Root orientation as normalized quaternion (3D mode). */
  readonly rootRotation?: Quat;
  /** Per-joint local rotation deltas (3D mode). */
  readonly localRotationDeltas?: readonly Quat[];
}

export interface PetWindowOptions {
  readonly initialFootDip: Point;
  readonly projectRoot: string;
  readonly characterRig: CharacterRigConfig;
  readonly onClick: (button: "left" | "right" | "middle") => void;
  readonly onPointerOpaque: (opaque: boolean) => void;
}

export class PetWindow {
  readonly #window: ElectronBrowserWindow;
  readonly #options: PetWindowOptions;
  #assetDataUrl: string | null = null;
  #assetParts: unknown = null;
  #footAnchorDip: Point = { ...PET_FOOT_ANCHOR_DIP };
  #lastInteractive = false;
  #lastPassthroughRearmMs = 0;
  #rendererReady = false;
  #probeTimer: NodeJS.Timeout | null = null;
  #topmostTimer: NodeJS.Timeout | null = null;
  #rendererRecoveryTimer: NodeJS.Timeout | null = null;
  #rendererStableTimer: NodeJS.Timeout | null = null;
  #rendererFailureCount = 0;
  #sceneAllowed = true;
  #disposed = false;
  #lastSkeletalConfig: Record<string, unknown> | null = null;

  constructor(options: PetWindowOptions) {
    this.#options = options;
    this.#window = new BrowserWindow({
      title: "PET",
      width: PET_WINDOW_DIP,
      height: PET_WINDOW_DIP,
      x: Math.round(options.initialFootDip.x - PET_FOOT_ANCHOR_DIP.x),
      y: Math.round(options.initialFootDip.y - PET_FOOT_ANCHOR_DIP.y),
      frame: false,
      transparent: true,
      resizable: false,
      maximizable: false,
      minimizable: false,
      fullscreenable: false,
      skipTaskbar: true,
      alwaysOnTop: true,
      focusable: false,
      show: false,
      hasShadow: false,
      backgroundColor: "#00000000",
      webPreferences: {
        nodeIntegration: false,
        contextIsolation: true,
        sandbox: true,
        webSecurity: true,
        preload: join(app.getAppPath(), "src", "preload.cjs"),
      },
    });
    this.#window.setMenu(null);
    this.#applyAlwaysOnTop();
    this.#window.webContents.setWindowOpenHandler(() => ({ action: "deny" }));
    this.#window.webContents.on("will-navigate", (event) => event.preventDefault());
    this.#window.webContents.on("will-redirect", (event) => event.preventDefault());
    this.#window.webContents.on("render-process-gone", (_event, details) => this.#handleRendererGone(details.reason));

    ipcMain.on("pet:ready", this.#handleReady);
    ipcMain.on("pet:hit-test", this.#handleHitTest);
    ipcMain.on("pet:click", this.#handleClick);

    this.#topmostTimer = setInterval(() => {
      if (this.#sceneAllowed && this.#window.isVisible()) this.#applyAlwaysOnTop();
    }, 1_000);
    this.#topmostTimer.unref?.();
    this.#probeTimer = setInterval(() => this.#probeCursor(), HIT_TEST_PROBE_MS);
    this.#probeTimer.unref?.();
  }

  async load(): Promise<void> {
    const asset = await readPetAsset(this.#options.characterRig);
    const currentFoot = this.getFootDip();
    this.#assetDataUrl = asset.dataUrl;
    this.#assetParts = asset.parts;
    this.#footAnchorDip = { ...asset.footAnchorDip };
    this.setFootDip(currentFoot);
    await this.#window.loadFile(join(app.getAppPath(), "src", "renderer", "index.html"));
    if (this.#sceneAllowed) this.#window.showInactive();
  }

  get browserWindow(): ElectronBrowserWindow {
    return this.#window;
  }

  get visible(): boolean {
    return !this.#window.isDestroyed() && this.#window.isVisible();
  }

  get footAnchorDip(): Readonly<Point> {
    return this.#footAnchorDip;
  }

  getFootDip(): Point {
    const bounds = this.#window.getBounds();
    return { x: bounds.x + this.#footAnchorDip.x, y: bounds.y + this.#footAnchorDip.y };
  }

  setFootDip(foot: Point): void {
    if (this.#window.isDestroyed()) return;
    const x = Math.round(foot.x - this.#footAnchorDip.x);
    const y = Math.round(foot.y - this.#footAnchorDip.y);
    const [currentX, currentY] = this.#window.getPosition();
    if (x !== currentX || y !== currentY) this.#window.setPosition(x, y, false);
  }

  setFootPhysical(foot: Point, targetDisplayId?: string): void {
    if (this.#window.isDestroyed()) return;
    const displays = screen.getAllDisplays();
    const targetDisplay = displays.find((display) => `display-${display.id}` === targetDisplayId) ??
      nearestDisplayForPhysicalPoint(foot);
    const bounds = lockedDipBoundsForPhysicalFoot(
      foot,
      this.#footAnchorDip,
      { x: PET_WINDOW_DIP, y: PET_WINDOW_DIP },
      coordinateFrameForDisplay(targetDisplay),
    );
    const current = this.#window.getBounds();
    if (bounds.x !== current.x || bounds.y !== current.y || bounds.width !== current.width || bounds.height !== current.height) {
      this.#window.setBounds(bounds, false);
    }
  }

  setSceneAllowed(allowed: boolean): void {
    if (this.#sceneAllowed === allowed || this.#window.isDestroyed()) return;
    this.#sceneAllowed = allowed;
    if (allowed) {
      this.#applyAlwaysOnTop();
      this.#window.showInactive();
      this.#rearmPassthrough("scene-restored");
    } else {
      this.#window.hide();
    }
  }

  sendVisualState(state: PetVisualState): void {
    if (!this.#rendererReady || this.#window.isDestroyed() || this.#window.webContents.isDestroyed()) return;
    this.#window.webContents.send("pet:render-state", {
      ...state,
      ...(this.#assetDataUrl ? { assetDataUrl: this.#assetDataUrl } : {}),
      ...(this.#assetParts ? { assetParts: this.#assetParts } : {}),
    });
  }

  /** Send the skeleton definition to the renderer for FK rendering. Safe to call multiple times. */
  sendSkeletalConfig(config: Record<string, unknown> | null): void {
    this.#lastSkeletalConfig = config;
    if (!this.#rendererReady || this.#window.isDestroyed() || this.#window.webContents.isDestroyed()) return;
    this.#window.webContents.send("pet:skeletal-config", config);
  }

  dispose(): void {
    if (this.#disposed) return;
    this.#disposed = true;
    if (this.#probeTimer) clearInterval(this.#probeTimer);
    if (this.#topmostTimer) clearInterval(this.#topmostTimer);
    if (this.#rendererRecoveryTimer) clearTimeout(this.#rendererRecoveryTimer);
    if (this.#rendererStableTimer) clearTimeout(this.#rendererStableTimer);
    ipcMain.off("pet:ready", this.#handleReady);
    ipcMain.off("pet:hit-test", this.#handleHitTest);
    ipcMain.off("pet:click", this.#handleClick);
    if (!this.#window.isDestroyed()) this.#window.destroy();
  }

  readonly #handleReady = (event: IpcMainEvent): void => {
    if (event.sender !== this.#window.webContents) return;
    if (this.#rendererRecoveryTimer) clearTimeout(this.#rendererRecoveryTimer);
    this.#rendererRecoveryTimer = null;
    this.#rendererReady = true;
    this.#lastInteractive = false;
    this.#options.onPointerOpaque(false);
    this.#setPassthrough(true);
    if (this.#sceneAllowed && !this.#window.isVisible()) this.#window.showInactive();
    if (this.#rendererStableTimer) clearTimeout(this.#rendererStableTimer);
    this.#rendererStableTimer = setTimeout(() => {
      this.#rendererStableTimer = null;
      this.#rendererFailureCount = 0;
    }, RENDERER_STABLE_RESET_MS);
    this.#rendererStableTimer.unref?.();
    // Re-send skeletal config after renderer reload to prevent stale null state.
    if (this.#lastSkeletalConfig) this.sendSkeletalConfig(this.#lastSkeletalConfig);
    debug("pet.window", "renderer ready", { window_id: this.#window.id });
  };

  readonly #handleHitTest = (event: IpcMainEvent, opaque: unknown): void => {
    if (event.sender !== this.#window.webContents || typeof opaque !== "boolean") return;
    if (this.#lastInteractive === opaque) return;
    this.#lastInteractive = opaque;
    this.#options.onPointerOpaque(opaque);
    this.#setPassthrough(!opaque);
  };

  readonly #handleClick = (event: IpcMainEvent, button: unknown): void => {
    if (event.sender !== this.#window.webContents || (button !== "left" && button !== "right" && button !== "middle")) return;
    this.#options.onClick(button);
  };

  #handleRendererGone(reason: string): void {
    if (this.#disposed || this.#window.isDestroyed()) return;
    logError("pet.window", "renderer process gone", new Error(reason));
    this.#rendererReady = false;
    if (this.#rendererStableTimer) clearTimeout(this.#rendererStableTimer);
    this.#rendererStableTimer = null;
    this.#rendererFailureCount += 1;
    this.#lastInteractive = false;
    this.#options.onPointerOpaque(false);
    this.#setPassthrough(true);
    if (this.#window.isVisible()) this.#window.hide();
    this.#scheduleRendererRecovery();
  }

  #scheduleRendererRecovery(): void {
    if (this.#disposed || this.#window.isDestroyed() || this.#rendererRecoveryTimer) return;
    const delay = rendererRecoveryDelayMs(this.#rendererFailureCount);
    this.#rendererRecoveryTimer = setTimeout(() => {
      this.#rendererRecoveryTimer = null;
      if (this.#disposed || this.#window.isDestroyed()) return;
      try {
        this.#window.reload();
      } catch (error) {
        this.#rendererFailureCount += 1;
        logError("pet.window", "renderer reload failed", error);
        this.#scheduleRendererRecovery();
      }
    }, delay);
    this.#rendererRecoveryTimer.unref?.();
    debug("pet.window", "renderer recovery scheduled", { attempt: this.#rendererFailureCount, delay_ms: delay });
  }

  #setPassthrough(passthrough: boolean): void {
    if (this.#window.isDestroyed()) return;
    if (passthrough) this.#window.setIgnoreMouseEvents(true, { forward: true });
    else this.#window.setIgnoreMouseEvents(false);
  }

  #probeCursor(): void {
    if (!this.#rendererReady || this.#window.isDestroyed() || !this.#window.isVisible()) return;
    const cursor = screen.getCursorScreenPoint();
    const bounds = this.#window.getContentBounds();
    const clientX = cursor.x - bounds.x;
    const clientY = cursor.y - bounds.y;
    const inside = clientX >= 0 && clientX < bounds.width && clientY >= 0 && clientY < bounds.height;
    if (!inside) {
      if (this.#lastInteractive) {
        this.#lastInteractive = false;
        this.#options.onPointerOpaque(false);
        this.#setPassthrough(true);
      }
      return;
    }
    // Windows can silently stop forwarding after fullscreen transitions. Drop
    // and restore the ignored state before asking the renderer for an alpha hit.
    if (!this.#lastInteractive && Date.now() - this.#lastPassthroughRearmMs >= PASSTHROUGH_REARM_MS) {
      this.#rearmPassthrough("cursor-probe-watchdog");
    }
    this.#window.webContents.send("pet:probe-hit-test", { clientX, clientY });
  }

  #rearmPassthrough(reason: string): void {
    if (this.#window.isDestroyed() || this.#lastInteractive) return;
    this.#window.setIgnoreMouseEvents(false);
    this.#window.setIgnoreMouseEvents(true, { forward: true });
    this.#lastPassthroughRearmMs = Date.now();
    debug("pet.window", "mouse forwarding rearmed", { reason });
  }

  #applyAlwaysOnTop(): void {
    if (this.#window.isDestroyed()) return;
    // OpenPets found that Windows can strip HWND_TOPMOST while Electron keeps a
    // stale cached true value. Toggle it off before reasserting to force the OS
    // SetWindowPos call.
    if (process.platform === "win32" && this.#window.isAlwaysOnTop()) this.#window.setAlwaysOnTop(false);
    this.#window.setAlwaysOnTop(true, "floating");
  }
}

async function readPetAsset(character: CharacterRigConfig): Promise<{ readonly dataUrl: string | null; readonly parts: unknown; readonly footAnchorDip: Readonly<Point> }> {
  const render = character.render;
  const footAnchorDip = {
    x: render.footAnchor[0] * PET_WINDOW_DIP / render.canvas[0],
    y: render.footAnchor[1] * PET_WINDOW_DIP / render.canvas[1],
  };
  const path = render.spriteImagePath;
  if (!path) {
    debug("pet.window", "selected character has no sprite asset; using skeletal renderer fallback", {
      characterId: character.characterId,
      mode: render.mode,
    });
    return { dataUrl: null, parts: characterSpriteMetadata(character, null), footAnchorDip };
  }
  try {
    const bytes = await readFile(path);
    let parts: unknown = null;
    if (render.spriteMetadataPath) {
      try {
        const metadata = await readFile(render.spriteMetadataPath, "utf8");
        if (metadata.length <= 64 * 1024) parts = JSON.parse(metadata) as unknown;
      } catch {
        // The manifest still supplies canvas, facing and foot anchor metadata.
      }
    }
    return {
      dataUrl: `data:${imageMimeType(path)};base64,${bytes.toString("base64")}`,
      parts: characterSpriteMetadata(character, parts),
      footAnchorDip,
    };
  } catch {
    debug("pet.window", "selected character sprite unavailable; using skeletal renderer fallback", {
      characterId: character.characterId,
    });
    return { dataUrl: null, parts: characterSpriteMetadata(character, null), footAnchorDip };
  }
}

function characterSpriteMetadata(character: CharacterRigConfig, source: unknown): Record<string, unknown> {
  const sourceRecord = typeof source === "object" && source !== null && !Array.isArray(source)
    ? source as Record<string, unknown>
    : {};
  return {
    ...sourceRecord,
    canvas: [...character.render.canvas],
    footAnchor: [...character.render.footAnchor],
    sourceFacing: character.render.sourceFacing,
    characterId: character.characterId,
    rigFingerprint: character.sha256,
  };
}

function imageMimeType(path: string): string {
  const lower = path.toLowerCase();
  if (lower.endsWith(".webp")) return "image/webp";
  if (lower.endsWith(".jpg") || lower.endsWith(".jpeg")) return "image/jpeg";
  return "image/png";
}
