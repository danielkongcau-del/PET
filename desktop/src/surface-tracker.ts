/**
 * Windows-only surface tracker derived from OpenPets
 * apps/desktop/src/window-tracker.ts at
 * 806659a46ef5b6833a3fbf69d21801af736acbff.
 *
 * Changes for PET: expose every eligible top edge (not just a terminal), clip
 * covered edge intervals, emit physical-pixel protocol records, and classify
 * full-screen/system-shell suppression.
 */
import electron from "electron";

import { displayIdForPhysicalRect, displayToProtocol } from "./coordinates.js";
import { intersectionArea, rectApproximatelyEquals, rectCovers, subtractIntervals, type Interval } from "./geometry.js";
import { debug, logError } from "./logger.js";
import type { DisplayState, Rect, SceneState, SurfaceState, WindowState } from "./protocol.js";
import { createLatchedTick } from "./window-tracker-latch.js";

const { screen } = electron;

export interface RawWindow {
  readonly id: number;
  readonly ownerPid: number;
  readonly ownerName: string;
  readonly ownerPath: string;
  readonly title: string;
  readonly bounds: Rect;
}

export interface SurfaceSnapshot {
  readonly displays: readonly DisplayState[];
  readonly windows: readonly WindowState[];
  readonly surfaces: readonly SurfaceState[];
  readonly scene: SceneState;
  readonly capturedAtMs: number;
}

interface PreviousWindowPosition {
  readonly x: number;
  readonly y: number;
  readonly capturedAtMs: number;
}

export interface SurfaceTrackerOptions {
  readonly ownProcessId: number;
  readonly minimumSurfaceWidthDip?: number;
  readonly onSnapshot: (snapshot: SurfaceSnapshot) => void;
}

interface WindowEnumeration {
  readonly windows: readonly RawWindow[];
  readonly foreground: RawWindow | null;
}

const SYSTEM_UI_EXECUTABLES = [
  "startmenuexperiencehost",
  "shellexperiencehost",
  "searchhost",
  "lockapp",
  "logonui",
  "credentialuibroker",
  "securityhealthsystray",
];

// get-windows exposes the executable FileDescription as owner.name when one
// exists.  These are normalized descriptions observed on Windows 11 and do
// not necessarily contain the executable basename (for example Start and
// Credential UI).
const SYSTEM_UI_DESCRIPTIONS = [
  "windowsstartexperiencehost",
  "windowsshellexperiencehost",
  "windowslogonuserinterfacehost",
  "credentialmanageruihost",
  "windowssecuritynotificationicon",
];

const NEVER_SURFACE_OWNERS = [
  "applicationframehost",
  "textinputhost",
  "pet-electron-host",
];

// Carrier geometry is sampled faster than generator world-state publication.
// The local host can therefore keep a perched pet attached during an
// interactive drag without churning generator plans at the same rate.
export const SURFACE_POLL_INTERVAL_MS = 33;
const MINIMUM_WINDOW_HEIGHT_DIP = 48;

export class SurfaceTracker {
  readonly #options: SurfaceTrackerOptions;
  #timer: NodeJS.Timeout | null = null;
  #tick: (() => void) | null = null;
  #previous = new Map<number, PreviousWindowPosition>();
  #lastSnapshot: SurfaceSnapshot | null = null;

  constructor(options: SurfaceTrackerOptions) {
    this.#options = options;
  }

  get snapshot(): SurfaceSnapshot | null {
    return this.#lastSnapshot;
  }

  start(): void {
    if (this.#timer) return;
    this.#tick = createLatchedTick(() => this.refreshNow());
    this.#timer = setInterval(this.#tick, SURFACE_POLL_INTERVAL_MS);
    this.#timer.unref?.();
    screen.on("display-added", this.#handleTopologyChange);
    screen.on("display-removed", this.#handleTopologyChange);
    screen.on("display-metrics-changed", this.#handleTopologyChange);
    this.#tick();
  }

  dispose(): void {
    if (this.#timer) clearInterval(this.#timer);
    this.#timer = null;
    this.#tick = null;
    screen.off("display-added", this.#handleTopologyChange);
    screen.off("display-removed", this.#handleTopologyChange);
    screen.off("display-metrics-changed", this.#handleTopologyChange);
  }

  async refreshNow(): Promise<void> {
    try {
      const primaryId = screen.getPrimaryDisplay().id;
      const displays = screen.getAllDisplays().map((display) => displayToProtocol(display, primaryId));
      const enumeration = await enumerateWindows();
      const rawWindows = enumeration.windows;
      const capturedAtMs = Date.now();
      const snapshot = buildSurfaceSnapshot(
        rawWindows,
        displays,
        this.#options.ownProcessId,
        capturedAtMs,
        this.#previous,
        this.#options.minimumSurfaceWidthDip ?? 96,
        enumeration.foreground,
      );
      this.#previous = new Map(rawWindows.map((window) => [window.id, {
        x: window.bounds.x,
        y: window.bounds.y,
        capturedAtMs,
      }]));
      this.#lastSnapshot = snapshot;
      this.#options.onSnapshot(snapshot);
      debug("surface", "window surfaces refreshed", {
        windows: snapshot.windows.length,
        surfaces: snapshot.surfaces.length,
        pet_allowed: snapshot.scene.pet_allowed,
      });
    } catch (error) {
      logError("surface", "window enumeration failed", error);
    }
  }

  readonly #handleTopologyChange = (): void => {
    this.#tick?.();
  };
}

export function buildSurfaceSnapshot(
  rawWindows: readonly RawWindow[],
  displays: readonly DisplayState[],
  ownProcessId: number,
  capturedAtMs: number,
  previous: ReadonlyMap<number, PreviousWindowPosition> = new Map(),
  minimumSurfaceWidthDip = 96,
  foregroundWindow: RawWindow | null = null,
): SurfaceSnapshot {
  const external = rawWindows.filter((window) => window.ownerPid !== ownProcessId && !isMinimizedParkingBounds(window.bounds));
  const foreground = foregroundWindow && foregroundWindow.ownerPid !== ownProcessId &&
    !isMinimizedParkingBounds(foregroundWindow.bounds) ? foregroundWindow : null;
  const systemUiActive = foreground ? isSystemUiWindow(foreground) : false;

  const windows: WindowState[] = external.map((window, zOrder) => {
    const displayId = displayIdForPhysicalRect(window.bounds, displays);
    const display = displays.find((entry) => entry.id === displayId) ?? displays[0];
    const edgeTolerance = display ? Math.ceil(12 * display.scale_factor) : 12;
    const fullscreen = display ? rectCovers(window.bounds, display.bounds, 0.985, edgeTolerance) : false;
    const maximized = display ? rectApproximatelyEquals(window.bounds, display.work_area, Math.ceil(4 * display.scale_factor)) : false;
    const area = Math.max(1, window.bounds.width * window.bounds.height);
    const occluded = external.slice(0, zOrder).some((cover) => intersectionArea(window.bounds, cover.bounds) / area >= 0.9);
    const minimumSurfaceWidthPhysical = display ? Math.ceil(minimumSurfaceWidthDip * display.scale_factor) : minimumSurfaceWidthDip;
    const minimumWindowHeightPhysical = display ? Math.ceil(MINIMUM_WINDOW_HEIGHT_DIP * display.scale_factor) : MINIMUM_WINDOW_HEIGHT_DIP;
    const eligible = Boolean(display) && !fullscreen && !isNeverSurfaceWindow(window) &&
      window.bounds.width >= minimumSurfaceWidthPhysical && window.bounds.height >= minimumWindowHeightPhysical &&
      intersectionArea(window.bounds, display!.work_area) > 0;
    return {
      id: `window-${window.id}`,
      display_id: displayId,
      bounds: window.bounds,
      z_order: zOrder,
      visible: true,
      minimized: false,
      maximized,
      fullscreen,
      active: foreground?.id === window.id,
      occluded,
      eligible,
    };
  });

  const foregroundDisplayId = foreground ? displayIdForPhysicalRect(foreground.bounds, displays) : null;
  const foregroundDisplay = foregroundDisplayId ? displays.find((display) => display.id === foregroundDisplayId) : undefined;
  const fullscreenActive = Boolean(foreground && foregroundDisplay && rectCovers(
    foreground.bounds,
    foregroundDisplay.bounds,
    0.985,
    Math.ceil(12 * foregroundDisplay.scale_factor),
  ));
  const surfaces: SurfaceState[] = [];
  for (const display of displays) {
    surfaces.push({
      id: `${display.id}:floor`,
      kind: "work_area_floor",
      display_id: display.id,
      x1: display.work_area.x,
      x2: display.work_area.x + display.work_area.width,
      y: display.work_area.y + display.work_area.height,
      enabled: true,
      occluded: false,
      vx: 0,
      vy: 0,
    });
  }

  for (let index = 0; index < external.length; index += 1) {
    const raw = external[index];
    const state = windows[index];
    if (!raw || !state || !state.eligible) continue;
    const display = displays.find((entry) => entry.id === state.display_id);
    if (!display) continue;
    const minimumSurfaceWidthPhysical = Math.ceil(minimumSurfaceWidthDip * display.scale_factor);
    const minimumWindowHeightPhysical = Math.ceil(MINIMUM_WINDOW_HEIGHT_DIP * display.scale_factor);
    const x1 = Math.max(raw.bounds.x, display.work_area.x);
    const x2 = Math.min(raw.bounds.x + raw.bounds.width, display.work_area.x + display.work_area.width);
    const y = raw.bounds.y;
    if (x2 - x1 < minimumSurfaceWidthPhysical || y < display.work_area.y + minimumWindowHeightPhysical || y > display.work_area.y + display.work_area.height) continue;

    const blockers: Interval[] = [];
    for (const cover of external.slice(0, index)) {
      if (cover.bounds.y <= y + 2 && cover.bounds.y + cover.bounds.height >= y - 2) {
        blockers.push({ x1: cover.bounds.x, x2: cover.bounds.x + cover.bounds.width });
      }
    }
    const segments = subtractIntervals({ x1, x2 }, blockers, minimumSurfaceWidthPhysical);
    const before = previous.get(raw.id);
    const elapsedSeconds = before ? Math.max(0.001, (capturedAtMs - before.capturedAtMs) / 1_000) : 0;
    const vx = before ? (raw.bounds.x - before.x) / elapsedSeconds : 0;
    const vy = before ? (raw.bounds.y - before.y) / elapsedSeconds : 0;
    segments.forEach((segment, segmentIndex) => {
      surfaces.push({
        id: `window-${raw.id}:top:${segmentIndex}`,
        kind: "window_top",
        display_id: state.display_id,
        window_id: state.id,
        x1: segment.x1,
        x2: segment.x2,
        y,
        enabled: !state.occluded,
        occluded: state.occluded,
        vx,
        vy,
      });
    });
  }

  let scene: SceneState;
  if (systemUiActive) scene = { fullscreen_active: false, pet_allowed: false, suspend_reason: "system_ui" };
  else if (fullscreenActive) scene = { fullscreen_active: true, pet_allowed: false, suspend_reason: "fullscreen" };
  else scene = { fullscreen_active: false, pet_allowed: true };

  return { displays, windows, surfaces, scene, capturedAtMs };
}

async function enumerateWindows(): Promise<WindowEnumeration> {
  const module = await import("get-windows") as {
    openWindows: () => Promise<unknown[]>;
    activeWindow: () => Promise<unknown>;
  };
  const [rawWindows, rawForeground] = await Promise.all([module.openWindows(), module.activeWindow()]);
  const windows = rawWindows.flatMap((entry): RawWindow[] => {
    const parsed = toRawWindow(entry);
    return parsed ? [parsed] : [];
  });
  // openWindows is documented front-to-back, so it remains a degraded
  // fallback if foreground lookup fails. Normal suppression uses the explicit
  // foreground HWND returned by activeWindow.
  return { windows, foreground: toRawWindow(rawForeground) ?? windows[0] ?? null };
}

function toRawWindow(value: unknown): RawWindow | null {
  if (!isGetWindowsEntry(value)) return null;
  return {
    id: value.id,
    ownerPid: value.owner.processId,
    ownerName: value.owner.name,
    ownerPath: value.owner.path,
    title: value.title,
    bounds: {
      x: value.bounds.x,
      y: value.bounds.y,
      width: value.bounds.width,
      height: value.bounds.height,
    },
  };
}

function isGetWindowsEntry(value: unknown): value is {
  id: number;
  title: string;
  owner: { processId: number; name: string; path: string };
  bounds: Rect;
} {
  if (typeof value !== "object" || value === null) return false;
  const record = value as Record<string, unknown>;
  if (typeof record.id !== "number" || typeof record.title !== "string") return false;
  if (typeof record.owner !== "object" || record.owner === null || typeof record.bounds !== "object" || record.bounds === null) return false;
  const owner = record.owner as Record<string, unknown>;
  const bounds = record.bounds as Record<string, unknown>;
  return typeof owner.processId === "number" && typeof owner.name === "string" && typeof owner.path === "string" &&
    typeof bounds.x === "number" && typeof bounds.y === "number" &&
    typeof bounds.width === "number" && typeof bounds.height === "number";
}

function isMinimizedParkingBounds(bounds: Rect): boolean {
  return bounds.x <= -30_000 || bounds.y <= -30_000 || bounds.width <= 0 || bounds.height <= 0;
}

export function isSystemUiWindow(window: Pick<RawWindow, "ownerName" | "ownerPath">): boolean {
  const normalizedName = normalizeOwnerIdentity(window.ownerName);
  const executable = window.ownerPath.split(/[\\/]/).pop() ?? "";
  const normalizedExecutable = normalizeOwnerIdentity(executable).replace(/exe$/, "");
  return SYSTEM_UI_EXECUTABLES.some((name) => normalizedExecutable === name) ||
    SYSTEM_UI_DESCRIPTIONS.some((description) => normalizedName.includes(description)) ||
    SYSTEM_UI_EXECUTABLES.some((name) => normalizedName.includes(name));
}

function isNeverSurfaceWindow(window: RawWindow): boolean {
  const owner = normalizeOwnerIdentity(window.ownerName);
  if (isSystemUiWindow(window)) return true;
  if (NEVER_SURFACE_OWNERS.some((name) => owner.includes(name))) return true;
  // Explorer owns both real folder windows and invisible desktop/taskbar
  // surfaces. A title is used only for this local filter and never leaves the
  // tracker or enters logs/protocol messages.
  if (owner.includes("explorer") && window.title.trim() === "") return true;
  return false;
}

function normalizeOwnerIdentity(value: string): string {
  return value.toLowerCase().replace(/[^a-z0-9]/g, "");
}
