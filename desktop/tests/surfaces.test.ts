import assert from "node:assert/strict";
import test from "node:test";

import { lockedDipBoundsForPhysicalFoot } from "../src/coordinates.js";
import { findCrossedSurface, rectCovers, subtractIntervals } from "../src/geometry.js";
import { rendererRecoveryDelayMs } from "../src/pet-window.js";
import { resolveSurfaceAttachment, SUPPORT_HALF_WIDTH_DIP, surfaceIdForMotionSample } from "../src/surface-attachment.js";
import type { DisplayState, SurfaceState, WindowState } from "../src/protocol.js";
import { buildSurfaceSnapshot, isSystemUiWindow, SURFACE_POLL_INTERVAL_MS, type RawWindow } from "../src/surface-tracker.js";

test("covered portions of a window top are removed without tiny landing slivers", () => {
  assert.deepEqual(
    subtractIntervals({ x1: 0, x2: 1_000 }, [{ x1: 200, x2: 500 }, { x1: 700, x2: 950 }], 96),
    [{ x1: 0, x2: 200 }, { x1: 500, x2: 700 }],
  );
});

test("falling root lands on the highest crossed enabled surface", () => {
  const surfaces: SurfaceState[] = [
    { id: "floor", kind: "work_area_floor", display_id: "d", x1: 0, x2: 1_000, y: 800, enabled: true, occluded: false },
    { id: "window", kind: "window_top", display_id: "d", window_id: "w", x1: 300, x2: 700, y: 500, enabled: true, occluded: false },
  ];
  assert.equal(findCrossedSurface({ x: 400, y: 450 }, { x: 400, y: 850 }, surfaces)?.id, "window");
  assert.equal(findCrossedSurface({ x: 100, y: 450 }, { x: 100, y: 850 }, surfaces)?.id, "floor");
});

test("fullscreen coverage tolerates invisible Win32 frame margins but not the taskbar gap", () => {
  const display = { x: 0, y: 0, width: 1920, height: 1080 };
  assert.equal(rectCovers({ x: -8, y: -8, width: 1936, height: 1096 }, display), true);
  assert.equal(rectCovers({ x: -8, y: -8, width: 1936, height: 1048 }, display), false);
});

const display: DisplayState = {
  id: "display-1",
  bounds: { x: 0, y: 0, width: 1_920, height: 1_080 },
  work_area: { x: 0, y: 0, width: 1_920, height: 1_040 },
  scale_factor: 1,
  is_primary: true,
};

function rawWindow(id: number, ownerName: string, bounds: RawWindow["bounds"]): RawWindow {
  return { id, ownerPid: id + 1_000, ownerName, ownerPath: `C:\\Apps\\${ownerName}`, title: `window-${id}`, bounds };
}

test("explicit foreground drives fullscreen suppression instead of array position", () => {
  const fullscreen = rawWindow(1, "game.exe", { x: 0, y: 0, width: 1_920, height: 1_080 });
  const editor = rawWindow(2, "editor.exe", { x: 200, y: 120, width: 1_000, height: 700 });

  const backgroundFullscreen = buildSurfaceSnapshot([fullscreen, editor], [display], 999, 1_000, new Map(), 96, editor);
  assert.equal(backgroundFullscreen.windows.find((window) => window.id === "window-1")?.fullscreen, true);
  assert.equal(backgroundFullscreen.windows.find((window) => window.id === "window-1")?.active, false);
  assert.equal(backgroundFullscreen.windows.find((window) => window.id === "window-2")?.active, true);
  assert.deepEqual(backgroundFullscreen.scene, { fullscreen_active: false, pet_allowed: true });

  const foregroundFullscreen = buildSurfaceSnapshot([editor, fullscreen], [display], 999, 1_000, new Map(), 96, fullscreen);
  assert.deepEqual(foregroundFullscreen.scene, {
    fullscreen_active: true,
    pet_allowed: false,
    suspend_reason: "fullscreen",
  });
});

test("only foreground system UI suppresses the pet", () => {
  const editor = rawWindow(3, "editor.exe", { x: 200, y: 120, width: 1_000, height: 700 });
  const startMenu = rawWindow(4, "StartMenuExperienceHost.exe", { x: 500, y: 200, width: 700, height: 700 });

  const backgroundShell = buildSurfaceSnapshot([startMenu, editor], [display], 999, 1_000, new Map(), 96, editor);
  assert.equal(backgroundShell.scene.pet_allowed, true);

  const foregroundShell = buildSurfaceSnapshot([editor], [display], 999, 1_000, new Map(), 96, startMenu);
  assert.deepEqual(foregroundShell.scene, {
    fullscreen_active: false,
    pet_allowed: false,
    suspend_reason: "system_ui",
  });
});

test("Windows 11 executable paths and FileDescription owner names identify system UI", () => {
  assert.equal(isSystemUiWindow({
    ownerName: "Windows Start Experience Host",
    ownerPath: "C:\\Windows\\SystemApps\\StartMenuExperienceHost.exe",
  }), true);
  assert.equal(isSystemUiWindow({
    ownerName: "Windows Logon User Interface Host",
    ownerPath: "C:\\Windows\\System32\\renamed-host.exe",
  }), true);
  assert.equal(isSystemUiWindow({
    ownerName: "Credential Manager UI Host",
    ownerPath: "C:\\Windows\\System32\\CredentialUIBroker.exe",
  }), true);
  assert.equal(isSystemUiWindow({ ownerName: "Google Chrome", ownerPath: "C:\\Apps\\chrome.exe" }), false);
});

test("mixed-DPI foot placement keeps the target display transform for the whole window", () => {
  const bounds = lockedDipBoundsForPhysicalFoot(
    { x: 1_940, y: 900 },
    { x: 48, y: 92 },
    { x: 96, y: 96 },
    {
      dipBounds: { x: 1_920, y: 0, width: 1_707, height: 960 },
      physicalBounds: { x: 1_920, y: 0, width: 2_560, height: 1_440 },
      scaleFactor: 1.5,
    },
  );
  assert.deepEqual(bounds, { x: 1_885, y: 508, width: 96, height: 96 });
  const roundTripFoot = {
    x: 1_920 + (bounds.x + 48 - 1_920) * 1.5,
    y: (bounds.y + 92) * 1.5,
  };
  assert.ok(Math.abs(roundTripFoot.x - 1_940) <= 0.75);
  assert.ok(Math.abs(roundTripFoot.y - 900) <= 0.75);
});

test("renderer recovery is rate-limited with a bounded exponential delay", () => {
  assert.equal(rendererRecoveryDelayMs(1), 250);
  assert.equal(rendererRecoveryDelayMs(2), 500);
  assert.equal(rendererRecoveryDelayMs(6), 8_000);
  assert.equal(rendererRecoveryDelayMs(100), 8_000);
});

test("surface width threshold scales independently for each display", () => {
  const highDpiDisplay: DisplayState = {
    id: "display-2",
    bounds: { x: 1_920, y: 0, width: 2_560, height: 1_440 },
    work_area: { x: 1_920, y: 0, width: 2_560, height: 1_400 },
    scale_factor: 2,
    is_primary: false,
  };
  const lowDpiWindow = rawWindow(10, "low-dpi.exe", { x: 100, y: 100, width: 150, height: 300 });
  const tooNarrowHighDpiWindow = rawWindow(11, "high-dpi-narrow.exe", { x: 2_100, y: 200, width: 150, height: 400 });
  const wideHighDpiWindow = rawWindow(12, "high-dpi-wide.exe", { x: 2_400, y: 300, width: 220, height: 400 });
  const snapshot = buildSurfaceSnapshot(
    [lowDpiWindow, tooNarrowHighDpiWindow, wideHighDpiWindow],
    [display, highDpiDisplay],
    999,
    1_000,
    new Map(),
    96,
    lowDpiWindow,
  );

  assert.equal(snapshot.windows.find((window) => window.id === "window-10")?.eligible, true);
  assert.equal(snapshot.windows.find((window) => window.id === "window-11")?.eligible, false);
  assert.equal(snapshot.windows.find((window) => window.id === "window-12")?.eligible, true);
  assert.ok(snapshot.surfaces.some((surface) => surface.window_id === "window-10"));
  assert.ok(!snapshot.surfaces.some((surface) => surface.window_id === "window-11"));
  assert.ok(snapshot.surfaces.some((surface) => surface.window_id === "window-12"));
});

test("surface polling meets the replanning freshness budget", () => {
  assert.ok(SURFACE_POLL_INTERVAL_MS <= 50);
});

test("a pet follows its carrier window instead of losing support", () => {
  const previousSurface: SurfaceState = {
    id: "window-7:top:0", kind: "window_top", display_id: display.id, window_id: "window-7",
    x1: 100, x2: 700, y: 200, enabled: true, occluded: false,
  };
  const nextSurface: SurfaceState = {
    ...previousSurface, x1: 180, x2: 780, y: 260,
  };
  const previousWindow = windowState("window-7", { x: 100, y: 200, width: 600, height: 400 });
  const nextWindow = windowState("window-7", { x: 180, y: 260, width: 600, height: 400 });
  const attachment = resolveSurfaceAttachment({
    foot: { x: 300, y: 200 },
    surfaceId: previousSurface.id,
    previousSurfaces: [previousSurface],
    nextSurfaces: [nextSurface],
    previousWindows: [previousWindow],
    nextWindows: [nextWindow],
    nextDisplays: [display],
  });

  assert.ok(attachment);
  assert.deepEqual(attachment.foot, { x: 380, y: 260 });
  assert.equal(attachment.surface.id, previousSurface.id);
  assert.equal(attachment.changed, true);
});

test("attachment survives segment id changes and clamps to remaining space", () => {
  const previousSurface: SurfaceState = {
    id: "window-8:top:1", kind: "window_top", display_id: display.id, window_id: "window-8",
    x1: 100, x2: 700, y: 300, enabled: true, occluded: false,
  };
  const replacement: SurfaceState = {
    ...previousSurface, id: "window-8:top:0", x1: 400, x2: 700,
  };
  const carrier = windowState("window-8", { x: 100, y: 300, width: 600, height: 400 });
  const attachment = resolveSurfaceAttachment({
    foot: { x: 200, y: 300 },
    surfaceId: previousSurface.id,
    previousSurfaces: [previousSurface],
    nextSurfaces: [replacement],
    previousWindows: [carrier],
    nextWindows: [carrier],
    nextDisplays: [display],
  });

  assert.ok(attachment);
  assert.equal(attachment.surface.id, replacement.id);
  assert.deepEqual(attachment.foot, { x: 400 + SUPPORT_HALF_WIDTH_DIP, y: 300 });
});

test("attachment is released only when the carrier has no usable foot space", () => {
  const previousSurface: SurfaceState = {
    id: "window-9:top:0", kind: "window_top", display_id: display.id, window_id: "window-9",
    x1: 100, x2: 500, y: 300, enabled: true, occluded: false,
  };
  const tooNarrow: SurfaceState = { ...previousSurface, x1: 200, x2: 220 };
  const carrier = windowState("window-9", { x: 100, y: 300, width: 400, height: 400 });
  const attachment = resolveSurfaceAttachment({
    foot: { x: 300, y: 300 },
    surfaceId: previousSurface.id,
    previousSurfaces: [previousSurface],
    nextSurfaces: [tooNarrow],
    previousWindows: [carrier],
    nextWindows: [carrier],
    nextDisplays: [display],
  });

  assert.equal(attachment, null);
});

test("jump preparation stays attached but a real takeoff cannot be pulled back", () => {
  const source: SurfaceState = {
    id: "window-10:top:0", kind: "window_top", display_id: display.id, window_id: "window-10",
    x1: 100, x2: 500, y: 400, enabled: true, occluded: false,
  };
  assert.equal(surfaceIdForMotionSample(source.id, { x: 300, y: 400 }, "jump", [source]), source.id);
  assert.equal(surfaceIdForMotionSample(source.id, { x: 300, y: 397 }, "jump", [source]), null);
  assert.equal(surfaceIdForMotionSample(null, { x: 300, y: 399.8 }, "jump", [source]), null);
  assert.equal(surfaceIdForMotionSample(null, { x: 300, y: 400 }, "walk", [source]), source.id);
});

function windowState(id: string, bounds: DisplayState["bounds"]): WindowState {
  return {
    id,
    display_id: display.id,
    bounds,
    z_order: 0,
    visible: true,
    minimized: false,
    maximized: false,
    fullscreen: false,
    active: true,
    occluded: false,
    eligible: true,
  };
}
