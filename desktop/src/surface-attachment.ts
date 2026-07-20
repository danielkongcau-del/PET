import { clamp, nearestSupportingSurface, surfaceSupportsPoint, type Point } from "./geometry.js";
import type { Behavior, DisplayState, SurfaceState, WindowState } from "./protocol.js";

export const SUPPORT_HALF_WIDTH_DIP = 12;

export interface SurfaceAttachmentOptions {
  readonly foot: Point;
  readonly surfaceId: string;
  readonly previousSurfaces: readonly SurfaceState[];
  readonly nextSurfaces: readonly SurfaceState[];
  readonly previousWindows: readonly WindowState[];
  readonly nextWindows: readonly WindowState[];
  readonly nextDisplays: readonly DisplayState[];
}

export interface SurfaceAttachment {
  readonly foot: Point;
  readonly surface: SurfaceState;
  readonly windowRelativeX: number;
  readonly changed: boolean;
}

/**
 * A jump may remain attached while its preparation samples are exactly on the
 * carrier, but once it leaves that surface it cannot be reacquired by the
 * general proximity tolerance. Landing is owned by findCrossedSurface.
 */
export function surfaceIdForMotionSample(
  currentSurfaceId: string | null,
  foot: Point,
  behavior: Behavior,
  surfaces: readonly SurfaceState[],
): string | null {
  if (behavior === "falling") return null;
  if (behavior === "jump") {
    if (!currentSurfaceId) return null;
    const current = surfaces.find((surface) => surface.id === currentSurfaceId);
    return current && surfaceSupportsPoint(foot, current) ? current.id : null;
  }
  return nearestSupportingSurface(foot, surfaces, 0.5)?.id ?? null;
}

/**
 * Keeps a perched pet in the coordinate frame of its carrier window. Surface
 * segment ids can change when occlusion slicing changes, so window_id is the
 * stable identity and the closest usable top segment is selected each time.
 */
export function resolveSurfaceAttachment(options: SurfaceAttachmentOptions): SurfaceAttachment | null {
  const previousSurface = options.previousSurfaces.find((surface) => surface.id === options.surfaceId);
  if (previousSurface?.kind !== "window_top" || !previousSurface.window_id) return null;
  const previousWindow = options.previousWindows.find((window) => window.id === previousSurface.window_id);
  const nextWindow = options.nextWindows.find((window) => window.id === previousSurface.window_id);
  if (!previousWindow || !nextWindow || !nextWindow.visible || nextWindow.minimized || !nextWindow.eligible ||
    previousWindow.bounds.width <= 0 || nextWindow.bounds.width <= 0) return null;

  const windowRelativeX = clamp(
    (options.foot.x - previousWindow.bounds.x) / previousWindow.bounds.width,
    0,
    1,
  );
  const preferredX = nextWindow.bounds.x + windowRelativeX * nextWindow.bounds.width;
  let best: { readonly surface: SurfaceState; readonly x: number; readonly score: number } | null = null;

  for (const surface of options.nextSurfaces) {
    if (surface.kind !== "window_top" || surface.window_id !== nextWindow.id || !surface.enabled || surface.occluded) continue;
    const display = options.nextDisplays.find((candidate) => candidate.id === surface.display_id);
    const scale = display?.scale_factor ?? 1;
    const halfSupport = SUPPORT_HALF_WIDTH_DIP * scale;
    const safeX1 = surface.x1 + halfSupport;
    const safeX2 = surface.x2 - halfSupport;
    if (safeX2 < safeX1) continue;
    const x = clamp(preferredX, safeX1, safeX2);
    const score = Math.abs(x - preferredX);
    if (!best || score < best.score) best = { surface, x, score };
  }
  if (!best) return null;

  const foot = { x: best.x, y: best.surface.y };
  const changed = best.surface.id !== previousSurface.id ||
    best.surface.x1 !== previousSurface.x1 || best.surface.x2 !== previousSurface.x2 || best.surface.y !== previousSurface.y ||
    foot.x !== options.foot.x || foot.y !== options.foot.y;
  return { foot, surface: best.surface, windowRelativeX, changed };
}
