import type { Rect, SurfaceState } from "./protocol.js";

export interface Point {
  readonly x: number;
  readonly y: number;
}

export interface Interval {
  readonly x1: number;
  readonly x2: number;
}

export function clamp(value: number, min: number, max: number): number {
  return Math.min(Math.max(value, min), max);
}

export function intersectionArea(a: Rect, b: Rect): number {
  const width = Math.max(0, Math.min(a.x + a.width, b.x + b.width) - Math.max(a.x, b.x));
  const height = Math.max(0, Math.min(a.y + a.height, b.y + b.height) - Math.max(a.y, b.y));
  return width * height;
}

export function rectContainsPoint(rect: Rect, point: Point): boolean {
  return point.x >= rect.x && point.x <= rect.x + rect.width && point.y >= rect.y && point.y <= rect.y + rect.height;
}

export function rectApproximatelyEquals(a: Rect, b: Rect, tolerance = 3): boolean {
  return Math.abs(a.x - b.x) <= tolerance && Math.abs(a.y - b.y) <= tolerance &&
    Math.abs(a.width - b.width) <= tolerance && Math.abs(a.height - b.height) <= tolerance;
}

export function rectCovers(a: Rect, b: Rect, minimumCoverage = 0.985, edgeTolerance = 12): boolean {
  const area = Math.max(1, b.width * b.height);
  return intersectionArea(a, b) / area >= minimumCoverage &&
    a.x <= b.x + edgeTolerance && a.y <= b.y + edgeTolerance &&
    a.x + a.width >= b.x + b.width - edgeTolerance &&
    a.y + a.height >= b.y + b.height - edgeTolerance;
}

export function subtractIntervals(base: Interval, blockers: readonly Interval[], minimumWidth: number): Interval[] {
  let segments: Interval[] = [base];
  const sorted = [...blockers].sort((a, b) => a.x1 - b.x1);
  for (const blocker of sorted) {
    const next: Interval[] = [];
    for (const segment of segments) {
      if (blocker.x2 <= segment.x1 || blocker.x1 >= segment.x2) {
        next.push(segment);
        continue;
      }
      if (blocker.x1 - segment.x1 >= minimumWidth) next.push({ x1: segment.x1, x2: blocker.x1 });
      if (segment.x2 - blocker.x2 >= minimumWidth) next.push({ x1: blocker.x2, x2: segment.x2 });
    }
    segments = next;
    if (segments.length === 0) break;
  }
  return segments.filter((segment) => segment.x2 - segment.x1 >= minimumWidth);
}

export function findCrossedSurface(
  previous: Point,
  next: Point,
  surfaces: readonly SurfaceState[],
  halfFootWidth = 12,
): SurfaceState | null {
  // A contact is a crossing from above, not merely a sample that happens to
  // remain on the same y coordinate. Without the strict downward check, every
  // horizontal walk sample on a surface is reported as a fresh landing.
  if (next.y <= previous.y) return null;
  const candidates = surfaces.filter((surface) => surface.enabled && !surface.occluded &&
    previous.y < surface.y && next.y >= surface.y &&
    next.x + halfFootWidth >= surface.x1 && next.x - halfFootWidth <= surface.x2,
  );
  if (candidates.length === 0) return null;
  return candidates.reduce((highest, surface) => surface.y < highest.y ? surface : highest);
}

export function surfaceSupportsPoint(point: Point, surface: SurfaceState, tolerance = 0.5): boolean {
  return surface.enabled && !surface.occluded &&
    point.x >= surface.x1 && point.x <= surface.x2 &&
    Math.abs(point.y - surface.y) <= tolerance;
}

export function nearestSupportingSurface(point: Point, surfaces: readonly SurfaceState[], tolerance = 6): SurfaceState | null {
  const candidates = surfaces.filter((surface) => surfaceSupportsPoint(point, surface, tolerance));
  if (candidates.length === 0) return null;
  return candidates.reduce((best, surface) => Math.abs(surface.y - point.y) < Math.abs(best.y - point.y) ? surface : best);
}
