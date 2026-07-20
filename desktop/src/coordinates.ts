import electron, { type Display, type Point as ElectronPoint, type Rectangle } from "electron";

import type { DisplayState, Rect } from "./protocol.js";

const { screen } = electron;

export interface DisplayCoordinateFrame {
  readonly dipBounds: Rectangle;
  readonly physicalBounds: Rect;
  readonly scaleFactor: number;
}

/**
 * Map a physical foot anchor to a complete BrowserWindow DIP rectangle while
 * keeping one display's coordinate transform for the entire operation.
 *
 * Converting only the foot with screenToDipPoint and then subtracting a DIP
 * anchor lets the resulting top-left cross onto a neighbour with a different
 * scale factor.  This affine mapping deliberately remains relative to the
 * display containing the foot, even when part of the window straddles an edge.
 */
export function lockedDipBoundsForPhysicalFoot(
  foot: ElectronPoint,
  anchorDip: ElectronPoint,
  windowSizeDip: ElectronPoint,
  frame: DisplayCoordinateFrame,
): Rectangle {
  if (!Number.isFinite(frame.scaleFactor) || frame.scaleFactor <= 0) throw new RangeError("display scaleFactor must be positive");
  const x = frame.dipBounds.x + (foot.x - frame.physicalBounds.x) / frame.scaleFactor - anchorDip.x;
  const y = frame.dipBounds.y + (foot.y - frame.physicalBounds.y) / frame.scaleFactor - anchorDip.y;
  return {
    x: Math.round(x),
    y: Math.round(y),
    width: Math.round(windowSizeDip.x),
    height: Math.round(windowSizeDip.y),
  };
}

export function coordinateFrameForDisplay(display: Display): DisplayCoordinateFrame {
  return {
    dipBounds: display.bounds,
    physicalBounds: dipRectToPhysical(display.bounds),
    scaleFactor: display.scaleFactor,
  };
}

export function dipPointToPhysical(point: ElectronPoint): ElectronPoint {
  if (process.platform === "win32") return screen.dipToScreenPoint(point);
  const display = screen.getDisplayNearestPoint(point);
  return { x: Math.round(point.x * display.scaleFactor), y: Math.round(point.y * display.scaleFactor) };
}

export function physicalPointToDip(point: ElectronPoint): ElectronPoint {
  if (process.platform === "win32") return screen.screenToDipPoint(point);
  const display = nearestDisplayForPhysicalPoint(point);
  return { x: Math.round(point.x / display.scaleFactor), y: Math.round(point.y / display.scaleFactor) };
}

export function dipRectToPhysical(rect: Rectangle): Rect {
  if (process.platform === "win32") {
    // Let Electron select one display for the whole rectangle. Converting the
    // two corners independently can select different displays (and therefore
    // different scale factors) when a rectangle touches a mixed-DPI boundary.
    const converted = screen.dipToScreenRect(null, rect);
    return { x: converted.x, y: converted.y, width: converted.width, height: converted.height };
  }
  const topLeft = dipPointToPhysical({ x: rect.x, y: rect.y });
  const bottomRight = dipPointToPhysical({ x: rect.x + rect.width, y: rect.y + rect.height });
  return { x: topLeft.x, y: topLeft.y, width: bottomRight.x - topLeft.x, height: bottomRight.y - topLeft.y };
}

export function displayToProtocol(display: Display, primaryId: number): DisplayState {
  return {
    id: `display-${display.id}`,
    bounds: dipRectToPhysical(display.bounds),
    work_area: dipRectToPhysical(display.workArea),
    scale_factor: display.scaleFactor,
    is_primary: display.id === primaryId,
  };
}

export function displayIdForPhysicalRect(rect: Rect, displays: readonly DisplayState[]): string {
  const centre = { x: rect.x + rect.width / 2, y: rect.y + rect.height / 2 };
  let best = displays[0];
  if (!best) return "display-unknown";
  let bestDistance = Number.POSITIVE_INFINITY;
  for (const display of displays) {
    const nearestX = Math.max(display.bounds.x, Math.min(centre.x, display.bounds.x + display.bounds.width));
    const nearestY = Math.max(display.bounds.y, Math.min(centre.y, display.bounds.y + display.bounds.height));
    const distance = Math.hypot(centre.x - nearestX, centre.y - nearestY);
    if (distance < bestDistance) {
      best = display;
      bestDistance = distance;
    }
  }
  return best.id;
}

export function nearestDisplayForPhysicalPoint(point: ElectronPoint): Display {
  const displays = screen.getAllDisplays();
  let best = displays[0] ?? screen.getPrimaryDisplay();
  let bestDistance = Number.POSITIVE_INFINITY;
  for (const display of displays) {
    const bounds = dipRectToPhysical(display.bounds);
    const nearestX = Math.max(bounds.x, Math.min(point.x, bounds.x + bounds.width));
    const nearestY = Math.max(bounds.y, Math.min(point.y, bounds.y + bounds.height));
    const distance = Math.hypot(point.x - nearestX, point.y - nearestY);
    if (distance < bestDistance) {
      best = display;
      bestDistance = distance;
    }
  }
  return best;
}
