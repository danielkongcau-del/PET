"""Offline desktop environment for trajectory data generation.

Mirrors the host's geometry.ts collision and motion-controller.ts physics
so that teacher-generated plans produce physically consistent WorldState
sequences without requiring Electron or a real Windows desktop.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
import math
import random
from typing import Any, Sequence
import uuid

from .backend import MotionBackend, MotionPlan, MotionPoint
from .character_rig import SelectedCharacterRig, load_selected_character_rig
from .state import (
    ClickState, CursorState, DisplayState, PetState, SceneState,
    SurfaceState, WorldState,
)


# ═══════════════════════════════════════════════════════════════════════
#  Physics constants — mirrored from desktop/src/motion-controller.ts
# ═══════════════════════════════════════════════════════════════════════

MAX_ROOT_SPEED = 2200.0         # px / s
GRAVITY = 980.0                 # px / s^2
PLAN_DT_MS = 33                 # ms between plan points
WORLD_STATE_DT_MS = 50          # live host publishes at 20 Hz
MOTION_TICK_MS = 16             # live host's single-writer motion timer
# v1 host collision ABI. Character manifests normalize every render canvas to
# a 96 DIP window, while footAnchor remains character-specific.
PET_WINDOW_DIP = 96.0
MIN_WINDOW_TOP_Y_DIP = PET_WINDOW_DIP / 2.0
LANDING_HALF_FOOT_DIP = 12.0
PLAN_SAMPLING_SEMANTICS = "generated-at-wall-clock-linear-v1"
SCENARIO_EVENT_TYPES = (
    "window_move",
    "window_minimize",
    "window_maximize",
    "window_restore",
    "fullscreen_enter",
    "fullscreen_restore",
    "pet_click",
)
ALLOWED_BEHAVIORS = {
    "idle", "walk", "jump", "click_reaction", "landing", "falling",
    "hidden", "fallback",
}
FACIAL_PARAMETER_RANGES = {
    "eye_scale": (0.5, 1.5),
    "eye_squint": (0.0, 1.0),
    "mouth_open": (0.0, 1.0),
    "ear_angle": (-0.5, 0.5),
    "brow_tilt": (-1.0, 1.0),
}

CONDITION_FEATURE_ORDER = (
    "foot_x", "foot_y", "vx", "vy", "facing", "behavior", "on_surface",
    "surf_hash_0", "surf_hash_1", "surf_hash_2", "surf_hash_3", "surf_y_diff",
    *(
        f"surf{surface_index}_{field}"
        for surface_index in range(4)
        for field in ("dy", "x1_rel", "x2_rel", "kind", "width")
    ),
    "cur_x", "cur_y", "cur_over", "cur_down",
    "click_pending", "click_age", "click_x", "click_y",
    "pet_allowed", "fullscreen", "gen_status", "time_phase",
    "goal_behavior_0", "goal_behavior_1", "goal_behavior_2", "goal_surface_y",
)
TARGET_MOTION_FIELD_ORDER = (
    "dx", "dy", "vx", "vy", "facing", "lean", "squash", "bob",
)
QUATERNION_COMPONENT_ORDER = ("x", "y", "z", "w")
ROOT_TRANSLATION_ORDER = ("x", "y", "z")
FACIAL_PARAMETER_ORDER = (
    "eye_scale", "eye_squint", "mouth_open", "ear_angle", "brow_tilt",
)
TARGET_RECORD_FIELD_ORDER = (
    "t_ms", "behavior", *TARGET_MOTION_FIELD_ORDER, "expression",
    "quaternions", "root", "facial",
)
if len(CONDITION_FEATURE_ORDER) != 48:  # pragma: no cover - import-time ABI guard
    raise RuntimeError("Condition feature ABI must contain exactly 48 fields")


# ═══════════════════════════════════════════════════════════════════════
#  Geometry helpers — mirrored from desktop/src/geometry.ts
# ═══════════════════════════════════════════════════════════════════════

def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(value, hi))


def _surface_supports_point(
    foot_x: float, foot_y: float, surface: SurfaceState, tolerance: float = 0.5,
) -> bool:
    return (
        surface.enabled
        and not surface.occluded
        and surface.x1 <= foot_x <= surface.x2
        and abs(foot_y - surface.y) <= tolerance
    )


def _find_crossed_surface(
    prev_x: float, prev_y: float,
    next_x: float, next_y: float,
    surfaces: Sequence[SurfaceState],
    half_foot_width: float = LANDING_HALF_FOOT_DIP,
) -> SurfaceState | None:
    """Detect landing: foot crosses a surface from above."""
    if next_y <= prev_y:
        return None
    candidates = [
        s for s in surfaces
        if s.enabled and not s.occluded
        and prev_y < s.y and next_y >= s.y
        and next_x + half_foot_width >= s.x1
        and next_x - half_foot_width <= s.x2
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda s: s.y)


def _nearest_supporting_surface(
    foot_x: float, foot_y: float,
    surfaces: Sequence[SurfaceState],
    tolerance: float = 8.0,
) -> SurfaceState | None:
    candidates = [s for s in surfaces if _surface_supports_point(foot_x, foot_y, s, tolerance)]
    if not candidates:
        return None
    return min(candidates, key=lambda s: abs(s.y - foot_y))


def _clamp_to_work_area(
    x: float, y: float,
    display: DisplayState,
    anchor_x: float = 48, anchor_y: float = 92,
    window_w: float = 96, window_h: float = 96,
) -> tuple[float, float]:
    """Keep the pet window within one display's work area."""
    scale = display.scale_factor
    left = display.work_x + anchor_x * scale
    right = display.work_x + display.work_width - (window_w - anchor_x) * scale
    top = display.work_y + anchor_y * scale
    bottom = display.work_y + display.work_height
    return (_clamp(x, left, right), _clamp(y, top, bottom))


# ═══════════════════════════════════════════════════════════════════════
#  Scenario generation
# ═══════════════════════════════════════════════════════════════════════

@dataclass(slots=True)
class ScenarioConfig:
    num_displays: tuple[int, int] = (1, 2)
    display_scales: tuple[float, ...] = (1.0, 1.25, 1.5)
    num_windows: tuple[int, int] = (3, 10)
    window_min_w: int = 280
    window_max_w: int = 1400
    window_min_h: int = 100
    window_max_h: int = 900
    pet_start: str = "floor"   # "floor" | "window" | "mixed"
    duration_ms: int = 30_000
    window_move_probability: float = 0.02   # per tick
    window_minimize_probability: float = 0.002
    window_maximize_probability: float = 0.002
    window_restore_probability: float = 0.004
    fullscreen_probability: float = 0.001
    fullscreen_restore_probability: float = 0.02
    click_probability: float = 0.005        # per tick

    def __post_init__(self) -> None:
        probabilities = {
            "window_move_probability": self.window_move_probability,
            "window_minimize_probability": self.window_minimize_probability,
            "window_maximize_probability": self.window_maximize_probability,
            "window_restore_probability": self.window_restore_probability,
            "fullscreen_probability": self.fullscreen_probability,
            "fullscreen_restore_probability": self.fullscreen_restore_probability,
            "click_probability": self.click_probability,
        }
        for name, probability in probabilities.items():
            if not isinstance(probability, (int, float)) or isinstance(probability, bool):
                raise ValueError(f"{name} must be numeric")
            if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")
        if sum((
            self.window_move_probability,
            self.window_minimize_probability,
            self.window_maximize_probability,
            self.window_restore_probability,
        )) > 1.0:
            raise ValueError("window event probabilities must sum to at most 1")


def _generate_display_state(
    display_id: int, is_primary: bool, rng: random.Random,
    scale: float | None = None,
    origin_x: float | None = None,
) -> DisplayState:
    if scale is None:
        scale = rng.choice((1.0, 1.25, 1.5, 2.0))
    w = rng.randint(1366, 2560)
    h = rng.randint(768, 1440)
    # Leave some margin so windows can sit on the display.
    taskbar_h = int(48 * scale)
    x = float(display_id * 2000) if origin_x is None else float(origin_x)
    return DisplayState(
        id=f"display-{display_id}",
        x=x, y=0.0,
        width=float(w), height=float(h),
        work_x=x,
        work_y=0.0,
        work_width=float(w),
        work_height=float(h - taskbar_h),
        scale_factor=scale,
    )


def _subtract_intervals(
    x1: float,
    x2: float,
    blockers: Sequence[tuple[float, float]],
    minimum_width: float,
) -> list[tuple[float, float]]:
    """Return visible portions of one top edge after z-order blockers."""
    segments = [(x1, x2)]
    for blocker_x1, blocker_x2 in sorted(blockers):
        next_segments: list[tuple[float, float]] = []
        for segment_x1, segment_x2 in segments:
            if blocker_x2 <= segment_x1 or blocker_x1 >= segment_x2:
                next_segments.append((segment_x1, segment_x2))
                continue
            if blocker_x1 - segment_x1 >= minimum_width:
                next_segments.append((segment_x1, min(blocker_x1, segment_x2)))
            if segment_x2 - blocker_x2 >= minimum_width:
                next_segments.append((max(blocker_x2, segment_x1), segment_x2))
        segments = next_segments
    return [segment for segment in segments if segment[1] - segment[0] >= minimum_width]


def _window_intersection_area(first: SimWindow, second: SimWindow) -> float:
    width = max(0.0, min(first.x + first.width, second.x + second.width) - max(first.x, second.x))
    height = max(0.0, min(first.y + first.height, second.y + second.height) - max(first.y, second.y))
    return width * height


def _generate_surfaces(
    displays: Sequence[DisplayState],
    windows: list[SimWindow],
    min_surface_w: float = PET_WINDOW_DIP,
) -> list[SurfaceState]:
    """Generate work_area_floor for each display + window_top for each window."""
    surfaces: list[SurfaceState] = []
    for d in displays:
        surfaces.append(SurfaceState(
            id=f"{d.id}:floor", kind="work_area_floor",
            display_id=d.id, window_id=None,
            x1=d.work_x, x2=d.work_x + d.work_width,
            y=d.work_y + d.work_height,
            enabled=True, occluded=False,
        ))
    # SimWindow list order is z-order: index 0 is the front-most window.
    for index, w in enumerate(windows):
        if w.minimized:
            continue
        display = next((candidate for candidate in displays if candidate.id == w.display_id), None)
        if display is None:
            continue
        minimum_width = min_surface_w * display.scale_factor
        x1 = max(float(w.x), display.work_x)
        x2 = min(float(w.x + w.width), display.work_x + display.work_width)
        y = float(w.y)
        minimum_top = display.work_y + MIN_WINDOW_TOP_Y_DIP * display.scale_factor
        if (
            x2 - x1 < minimum_width
            or w.height < 48.0 * display.scale_factor
            or y < minimum_top
            or y > display.work_y + display.work_height
        ):
            continue
        covers = [cover for cover in windows[:index] if not cover.minimized]
        area = max(1.0, float(w.width * w.height))
        occluded = any(_window_intersection_area(w, cover) / area >= 0.9 for cover in covers)
        blockers: list[tuple[float, float]] = []
        for cover in covers:
            if cover.y <= y + 2 and cover.y + cover.height >= y - 2:
                blockers.append((float(cover.x), float(cover.x + cover.width)))
        segments = _subtract_intervals(x1, x2, blockers, minimum_width)
        for segment_index, (segment_x1, segment_x2) in enumerate(segments):
            surfaces.append(SurfaceState(
                id=f"window-{w.win_id}:top:{segment_index}", kind="window_top",
                display_id=w.display_id, window_id=f"window-{w.win_id}",
                x1=segment_x1, x2=segment_x2, y=y,
                enabled=not occluded, occluded=occluded,
            ))
    return surfaces


@dataclass
class SimWindow:
    win_id: int
    display_id: str
    x: int
    y: int
    width: int
    height: int
    minimized: bool = False
    maximized: bool = False
    restore_bounds: tuple[int, int, int, int] | None = None
    vx: float = 0.0
    vy: float = 0.0


@dataclass
class SimPet:
    foot_x: float
    foot_y: float
    vx: float = 0.0
    vy: float = 0.0
    facing: int = 1
    behavior: str = "idle"
    surface_id: str | None = None
    width: float = 96.0
    height: float = 96.0


@dataclass(frozen=True, slots=True)
class _WindowMove:
    window_id: str
    old_x: float
    old_y: float
    old_width: float
    new_x: float
    new_y: float


# ═══════════════════════════════════════════════════════════════════════
#  Desktop Simulator
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class TrainingSample:
    """One (condition, target) pair for model training."""
    condition_frames: list[dict[str, Any]]
    target_poses: list[dict[str, Any]]
    metadata: dict[str, Any] = field(default_factory=dict)


class DesktopSimulator:
    """Offline desktop environment with deterministic physics."""

    def __init__(
        self,
        seed: int = 0,
        *,
        context_steps: int = 8,
        horizon_steps: int = 12,
        dt_ms: int = PLAN_DT_MS,
        world_state_dt_ms: int = WORLD_STATE_DT_MS,
        character_rig: SelectedCharacterRig | None = None,
    ):
        self.rng = random.Random(seed)
        self.displays: list[DisplayState] = []
        self.windows: list[SimWindow] = []
        self.surfaces: list[SurfaceState] = []
        self.pet: SimPet = SimPet(0.0, 0.0)
        self.clicks: list[ClickState] = []
        self.cursor: CursorState | None = None
        self.time_ms: int = 0
        self.scene: SceneState = SceneState(
            fullscreen_active=False, pet_allowed=True, suspend_reason=None,
        )
        self.seq: int = 0
        self.session_id: str = "sim-session"
        self._session_namespace = uuid.uuid4().hex
        self._episode_serial = 0
        self.character_rig = character_rig
        self.context_steps = 0
        self.horizon_steps = 0
        self.plan_dt_ms = 0
        self.plan_horizon_ms = 0
        if not isinstance(world_state_dt_ms, int) or isinstance(world_state_dt_ms, bool):
            raise ValueError("world_state_dt_ms must be an integer")
        if world_state_dt_ms < 8 or world_state_dt_ms > 1_000:
            raise ValueError("world_state_dt_ms must be between 8 and 1000")
        self.world_state_dt_ms = world_state_dt_ms
        self.world_state_dt_s = world_state_dt_ms / 1000.0
        # The desktop owns one long-lived 16 ms motion timer and a separate
        # world-state timer. Keep their nominal phases independent: a 50 ms
        # world interval must never be synthesized as 16+16+16+2 ms ticks.
        self._last_motion_tick_ms = 0
        self._next_motion_tick_ms = MOTION_TICK_MS
        self.set_timing(dt_ms, horizon_steps * dt_ms, context_steps=context_steps)

    # ── Configuration ──────────────────────────────────────────────────

    def set_timing(
        self,
        dt_ms: int,
        horizon_ms: int,
        *,
        context_steps: int | None = None,
    ) -> None:
        """Configure this simulator without mutating other instances."""
        if dt_ms < 8 or dt_ms > 250:
            raise ValueError("dt_ms must be between 8 and 250")
        if horizon_ms < dt_ms:
            raise ValueError("horizon_ms must contain at least one step")
        horizon_steps = int(round(horizon_ms / dt_ms))
        if horizon_steps < 2 or horizon_steps > 120:
            raise ValueError("horizon_steps must be between 2 and 120")
        if horizon_steps * dt_ms < self.world_state_dt_ms:
            raise ValueError("plan horizon must cover one world-state interval")
        if context_steps is not None:
            if context_steps < 1:
                raise ValueError("context_steps must be positive")
            self.context_steps = context_steps
        elif self.context_steps < 1:
            self.context_steps = 8
        self.plan_dt_ms = dt_ms
        self.horizon_steps = horizon_steps
        self.plan_horizon_ms = horizon_ms

    # ── Scene generation ──────────────────────────────────────────────

    def reset(self, config: ScenarioConfig | None = None) -> WorldState:
        cfg = config or ScenarioConfig()
        self._selected_character_rig()
        self.time_ms = 0
        self.seq = 0
        self.clicks = []
        self._last_motion_tick_ms = 0
        self._next_motion_tick_ms = MOTION_TICK_MS

        # Displays
        n_displays = self.rng.randint(*cfg.num_displays)
        scales = cfg.display_scales
        self.displays = []
        next_display_x = 0.0
        for i in range(n_displays):
            display = _generate_display_state(
                i,
                i == 0,
                self.rng,
                self.rng.choice(scales) if scales else None,
                origin_x=next_display_x,
            )
            self.displays.append(display)
            next_display_x += display.width

        # Windows
        n_windows = self.rng.randint(*cfg.num_windows)
        self.windows = []
        for i in range(n_windows):
            d = self.rng.choice(self.displays)
            max_w = max(48, int(d.work_width) - 4)
            top_clearance = int(math.ceil(MIN_WINDOW_TOP_Y_DIP * d.scale_factor))
            max_h = max(48, int(d.work_height) - top_clearance - 4)
            w_min = min(cfg.window_min_w, max_w)
            w_max = min(cfg.window_max_w, max_w)
            h_min = min(cfg.window_min_h, max_h)
            h_max = min(cfg.window_max_h, max_h)
            if w_max < w_min:
                w_max = w_min
            if h_max < h_min:
                h_max = h_min
            w = self.rng.randint(w_min, w_max)
            h = self.rng.randint(h_min, h_max)
            x_max = int(d.work_x + d.work_width - w)
            x_min = int(d.work_x)
            if x_max <= x_min:
                x = x_min
            else:
                x = self.rng.randint(x_min, x_max)
            y_max = int(d.work_y + d.work_height - h)
            y_min = int(d.work_y + top_clearance)
            if y_max <= y_min:
                y = y_min
            else:
                y = self.rng.randint(y_min, y_max)
            self.windows.append(SimWindow(
                win_id=i, display_id=d.id, x=x, y=y, width=w, height=h,
            ))

        # Surfaces
        self.surfaces = _generate_surfaces(self.displays, self.windows)

        # Pet placement
        self._place_pet(cfg.pet_start)
        self.cursor = CursorState(
            x=self.pet.foot_x + self.rng.randint(-200, 200),
            y=self.pet.foot_y + self.rng.randint(-200, 200),
            left_down=False,
        )
        self.scene = SceneState(fullscreen_active=False, pet_allowed=True, suspend_reason=None)
        return self._build_world_state()

    def _place_pet(self, mode: str) -> None:
        if mode == "mixed":
            mode = self.rng.choice(("floor", "window")) if self.windows else "floor"
        if mode == "floor" or not self.windows:
            floor = next((s for s in self.surfaces if s.kind == "work_area_floor"), None)
            if floor:
                self.pet = self._pet_on_surface(floor)
                return
        elif mode == "window":
            windows = [
                surface
                for surface in self.surfaces
                if surface.kind == "window_top" and surface.enabled and not surface.occluded
            ]
            if windows:
                s = self.rng.choice(windows)
                self.pet = self._pet_on_surface(s)
                return
        # Fallback: use first floor
        floor = next((s for s in self.surfaces if s.kind == "work_area_floor"), None)
        if floor:
            self.pet = self._pet_on_surface(floor)
        else:
            self.pet = SimPet(foot_x=400.0, foot_y=700.0)

    def _pet_on_surface(self, surface: SurfaceState) -> SimPet:
        display = next((candidate for candidate in self.displays if candidate.id == surface.display_id), None)
        scale = display.scale_factor if display else 1.0
        anchor_x, _, window_w, window_h = self._render_geometry_dip()
        if surface.kind == "window_top":
            low = surface.x1 + LANDING_HALF_FOOT_DIP * scale
            high = surface.x2 - LANDING_HALF_FOOT_DIP * scale
        else:
            low = surface.x1 + anchor_x * scale
            high = surface.x2 - (window_w - anchor_x) * scale
        x = (surface.x1 + surface.x2) / 2.0 if high < low else self.rng.uniform(low, high)
        return SimPet(
            foot_x=x,
            foot_y=surface.y,
            surface_id=surface.id,
            width=window_w * scale,
            height=window_h * scale,
        )

    # ── Physics tick ──────────────────────────────────────────────────

    def _apply_plan_point(
        self,
        origin_x: float,
        origin_y: float,
        point: Any,
        behavior: str,
        dt_s: float | None = None,
    ) -> SurfaceState | None:
        """Apply one fixed-origin plan point using the host safety order."""
        if dt_s is None:
            dt_s = MOTION_TICK_MS / 1000.0
        previous_x = self.pet.foot_x
        previous_y = self.pet.foot_y
        target_x = origin_x + point.dx
        target_y = origin_y + point.dy

        # limitStep
        dx = target_x - previous_x
        dy = target_y - previous_y
        dist = math.hypot(dx, dy)
        max_dist = MAX_ROOT_SPEED * dt_s
        if dist > max_dist and dist > 0:
            scale = max_dist / dist
            target_x = self.pet.foot_x + dx * scale
            target_y = self.pet.foot_y + dy * scale

        # clampToWorkArea
        display = self._display_for_point(target_x, target_y)
        if display:
            anchor_x, anchor_y, window_w, window_h = self._render_geometry_dip()
            target_x, target_y = _clamp_to_work_area(
                target_x,
                target_y,
                display,
                anchor_x=anchor_x,
                anchor_y=anchor_y,
                window_w=window_w,
                window_h=window_h,
            )

        # findCrossedSurface
        landed = _find_crossed_surface(
            self.pet.foot_x, self.pet.foot_y, target_x, target_y, self.surfaces,
        )
        if landed:
            self.pet.foot_x = _clamp(target_x, landed.x1, landed.x2)
            self.pet.foot_y = landed.y
            self.pet.surface_id = landed.id
            self.pet.behavior = "landing"
        else:
            self.pet.foot_x = target_x
            self.pet.foot_y = target_y
            if behavior == "falling":
                self.pet.surface_id = None
            elif behavior == "jump":
                current = next(
                    (surface for surface in self.surfaces if surface.id == self.pet.surface_id),
                    None,
                )
                if not (current and _surface_supports_point(target_x, target_y, current)):
                    self.pet.surface_id = None
            else:
                support = _nearest_supporting_surface(target_x, target_y, self.surfaces, 0.5)
                self.pet.surface_id = support.id if support else None
            self.pet.behavior = behavior
        self.pet.vx = (self.pet.foot_x - previous_x) / dt_s
        self.pet.vy = (self.pet.foot_y - previous_y) / dt_s
        if point.facing in (-1, 1):
            self.pet.facing = point.facing
        self._sync_pet_display_size()
        return landed

    def _apply_fallback_tick(self, dt_s: float) -> None:
        """Mirror MotionController._applyFallback for a cancelled plan tick."""
        current = next(
            (
                surface
                for surface in self.surfaces
                if surface.id == self.pet.surface_id
                and surface.enabled
                and not surface.occluded
            ),
            None,
        )
        if current is not None and _surface_supports_point(
            self.pet.foot_x, self.pet.foot_y, current,
        ):
            dy = current.y - self.pet.foot_y
            max_dist = MAX_ROOT_SPEED * dt_s
            self.pet.foot_y += _clamp(dy, -max_dist, max_dist)
            self.pet.vx = 0.0
            self.pet.vy = 0.0
            self.pet.behavior = "fallback"
            self._sync_pet_display_size()
            return

        support = _nearest_supporting_surface(
            self.pet.foot_x, self.pet.foot_y, self.surfaces, 8.0,
        )
        if support is not None:
            self.pet.surface_id = support.id
            self.pet.foot_y = support.y
            self.pet.vx = 0.0
            self.pet.vy = 0.0
            self.pet.behavior = "fallback"
            self._sync_pet_display_size()
            return

        next_velocity_y = self.pet.vy + GRAVITY * dt_s
        target_x = self.pet.foot_x
        target_y = self.pet.foot_y + next_velocity_y * dt_s
        display = self._display_for_point(target_x, target_y)
        if display is not None:
            anchor_x, anchor_y, window_w, window_h = self._render_geometry_dip()
            target_x, target_y = _clamp_to_work_area(
                target_x,
                target_y,
                display,
                anchor_x=anchor_x,
                anchor_y=anchor_y,
                window_w=window_w,
                window_h=window_h,
            )
        landed = _find_crossed_surface(
            self.pet.foot_x,
            self.pet.foot_y,
            target_x,
            target_y,
            self.surfaces,
        )
        if landed is not None:
            self.pet.foot_x = _clamp(target_x, landed.x1, landed.x2)
            self.pet.foot_y = landed.y
            self.pet.surface_id = landed.id
            self.pet.vx = 0.0
            self.pet.vy = 0.0
            self.pet.behavior = "landing"
        else:
            self.pet.foot_x = target_x
            self.pet.foot_y = target_y
            self.pet.surface_id = None
            self.pet.vx = 0.0
            self.pet.vy = next_velocity_y
            self.pet.behavior = "falling"
        self._sync_pet_display_size()

    def _sample_plan_at(self, plan: MotionPlan, elapsed_ms: float) -> MotionPoint | None:
        """Sample a plan using the host's generated-at wall-clock interpolation."""
        if not plan.points:
            return None
        if elapsed_ms > plan.points[-1].t_ms + plan.dt_ms:
            return None
        left = plan.points[0]
        right = left
        for candidate in plan.points[1:]:
            if candidate.t_ms >= elapsed_ms:
                right = candidate
                break
            left = candidate
            right = candidate
        span = max(1.0, float(right.t_ms - left.t_ms))
        alpha = _clamp((elapsed_ms - left.t_ms) / span, 0.0, 1.0)
        source = left if alpha < 0.5 else right

        def lerp(left_value: float, right_value: float) -> float:
            return left_value + (right_value - left_value) * alpha

        return replace(
            source,
            t_ms=round(elapsed_ms),
            dx=lerp(left.dx, right.dx),
            dy=lerp(left.dy, right.dy),
            vx=lerp(left.vx, right.vx),
            vy=lerp(left.vy, right.vy),
            facing=source.facing,
            lean=lerp(left.lean, right.lean),
            squash=lerp(left.squash, right.squash),
            bob=lerp(left.bob, right.bob),
            expression=source.expression,
        )

    def _advance_plan_to_next_world_state(
        self,
        plan: MotionPlan,
        *,
        backend: MotionBackend,
        origin_x: float,
        origin_y: float,
    ) -> None:
        """Advance on the long-lived motion timer until the next 20 Hz state.

        The online executor samples relative to ``generated_at_ms`` on every
        motion timer callback. The 16 ms timer was started before the separate
        50 ms publisher, so its absolute phase survives every replan. If both
        timers share a nominal deadline, motion runs first, matching their host
        registration order.
        """

        world_start_ms = plan.generated_at_ms
        world_end_ms = world_start_ms + self.world_state_dt_ms

        # The first training plan is produced by the first world publication;
        # advance the already-running timer's phase through earlier callbacks
        # without inventing plan execution before that plan existed.
        while self._next_motion_tick_ms <= world_start_ms:
            self._last_motion_tick_ms = self._next_motion_tick_ms
            self._next_motion_tick_ms += MOTION_TICK_MS

        target_surface_id = (
            plan.target.get("surface_id")
            if isinstance(plan.target, Mapping)
            else None
        )
        plan_cancelled = False
        while self._next_motion_tick_ms <= world_end_ms:
            tick_ms = self._next_motion_tick_ms
            step_ms = tick_ms - self._last_motion_tick_ms
            elapsed_ms = tick_ms - world_start_ms
            landed = None
            if plan_cancelled:
                self._apply_fallback_tick(step_ms / 1000.0)
            else:
                point = self._sample_plan_at(plan, elapsed_ms)
                if point is None:
                    raise RuntimeError("validated teacher plan expired before the next world state")
                landed = self._apply_plan_point(
                    origin_x,
                    origin_y,
                    point,
                    plan.behavior,
                    dt_s=step_ms / 1000.0,
                )
            self._last_motion_tick_ms = tick_ms
            self._next_motion_tick_ms += MOTION_TICK_MS
            if landed is not None and target_surface_id and landed.id != target_surface_id:
                # The host cancels the plan here, then continues running its
                # safety fallback on every remaining absolute motion tick.
                backend.cancel(plan.plan_id)
                plan_cancelled = True

    def _display_for_point(self, x: float, y: float) -> DisplayState | None:
        for d in self.displays:
            if d.work_x <= x <= d.work_x + d.work_width and d.work_y <= y <= d.work_y + d.work_height:
                return d
        best: DisplayState | None = None
        best_distance = math.inf
        for display in self.displays:
            nearest_x = _clamp(x, display.work_x, display.work_x + display.work_width)
            nearest_y = _clamp(y, display.work_y, display.work_y + display.work_height)
            distance = math.hypot(x - nearest_x, y - nearest_y)
            if distance < best_distance:
                best = display
                best_distance = distance
        return best

    def _sync_pet_display_size(self) -> None:
        display = self._display_for_point(self.pet.foot_x, self.pet.foot_y)
        scale = display.scale_factor if display else 1.0
        _, _, window_w, window_h = self._render_geometry_dip()
        self.pet.width = window_w * scale
        self.pet.height = window_h * scale

    def _render_geometry_dip(self) -> tuple[float, float, float, float]:
        """Return foot anchor and window size from the selected character manifest."""
        render = self._selected_character_rig().raw.get("render")
        if isinstance(render, Mapping):
            canvas = render.get("canvas")
            foot_anchor = render.get("footAnchor")
            display_scale = render.get("displayScale")
            if (
                isinstance(canvas, list)
                and len(canvas) == 2
                and isinstance(foot_anchor, list)
                and len(foot_anchor) == 2
                and isinstance(display_scale, (int, float))
                and not isinstance(display_scale, bool)
            ):
                return (
                    float(foot_anchor[0]) * float(display_scale),
                    float(foot_anchor[1]) * float(display_scale),
                    float(canvas[0]) * float(display_scale),
                    float(canvas[1]) * float(display_scale),
                )
        # Legacy rigs have no render block and retain the normalized v1 ABI.
        return (48.0, 92.0, 96.0, 96.0)

    # ── World state construction ──────────────────────────────────────

    def _build_world_state(self) -> WorldState:
        display = self._display_for_point(self.pet.foot_x, self.pet.foot_y)
        scale = display.scale_factor if display else 1.0
        anchor_x, anchor_y, _, _ = self._render_geometry_dip()
        self._sync_pet_display_size()
        return WorldState(
            seq=self.seq,
            timestamp_ms=self.time_ms,
            session_id=self.session_id,
            coordinate_space="physical_px",
            displays=tuple(self.displays),
            surfaces=tuple(self.surfaces),
            pet=PetState(
                x=self.pet.foot_x - anchor_x * scale,
                y=self.pet.foot_y - anchor_y * scale,
                width=self.pet.width, height=self.pet.height,
                foot_x=self.pet.foot_x, foot_y=self.pet.foot_y,
                vx=self.pet.vx, vy=self.pet.vy,
                facing=self.pet.facing,
                behavior=self.pet.behavior,
                visible=self.scene.pet_allowed, user_dragging=False,
                surface_id=self.pet.surface_id,
            ),
            cursor=self.cursor,
            clicks=tuple(self.clicks),
            scene=self.scene,
            requested_seed=None,
        )

    # ── Episode generation ────────────────────────────────────────────

    def generate_episode(
        self,
        backend: MotionBackend,
        config: ScenarioConfig | None = None,
        episode_seed: int = 0,
        *,
        sample_metadata_provider: Callable[[str], Mapping[str, Any]] | None = None,
    ) -> list[TrainingSample]:
        """Run one episode and emit one aligned sample per teacher plan.

        A fresh horizon is generated every 50 ms world-state tick, matching the
        live host's 20 Hz publisher.  Between states the plan is sampled from
        its generated-at wall-clock origin on deterministic 16 ms motion ticks.
        Provenance is consumed immediately even while context history is still
        warming, so backend-side state remains bounded.
        """
        cfg = config or ScenarioConfig()
        episode_rng = random.Random(episode_seed)
        self.rng = random.Random(episode_rng.randint(0, 2**31 - 1))
        self._episode_serial += 1
        self.session_id = (
            f"sim-{self._session_namespace}-run-{self._episode_serial:08x}-seed-{episode_seed:x}"
        )
        self.reset(cfg)
        # Custom deterministic test/world builders may override reset(). The
        # two host timers are process-lifetime clocks, so every episode starts
        # from the same nominal phase regardless of the reset implementation.
        self._last_motion_tick_ms = 0
        self._next_motion_tick_ms = MOTION_TICK_MS
        backend.cancel()
        backend.configure_timing(self.plan_horizon_ms, self.plan_dt_ms)
        backend.set_skeletal_3d(True)
        backend.set_skeletal_enabled(True)

        samples: list[TrainingSample] = []
        condition_history: list[dict[str, Any]] = []

        while self.time_ms < cfg.duration_ms:
            self.time_ms += self.world_state_dt_ms
            self.seq += 1

            # Inject deterministic seeded desktop events.
            window_event = self._random_window_event(episode_rng, cfg)
            if window_event is not None:
                previous_surfaces = tuple(self.surfaces)
                move = self._apply_window_transition(window_event, episode_rng)
                self.surfaces = _generate_surfaces(self.displays, self.windows)
                if move is not None:
                    self._reattach_pet_after_window_move(move, previous_surfaces)
            self._random_fullscreen_event(episode_rng, cfg)
            if episode_rng.random() < cfg.click_probability:
                self.clicks.append(ClickState(
                    id=f"click-{self._episode_serial}-{self.time_ms}",
                    button=episode_rng.choice(("left", "right", "middle")),
                    x=self.pet.foot_x + episode_rng.randint(-30, 30),
                    y=self.pet.foot_y + episode_rng.randint(-30, 30),
                    target="pet",
                    timestamp_ms=self.time_ms,
                ))

            world = self._build_world_state()
            condition_history.append(self._encode_world_state(world))
            if len(condition_history) > self.context_steps:
                condition_history.pop(0)

            plan_seed = episode_rng.randint(0, 2**31 - 1)
            plan = backend.generate(world, plan_seed, self.time_ms)
            self._validate_training_plan(plan, world)
            extra_metadata: Mapping[str, Any] = {}
            if sample_metadata_provider is not None:
                extra_metadata = sample_metadata_provider(plan.plan_id)
                if not isinstance(extra_metadata, Mapping):
                    raise ValueError("sample metadata provider must return a mapping")
            if len(condition_history) == self.context_steps:
                sample = self._training_sample_for_plan(
                    condition_history,
                    plan,
                    world,
                    episode_seed,
                )
                collisions = set(sample.metadata).intersection(extra_metadata)
                if collisions:
                    raise ValueError(
                        "sample metadata provider attempted to replace fields: "
                        f"{sorted(collisions)}"
                    )
                sample.metadata.update(extra_metadata)
                samples.append(sample)

            self._advance_plan_to_next_world_state(
                plan,
                backend=backend,
                origin_x=world.pet.foot_x,
                origin_y=world.pet.foot_y,
            )
            self.clicks = []

        return samples

    def _selected_character_rig(self) -> SelectedCharacterRig:
        if self.character_rig is None:
            self.character_rig = load_selected_character_rig()
        return self.character_rig

    def _validate_training_plan(self, plan: MotionPlan, world: WorldState) -> None:
        def is_integer(value: Any) -> bool:
            return isinstance(value, int) and not isinstance(value, bool)

        def finite_number(value: Any) -> bool:
            return (
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(value)
            )

        def bounded(value: Any, low: float, high: float) -> bool:
            return finite_number(value) and low <= float(value) <= high

        if not isinstance(plan.plan_id, str) or not 1 <= len(plan.plan_id) <= 128:
            raise ValueError("teacher plan_id must contain 1 to 128 characters")
        if plan.behavior not in ALLOWED_BEHAVIORS:
            raise ValueError(f"teacher plan has unsupported behavior {plan.behavior!r}")
        if plan.based_on_seq != world.seq:
            raise ValueError(
                f"teacher plan based_on_seq {plan.based_on_seq} does not match world {world.seq}"
            )
        if plan.generated_at_ms != world.timestamp_ms:
            raise ValueError(
                "teacher plan generated_at_ms does not match its triggering world"
            )
        if not is_integer(plan.based_on_seq) or plan.based_on_seq < 0:
            raise ValueError("teacher plan based_on_seq must be a non-negative integer")
        if not is_integer(plan.generated_at_ms) or plan.generated_at_ms < 0:
            raise ValueError("teacher plan generated_at_ms must be a non-negative integer")
        if not is_integer(plan.valid_until_ms) or plan.valid_until_ms < 0:
            raise ValueError("teacher plan valid_until_ms must be a non-negative integer")
        if plan.valid_until_ms > plan.generated_at_ms + 5_000:
            raise ValueError("teacher plan validity exceeds the protocol five-second ceiling")
        if not is_integer(plan.dt_ms) or not 8 <= plan.dt_ms <= 250:
            raise ValueError("teacher plan dt_ms is outside the protocol range")
        if plan.dt_ms != self.plan_dt_ms:
            raise ValueError(
                f"teacher plan dt_ms {plan.dt_ms} does not match simulator {self.plan_dt_ms}"
            )
        if not bounded(plan.confidence, 0.0, 1.0):
            raise ValueError("teacher plan confidence must be between 0 and 1")
        if not is_integer(plan.seed) or not 0 <= plan.seed <= 0xFFFF_FFFF:
            raise ValueError("teacher plan seed must be an unsigned 32-bit integer")
        if plan.target is not None:
            if not isinstance(plan.target, Mapping) or set(plan.target) != {
                "surface_id", "foot_x", "foot_y",
            }:
                raise ValueError("teacher plan target does not match the protocol shape")
            surface_id = plan.target["surface_id"]
            if not isinstance(surface_id, str) or not 1 <= len(surface_id) <= 128:
                raise ValueError("teacher plan target surface_id is invalid")
            if not finite_number(plan.target["foot_x"]) or not finite_number(plan.target["foot_y"]):
                raise ValueError("teacher plan target coordinates must be finite")
        if len(plan.points) != self.horizon_steps:
            raise ValueError(
                f"teacher plan has {len(plan.points)} points; expected {self.horizon_steps}"
            )
        expected_times = [index * self.plan_dt_ms for index in range(self.horizon_steps)]
        actual_times = [point.t_ms for point in plan.points]
        if actual_times != expected_times:
            raise ValueError(
                f"teacher plan times must be monotonic configured ticks: {actual_times!r}"
            )
        if plan.valid_until_ms <= plan.generated_at_ms + expected_times[-1]:
            raise ValueError("teacher plan expires before its final target tick")
        expected_locals = len(self._selected_character_rig().driven_joint_order)
        for index, point in enumerate(plan.points):
            if point.bone_rotations is not None:
                raise ValueError(
                    f"teacher point {index} contains legacy 2D bone_rotations in a 3D dataset"
                )
            if point.root_translation is None or len(point.root_translation) != 3:
                raise ValueError(f"teacher point {index} has no 3D root translation")
            if point.root_rotation is None or len(point.root_rotation) != 4:
                raise ValueError(f"teacher point {index} has no 3D root rotation")
            if (
                point.local_rotation_deltas is None
                or len(point.local_rotation_deltas) != expected_locals
                or any(len(quaternion) != 4 for quaternion in point.local_rotation_deltas)
            ):
                actual = (
                    None
                    if point.local_rotation_deltas is None
                    else len(point.local_rotation_deltas)
                )
                raise ValueError(
                    f"teacher point {index} has {actual} local rotations; "
                    f"selected rig requires {expected_locals}"
                )
            if not is_integer(point.t_ms) or point.t_ms < 0:
                raise ValueError(f"teacher point {index} t_ms must be a non-negative integer")
            for name, value, low, high in (
                ("dx", point.dx, -16_384.0, 16_384.0),
                ("dy", point.dy, -16_384.0, 16_384.0),
                ("vx", point.vx, -20_000.0, 20_000.0),
                ("vy", point.vy, -20_000.0, 20_000.0),
                ("lean", point.lean, -1.0, 1.0),
                ("squash", point.squash, 0.5, 1.5),
                ("bob", point.bob, -48.0, 48.0),
            ):
                if not finite_number(value):
                    raise ValueError(f"teacher point {index} contains a non-finite motion value")
                if not low <= float(value) <= high:
                    raise ValueError(
                        f"teacher point {index} {name} is outside the protocol range"
                    )
            if point.facing not in (-1, 1):
                raise ValueError(f"teacher point {index} facing must be -1 or 1")
            if not isinstance(point.expression, str) or not 1 <= len(point.expression) <= 32:
                raise ValueError(f"teacher point {index} expression is invalid")
            if any(not finite_number(value) for value in point.root_translation):
                raise ValueError(f"teacher point {index} contains a non-finite motion value")
            quaternions = (point.root_rotation, *point.local_rotation_deltas)
            for quaternion_index, quaternion in enumerate(quaternions):
                if any(
                    not bounded(value, -1.0, 1.0)
                    for value in quaternion
                ):
                    raise ValueError(
                        f"teacher point {index} quaternion {quaternion_index} is outside the protocol range"
                    )
                norm = math.sqrt(sum(float(value) ** 2 for value in quaternion))
                if norm <= 1e-3 or abs(norm - 1.0) >= 1e-3:
                    raise ValueError(
                        f"teacher point {index} quaternion {quaternion_index} is not unit length"
                    )
            if point.facial_params is not None:
                if not isinstance(point.facial_params, Mapping):
                    raise ValueError(
                        f"teacher point {index} facial_params must be an object"
                    )
                unknown_fields = set(point.facial_params) - set(FACIAL_PARAMETER_RANGES)
                if unknown_fields:
                    raise ValueError(
                        f"teacher point {index} facial_params contains unknown fields: "
                        f"{sorted(unknown_fields)}"
                    )
                for name, value in point.facial_params.items():
                    low, high = FACIAL_PARAMETER_RANGES[name]
                    if not bounded(value, low, high):
                        raise ValueError(
                            f"teacher point {index} facial_params.{name} is outside the protocol range"
                        )

    def _training_sample_for_plan(
        self,
        condition_history: Sequence[dict[str, Any]],
        plan: MotionPlan,
        world: WorldState,
        episode_seed: int,
    ) -> TrainingSample:
        goal = self._goal_features(plan, world.pet.foot_y)
        condition = [{**frame, **goal} for frame in condition_history]
        target = [
            self._encode_plan_point_full(point, behavior=plan.behavior)
            for point in plan.points
        ]
        rig = self._selected_character_rig()
        quaternion_widths = {len(point["quaternions"]) for point in target}
        expected_width = (len(rig.driven_joint_order) + 1) * 4
        if quaternion_widths != {expected_width}:
            raise ValueError(
                f"teacher target quaternion widths {sorted(quaternion_widths)} "
                f"do not match selected rig width {expected_width}"
            )
        return TrainingSample(
            condition_frames=condition,
            target_poses=target,
            metadata={
                "seed": episode_seed,
                "time_ms": self.time_ms,
                "plan_id": plan.plan_id,
                "behavior": plan.behavior,
                "world_state_dt_ms": self.world_state_dt_ms,
                "plan_dt_ms": self.plan_dt_ms,
                "context_steps": self.context_steps,
                "horizon_steps": self.horizon_steps,
                "quaternion_count": expected_width // 4,
                "character_id": rig.character_id,
                "rig_id": rig.rig_id,
                "rig_fingerprint": rig.fingerprint,
                "rig_source": rig.source,
                "driven_joint_order": list(rig.driven_joint_order),
                "context_anchor": {
                    "foot_x": world.pet.foot_x,
                    "foot_y": world.pet.foot_y,
                },
            },
        )

    @staticmethod
    def _goal_features(plan: MotionPlan, anchor_y: float) -> dict[str, float]:
        if plan.behavior == "walk":
            behavior = (1.0, 0.0, 0.0)
        elif plan.behavior in {"jump", "falling", "landing"}:
            behavior = (0.0, 1.0, 0.0)
        else:
            behavior = (0.0, 0.0, 1.0)
        target_y = plan.target.get("foot_y") if plan.target else None
        relative_target_y = 0.0
        if isinstance(target_y, (int, float)) and math.isfinite(target_y):
            relative_target_y = (float(target_y) - anchor_y) / 200.0
        return {
            "goal_behavior_0": behavior[0],
            "goal_behavior_1": behavior[1],
            "goal_behavior_2": behavior[2],
            "goal_surface_y": relative_target_y,
        }

    def _random_window_move(self, rng: random.Random) -> _WindowMove | None:
        movable = [window for window in self.windows if not window.minimized and not window.maximized]
        if not movable:
            return None
        w = rng.choice(movable)
        old_x = float(w.x)
        old_y = float(w.y)
        old_width = float(w.width)
        dx = rng.randint(-60, 60)
        dy = rng.randint(-20, 20)
        d = next((d for d in self.displays if d.id == w.display_id), None)
        if d:
            minimum_y = int(d.work_y + math.ceil(MIN_WINDOW_TOP_Y_DIP * d.scale_factor))
            w.x = int(_clamp(
                int(w.x + dx),
                int(d.work_x),
                int(d.work_x + d.work_width - w.width),
            ))
            w.y = int(_clamp(
                int(w.y + dy),
                minimum_y,
                int(d.work_y + d.work_height - w.height),
            ))
            w.vx = (w.x - old_x) / self.world_state_dt_s
            w.vy = (w.y - old_y) / self.world_state_dt_s
            return _WindowMove(
                window_id=f"window-{w.win_id}",
                old_x=old_x,
                old_y=old_y,
                old_width=old_width,
                new_x=float(w.x),
                new_y=float(w.y),
            )
        return None

    @staticmethod
    def _random_window_event(rng: random.Random, config: ScenarioConfig) -> str | None:
        roll = rng.random()
        cumulative = 0.0
        for event, probability in (
            ("move", config.window_move_probability),
            ("minimize", config.window_minimize_probability),
            ("maximize", config.window_maximize_probability),
            ("restore", config.window_restore_probability),
        ):
            cumulative += probability
            if roll < cumulative:
                return event
        return None

    def _apply_window_transition(
        self,
        event: str,
        rng: random.Random,
    ) -> _WindowMove | None:
        """Apply one explicit window lifecycle transition for this tick."""
        if event == "move":
            return self._random_window_move(rng)
        if event == "restore":
            candidates = [window for window in self.windows if window.minimized or window.maximized]
        elif event in {"minimize", "maximize"}:
            candidates = [
                window for window in self.windows
                if not window.minimized and not window.maximized
            ]
        else:
            raise ValueError(f"unsupported simulator window event: {event}")
        if not candidates:
            return None

        window = rng.choice(candidates)
        old_x = float(window.x)
        old_y = float(window.y)
        old_width = float(window.width)
        if event == "minimize":
            window.restore_bounds = (window.x, window.y, window.width, window.height)
            window.minimized = True
            window.vx = 0.0
            window.vy = 0.0
        elif event == "maximize":
            display = next(
                (candidate for candidate in self.displays if candidate.id == window.display_id),
                None,
            )
            if display is None:
                return None
            window.restore_bounds = (window.x, window.y, window.width, window.height)
            window.x = int(display.work_x)
            window.y = int(display.work_y)
            window.width = int(display.work_width)
            window.height = int(display.work_height)
            window.maximized = True
        else:
            if window.restore_bounds is not None:
                window.x, window.y, window.width, window.height = window.restore_bounds
            window.minimized = False
            window.maximized = False
            window.restore_bounds = None
        window.vx = (window.x - old_x) / self.world_state_dt_s
        window.vy = (window.y - old_y) / self.world_state_dt_s
        return _WindowMove(
            window_id=f"window-{window.win_id}",
            old_x=old_x,
            old_y=old_y,
            old_width=old_width,
            new_x=float(window.x),
            new_y=float(window.y),
        )

    def _random_fullscreen_event(
        self,
        rng: random.Random,
        config: ScenarioConfig,
    ) -> str | None:
        if self.scene.fullscreen_active:
            if rng.random() >= config.fullscreen_restore_probability:
                return None
            self.scene = SceneState(
                fullscreen_active=False,
                pet_allowed=True,
                suspend_reason=None,
            )
            return "fullscreen_restore"
        if rng.random() >= config.fullscreen_probability:
            return None
        self.scene = SceneState(
            fullscreen_active=True,
            pet_allowed=False,
            suspend_reason="fullscreen",
        )
        return "fullscreen_enter"

    def _reattach_pet_after_window_move(
        self,
        move: _WindowMove,
        previous_surfaces: Sequence[SurfaceState],
    ) -> bool:
        previous = next(
            (surface for surface in previous_surfaces if surface.id == self.pet.surface_id),
            None,
        )
        if previous is None or previous.kind != "window_top" or previous.window_id is None:
            return False
        carrier = next(
            (window for window in self.windows if f"window-{window.win_id}" == previous.window_id),
            None,
        )
        if carrier is None or carrier.minimized:
            self.pet.surface_id = None
            self.pet.behavior = "falling"
            self.pet.vx = 0.0
            self.pet.vy = max(0.0, self.pet.vy)
            return False
        carrier_moved = previous.window_id == move.window_id
        old_x = move.old_x if carrier_moved else float(carrier.x)
        old_width = move.old_width if carrier_moved else float(carrier.width)
        new_x = move.new_x if carrier_moved else float(carrier.x)
        relative_x = 0.5
        if old_width > 0:
            relative_x = _clamp((self.pet.foot_x - old_x) / old_width, 0.0, 1.0)
        preferred_x = new_x + relative_x * float(carrier.width)
        candidates: list[tuple[float, SurfaceState, float]] = []
        for surface in self.surfaces:
            if (
                surface.kind != "window_top"
                or surface.window_id != previous.window_id
                or not surface.enabled
                or surface.occluded
            ):
                continue
            display = next(
                (candidate for candidate in self.displays if candidate.id == surface.display_id),
                None,
            )
            half_support = LANDING_HALF_FOOT_DIP * (display.scale_factor if display else 1.0)
            safe_x1 = surface.x1 + half_support
            safe_x2 = surface.x2 - half_support
            if safe_x2 < safe_x1:
                continue
            x = _clamp(preferred_x, safe_x1, safe_x2)
            candidates.append((abs(x - preferred_x), surface, x))
        if not candidates:
            self.pet.surface_id = None
            self.pet.behavior = "falling"
            self.pet.vx = 0.0
            self.pet.vy = max(0.0, self.pet.vy)
            return False
        _, surface, x = min(candidates, key=lambda candidate: candidate[0])
        self.pet.foot_x = x
        self.pet.foot_y = surface.y
        self.pet.surface_id = surface.id
        # A carrier translation changes the coordinate frame, not the pet's
        # locomotion velocity.  Preserve the in-progress motion exactly as the
        # desktop host does when it rebases a grounded plan.
        self._sync_pet_display_size()
        return True

    # ── Encoding helpers (match neural-motion-model.md §3) ────────────

    def _encode_world_state(self, world: WorldState) -> dict[str, Any]:
        """Encode WorldState into the 48-dim feature vector expected by the model."""
        pet = world.pet
        # Pet state (12 dims: §3.1)
        # surface_id hash (4 dims): deterministic bucket from id string
        surf_hash = [0.0] * 4
        if pet.surface_id:
            # FNV-1a 32-bit hash — deterministic across processes.
            h = 0x811c9dc5
            for c in pet.surface_id:
                h = ((h ^ ord(c)) * 0x01000193) & 0xFFFFFFFF
            bucket = h & 0x3  # 4 buckets
            surf_hash[bucket] = 1.0
        # current surface y-diff (1 dim)
        surf_y_diff = 0.0
        if pet.surface_id:
            surf = next((s for s in world.surfaces if s.id == pet.surface_id), None)
            if surf:
                surf_y_diff = (pet.foot_y - surf.y) / 100.0
        pet_feats = {
            "foot_x": pet.foot_x / 2000.0,
            "foot_y": pet.foot_y / 2000.0,
            "vx": _clamp(pet.vx / 2000.0, -1, 1),
            "vy": _clamp(pet.vy / 2000.0, -1, 1),
            "facing": float(pet.facing),
            "behavior": {"idle": 0, "walk": 1, "jump": 2, "click_reaction": 3,
                          "falling": 4, "landing": 5, "hidden": 6, "fallback": 7}.get(pet.behavior, 0),
            "on_surface": 1.0 if pet.surface_id else 0.0,
            "surf_hash_0": surf_hash[0], "surf_hash_1": surf_hash[1],
            "surf_hash_2": surf_hash[2], "surf_hash_3": surf_hash[3],
            "surf_y_diff": surf_y_diff,
        }
        # Nearest 4 surfaces (5 dims each)
        usable = [s for s in world.surfaces if s.enabled and not s.occluded]
        usable.sort(key=lambda s: abs(s.y - pet.foot_y))
        for i in range(4):
            prefix = f"surf{i}"
            if i < len(usable):
                s = usable[i]
                pet_feats[f"{prefix}_dy"] = (s.y - pet.foot_y) / 200.0
                pet_feats[f"{prefix}_x1_rel"] = (s.x1 - pet.foot_x) / 200.0
                pet_feats[f"{prefix}_x2_rel"] = (s.x2 - pet.foot_x) / 200.0
                pet_feats[f"{prefix}_kind"] = 1.0 if s.kind == "window_top" else 0.0
                pet_feats[f"{prefix}_width"] = math.log(s.x2 - s.x1 + 1) / math.log(2000)
            else:
                pet_feats[f"{prefix}_dy"] = 0.0
                pet_feats[f"{prefix}_x1_rel"] = 0.0
                pet_feats[f"{prefix}_x2_rel"] = 0.0
                pet_feats[f"{prefix}_kind"] = 0.0
                pet_feats[f"{prefix}_width"] = 0.0
        # Cursor (4 dims: §3.3)
        if world.cursor:
            pet_feats["cur_x"] = (world.cursor.x - pet.foot_x) / 200.0
            pet_feats["cur_y"] = (world.cursor.y - pet.foot_y) / 200.0
            pet_feats["cur_over"] = 0.0  # simulator has no hit-test; always 0
            pet_feats["cur_down"] = 1.0 if world.cursor.left_down else 0.0
        else:
            pet_feats["cur_x"] = pet_feats["cur_y"] = pet_feats["cur_over"] = pet_feats["cur_down"] = 0.0
        # Click (4 dims: §3.3)
        if world.clicks:
            c = world.clicks[-1]
            pending = min(len(world.clicks), 3)
            pet_feats["click_pending"] = pending / 3.0
            pet_feats["click_age"] = min((world.timestamp_ms - c.timestamp_ms) / 500.0, 1.0)
            pet_feats["click_x"] = (c.x - pet.foot_x) / 200.0
            pet_feats["click_y"] = (c.y - pet.foot_y) / 200.0
        else:
            pet_feats["click_pending"] = pet_feats["click_age"] = pet_feats["click_x"] = pet_feats["click_y"] = 0.0
        # Scene (4 dims: §3.4)
        pet_feats["pet_allowed"] = 1.0 if world.scene.pet_allowed else 0.0
        pet_feats["fullscreen"] = 1.0 if world.scene.fullscreen_active else 0.0
        pet_feats["gen_status"] = 1.0  # simulator always 'ready'
        pet_feats["time_phase"] = math.sin(world.timestamp_ms / 1000.0)
        # Behavior goal (4 dims: §3.5) — overlaid from each teacher plan per sample.
        pet_feats["goal_behavior_0"] = 0.0
        pet_feats["goal_behavior_1"] = 0.0
        pet_feats["goal_behavior_2"] = 0.0
        pet_feats["goal_surface_y"] = 0.0
        if tuple(pet_feats) != CONDITION_FEATURE_ORDER:
            raise RuntimeError(
                "Encoded condition feature order drifted from CONDITION_FEATURE_ORDER"
            )
        return pet_feats

    def _encode_plan_point_full(self, point: Any, *, behavior: str) -> dict[str, Any]:
        """Extract ALL motion fields as training targets, not just skeletal placeholders."""
        p = point
        quats = []
        # root_rotation first, then local_rotation_deltas
        if p.root_rotation:
            quats.extend(p.root_rotation[:4])
        else:
            quats.extend([0.0, 0.0, 0.0, 1.0])
        if p.local_rotation_deltas:
            for q in p.local_rotation_deltas:
                quats.extend(q[:4])
        root = list(p.root_translation) if p.root_translation else [0.0, 0.0, 0.0]
        facial = [
            p.facial_params.get("eye_scale", 1.0) if p.facial_params else 1.0,
            p.facial_params.get("eye_squint", 0.0) if p.facial_params else 0.0,
            p.facial_params.get("mouth_open", 0.0) if p.facial_params else 0.0,
            p.facial_params.get("ear_angle", 0.0) if p.facial_params else 0.0,
            p.facial_params.get("brow_tilt", 0.0) if p.facial_params else 0.0,
        ]
        encoded = {
            "t_ms": p.t_ms,
            "behavior": behavior,
            "dx": p.dx,
            "dy": p.dy,
            "vx": p.vx,
            "vy": p.vy,
            "facing": p.facing,
            "lean": p.lean,
            "squash": p.squash,
            "bob": p.bob,
            "expression": p.expression,
            "quaternions": quats,
            "root": root,
            "facial": facial,
        }
        if tuple(encoded) != TARGET_RECORD_FIELD_ORDER:
            raise RuntimeError(
                "Encoded target field order drifted from TARGET_RECORD_FIELD_ORDER"
            )
        return encoded
