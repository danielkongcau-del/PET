"""Offline desktop environment for trajectory data generation.

Mirrors the host's geometry.ts collision and motion-controller.ts physics
so that teacher-generated plans produce physically consistent WorldState
sequences without requiring Electron or a real Windows desktop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import random
from typing import Any, Sequence

from .backend import MotionBackend, MotionPlan
from .state import (
    ClickState, CursorState, DisplayState, PetState, SceneState,
    SurfaceState, WorldState,
)


# ═══════════════════════════════════════════════════════════════════════
#  Physics constants — mirrored from desktop/src/motion-controller.ts
# ═══════════════════════════════════════════════════════════════════════

GRAVITY = 980.0                 # px / s^2
MAX_ROOT_SPEED = 2200.0         # px / s
PLAN_DT_MS = 33                 # ms between plan points
PLAN_HORIZON_MS = 400           # total plan horizon
TICK_DT_S = 33 / 1000.0         # physics tick in seconds


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
    half_foot_width: float = 12.0,
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
    anchor_x: float = 24, anchor_y: float = 46,
    window_w: float = 96, window_h: float = 96,
) -> tuple[float, float]:
    """Keep the pet window within one display's work area."""
    left = display.work_x + anchor_x
    right = display.work_x + display.work_width - (window_w - anchor_x)
    top = display.work_y + anchor_y
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
    click_probability: float = 0.005        # per tick


def _generate_display_state(
    display_id: int, is_primary: bool, rng: random.Random,
    scale: float | None = None,
) -> DisplayState:
    if scale is None:
        scale = rng.choice((1.0, 1.25, 1.5, 2.0))
    w = rng.randint(1366, 2560)
    h = rng.randint(768, 1440)
    # Leave some margin so windows can sit on the display.
    taskbar_h = int(48 * scale)
    return DisplayState(
        id=f"display-{display_id}",
        x=float(display_id * 2000), y=0.0,
        width=float(w), height=float(h),
        work_x=float(display_id * 2000),
        work_y=0.0,
        work_width=float(w),
        work_height=float(h - taskbar_h),
        scale_factor=scale,
    )


def _generate_surfaces(
    displays: Sequence[DisplayState],
    windows: list[SimWindow],
    min_surface_w: float = 48.0,
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
    for w in windows:
        if w.width < min_surface_w:
            continue
        surfaces.append(SurfaceState(
            id=f"window-{w.win_id}:top", kind="window_top",
            display_id=w.display_id, window_id=f"window-{w.win_id}",
            x1=float(w.x), x2=float(w.x + w.width),
            y=float(w.y),
            enabled=True, occluded=w.minimized,
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
    vx: float = 0.0
    vy: float = 0.0


@dataclass
class SimPet:
    foot_x: float
    foot_y: float
    vx: float = 0.0
    vy: float = 0.0
    facing: int = 1
    surface_id: str | None = None
    width: float = 96.0
    height: float = 96.0


# ═══════════════════════════════════════════════════════════════════════
#  Desktop Simulator
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class TrainingSample:
    """One (condition, target) pair for model training."""
    condition_frames: list[dict[str, Any]]   # K=8 encoded world states
    target_poses: list[dict[str, Any]]       # H=12 MotionPlan points
    metadata: dict[str, Any] = field(default_factory=dict)


class DesktopSimulator:
    """Offline desktop environment with deterministic physics."""

    def __init__(self, seed: int = 0):
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

    # ── Scene generation ──────────────────────────────────────────────

    def reset(self, config: ScenarioConfig | None = None) -> WorldState:
        cfg = config or ScenarioConfig()
        self.time_ms = 0
        self.seq = 0
        self.clicks = []

        # Displays
        n_displays = self.rng.randint(*cfg.num_displays)
        scales = cfg.display_scales
        self.displays = [
            _generate_display_state(i, i == 0, self.rng,
                                    self.rng.choice(scales) if scales else None)
            for i in range(n_displays)
        ]

        # Windows
        n_windows = self.rng.randint(*cfg.num_windows)
        self.windows = []
        for i in range(n_windows):
            d = self.rng.choice(self.displays)
            w = self.rng.randint(cfg.window_min_w, cfg.window_max_w)
            h = self.rng.randint(cfg.window_min_h, cfg.window_max_h)
            x = self.rng.randint(int(d.work_x), int(d.work_x + d.work_width - w))
            y = self.rng.randint(int(d.work_y + 48), int(d.work_y + d.work_height - h))
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
        if mode == "floor" or not self.windows:
            floor = next((s for s in self.surfaces if s.kind == "work_area_floor"), None)
            if floor:
                x = self.rng.uniform(floor.x1 + 48, floor.x2 - 48)
                self.pet = SimPet(foot_x=x, foot_y=floor.y, surface_id=floor.id)
                return
        elif mode == "window":
            windows = [s for s in self.surfaces if s.kind == "window_top"]
            if windows:
                s = self.rng.choice(windows)
                x = self.rng.uniform(s.x1 + 48, s.x2 - 48)
                self.pet = SimPet(foot_x=x, foot_y=s.y, surface_id=s.id)
                return
        # Fallback: use first floor
        floor = next((s for s in self.surfaces if s.kind == "work_area_floor"), None)
        if floor:
            x = self.rng.uniform(floor.x1 + 48, floor.x2 - 48)
            self.pet = SimPet(foot_x=x, foot_y=floor.y, surface_id=floor.id)
        else:
            self.pet = SimPet(foot_x=400.0, foot_y=700.0)

    # ── Physics tick ──────────────────────────────────────────────────

    def _apply_plan_to_pet(
        self, plan: MotionPlan, dt_s: float = TICK_DT_S,
    ) -> None:
        """Advance pet position using the first point(s) of a teacher plan."""
        if not plan.points:
            return
        point = plan.points[0]
        # Absolute target position = current foot + plan offset
        target_x = self.pet.foot_x + point.dx
        target_y = self.pet.foot_y + point.dy

        # Limit speed (mirror motion-controller.ts limitStep)
        dx = target_x - self.pet.foot_x
        dy = target_y - self.pet.foot_y
        dist = math.hypot(dx, dy)
        max_dist = MAX_ROOT_SPEED * dt_s
        if dist > max_dist and dist > 0:
            scale = max_dist / dist
            target_x = self.pet.foot_x + dx * scale
            target_y = self.pet.foot_y + dy * scale

        # Work area clamp
        display = self._display_for_point(target_x, target_y)
        if display:
            target_x, target_y = _clamp_to_work_area(target_x, target_y, display)

        # Detect landing
        landed = _find_crossed_surface(
            self.pet.foot_x, self.pet.foot_y, target_x, target_y, self.surfaces,
        )
        if landed:
            self.pet.foot_x = _clamp(target_x, landed.x1, landed.x2)
            self.pet.foot_y = landed.y
            self.pet.surface_id = landed.id
            self.pet.vy = 0.0
            return

        # Check surface support
        support = _nearest_supporting_surface(target_x, target_y, self.surfaces)
        if support:
            self.pet.foot_x = target_x
            self.pet.foot_y = support.y
            self.pet.surface_id = support.id
            self.pet.vy = 0.0
            return

        # No support — falling
        self.pet.vy += GRAVITY * dt_s
        target_y_fall = self.pet.foot_y + self.pet.vy * dt_s

        # Check if falling crosses a surface
        fall_landed = _find_crossed_surface(
            self.pet.foot_x, self.pet.foot_y, target_x, target_y_fall, self.surfaces,
        )
        if fall_landed:
            self.pet.foot_x = _clamp(target_x, fall_landed.x1, fall_landed.x2)
            self.pet.foot_y = fall_landed.y
            self.pet.surface_id = fall_landed.id
            self.pet.vy = 0.0
        else:
            self.pet.foot_x = target_x
            self.pet.foot_y = target_y_fall
            self.pet.surface_id = None

    def _display_for_point(self, x: float, y: float) -> DisplayState | None:
        for d in self.displays:
            if d.work_x <= x <= d.work_x + d.work_width and d.work_y <= y <= d.work_y + d.work_height:
                return d
        return self.displays[0] if self.displays else None

    # ── World state construction ──────────────────────────────────────

    def _build_world_state(self) -> WorldState:
        display = self._display_for_point(self.pet.foot_x, self.pet.foot_y)
        return WorldState(
            seq=self.seq,
            timestamp_ms=self.time_ms,
            session_id=self.session_id,
            coordinate_space="physical_px",
            displays=tuple(self.displays),
            surfaces=tuple(self.surfaces),
            pet=PetState(
                x=self.pet.foot_x - 48, y=self.pet.foot_y - 92,
                width=self.pet.width, height=self.pet.height,
                foot_x=self.pet.foot_x, foot_y=self.pet.foot_y,
                vx=self.pet.vx, vy=self.pet.vy,
                facing=self.pet.facing,
                behavior="idle",
                visible=True, user_dragging=False,
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
    ) -> list[TrainingSample]:
        """Run one episode and collect training samples."""
        cfg = config or ScenarioConfig()
        episode_rng = random.Random(episode_seed)
        self.rng = random.Random(episode_rng.randint(0, 2**31 - 1))

        world = self.reset(cfg)
        backend.cancel()
        backend.configure_timing(PLAN_HORIZON_MS, PLAN_DT_MS)
        if not hasattr(backend, '_skeletal_3d') or not getattr(backend, '_skeletal_3d'):
            backend.set_skeletal_3d(True)
            backend.set_skeletal_enabled(True)

        samples: list[TrainingSample] = []
        condition_buffer: list[dict[str, Any]] = []
        target_buffer: list[dict[str, Any]] = []

        plan: MotionPlan | None = None
        plan_consumed_ms = 0

        K = 8   # context frames
        H = 12  # horizon frames

        while self.time_ms < cfg.duration_ms:
            self.time_ms += PLAN_DT_MS
            self.seq += 1

            # Inject random events
            if self.rng.random() < cfg.window_move_probability:
                self._random_window_move(episode_rng)
                self.surfaces = _generate_surfaces(self.displays, self.windows)
            if self.rng.random() < cfg.click_probability:
                self.clicks.append(ClickState(
                    id=f"click-{self.time_ms}",
                    button=self.rng.choice(("left", "right", "middle")),
                    x=self.pet.foot_x + self.rng.randint(-30, 30),
                    y=self.pet.foot_y + self.rng.randint(-30, 30),
                    target="pet",
                    timestamp_ms=self.time_ms,
                ))

            world = self._build_world_state()

            # Get teacher plan
            plan_seed = episode_rng.randint(0, 2**31 - 1)
            generated_at = self.time_ms
            plan = backend.generate(world, plan_seed, generated_at)
            plan_consumed_ms = 0

            # Encode condition frame
            cond = self._encode_world_state(world)
            condition_buffer.append(cond)
            if len(condition_buffer) > K:
                condition_buffer.pop(0)

            # Apply plan to pet
            self._apply_plan_to_pet(plan)

            # Collect target pose from plan's first point
            if plan.points:
                target_buffer.append(self._encode_plan_point(plan.points[0]))
                if len(target_buffer) >= H:
                    if len(condition_buffer) >= K:
                        samples.append(TrainingSample(
                            condition_frames=list(condition_buffer),
                            target_poses=list(target_buffer),
                            metadata={"seed": episode_seed, "time_ms": self.time_ms},
                        ))
                    target_buffer = []

            # Clear consumed clicks
            self.clicks = []

        return samples

    def _random_window_move(self, rng: random.Random) -> None:
        if not self.windows:
            return
        w = rng.choice(self.windows)
        dx = rng.randint(-60, 60)
        dy = rng.randint(-20, 20)
        d = next((d for d in self.displays if d.id == w.display_id), None)
        if d:
            w.x = _clamp(int(w.x + dx), int(d.work_x), int(d.work_x + d.work_width - w.width))
            w.y = _clamp(int(w.y + dy), int(d.work_y + 48), int(d.work_y + d.work_height - w.height))

    # ── Encoding helpers (match neural-motion-model.md §3) ────────────

    def _encode_world_state(self, world: WorldState) -> dict[str, Any]:
        """Encode WorldState into the 48-dim feature vector expected by the model."""
        pet = world.pet
        # Pet state (12 dims: §3.1)
        # surface_id hash (4 dims): one-hot of 16 buckets
        surf_hash = [0.0] * 4
        if pet.surface_id:
            h = hash(pet.surface_id) & 0xF
            surf_hash[h >> 2] = 1.0  # 4 buckets, simplified from 16 one-hot
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
        # Behavior goal (4 dims: §3.5) — placeholder; training loop fills from teacher
        pet_feats["goal_behavior_0"] = 0.0
        pet_feats["goal_behavior_1"] = 0.0
        pet_feats["goal_behavior_2"] = 0.0
        pet_feats["goal_surface_y"] = 0.0
        return pet_feats

    def _encode_plan_point(self, point: Any) -> dict[str, Any]:
        """Extract the 48-dim target vector from a MotionPoint."""
        p = point
        quats = []
        if p.local_rotation_deltas:
            for q in p.local_rotation_deltas:
                quats.extend(q[:4])
        else:
            quats = [0.0] * 40  # 10 quaternions × 4
        root = list(p.root_translation) if p.root_translation else [0.0, 0.0, 0.0]
        facial = [
            p.facial_params.get("eye_scale", 1.0) if p.facial_params else 1.0,
            p.facial_params.get("eye_squint", 0.0) if p.facial_params else 0.0,
            p.facial_params.get("mouth_open", 0.0) if p.facial_params else 0.0,
            p.facial_params.get("ear_angle", 0.0) if p.facial_params else 0.0,
            p.facial_params.get("brow_tilt", 0.0) if p.facial_params else 0.0,
        ]
        return {
            "t_ms": p.t_ms,
            "quaternions": quats,
            "root": root,
            "facial": facial,
        }
