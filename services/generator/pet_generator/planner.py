"""Small stochastic autoregressive motion backend used for the first slice."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
import math
import random
from typing import Any

from .backend import MotionBackend, MotionPlan, MotionPoint
from .state import ClickState, SurfaceState, WorldState


@dataclass(frozen=True, slots=True)
class PlannerConfig:
    horizon_steps: int = 12
    dt_ms: int = 33
    walk_speed_min: float = 48.0
    walk_speed_max: float = 86.0
    walk_noise: float = 9.0
    velocity_memory: float = 0.72
    initial_jump_delay_min_ms: int = 1_800
    initial_jump_delay_max_ms: int = 3_500
    jump_delay_min_ms: int = 3_000
    jump_delay_max_ms: int = 7_000
    # A pet starts on the work-area floor, so the default reach must cover a
    # normal 1080p desktop window.  The duration calculation below stretches
    # these larger jumps across several rolling horizons.
    max_jump_distance: float = 1_100.0
    max_jump_vertical: float = 820.0
    support_tolerance: float = 20.0
    voluntary_drop_probability: float = 0.28
    # The desktop host confirms a new facing for 96 ms. Keep the root planted
    # for at least four default 33 ms samples so the cat visibly turns and
    # anticipates the jump before takeoff.
    jump_prepare_ms: int = 132

    def __post_init__(self) -> None:
        if self.horizon_steps < 2 or self.horizon_steps > 120:
            raise ValueError("horizon_steps must be between 2 and 120")
        if self.dt_ms < 8 or self.dt_ms > 250:
            raise ValueError("dt_ms must be between 8 and 250")
        if self.initial_jump_delay_min_ms > self.initial_jump_delay_max_ms:
            raise ValueError("invalid initial jump delay range")
        if self.jump_delay_min_ms > self.jump_delay_max_ms:
            raise ValueError("invalid jump delay range")
        if self.jump_prepare_ms < 0 or self.jump_prepare_ms > 1_000:
            raise ValueError("jump_prepare_ms must be between 0 and 1000")
        if not 0.0 <= self.voluntary_drop_probability <= 1.0:
            raise ValueError("voluntary_drop_probability must be between 0 and 1")


@dataclass(slots=True)
class _JumpState:
    source_surface_id: str | None
    target: SurfaceState
    start_ms: int
    prepare_ms: int
    duration_ms: int
    start_x: float
    start_y: float
    target_x: float
    arc_height: float
    facing: int


def _clamp(value: float, low: float, high: float) -> float:
    return min(max(value, low), high)


def _smoothstep(value: float) -> float:
    value = _clamp(value, 0.0, 1.0)
    return value * value * (3.0 - 2.0 * value)


class AutoregressiveMotionBackend(MotionBackend):
    """Generates a fresh short horizon from state and seeded stochastic noise.

    This is intentionally a plumbing backend, not a claim of learned motion.
    It supplies continuous, non-recorded trajectories while preserving the
    backend boundary that PetMotionDiffusion will later replace.
    """

    name = "autoregressive-v0"

    def __init__(self, config: PlannerConfig | None = None):
        self.config = config or PlannerConfig()
        self._session_id: str | None = None
        self._skeletal_enabled = False
        self._skeletal_3d = False
        self._model_driven_count = 0
        self._next_jump_due_ms: int | None = None
        self._active_jump: _JumpState | None = None
        self._walk_direction = 1
        self._walk_velocity = 0.0
        self._last_surface_id: str | None = None
        self._last_plan_id: str | None = None
        self._seen_click_ids: set[str] = set()
        self._seen_click_order: deque[str] = deque()
        self._pending_pet_clicks: deque[ClickState] = deque()
        self._pending_pet_click_ids: set[str] = set()
        self._generated_count = 0
        self._cancelled_count = 0

    def _reset_session(self, session_id: str) -> None:
        self._session_id = session_id
        self._next_jump_due_ms = None
        self._active_jump = None
        self._walk_direction = 1
        self._walk_velocity = 0.0
        self._last_surface_id = None
        self._last_plan_id = None
        self._seen_click_ids.clear()
        self._seen_click_order.clear()
        self._pending_pet_clicks.clear()
        self._pending_pet_click_ids.clear()

    def set_skeletal_enabled(self, enabled: bool) -> None:
        self._skeletal_enabled = enabled
        if enabled:
            self._load_skeletal_metadata()

    def set_skeletal_3d(self, enabled_3d: bool) -> None:
        """Called when 3D capability is specifically negotiated (not just 2D)."""
        self._skeletal_3d = enabled_3d

    def _load_skeletal_metadata(self) -> None:
        """Read cat-skeleton-3d.json to discover the expected 3D joint count."""
        try:
            import json
            from pathlib import Path
            skeleton_path = Path(__file__).resolve().parents[3] / "assets" / "pet" / "runtime" / "cat-skeleton-3d.json"
            if not skeleton_path.is_file():
                return
            data = json.loads(skeleton_path.read_text(encoding="utf-8"))
            joints = data.get("joints", [])
            model_driven = [
                j for j in joints
                if isinstance(j, dict)
                and not (j.get("deform") is False and j.get("parent") is None)  # exclude __motion_root__
                and j.get("poseDofs", {}).get("rotation") is True
                and j.get("physics", {}).get("mode") not in ("secondary", "static")
            ]
            self._model_driven_count = len(model_driven)
        except Exception:
            self._model_driven_count = 0

    def cancel(self, plan_id: str | None = None) -> bool:
        if plan_id is not None and plan_id != self._last_plan_id:
            return False
        was_active = self._active_jump is not None or self._last_plan_id is not None
        self._active_jump = None
        self._next_jump_due_ms = None
        self._walk_velocity = 0.0
        self._last_plan_id = None
        if was_active:
            self._cancelled_count += 1
        return was_active

    def configure_timing(self, plan_horizon_ms: int, plan_dt_ms: int) -> None:
        steps = int(round(plan_horizon_ms / plan_dt_ms))
        steps = max(2, min(128, steps))
        if steps == self.config.horizon_steps and plan_dt_ms == self.config.dt_ms:
            return
        self.config = replace(self.config, horizon_steps=steps, dt_ms=plan_dt_ms)
        self.cancel()

    def metrics(self) -> dict[str, Any]:
        return {
            "backend": self.name,
            "plans_generated": self._generated_count,
            "plans_cancelled": self._cancelled_count,
            "active_jump": self._active_jump is not None,
        }

    def generate(self, world: WorldState, seed: int, generated_at_ms: int) -> MotionPlan:
        if world.session_id != self._session_id:
            self._reset_session(world.session_id)
        rng = random.Random(seed)
        self._generated_count += 1

        if world.pet.user_dragging:
            self._active_jump = None
            plan = self._stationary_plan(world, seed, generated_at_ms, "fallback", "neutral", 1.0)
        elif not world.pet.visible or not world.scene.pet_allowed:
            self._active_jump = None
            plan = self._stationary_plan(world, seed, generated_at_ms, "hidden", "neutral", 1.0)
        else:
            click = self._next_pet_click(world.clicks)
            if click is not None:
                self._active_jump = None
                self._schedule_next_jump(world.timestamp_ms, rng)
                plan = self._reaction_plan(world, click, seed, generated_at_ms, rng)
            else:
                support = self._support_surface(world)
                plan = self._locomotion_plan(world, support, seed, generated_at_ms, rng)

        self._last_plan_id = plan.plan_id
        return plan

    def _next_pet_click(self, clicks: tuple[ClickState, ...]) -> ClickState | None:
        for click in sorted(clicks, key=lambda item: (item.timestamp_ms, item.id)):
            if click.id in self._seen_click_ids or click.id in self._pending_pet_click_ids:
                continue
            if click.target == "pet":
                self._pending_pet_clicks.append(click)
                self._pending_pet_click_ids.add(click.id)
            else:
                self._remember_click(click.id)

        # Bound adversarial input without marking ordinary queued clicks as
        # consumed before their reactions are generated.
        while len(self._pending_pet_clicks) > 256:
            dropped = self._pending_pet_clicks.popleft()
            self._pending_pet_click_ids.discard(dropped.id)
            self._remember_click(dropped.id)

        if not self._pending_pet_clicks:
            return None
        selected = self._pending_pet_clicks.popleft()
        self._pending_pet_click_ids.discard(selected.id)
        self._remember_click(selected.id)
        return selected

    def _remember_click(self, click_id: str) -> None:
        self._seen_click_ids.add(click_id)
        self._seen_click_order.append(click_id)
        while len(self._seen_click_order) > 256:
            stale = self._seen_click_order.popleft()
            self._seen_click_ids.discard(stale)

    def _support_surface(self, world: WorldState) -> SurfaceState | None:
        usable = [surface for surface in world.surfaces if surface.enabled and not surface.occluded]
        if world.pet.surface_id:
            direct = next((surface for surface in usable if surface.id == world.pet.surface_id), None)
            # Surface ids describe the host's last observation, not an
            # unconditional attachment.  A window may have moved since that
            # observation, so its current geometry must still support the pet.
            if direct is not None and self._surface_supports_pet(world, direct):
                return direct
        candidates = [surface for surface in usable if self._surface_supports_pet(world, surface)]
        if not candidates:
            return None
        return min(candidates, key=lambda surface: abs(surface.y - world.pet.foot_y))

    def _surface_supports_pet(self, world: WorldState, surface: SurfaceState) -> bool:
        tolerance = self.config.support_tolerance
        return (
            surface.x1 - tolerance <= world.pet.foot_x <= surface.x2 + tolerance
            and abs(surface.y - world.pet.foot_y) <= tolerance
        )

    def _locomotion_plan(
        self,
        world: WorldState,
        support: SurfaceState | None,
        seed: int,
        generated_at_ms: int,
        rng: random.Random,
    ) -> MotionPlan:
        active = self._active_jump
        if active is not None:
            target_still_exists = next(
                (
                    surface
                    for surface in world.surfaces
                    if surface.id == active.target.id and surface.enabled and not surface.occluded
                ),
                None,
            )
            intercepted = (
                support is not None
                and support.id not in {active.source_surface_id, active.target.id}
                and world.pet.surface_id == support.id
                and self._surface_supports_pet(world, support)
            )
            landed = (
                target_still_exists is not None
                and world.pet.surface_id == target_still_exists.id
                and self._surface_supports_pet(world, target_still_exists)
            )
            expired = world.timestamp_ms >= active.start_ms + active.prepare_ms + active.duration_ms + 100
            if target_still_exists is None or landed or intercepted or expired:
                self._active_jump = None
                self._schedule_next_jump(world.timestamp_ms, rng)
            else:
                active.target = target_still_exists
                return self._jump_plan(world, active, seed, generated_at_ms)

        if support is None:
            return self._fall_plan(world, seed, generated_at_ms)

        if self._next_jump_due_ms is None:
            self._next_jump_due_ms = world.timestamp_ms + rng.randint(
                self.config.initial_jump_delay_min_ms,
                self.config.initial_jump_delay_max_ms,
            )

        if world.timestamp_ms >= self._next_jump_due_ms:
            jump = self._choose_jump(world, support, rng)
            drop_rng = random.Random(seed ^ 0xD09F00D)
            drop = self._choose_drop(world, support, drop_rng)
            if drop is not None and (
                jump is None or drop_rng.random() < self.config.voluntary_drop_probability
            ):
                jump = drop
            if jump is not None:
                self._active_jump = jump
                return self._jump_plan(world, jump, seed, generated_at_ms)
            self._next_jump_due_ms = world.timestamp_ms + 1_000

        return self._walk_plan(world, support, seed, generated_at_ms, rng)

    def _schedule_next_jump(self, now_ms: int, rng: random.Random) -> None:
        self._next_jump_due_ms = now_ms + rng.randint(
            self.config.jump_delay_min_ms,
            self.config.jump_delay_max_ms,
        )

    def _choose_jump(self, world: WorldState, support: SurfaceState, rng: random.Random) -> _JumpState | None:
        margin = max(12.0, world.pet.width * 0.18)
        ranked: list[tuple[float, SurfaceState, float]] = []
        for candidate in world.surfaces:
            if (
                candidate.kind != "window_top"
                or candidate.id == support.id
                or not candidate.enabled
                or candidate.occluded
                or candidate.display_id != support.display_id
                or candidate.width < max(24.0, margin * 2.0)
            ):
                continue
            low = candidate.x1 + margin
            high = candidate.x2 - margin
            if high < low:
                landing_x = (candidate.x1 + candidate.x2) / 2.0
            else:
                nearest = _clamp(world.pet.foot_x, low, high)
                landing_x = _clamp(nearest + rng.uniform(-candidate.width * 0.18, candidate.width * 0.18), low, high)
            horizontal = abs(landing_x - world.pet.foot_x)
            vertical = abs(candidate.y - world.pet.foot_y)
            distance = math.hypot(horizontal, vertical)
            if (
                distance < 28.0
                or distance > self.config.max_jump_distance
                or vertical > self.config.max_jump_vertical
            ):
                continue
            ranked.append((distance * rng.uniform(0.88, 1.12), candidate, landing_x))
        if not ranked:
            return None
        _, target, target_x = min(ranked, key=lambda item: item[0])
        distance = math.hypot(target_x - world.pet.foot_x, target.y - world.pet.foot_y)
        arc = _clamp(
            55.0 + distance * 0.18 + max(0.0, world.pet.foot_y - target.y) * 0.16,
            58.0,
            210.0,
        )
        ascent = max(0.0, world.pet.foot_y - target.y)
        natural_duration_ms = 390.0 + distance * 0.72 + ascent * 0.18
        # The smoothstep component peaks at 1.5 * distance / duration and the
        # parabolic arc at 4 * arc / duration.  This conservative sum keeps
        # proposed root speed below the host's 2,200 physical-px/s limiter, so
        # a large jump does not permanently lag its internal clock.
        speed_limited_duration_ms = (1.5 * distance + 4.0 * arc) / 2_000.0 * 1_000.0
        duration = math.ceil(
            _clamp(max(natural_duration_ms, speed_limited_duration_ms), 430.0, 1_300.0)
        )
        horizontal_delta = target_x - world.pet.foot_x
        facing = world.pet.facing if abs(horizontal_delta) < 1.0 else (1 if horizontal_delta > 0 else -1)
        prepare_steps = math.ceil(self.config.jump_prepare_ms / self.config.dt_ms)
        return _JumpState(
            source_surface_id=support.id,
            target=target,
            start_ms=world.timestamp_ms,
            prepare_ms=prepare_steps * self.config.dt_ms,
            duration_ms=duration,
            start_x=world.pet.foot_x,
            start_y=world.pet.foot_y,
            target_x=target_x,
            arc_height=arc,
            facing=facing,
        )

    def _choose_drop(
        self,
        world: WorldState,
        support: SurfaceState,
        rng: random.Random,
    ) -> _JumpState | None:
        if support.kind != "window_top":
            return None
        floors = [
            surface
            for surface in world.surfaces
            if surface.kind == "work_area_floor"
            and surface.display_id == support.display_id
            and surface.enabled
            and not surface.occluded
            and surface.y > world.pet.foot_y + 28.0
        ]
        if not floors:
            return None
        target = min(floors, key=lambda surface: surface.y)
        margin = max(12.0, world.pet.width * 0.5)
        low = target.x1 + margin
        high = target.x2 - margin
        if high < low:
            return None
        vertical = target.y - world.pet.foot_y
        horizontal_step = _clamp(48.0 + vertical * 0.08, 48.0, 180.0)
        target_x = _clamp(
            world.pet.foot_x + world.pet.facing * horizontal_step + rng.uniform(-18.0, 18.0),
            low,
            high,
        )
        distance = math.hypot(target_x - world.pet.foot_x, vertical)
        speed_limited_duration_ms = 1.5 * distance / 1_800.0 * 1_000.0
        duration = math.ceil(
            _clamp(max(420.0 + distance * 0.55, speed_limited_duration_ms), 520.0, 3_200.0)
        )
        prepare_steps = math.ceil(self.config.jump_prepare_ms / self.config.dt_ms)
        horizontal_delta = target_x - world.pet.foot_x
        facing = world.pet.facing if abs(horizontal_delta) < 1.0 else (1 if horizontal_delta > 0 else -1)
        return _JumpState(
            source_surface_id=support.id,
            target=target,
            start_ms=world.timestamp_ms,
            prepare_ms=prepare_steps * self.config.dt_ms,
            duration_ms=duration,
            start_x=world.pet.foot_x,
            start_y=world.pet.foot_y,
            target_x=target_x,
            arc_height=0.0,
            facing=facing,
        )

    def _jump_plan(self, world: WorldState, jump: _JumpState, seed: int, generated_at_ms: int) -> MotionPlan:
        points: list[MotionPoint] = []
        elapsed_ms = max(0, world.timestamp_ms - jump.start_ms)
        duration = float(jump.duration_ms)
        delta_x = jump.target_x - jump.start_x
        delta_y = jump.target.y - jump.start_y
        motion_elapsed_ms = max(0, elapsed_ms - jump.prepare_ms)
        last_u = _clamp(motion_elapsed_ms / duration, 0.0, 1.0)

        for index in range(self.config.horizon_steps):
            t_ms = index * self.config.dt_ms
            total_elapsed_ms = elapsed_ms + t_ms
            trajectory_elapsed_ms = max(0, total_elapsed_ms - jump.prepare_ms)
            u = _clamp(trajectory_elapsed_ms / duration, 0.0, 1.0)
            eased = _smoothstep(u)
            desired_x = jump.start_x + delta_x * eased
            desired_y = jump.start_y + delta_y * eased - 4.0 * jump.arc_height * u * (1.0 - u)
            if total_elapsed_ms < jump.prepare_ms:
                vx = 0.0
                vy = 0.0
            elif u >= 1.0:
                vx = 0.0
                vy = 0.0
            else:
                smooth_derivative = 6.0 * u * (1.0 - u)
                du_per_second = 1000.0 / duration
                vx = delta_x * smooth_derivative * du_per_second
                vy = (delta_y * smooth_derivative - 4.0 * jump.arc_height * (1.0 - 2.0 * u)) * du_per_second
            edge_squash = math.exp(-((u - 0.04) / 0.08) ** 2) + math.exp(-((u - 0.96) / 0.08) ** 2)
            points.append(
                MotionPoint(
                    t_ms=t_ms,
                    dx=desired_x - world.pet.foot_x,
                    dy=desired_y - world.pet.foot_y,
                    vx=vx,
                    vy=vy,
                    facing=jump.facing,
                    lean=_clamp(vx / 480.0, -0.35, 0.35),
                    squash=_clamp(1.10 - 0.18 * edge_squash, 0.82, 1.12),
                    bob=0.0,
                    expression="focused" if u < 0.75 else "happy",
                )
            )
            last_u = u

        behavior = "jump" if last_u < 1.0 else "landing"
        return self._make_plan(
            world,
            seed,
            generated_at_ms,
            behavior,
            points,
            confidence=0.92,
            target={
                "surface_id": jump.target.id,
                "foot_x": round(jump.target_x, 4),
                "foot_y": round(jump.target.y, 4),
            },
        )

    def _walk_plan(
        self,
        world: WorldState,
        surface: SurfaceState,
        seed: int,
        generated_at_ms: int,
        rng: random.Random,
    ) -> MotionPlan:
        if surface.id != self._last_surface_id:
            self._last_surface_id = surface.id
            self._walk_direction = world.pet.facing
            self._walk_velocity = world.pet.vx

        margin = max(10.0, world.pet.width * 0.16)
        low = surface.x1 + margin
        high = surface.x2 - margin
        if high <= low:
            low = high = (surface.x1 + surface.x2) / 2.0
        if world.pet.foot_x <= low + 2.0:
            self._walk_direction = 1
        elif world.pet.foot_x >= high - 2.0:
            self._walk_direction = -1
        target_speed = self._walk_direction * rng.uniform(self.config.walk_speed_min, self.config.walk_speed_max)
        x = world.pet.foot_x
        velocity = self._walk_velocity
        dt_s = self.config.dt_ms / 1000.0
        points: list[MotionPoint] = []

        for index in range(self.config.horizon_steps):
            t_ms = index * self.config.dt_ms
            velocity = (
                self.config.velocity_memory * velocity
                + (1.0 - self.config.velocity_memory) * target_speed
                + rng.gauss(0.0, self.config.walk_noise)
            )
            if index > 0:
                x += velocity * dt_s
            if x < low:
                x = low + (low - x)
                velocity = abs(velocity)
                self._walk_direction = 1
                target_speed = abs(target_speed)
            elif x > high:
                x = high - (x - high)
                velocity = -abs(velocity)
                self._walk_direction = -1
                target_speed = -abs(target_speed)
            phase = (world.timestamp_ms + t_ms) / 150.0
            bob = abs(math.sin(phase * math.pi)) * 2.0
            points.append(
                MotionPoint(
                    t_ms=t_ms,
                    dx=x - world.pet.foot_x,
                    dy=surface.y - world.pet.foot_y,
                    vx=velocity,
                    vy=0.0,
                    facing=1 if velocity >= 0 else -1,
                    lean=_clamp(velocity / 520.0, -0.22, 0.22),
                    squash=1.0 + abs(math.sin(phase * math.pi)) * 0.035,
                    bob=bob,
                    expression="curious" if index % 5 == 0 else "neutral",
                )
            )
        self._walk_velocity = velocity
        return self._make_plan(
            world,
            seed,
            generated_at_ms,
            "walk",
            points,
            confidence=0.88,
            target={"surface_id": surface.id, "foot_x": round(x, 4), "foot_y": round(surface.y, 4)},
        )

    def _fall_plan(self, world: WorldState, seed: int, generated_at_ms: int) -> MotionPlan:
        floors = [
            surface
            for surface in world.surfaces
            if surface.kind == "work_area_floor"
            and surface.enabled
            and not surface.occluded
            and surface.x1 <= world.pet.foot_x <= surface.x2
            and surface.y >= world.pet.foot_y
        ]
        floor = min(floors, key=lambda item: item.y, default=None)
        gravity = 980.0
        points: list[MotionPoint] = []
        for index in range(self.config.horizon_steps):
            t_ms = index * self.config.dt_ms
            t_s = t_ms / 1000.0
            dy = world.pet.vy * t_s + 0.5 * gravity * t_s * t_s
            vy = world.pet.vy + gravity * t_s
            behavior_expression = "surprised"
            if floor is not None and world.pet.foot_y + dy >= floor.y:
                dy = floor.y - world.pet.foot_y
                vy = 0.0
                behavior_expression = "relieved"
            points.append(
                MotionPoint(
                    t_ms=t_ms,
                    dx=world.pet.vx * t_s,
                    dy=dy,
                    vx=world.pet.vx,
                    vy=vy,
                    facing=world.pet.facing,
                    lean=_clamp(world.pet.vx / 500.0, -0.25, 0.25),
                    squash=0.96,
                    bob=0.0,
                    expression=behavior_expression,
                )
            )
        target = (
            None
            if floor is None
            else {"surface_id": floor.id, "foot_x": world.pet.foot_x, "foot_y": floor.y}
        )
        return self._make_plan(world, seed, generated_at_ms, "falling", points, confidence=0.72, target=target)

    def _reaction_plan(
        self,
        world: WorldState,
        click: ClickState,
        seed: int,
        generated_at_ms: int,
        rng: random.Random,
    ) -> MotionPlan:
        reaction = rng.choice(("squash", "hop", "recoil", "turn"))
        support = self._support_surface(world)
        points: list[MotionPoint] = []
        duration_ms = self.config.horizon_steps * self.config.dt_ms
        facing = world.pet.facing
        if reaction == "turn":
            facing *= -1

        for index in range(self.config.horizon_steps):
            t_ms = index * self.config.dt_ms
            u = _clamp(t_ms / duration_ms, 0.0, 1.0)
            pulse = math.sin(math.pi * u)
            dx = 0.0
            dy = 0.0 if support is None else support.y - world.pet.foot_y
            vx = 0.0
            vy = 0.0
            lean = 0.0
            squash = 1.0
            bob = 0.0
            expression = "surprised"
            if reaction == "squash":
                squash = 1.0 - 0.34 * pulse
                bob = 2.0 * pulse
                expression = "annoyed"
            elif reaction == "hop":
                hop = 30.0 * pulse
                dy -= hop
                vy = -30.0 * math.pi / (duration_ms / 1000.0) * math.cos(math.pi * u)
                squash = 0.90 + 0.18 * pulse
                expression = "surprised"
            elif reaction == "recoil":
                distance = -world.pet.facing * 22.0 * _smoothstep(u)
                if support is not None:
                    margin = max(10.0, world.pet.width * 0.16)
                    target_x = _clamp(world.pet.foot_x + distance, support.x1 + margin, support.x2 - margin)
                    dx = target_x - world.pet.foot_x
                else:
                    dx = distance
                vx = -world.pet.facing * 22.0 * 6.0 * u * (1.0 - u) / (duration_ms / 1000.0)
                lean = -world.pet.facing * 0.28 * pulse
                expression = "annoyed"
            elif reaction == "turn":
                lean = facing * 0.12 * pulse
                expression = "curious"
            points.append(
                MotionPoint(
                    t_ms=t_ms,
                    dx=dx,
                    dy=dy,
                    vx=vx,
                    vy=vy,
                    facing=facing,
                    lean=lean,
                    squash=squash,
                    bob=bob,
                    expression=expression,
                )
            )
        return self._make_plan(
            world,
            seed,
            generated_at_ms,
            "click_reaction",
            points,
            confidence=0.96,
            target=None,
        )

    def _stationary_plan(
        self,
        world: WorldState,
        seed: int,
        generated_at_ms: int,
        behavior: str,
        expression: str,
        confidence: float,
    ) -> MotionPlan:
        points = tuple(
            MotionPoint(
                t_ms=index * self.config.dt_ms,
                dx=0.0,
                dy=0.0,
                vx=0.0,
                vy=0.0,
                facing=world.pet.facing,
                lean=0.0,
                squash=1.0,
                bob=0.0,
                expression=expression,
            )
            for index in range(self.config.horizon_steps)
        )
        return self._make_plan(world, seed, generated_at_ms, behavior, list(points), confidence=confidence)

    def _make_plan(
        self,
        world: WorldState,
        seed: int,
        generated_at_ms: int,
        behavior: str,
        points: list[MotionPoint],
        *,
        confidence: float,
        target: dict[str, Any] | None = None,
    ) -> MotionPlan:
        # When 3D skeletal motion is negotiated, emit identity quaternion deltas.
        # The learned model will later replace these with real predictions.
        if self._skeletal_3d and self._model_driven_count > 0:
            identity_quat = (0.0, 0.0, 0.0, 1.0)
            zero_rotations = tuple(identity_quat for _ in range(self._model_driven_count))
            default_facial = {"eye_scale": 1.0, "eye_squint": 0.0, "mouth_open": 0.0, "ear_angle": 0.0, "brow_tilt": 0.0}
            points = [
                replace(
                    point,
                    root_translation=(float(point.dx), float(-point.dy), 0.0),
                    root_rotation=(0.0, 0.0, 0.0, 1.0),
                    local_rotation_deltas=zero_rotations,
                    facial_params=dict(default_facial),
                )
                for point in points
            ]

        plan_id = f"ar-{world.seq}-{seed:016x}"
        validity_ms = max(250, self.config.dt_ms * (self.config.horizon_steps + 3))
        return MotionPlan(
            plan_id=plan_id,
            based_on_seq=world.seq,
            behavior=behavior,
            generated_at_ms=generated_at_ms,
            valid_until_ms=generated_at_ms + validity_ms,
            dt_ms=self.config.dt_ms,
            confidence=confidence,
            seed=seed,
            points=tuple(points),
            target=target,
        )
