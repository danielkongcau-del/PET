from __future__ import annotations

from dataclasses import asdict, replace
import json
from pathlib import Path
import tempfile
import unittest

from pet_generator.backend import MotionBackend, MotionPlan, MotionPoint
from pet_generator.character_rig import checkpoint_target_path, load_selected_character_rig
from pet_generator.data_gen import build_parser, ensure_dataset_manifest
from pet_generator.planner import AutoregressiveMotionBackend
from pet_generator.simulator import (
    DesktopSimulator,
    ScenarioConfig,
    SimPet,
    SimWindow,
    _clamp_to_work_area,
    _generate_surfaces,
)
from pet_generator.state import CursorState, DisplayState, SceneState, SurfaceState, WorldState


def point(
    t_ms: int,
    *,
    dx: float = 0.0,
    dy: float = 0.0,
    vx: float = 0.0,
    vy: float = 0.0,
    facing: int = 1,
    expression: str = "neutral",
    locals_count: int = 0,
) -> MotionPoint:
    identity = (0.0, 0.0, 0.0, 1.0)
    return MotionPoint(
        t_ms=t_ms,
        dx=dx,
        dy=dy,
        vx=vx,
        vy=vy,
        facing=facing,
        lean=0.0,
        squash=1.0,
        bob=0.0,
        expression=expression,
        root_translation=(0.0, 0.0, 0.0),
        root_rotation=identity,
        local_rotation_deltas=tuple(identity for _ in range(locals_count)),
    )


class FixedPlanBackend(MotionBackend):
    name = "fixed-plan-test"

    def __init__(
        self,
        behavior: str = "walk",
        target_dy: float = 0.0,
        locals_count: int | None = None,
    ):
        self.behavior = behavior
        self.target_dy = target_dy
        self.locals_count = (
            len(load_selected_character_rig().driven_joint_order)
            if locals_count is None
            else locals_count
        )
        self.dt_ms = 33
        self.horizon_steps = 12
        self.generated_worlds: list[WorldState] = []
        self.cancelled_plan_ids: list[str | None] = []

    def configure_timing(self, plan_horizon_ms: int, plan_dt_ms: int) -> None:
        self.dt_ms = plan_dt_ms
        self.horizon_steps = int(round(plan_horizon_ms / plan_dt_ms))

    def cancel(self, plan_id: str | None = None) -> bool:
        self.cancelled_plan_ids.append(plan_id)
        return False

    def generate(self, world: WorldState, seed: int, generated_at_ms: int) -> MotionPlan:
        self.generated_worlds.append(world)
        tag = f"seq-{world.seq}"
        points = tuple(
            point(
                index * self.dt_ms,
                dx=index * 2.0,
                dy=self.target_dy * index / max(1, self.horizon_steps - 1),
                vx=2_000.0 / self.dt_ms,
                vy=self.target_dy * 1_000.0 / max(1, self.horizon_steps - 1) / self.dt_ms,
                expression=f"{tag}:{index}",
                locals_count=self.locals_count,
            )
            for index in range(self.horizon_steps)
        )
        return MotionPlan(
            plan_id=f"plan-{world.seq}-{seed}",
            based_on_seq=world.seq,
            behavior=self.behavior,
            generated_at_ms=generated_at_ms,
            valid_until_ms=generated_at_ms + self.dt_ms * (self.horizon_steps + 3),
            dt_ms=self.dt_ms,
            confidence=1.0,
            seed=seed,
            points=points,
            target={
                "surface_id": world.pet.surface_id or "none",
                "foot_x": world.pet.foot_x + points[-1].dx,
                "foot_y": world.pet.foot_y + self.target_dy,
            },
        )


class CollisionAwareBackend(FixedPlanBackend):
    def generate(self, world: WorldState, seed: int, generated_at_ms: int) -> MotionPlan:
        self.behavior = "walk" if world.pet.surface_id == "middle" else "falling"
        self.target_dy = 0.0 if world.pet.surface_id == "middle" else 400.0
        return super().generate(world, seed, generated_at_ms)


class IntermediateCollisionSimulator(DesktopSimulator):
    def reset(self, config: ScenarioConfig | None = None) -> WorldState:
        self._selected_character_rig()
        self.time_ms = 0
        self.seq = 0
        self.clicks = []
        self.displays = [
            DisplayState("display", 0, 0, 1000, 900, 0, 0, 1000, 800, 1.0)
        ]
        self.windows = []
        self.surfaces = [
            SurfaceState("middle", "window_top", "display", "window-middle", 200, 800, 600, True, False),
            SurfaceState("floor", "work_area_floor", "display", None, 0, 1000, 800, True, False),
        ]
        # A plan emitted at t=50 first runs at the absolute t=64 motion tick.
        # Start close enough to land there, leaving t=80/96 to exercise the
        # same post-cancel fallback ticks as the online host.
        self.pet = SimPet(400, 590, surface_id=None)
        self.cursor = CursorState(400, 590, False)
        self.scene = SceneState(False, True, None)
        return self._build_world_state()


class SequenceRng:
    def __init__(self, dx: int, dy: int):
        self.values = iter((dx, dy))

    @staticmethod
    def choice(values):
        return values[0]

    def randint(self, low: int, high: int) -> int:
        value = next(self.values)
        if not low <= value <= high:
            raise AssertionError(f"test value {value} outside [{low}, {high}]")
        return value


class SimulatorTests(unittest.TestCase):
    @staticmethod
    def empty_config(**overrides) -> ScenarioConfig:
        values = {
            "num_displays": (1, 1),
            "num_windows": (0, 0),
            "duration_ms": 1_000,
            "window_move_probability": 0.0,
            "window_minimize_probability": 0.0,
            "window_maximize_probability": 0.0,
            "window_restore_probability": 0.0,
            "fullscreen_probability": 0.0,
            "fullscreen_restore_probability": 0.0,
            "click_probability": 0.0,
        }
        values.update(overrides)
        return ScenarioConfig(**values)

    def test_default_seed_42_no_longer_crashes(self) -> None:
        simulator = DesktopSimulator()
        samples = simulator.generate_episode(
            AutoregressiveMotionBackend(),
            ScenarioConfig(duration_ms=0),
            episode_seed=42,
        )
        self.assertEqual(samples, [])

    def test_plan_points_keep_one_fixed_origin(self) -> None:
        simulator = DesktopSimulator()
        simulator.reset(self.empty_config(duration_ms=0))
        origin_x = simulator.pet.foot_x
        origin_y = simulator.pet.foot_y

        simulator._apply_plan_point(origin_x, origin_y, point(33, dx=10, vx=10), "walk")
        simulator._apply_plan_point(origin_x, origin_y, point(66, dx=20, vx=10), "walk")

        self.assertAlmostEqual(simulator.pet.foot_x, origin_x + 20)
        self.assertAlmostEqual(simulator.pet.foot_y, origin_y)
        self.assertEqual(simulator.pet.behavior, "walk")

    def test_sample_targets_are_one_monotonic_plan_with_context_anchor(self) -> None:
        simulator = DesktopSimulator(context_steps=2, horizon_steps=4, dt_ms=25)
        backend = FixedPlanBackend()
        samples = simulator.generate_episode(
            backend,
            self.empty_config(duration_ms=150),
            episode_seed=7,
        )

        self.assertGreaterEqual(len(samples), 1)
        for sample in samples:
            self.assertEqual(len(sample.condition_frames), 2)
            self.assertEqual([target["t_ms"] for target in sample.target_poses], [0, 25, 50, 75])
            prefixes = {target["expression"].split(":", 1)[0] for target in sample.target_poses}
            self.assertEqual(len(prefixes), 1)
            self.assertEqual([target["dx"] for target in sample.target_poses], [0, 2, 4, 6])
            self.assertEqual(sample.target_poses[0]["dx"], 0)
            self.assertIn("context_anchor", sample.metadata)
            self.assertEqual(sample.metadata["world_state_dt_ms"], 50)
            self.assertEqual(sample.metadata["plan_dt_ms"], 25)
            self.assertEqual(
                sample.metadata["driven_joint_order"],
                list(load_selected_character_rig().driven_joint_order),
            )
            self.assertEqual(
                sample.metadata["quaternion_count"],
                len(load_selected_character_rig().driven_joint_order) + 1,
            )
            self.assertEqual(sample.metadata["plan_id"].split("-", 2)[1], prefixes.pop().split("-", 1)[1])

    def test_each_50ms_world_tick_replans_and_advances_continuous_plan_time(self) -> None:
        class PhaseRecordingSimulator(DesktopSimulator):
            def __init__(self, *args: object, **kwargs: object) -> None:
                super().__init__(*args, **kwargs)
                self.sample_offsets: list[tuple[int, int]] = []

            def _sample_plan_at(self, plan: MotionPlan, elapsed_ms: float) -> MotionPoint | None:
                self.sample_offsets.append((plan.generated_at_ms, round(elapsed_ms)))
                return super()._sample_plan_at(plan, elapsed_ms)

        simulator = PhaseRecordingSimulator(context_steps=2, horizon_steps=4, dt_ms=25)
        backend = FixedPlanBackend()
        consumed: list[str] = []

        samples = simulator.generate_episode(
            backend,
            self.empty_config(duration_ms=100),
            episode_seed=17,
            sample_metadata_provider=lambda plan_id: consumed.append(plan_id) or {},
        )

        self.assertEqual(len(backend.generated_worlds), 2)
        self.assertEqual(
            [world.timestamp_ms for world in backend.generated_worlds],
            [50, 100],
        )
        self.assertEqual(len(consumed), 2, "warm-up plans must release provenance immediately")
        self.assertEqual(len(samples), 1)
        self.assertAlmostEqual(
            simulator.pet.foot_x,
            backend.generated_worlds[0].pet.foot_x + 7.2,
        )
        self.assertAlmostEqual(
            backend.generated_worlds[1].pet.foot_x - backend.generated_worlds[0].pet.foot_x,
            3.68,
        )
        self.assertEqual(
            simulator.sample_offsets,
            [(50, 14), (50, 30), (50, 46), (100, 12), (100, 28), (100, 44)],
            "the absolute 16 ms timer phase must continue across 50 ms replans",
        )

    def test_plan_sampling_linearly_interpolates_wall_clock_phase(self) -> None:
        simulator = DesktopSimulator(context_steps=1, horizon_steps=4, dt_ms=25)
        world = simulator.reset(self.empty_config(duration_ms=0))
        backend = FixedPlanBackend()
        backend.configure_timing(100, 25)
        plan = backend.generate(world, 1, world.timestamp_ms)

        sampled = simulator._sample_plan_at(plan, 40)

        self.assertIsNotNone(sampled)
        self.assertAlmostEqual(sampled.dx, 3.2)
        self.assertEqual(sampled.t_ms, 40)

    def test_intermediate_landing_is_observed_before_next_horizon(self) -> None:
        simulator = IntermediateCollisionSimulator(context_steps=1, horizon_steps=3, dt_ms=100)
        backend = CollisionAwareBackend()

        simulator.generate_episode(
            backend,
            self.empty_config(duration_ms=200),
            episode_seed=29,
        )

        self.assertEqual(len(backend.generated_worlds), 4)
        self.assertEqual(backend.generated_worlds[1].pet.surface_id, "middle")
        self.assertEqual(backend.generated_worlds[1].pet.vy, 0.0)
        self.assertEqual(backend.generated_worlds[1].pet.behavior, "fallback")
        self.assertEqual(simulator.pet.surface_id, "middle")
        self.assertEqual(simulator.pet.foot_y, 600)
        self.assertEqual(len(backend.cancelled_plan_ids), 2)
        self.assertIsNone(backend.cancelled_plan_ids[0])
        self.assertRegex(backend.cancelled_plan_ids[1] or "", r"^plan-1-\d+$")

    def test_behavior_and_goal_labels_come_from_teacher_plan(self) -> None:
        simulator = DesktopSimulator(context_steps=2, horizon_steps=4, dt_ms=25)
        samples = simulator.generate_episode(
            FixedPlanBackend(behavior="jump", target_dy=-100.0),
            self.empty_config(duration_ms=150),
            episode_seed=8,
        )

        sample = samples[0]
        self.assertEqual(sample.metadata["behavior"], "jump")
        self.assertTrue(all(target["behavior"] == "jump" for target in sample.target_poses))
        self.assertTrue(all(frame["goal_behavior_1"] == 1.0 for frame in sample.condition_frames))
        self.assertTrue(all(frame["goal_surface_y"] == -0.5 for frame in sample.condition_frames))
        self.assertIn(2, {frame["behavior"] for frame in sample.condition_frames})

    def test_repeating_same_seed_is_episode_isolated(self) -> None:
        simulator = DesktopSimulator()
        backend = AutoregressiveMotionBackend()
        config = self.empty_config(duration_ms=1_000, click_probability=1.0)

        first = simulator.generate_episode(backend, config, episode_seed=123)
        second = simulator.generate_episode(backend, config, episode_seed=123)

        encode = lambda samples: json.dumps(
            [asdict(sample) for sample in samples], sort_keys=True, separators=(",", ":")
        )
        self.assertEqual(encode(first), encode(second))

    def test_same_seed_is_isolated_across_simulator_instances(self) -> None:
        backend = AutoregressiveMotionBackend()
        config = self.empty_config(duration_ms=1_000, click_probability=1.0)

        first = DesktopSimulator().generate_episode(backend, config, episode_seed=123)
        second = DesktopSimulator().generate_episode(backend, config, episode_seed=123)

        encode = lambda samples: json.dumps(
            [asdict(sample) for sample in samples], sort_keys=True, separators=(",", ":")
        )
        self.assertEqual(encode(first), encode(second))

    def test_timing_and_context_are_instance_local(self) -> None:
        first = DesktopSimulator(context_steps=3, horizon_steps=4, dt_ms=25)
        second = DesktopSimulator(context_steps=5, horizon_steps=2, dt_ms=50)
        first_backend = FixedPlanBackend()
        second_backend = FixedPlanBackend()

        first_samples = first.generate_episode(
            first_backend, self.empty_config(duration_ms=150), episode_seed=1,
        )
        second_samples = second.generate_episode(
            second_backend, self.empty_config(duration_ms=300), episode_seed=2,
        )

        self.assertEqual((first_backend.horizon_steps, first_backend.dt_ms), (4, 25))
        self.assertEqual((second_backend.horizon_steps, second_backend.dt_ms), (2, 50))
        self.assertTrue(first_samples)
        self.assertTrue(second_samples)
        self.assertEqual({len(sample.condition_frames) for sample in first_samples}, {3})
        self.assertEqual({len(sample.target_poses) for sample in first_samples}, {4})
        self.assertEqual({len(sample.condition_frames) for sample in second_samples}, {5})
        self.assertEqual({len(sample.target_poses) for sample in second_samples}, {2})
        with self.assertRaisesRegex(ValueError, "between 2 and 120"):
            DesktopSimulator(horizon_steps=1)
        with self.assertRaisesRegex(ValueError, "between 2 and 120"):
            DesktopSimulator(horizon_steps=121)

    def test_teacher_plan_must_match_rig_and_triggering_world(self) -> None:
        simulator = DesktopSimulator(context_steps=2, horizon_steps=2, dt_ms=25)
        world = simulator.reset(self.empty_config(duration_ms=0))
        backend = FixedPlanBackend()
        backend.configure_timing(50, 25)
        plan = backend.generate(world, 1, world.timestamp_ms)

        simulator._validate_training_plan(plan, world)
        with self.assertRaisesRegex(ValueError, "based_on_seq"):
            simulator._validate_training_plan(replace(plan, based_on_seq=world.seq + 1), world)
        with self.assertRaisesRegex(ValueError, "expires before"):
            simulator._validate_training_plan(
                replace(plan, valid_until_ms=plan.generated_at_ms + plan.points[-1].t_ms),
                world,
            )
        ragged = replace(
            plan,
            points=(plan.points[0], replace(plan.points[1], local_rotation_deltas=())),
        )
        with self.assertRaisesRegex(ValueError, "selected rig requires"):
            simulator._validate_training_plan(ragged, world)
        zero_quaternion = replace(
            plan,
            points=(replace(plan.points[0], root_rotation=(0.0, 0.0, 0.0, 0.0)), plan.points[1]),
        )
        with self.assertRaisesRegex(ValueError, "not unit length"):
            simulator._validate_training_plan(zero_quaternion, world)
        non_finite = replace(
            plan,
            points=(replace(plan.points[0], dx=float("nan")), plan.points[1]),
        )
        with self.assertRaisesRegex(ValueError, "non-finite motion"):
            simulator._validate_training_plan(non_finite, world)
        bad_facing = replace(
            plan,
            points=(replace(plan.points[0], facing=0), plan.points[1]),
        )
        with self.assertRaisesRegex(ValueError, "facing"):
            simulator._validate_training_plan(bad_facing, world)
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            replace(plan.points[0], bone_rotations=(0.0,))
        out_of_range = replace(
            plan,
            points=(replace(plan.points[0], dx=16_385.0), plan.points[1]),
        )
        with self.assertRaisesRegex(ValueError, "protocol range"):
            simulator._validate_training_plan(out_of_range, world)
        invalid_facial = replace(
            plan,
            points=(replace(plan.points[0], facial_params={
                "eye_scale": 1.0,
                "eye_squint": 0.0,
                "mouth_open": 1.1,
                "ear_angle": 0.0,
                "brow_tilt": 0.0,
            }), plan.points[1]),
        )
        with self.assertRaisesRegex(ValueError, "facial_params.mouth_open"):
            simulator._validate_training_plan(invalid_facial, world)
        partial_facial = replace(
            plan,
            points=(replace(plan.points[0], facial_params={"mouth_open": 0.5}), plan.points[1]),
        )
        simulator._validate_training_plan(partial_facial, world)
        unknown_facial = replace(
            plan,
            points=(replace(plan.points[0], facial_params={"unknown": 0.5}), plan.points[1]),
        )
        with self.assertRaisesRegex(ValueError, "unknown fields"):
            simulator._validate_training_plan(unknown_facial, world)
        non_finite_facial = replace(
            plan,
            points=(replace(plan.points[0], facial_params={"mouth_open": float("nan")}), plan.points[1]),
        )
        with self.assertRaisesRegex(ValueError, "facial_params.mouth_open"):
            simulator._validate_training_plan(non_finite_facial, world)

    def test_mixed_dpi_scales_anchor_and_pet_state(self) -> None:
        display = DisplayState("display", 0, 0, 2880, 1620, 0, 0, 2880, 1548, 1.5)
        self.assertEqual(_clamp_to_work_area(0, 0, display), (72.0, 138.0))

        simulator = DesktopSimulator()
        simulator.displays = [display]
        simulator.surfaces = [
            SurfaceState("display:floor", "work_area_floor", "display", None, 0, 2880, 1548, True, False)
        ]
        simulator.pet = SimPet(300, 1548, surface_id="display:floor")
        world = simulator._build_world_state()
        self.assertEqual((world.pet.width, world.pet.height), (144.0, 144.0))
        self.assertEqual((world.pet.x, world.pet.y), (228.0, 1410.0))

    def test_pet_geometry_comes_from_selected_character_manifest(self) -> None:
        rig = load_selected_character_rig()
        custom_render = dict(rig.raw["render"])
        custom_render["footAnchor"] = [10, 40]
        custom_rig = replace(rig, raw={**rig.raw, "render": custom_render})
        display = DisplayState("display", 0, 0, 2880, 1620, 0, 0, 2880, 1548, 1.5)
        simulator = DesktopSimulator(character_rig=custom_rig)
        simulator.displays = [display]
        simulator.surfaces = [
            SurfaceState("display:floor", "work_area_floor", "display", None, 0, 2880, 1548, True, False)
        ]
        simulator.pet = SimPet(300, 1548, surface_id="display:floor")

        world = simulator._build_world_state()
        self.assertEqual((world.pet.width, world.pet.height), (144.0, 144.0))
        self.assertEqual((world.pet.x, world.pet.y), (270.0, 1428.0))
        self.assertEqual(
            _clamp_to_work_area(
                0, 0, display, anchor_x=20, anchor_y=80, window_w=96, window_h=96,
            ),
            (30.0, 120.0),
        )

    def test_surface_width_threshold_scales_per_display(self) -> None:
        displays = [
            DisplayState("one", 0, 0, 1920, 1080, 0, 0, 1920, 1040, 1.0),
            DisplayState("one-half", 1920, 0, 1920, 1080, 1920, 0, 1920, 1020, 1.5),
        ]
        windows = [
            SimWindow(0, "one", 100, 100, 120, 200),
            SimWindow(1, "one-half", 2020, 100, 120, 200),
        ]

        tops = [surface for surface in _generate_surfaces(displays, windows) if surface.kind == "window_top"]

        self.assertEqual([surface.window_id for surface in tops], ["window-0"])

    def test_window_move_preserves_carrier_relative_position(self) -> None:
        simulator = DesktopSimulator(seed=2)
        simulator.reset(ScenarioConfig(
            num_displays=(1, 1),
            num_windows=(1, 1),
            window_min_w=400,
            window_max_w=400,
            window_min_h=200,
            window_max_h=200,
            pet_start="window",
        ))
        before_surfaces = tuple(simulator.surfaces)
        before_foot = (simulator.pet.foot_x, simulator.pet.foot_y)
        simulator.pet.vx = 37.0
        simulator.pet.vy = -4.0

        move = simulator._random_window_move(SequenceRng(60, 20))
        self.assertIsNotNone(move)
        simulator.surfaces = _generate_surfaces(simulator.displays, simulator.windows)
        assert move is not None
        self.assertTrue(simulator._reattach_pet_after_window_move(move, before_surfaces))

        self.assertAlmostEqual(simulator.pet.foot_x - before_foot[0], 60.0)
        self.assertAlmostEqual(simulator.pet.foot_y - before_foot[1], 20.0)
        self.assertEqual((simulator.pet.vx, simulator.pet.vy), (37.0, -4.0))
        self.assertIsNotNone(simulator.pet.surface_id)

    def test_window_move_releases_attachment_only_without_usable_segment(self) -> None:
        simulator = DesktopSimulator(seed=2)
        simulator.reset(ScenarioConfig(
            num_displays=(1, 1),
            num_windows=(1, 1),
            window_min_w=400,
            window_max_w=400,
            window_min_h=200,
            window_max_h=200,
            pet_start="window",
        ))
        before_surfaces = tuple(simulator.surfaces)
        simulator.pet.vx = 37.0
        simulator.pet.vy = -4.0

        move = simulator._random_window_move(SequenceRng(60, 20))
        assert move is not None
        simulator.surfaces = [
            surface for surface in _generate_surfaces(simulator.displays, simulator.windows)
            if surface.kind == "work_area_floor"
        ]

        self.assertFalse(simulator._reattach_pet_after_window_move(move, before_surfaces))
        self.assertIsNone(simulator.pet.surface_id)
        self.assertEqual(simulator.pet.behavior, "falling")
        self.assertEqual((simulator.pet.vx, simulator.pet.vy), (0.0, 0.0))

    def test_minimize_maximize_restore_and_fullscreen_transitions(self) -> None:
        simulator = DesktopSimulator(seed=2)
        simulator.reset(ScenarioConfig(
            num_displays=(1, 1),
            num_windows=(1, 1),
            window_min_w=400,
            window_max_w=400,
            window_min_h=200,
            window_max_h=200,
            pet_start="window",
        ))
        original_bounds = (
            simulator.windows[0].x,
            simulator.windows[0].y,
            simulator.windows[0].width,
            simulator.windows[0].height,
        )
        previous_surfaces = tuple(simulator.surfaces)
        move = simulator._apply_window_transition("minimize", simulator.rng)
        assert move is not None
        simulator.surfaces = _generate_surfaces(simulator.displays, simulator.windows)
        self.assertFalse(simulator._reattach_pet_after_window_move(move, previous_surfaces))
        self.assertTrue(simulator.windows[0].minimized)
        self.assertEqual(simulator.pet.behavior, "falling")

        simulator._apply_window_transition("restore", simulator.rng)
        simulator.surfaces = _generate_surfaces(simulator.displays, simulator.windows)
        self.assertEqual(
            (
                simulator.windows[0].x,
                simulator.windows[0].y,
                simulator.windows[0].width,
                simulator.windows[0].height,
            ),
            original_bounds,
        )
        self.assertTrue(any(surface.window_id == "window-0" for surface in simulator.surfaces))

        maximize = simulator._apply_window_transition("maximize", simulator.rng)
        self.assertIsNotNone(maximize)
        simulator.surfaces = _generate_surfaces(simulator.displays, simulator.windows)
        self.assertTrue(simulator.windows[0].maximized)
        self.assertFalse(any(surface.window_id == "window-0" for surface in simulator.surfaces))
        simulator._apply_window_transition("restore", simulator.rng)
        self.assertEqual(
            (
                simulator.windows[0].x,
                simulator.windows[0].y,
                simulator.windows[0].width,
                simulator.windows[0].height,
            ),
            original_bounds,
        )

        fullscreen_config = self.empty_config(
            fullscreen_probability=1.0,
            fullscreen_restore_probability=1.0,
        )
        self.assertEqual(
            simulator._random_fullscreen_event(simulator.rng, fullscreen_config),
            "fullscreen_enter",
        )
        self.assertFalse(simulator._build_world_state().pet.visible)
        self.assertEqual(
            simulator._random_fullscreen_event(simulator.rng, fullscreen_config),
            "fullscreen_restore",
        )
        self.assertTrue(simulator._build_world_state().pet.visible)

    def test_foreground_move_reclamps_pet_on_its_unchanged_carrier(self) -> None:
        display = DisplayState("display", 0, 0, 1920, 1080, 0, 0, 1920, 1040, 1.0)
        simulator = DesktopSimulator()
        simulator.displays = [display]
        simulator.windows = [
            SimWindow(0, "display", 500, 100, 200, 200),
            SimWindow(1, "display", 100, 100, 500, 200),
        ]
        simulator.surfaces = _generate_surfaces(simulator.displays, simulator.windows)
        carrier = next(surface for surface in simulator.surfaces if surface.window_id == "window-1")
        simulator.pet = SimPet(450, carrier.y, surface_id=carrier.id)
        before_surfaces = tuple(simulator.surfaces)

        move = simulator._random_window_move(SequenceRng(-60, 0))
        assert move is not None
        simulator.surfaces = _generate_surfaces(simulator.displays, simulator.windows)

        self.assertTrue(simulator._reattach_pet_after_window_move(move, before_surfaces))
        self.assertEqual(simulator.pet.surface_id, "window-1:top:0")
        self.assertAlmostEqual(simulator.pet.foot_x, 428.0)

    def test_fully_occluded_window_has_no_surface(self) -> None:
        display = DisplayState("display", 0, 0, 1920, 1080, 0, 0, 1920, 1040, 1.0)
        windows = [
            SimWindow(0, "display", 100, 100, 500, 300),
            SimWindow(1, "display", 100, 100, 500, 300),
        ]

        tops = [surface for surface in _generate_surfaces([display], windows) if surface.kind == "window_top"]

        self.assertEqual([surface.window_id for surface in tops], ["window-0"])

    def test_ninety_percent_occluded_window_keeps_only_disabled_segments(self) -> None:
        display = DisplayState("display", 0, 0, 1920, 1080, 0, 0, 1920, 1040, 1.0)
        windows = [
            SimWindow(0, "display", 100, 100, 1260, 300),
            SimWindow(1, "display", 100, 100, 1400, 300),
        ]

        lower = [
            surface
            for surface in _generate_surfaces([display], windows)
            if surface.window_id == "window-1"
        ]

        self.assertEqual(len(lower), 1)
        self.assertFalse(lower[0].enabled)
        self.assertTrue(lower[0].occluded)

    def test_target_quaternion_count_comes_from_teacher(self) -> None:
        simulator = DesktopSimulator()
        encoded_two = simulator._encode_plan_point_full(point(0, locals_count=2), behavior="idle")
        encoded_none = simulator._encode_plan_point_full(point(0, locals_count=0), behavior="idle")

        self.assertEqual(len(encoded_two["quaternions"]), 12)
        self.assertEqual(len(encoded_none["quaternions"]), 4)

    def test_data_gen_parser_exposes_context_steps(self) -> None:
        args = build_parser().parse_args([
            "--context-steps", "5",
            "--horizon-steps", "7",
            "--dt-ms", "40",
            "--world-state-dt-ms", "60",
        ])
        self.assertEqual(
            (args.context_steps, args.horizon_steps, args.dt_ms, args.world_state_dt_ms),
            (5, 7, 40, 60),
        )

    def test_dataset_manifest_binds_output_to_rig_and_shape(self) -> None:
        rig = load_selected_character_rig()
        teacher_provenance = {
            "teacherBackend": "fixed-plan-test",
            "poseSource": "character_animation",
            "poseSpace": "rest_local_delta",
            "poseSelector": "test-world-time-selector-v1",
            "clipFingerprints": [
                {"clipId": "test-clip", "fingerprint": "c" * 64},
            ],
            "unsupportedPoseChannels": ["localTranslations"],
        }
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            path = ensure_dataset_manifest(
                output, rig, context_steps=8, horizon_steps=12, dt_ms=33,
                teacher_provenance=teacher_provenance,
            )
            payload = json.loads(path.read_text(encoding="utf-8"))

            simulator = DesktopSimulator(seed=0)
            world = simulator.reset(
                ScenarioConfig(num_displays=(1, 1), num_windows=(0, 0))
            )
            encoded_condition = simulator._encode_world_state(world)
            encoded_target = simulator._encode_plan_point_full(
                point(0, locals_count=len(rig.driven_joint_order)),
                behavior="idle",
            )

            self.assertEqual(payload["characterId"], rig.character_id)
            self.assertEqual(payload["rigFingerprint"], rig.fingerprint)
            self.assertEqual(payload["drivenJointOrder"], list(rig.driven_joint_order))
            self.assertEqual(
                payload["checkpointTarget"],
                {
                    "format": rig.checkpoint.format,
                    "metadataSchema": rig.checkpoint.metadata_schema,
                    "path": rig.checkpoint.path,
                    "characterId": rig.character_id,
                    "rigFingerprint": rig.fingerprint,
                    "drivenJointOrder": list(rig.driven_joint_order),
                    "manifestDeclared": True,
                },
            )
            self.assertEqual(
                payload["targetQuaternionOrder"],
                ["__root_rotation__", *rig.driven_joint_order],
            )
            self.assertEqual(len(payload["conditionFeatureOrder"]), 48)
            self.assertEqual(payload["conditionFeatureOrder"], list(encoded_condition))
            self.assertEqual(payload["targetRecordFieldOrder"], list(encoded_target))
            self.assertEqual(
                payload["targetMotionFieldOrder"],
                ["dx", "dy", "vx", "vy", "facing", "lean", "squash", "bob"],
            )
            self.assertEqual(payload["quaternionComponentOrder"], ["x", "y", "z", "w"])
            self.assertEqual(payload["rootTranslationOrder"], ["x", "y", "z"])
            self.assertEqual(
                payload["facialOrder"],
                ["eye_scale", "eye_squint", "mouth_open", "ear_angle", "brow_tilt"],
            )
            self.assertEqual(payload["teacherBackend"], "fixed-plan-test")
            self.assertEqual(payload["clipFingerprints"], teacher_provenance["clipFingerprints"])
            self.assertEqual(payload["worldStateDtMs"], 50)
            self.assertEqual(payload["planDtMs"], 33)
            self.assertEqual(
                payload["executionClock"],
                {
                    "sampling": "generated-at-wall-clock-linear-v1",
                    "origin": "plan.generated_at_ms",
                    "motionTickMs": 16,
                    "motionTimerPhase": "absolute-from-simulator-reset",
                    "coincidentTimerOrder": "motion-before-world-state",
                    "advanceToNextWorldStateMs": 50,
                },
            )
            self.assertEqual(payload["scenarioEvents"]["cadenceMs"], 50)
            self.assertEqual(
                set(payload["scenarioEvents"]["supported"]),
                {
                    "window_move", "window_minimize", "window_maximize",
                    "window_restore", "fullscreen_enter", "fullscreen_restore",
                    "pet_click",
                },
            )
            with self.assertRaisesRegex(ValueError, "different dataset ABI"):
                ensure_dataset_manifest(
                    output, rig, context_steps=8, horizon_steps=13, dt_ms=33,
                    teacher_provenance=teacher_provenance,
                )
            with self.assertRaisesRegex(ValueError, "different dataset ABI"):
                ensure_dataset_manifest(
                    output, rig, context_steps=8, horizon_steps=12, dt_ms=33,
                    world_state_dt_ms=40,
                    teacher_provenance=teacher_provenance,
                )
            other_character = "same-rig-different-character"
            other_checkpoint = replace(
                rig.checkpoint,
                character_id=other_character,
                path=checkpoint_target_path(other_character, rig.fingerprint),
            )
            with self.assertRaisesRegex(ValueError, "different dataset ABI"):
                ensure_dataset_manifest(
                    output,
                    replace(
                        rig,
                        character_id=other_character,
                        checkpoint=other_checkpoint,
                    ),
                    context_steps=8,
                    horizon_steps=12,
                    dt_ms=33,
                    teacher_provenance=teacher_provenance,
                )
            (output / "episode-00000.ndjson").write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "already contains episode data"):
                ensure_dataset_manifest(
                    output, rig, context_steps=8, horizon_steps=12, dt_ms=33,
                    teacher_provenance=teacher_provenance,
                )


if __name__ == "__main__":
    unittest.main()
