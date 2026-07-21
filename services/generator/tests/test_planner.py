from __future__ import annotations

from dataclasses import replace
import unittest
from unittest.mock import patch

from pet_generator.character_rig import load_selected_character_rig
from pet_generator.planner import (
    EXPRESSION_FACIAL_PARAMS,
    AutoregressiveMotionBackend,
    PlannerConfig,
)
from pet_generator.protocol import decode_line, validate_shared_message
from pet_generator.state import parse_world_state

from tests.helpers import line, world_state


def parsed_world(**kwargs: object):
    return parse_world_state(decode_line(line(world_state(**kwargs))))


def parsed_floor_world(target_surface_id: str, *, foot_x: float):
    raw = world_state(surface_id="display:1:floor", foot_x=foot_x, foot_y=1040)
    raw["payload"]["surfaces"] = [
        surface
        for surface in raw["payload"]["surfaces"]
        if surface["id"] in {target_surface_id, "display:1:floor"}
    ]
    return parse_world_state(decode_line(line(raw)))


def validate_plan(plan) -> None:
    validate_shared_message(
        {
            "protocol": "pet-motion",
            "version": 1,
            "type": "horizon_plan",
            "seq": 0,
            "timestamp_ms": plan.generated_at_ms,
            "payload": plan.to_payload(),
        }
    )


class PlannerTests(unittest.TestCase):
    def test_skeletal_plans_encode_expression_as_informative_facial_params(self) -> None:
        config = PlannerConfig(
            initial_jump_delay_min_ms=60_000,
            initial_jump_delay_max_ms=60_000,
        )
        neutral = EXPRESSION_FACIAL_PARAMS["neutral"]

        for mode in ("full_2d", "full_3d"):
            with self.subTest(mode=mode):
                backend = AutoregressiveMotionBackend(config)
                if mode == "full_3d":
                    backend.set_skeletal_3d(True)
                backend.set_skeletal_enabled(True)

                plan = backend.generate(
                    parsed_world(click_id=f"click:facial:{mode}"),
                    7,
                    1_750_000_002_000,
                )

                self.assertEqual(plan.behavior, "click_reaction")
                self.assertTrue(
                    all(
                        point.facial_params
                        == EXPRESSION_FACIAL_PARAMS.get(point.expression, neutral)
                        for point in plan.points
                    )
                )
                self.assertTrue(
                    any(point.facial_params != neutral for point in plan.points)
                )
                validate_plan(plan)

    def test_cancel_preserves_negotiated_skeletal_encoding_in_same_session(self) -> None:
        rig = load_selected_character_rig()
        config = PlannerConfig(
            initial_jump_delay_min_ms=60_000,
            initial_jump_delay_max_ms=60_000,
        )

        for mode in ("full_2d", "full_3d"):
            with self.subTest(mode=mode):
                backend = AutoregressiveMotionBackend(config)
                if mode == "full_3d":
                    backend.set_skeletal_3d(True)
                backend.set_skeletal_enabled(True)
                first = backend.generate(parsed_world(), 91, 1_750_000_002_000)

                self.assertTrue(backend.cancel(first.plan_id))
                second = backend.generate(
                    parsed_world(seq=2, timestamp_ms=1_750_000_001_100),
                    92,
                    1_750_000_002_050,
                )

                if mode == "full_2d":
                    self.assertEqual(
                        len(second.points[0].bone_rotations or ()),
                        len(rig.driven_joint_order),
                    )
                    self.assertIsNone(second.points[0].root_rotation)
                else:
                    self.assertIsNone(second.points[0].bone_rotations)
                    self.assertEqual(second.points[0].root_rotation, (0.0, 0.0, 0.0, 1.0))
                    self.assertEqual(
                        len(second.points[0].local_rotation_deltas or ()),
                        len(rig.driven_joint_order),
                    )

    def test_legacy_pose_uses_selected_character_order_and_fails_closed_above_32(self) -> None:
        rig = load_selected_character_rig()
        backend = AutoregressiveMotionBackend(
            PlannerConfig(initial_jump_delay_min_ms=60_000, initial_jump_delay_max_ms=60_000)
        )
        backend.set_skeletal_enabled(True)
        plan = backend.generate(parsed_world(), 91, 1_750_000_002_000)
        self.assertEqual(len(plan.points[0].bone_rotations or ()), len(rig.driven_joint_order))

        large_rig = replace(
            rig,
            driven_joint_order=tuple(f"joint-{index}" for index in range(33)),
        )
        with patch(
            "pet_generator.character_rig.load_selected_character_rig",
            return_value=large_rig,
        ):
            oversized = AutoregressiveMotionBackend(
                PlannerConfig(
                    initial_jump_delay_min_ms=60_000,
                    initial_jump_delay_max_ms=60_000,
                )
            )
            oversized.set_skeletal_enabled(True)
            oversized_plan = oversized.generate(parsed_world(), 92, 1_750_000_002_000)
        self.assertTrue(all(point.bone_rotations is None for point in oversized_plan.points))

    def test_walk_is_deterministic_for_seed_and_starts_at_zero(self) -> None:
        config = PlannerConfig(initial_jump_delay_min_ms=60_000, initial_jump_delay_max_ms=60_000)
        first = AutoregressiveMotionBackend(config).generate(parsed_world(), 12345, 1_750_000_002_000)
        second = AutoregressiveMotionBackend(config).generate(parsed_world(), 12345, 1_750_000_002_000)
        self.assertEqual(first.to_payload(), second.to_payload())
        self.assertEqual(first.behavior, "walk")
        self.assertEqual(first.points[0].t_ms, 0)
        self.assertEqual(first.points[0].dx, 0)
        self.assertEqual([point.t_ms for point in first.points], [index * 33 for index in range(12)])
        validate_plan(first)

    def test_click_is_edge_triggered(self) -> None:
        backend = AutoregressiveMotionBackend(
            PlannerConfig(initial_jump_delay_min_ms=60_000, initial_jump_delay_max_ms=60_000)
        )
        first = backend.generate(parsed_world(click_id="click:1"), 7, 1_750_000_002_000)
        repeated = backend.generate(parsed_world(seq=2, click_id="click:1"), 8, 1_750_000_002_050)
        self.assertEqual(first.behavior, "click_reaction")
        self.assertEqual(repeated.behavior, "walk")
        self.assertTrue(all(point.expression for point in first.points))
        validate_plan(first)

    def test_multiple_pet_clicks_are_consumed_one_per_plan(self) -> None:
        raw = world_state(seq=1, click_id="click:1")
        second_click = dict(raw["payload"]["clicks"][0])
        second_click["id"] = "click:2"
        second_click["timestamp_ms"] += 1
        raw["payload"]["clicks"].append(second_click)
        backend = AutoregressiveMotionBackend(
            PlannerConfig(initial_jump_delay_min_ms=60_000, initial_jump_delay_max_ms=60_000)
        )

        first = backend.generate(parse_world_state(decode_line(line(raw))), 8, 1_750_000_002_000)
        second = backend.generate(parsed_world(seq=2, timestamp_ms=1_750_000_001_100), 9, 1_750_000_002_050)
        third = backend.generate(parsed_world(seq=3, timestamp_ms=1_750_000_001_200), 10, 1_750_000_002_100)

        self.assertEqual(first.behavior, "click_reaction")
        self.assertEqual(second.behavior, "click_reaction")
        self.assertEqual(third.behavior, "walk")
        validate_plan(first)
        validate_plan(second)
        validate_plan(third)

    def test_can_select_nearby_window_and_generate_jump(self) -> None:
        backend = AutoregressiveMotionBackend(
            PlannerConfig(
                initial_jump_delay_min_ms=0,
                initial_jump_delay_max_ms=0,
                voluntary_drop_probability=0.0,
            )
        )
        plan = backend.generate(parsed_world(), 22, 1_750_000_002_000)
        self.assertEqual(plan.behavior, "jump")
        self.assertIsNotNone(plan.target)
        self.assertEqual(plan.target["surface_id"], "window:b:top")
        self.assertAlmostEqual(plan.points[0].dx, 0.0)
        self.assertAlmostEqual(plan.points[0].dy, 0.0)
        validate_plan(plan)

    def test_can_voluntarily_drop_from_window_to_taskbar_edge(self) -> None:
        backend = AutoregressiveMotionBackend(
            PlannerConfig(
                initial_jump_delay_min_ms=0,
                initial_jump_delay_max_ms=0,
                voluntary_drop_probability=1.0,
            )
        )
        plan = backend.generate(parsed_world(), 31, 1_750_000_002_000)
        jump = backend._active_jump
        self.assertIsNotNone(jump)
        assert jump is not None

        self.assertEqual(plan.behavior, "jump")
        self.assertEqual(plan.target["surface_id"], "display:1:floor")
        self.assertEqual(jump.arc_height, 0.0)
        self.assertGreater(plan.target["foot_x"], 240.0)
        preparation = [point for point in plan.points if point.t_ms < jump.prepare_ms]
        self.assertTrue(preparation)
        self.assertTrue(all(point.dx == 0 and point.dy == 0 for point in preparation))
        first_motion = next(
            point for point in plan.points if abs(point.dx) > 1e-9 or abs(point.dy) > 1e-9
        )
        self.assertGreater(first_motion.dy, 0.0)
        validate_plan(plan)

    def test_floor_is_drop_fallback_when_no_other_window_is_reachable(self) -> None:
        raw = world_state()
        raw["payload"]["surfaces"] = [
            surface
            for surface in raw["payload"]["surfaces"]
            if surface["id"] in {"window:a:top", "display:1:floor"}
        ]
        world = parse_world_state(decode_line(line(raw)))
        backend = AutoregressiveMotionBackend(
            PlannerConfig(
                initial_jump_delay_min_ms=0,
                initial_jump_delay_max_ms=0,
                voluntary_drop_probability=0.0,
            )
        )

        plan = backend.generate(world, 32, 1_750_000_002_000)
        self.assertEqual(plan.target["surface_id"], "display:1:floor")
        validate_plan(plan)

    def test_voluntary_drop_stops_after_landing_on_an_intermediate_window(self) -> None:
        backend = AutoregressiveMotionBackend(
            PlannerConfig(
                initial_jump_delay_min_ms=0,
                initial_jump_delay_max_ms=0,
                voluntary_drop_probability=1.0,
            )
        )
        first = backend.generate(parsed_world(), 33, 1_750_000_002_000)
        self.assertEqual(first.target["surface_id"], "display:1:floor")

        raw = world_state(
            seq=2,
            timestamp_ms=1_750_000_001_400,
            surface_id="window:c:top",
            foot_x=360,
            foot_y=700,
        )
        raw["payload"]["surfaces"].append(
            {
                "id": "window:c:top",
                "kind": "window_top",
                "display_id": "display:1",
                "window_id": "window:c",
                "x1": 200,
                "x2": 700,
                "y": 700,
                "enabled": True,
                "occluded": False,
            }
        )
        landed = backend.generate(
            parse_world_state(decode_line(line(raw))),
            34,
            1_750_000_002_050,
        )

        self.assertEqual(landed.behavior, "walk")
        self.assertIsNone(backend._active_jump)
        self.assertEqual(landed.target["surface_id"], "window:c:top")
        validate_plan(landed)

    def test_jump_turns_toward_target_before_takeoff(self) -> None:
        raw = world_state()
        raw["payload"]["pet"]["facing"] = -1
        world = parse_world_state(decode_line(line(raw)))
        backend = AutoregressiveMotionBackend(
            PlannerConfig(initial_jump_delay_min_ms=0, initial_jump_delay_max_ms=0)
        )

        plan = backend.generate(world, 22, 1_750_000_002_000)
        jump = backend._active_jump
        self.assertIsNotNone(jump)
        assert jump is not None
        self.assertEqual(jump.facing, 1)
        self.assertGreaterEqual(jump.prepare_ms, 96)
        preparation = [point for point in plan.points if point.t_ms < jump.prepare_ms]
        self.assertTrue(preparation)
        self.assertTrue(
            all(
                point.dx == 0
                and point.dy == 0
                and point.vx == 0
                and point.vy == 0
                and point.facing == 1
                for point in preparation
            )
        )
        first_motion = next(
            point for point in plan.points if abs(point.dx) > 1e-9 or abs(point.dy) > 1e-9
        )
        self.assertGreater(first_motion.t_ms, jump.prepare_ms)
        validate_plan(plan)

    def test_can_jump_from_1080p_floor_to_window_top_at_420(self) -> None:
        backend = AutoregressiveMotionBackend(
            PlannerConfig(initial_jump_delay_min_ms=0, initial_jump_delay_max_ms=0)
        )
        plan = backend.generate(
            parsed_floor_world("window:a:top", foot_x=240),
            23,
            1_750_000_002_000,
        )

        self.assertEqual(plan.behavior, "jump")
        self.assertEqual(plan.target["surface_id"], "window:a:top")
        self.assertEqual(plan.target["foot_y"], 420)
        self.assertEqual((plan.points[0].dx, plan.points[0].dy), (0.0, 0.0))
        self.assertLess(plan.points[-1].dy, 0)
        self.assertGreater(plan.points[-1].dy, 420 - 1040)
        self.assertTrue(
            all((point.vx ** 2 + point.vy ** 2) ** 0.5 < 2_200 for point in plan.points)
        )
        active_jump = backend._active_jump
        self.assertIsNotNone(active_jump)
        assert active_jump is not None
        self.assertGreaterEqual(active_jump.duration_ms, 900)
        self.assertLessEqual(active_jump.duration_ms, 1_300)
        validate_plan(plan)

    def test_can_jump_from_1080p_floor_to_window_top_at_300(self) -> None:
        backend = AutoregressiveMotionBackend(
            PlannerConfig(initial_jump_delay_min_ms=0, initial_jump_delay_max_ms=0)
        )
        plan = backend.generate(
            parsed_floor_world("window:b:top", foot_x=900),
            24,
            1_750_000_002_000,
        )

        self.assertEqual(plan.behavior, "jump")
        self.assertEqual(plan.target["surface_id"], "window:b:top")
        self.assertEqual(plan.target["foot_y"], 300)
        self.assertEqual((plan.points[0].dx, plan.points[0].dy), (0.0, 0.0))
        self.assertLess(plan.points[-1].dy, 0)
        self.assertGreater(plan.points[-1].dy, 300 - 1040)
        self.assertTrue(
            all((point.vx ** 2 + point.vy ** 2) ** 0.5 < 2_200 for point in plan.points)
        )
        active_jump = backend._active_jump
        self.assertIsNotNone(active_jump)
        assert active_jump is not None
        self.assertGreaterEqual(active_jump.duration_ms, 1_000)
        self.assertLessEqual(active_jump.duration_ms, 1_300)
        validate_plan(plan)

    def test_stale_surface_id_does_not_walk_on_moved_surface(self) -> None:
        raw = world_state(surface_id="window:a:top", foot_x=240, foot_y=420)
        moved = next(surface for surface in raw["payload"]["surfaces"] if surface["id"] == "window:a:top")
        moved["x1"] = 1_000
        moved["x2"] = 1_500
        world = parse_world_state(decode_line(line(raw)))

        plan = AutoregressiveMotionBackend().generate(world, 25, 1_750_000_002_000)

        self.assertEqual(plan.behavior, "falling")
        self.assertEqual(plan.target["surface_id"], "display:1:floor")
        self.assertEqual((plan.points[0].dx, plan.points[0].dy), (0.0, 0.0))
        validate_plan(plan)

    def test_falls_toward_work_area_floor_without_support(self) -> None:
        world = parsed_world(surface_id=None, foot_x=1500, foot_y=500)
        plan = AutoregressiveMotionBackend().generate(world, 3, 1_750_000_002_000)
        self.assertEqual(plan.behavior, "falling")
        self.assertEqual(plan.points[0].dy, 0)
        self.assertGreater(plan.points[-1].dy, 0)
        self.assertEqual(plan.target["surface_id"], "display:1:floor")
        validate_plan(plan)

    def test_scene_suspend_generates_hidden_plan(self) -> None:
        plan = AutoregressiveMotionBackend().generate(parsed_world(allowed=False), 4, 1_750_000_002_000)
        self.assertEqual(plan.behavior, "hidden")
        self.assertTrue(all(point.dx == 0 and point.dy == 0 for point in plan.points))
        validate_plan(plan)


if __name__ == "__main__":
    unittest.main()
