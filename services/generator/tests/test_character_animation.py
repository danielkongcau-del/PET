from __future__ import annotations

from contextlib import redirect_stdout
from dataclasses import replace
import io
import json
import math
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

from pet_generator.backend import MotionBackend, MotionPlan, MotionPoint
from pet_generator.character_animation import (
    MAX_PLAN_PROVENANCE,
    CharacterAnimationTeacher,
)
from pet_generator.character_rig import (
    CHECKPOINT_FORMAT,
    CHECKPOINT_METADATA_SCHEMA,
    CharacterCheckpointContract,
    RIG_FINGERPRINT_ALGORITHM,
    RIG_FINGERPRINT_CANONICALIZATION,
    SelectedCharacterRig,
    rig_fingerprint,
)
from pet_generator.data_gen import (
    attach_teacher_provenance,
    ensure_dataset_manifest,
    main as data_gen_main,
)
from pet_generator.simulator import (
    WORLD_STATE_DT_MS,
    DesktopSimulator,
    ScenarioConfig,
    TrainingSample,
)


IDENTITY = (0.0, 0.0, 0.0, 1.0)


def quat(axis: tuple[float, float, float], degrees: float) -> tuple[float, float, float, float]:
    radians = math.radians(degrees) / 2.0
    sine = math.sin(radians)
    return (axis[0] * sine, axis[1] * sine, axis[2] * sine, math.cos(radians))


def multiply(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    lx, ly, lz, lw = left
    rx, ry, rz, rw = right
    return (
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
        lw * rw - lx * rx - ly * ry - lz * rz,
    )


def same_rotation(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
    tolerance: float = 1e-7,
) -> bool:
    return abs(abs(sum(a * b for a, b in zip(left, right))) - 1.0) <= tolerance


class StubBackend(MotionBackend):
    name = "stub-motion"

    def __init__(self, *, dt_ms: int = 250, point_count: int = 3):
        self.dt_ms = dt_ms
        self.point_count = point_count

    def generate(self, world, seed: int, generated_at_ms: int) -> MotionPlan:
        points = tuple(
            MotionPoint(
                t_ms=index * self.dt_ms,
                dx=float(index * 3),
                dy=float(-index),
                vx=12.0,
                vy=-4.0,
                facing=-1,
                lean=0.1,
                squash=0.9,
                bob=0.2,
                expression="happy",
                bone_rotations=(0.25,),
            )
            for index in range(self.point_count)
        )
        return MotionPlan(
            plan_id=f"stub-{world.seq}-{seed}",
            based_on_seq=world.seq,
            behavior="walk",
            generated_at_ms=generated_at_ms,
            valid_until_ms=generated_at_ms + self.dt_ms * (self.point_count + 2),
            dt_ms=self.dt_ms,
            confidence=0.8,
            seed=seed,
            points=points,
            target={"surface_id": "target", "foot_x": 50.0, "foot_y": 80.0},
        )

    def cancel(self, plan_id: str | None = None) -> bool:
        return False

    def configure_timing(self, plan_horizon_ms: int, plan_dt_ms: int) -> None:
        self.dt_ms = plan_dt_ms
        self.point_count = int(round(plan_horizon_ms / plan_dt_ms))


def world(*, timestamp_ms: int, seq: int = 7):
    return SimpleNamespace(
        seq=seq,
        timestamp_ms=timestamp_ms,
        pet=SimpleNamespace(
            foot_x=320.0,
            foot_y=700.0,
            behavior="walk",
            surface_id="window-1:top:0",
        ),
    )


def make_fixture(directory: Path, joint_count: int = 3) -> SelectedCharacterRig:
    joint_ids = [f"joint_{index}" for index in range(joint_count)]
    driven_order = tuple(reversed(joint_ids))
    rest_by_id = {
        joint_id: quat((0, 0, 1), 10.0 * (index + 1))
        for index, joint_id in enumerate(joint_ids)
    }
    joints = [
        {
            "id": "motion_root",
            "parentIndex": -1,
            "restLocal": {
                "translation": [0.0, 0.0, 0.0],
                "rotation": list(IDENTITY),
                "scale": [1.0, 1.0, 1.0],
            },
        }
    ]
    for index, joint_id in enumerate(joint_ids):
        joints.append(
            {
                "id": joint_id,
                "parentIndex": index,
                "restLocal": {
                    "translation": [0.0, 1.0, 0.0],
                    "rotation": list(rest_by_id[joint_id]),
                    "scale": [1.0, 1.0, 1.0],
                },
            }
        )

    start_deltas = []
    end_deltas = []
    for driven_index in range(joint_count):
        if driven_index == 1:
            fixed = quat((0, 1, 0), 60)
            start_deltas.append(fixed)
            end_deltas.append(tuple(-value for value in fixed))
        else:
            start_deltas.append(quat((1, 0, 0), 5.0 * (driven_index + 1)))
            end_deltas.append(quat((1, 0, 0), 65.0 + 5.0 * driven_index))
    authored_start = [
        multiply(rest_by_id[joint_id], start_deltas[index])
        for index, joint_id in enumerate(driven_order)
    ]
    authored_end = [
        multiply(rest_by_id[joint_id], end_deltas[index])
        for index, joint_id in enumerate(driven_order)
    ]
    unsigned = {
        "schema": "pet-character-animation-v1",
        "characterId": "synthetic",
        "rigId": f"synthetic-{joint_count}",
        "rigFingerprint": "a" * 64,
        "clipId": "motion",
        "name": "Synthetic Motion",
        "durationMs": 1000.0,
        "sampleRateHz": 4.0,
        "playbackMode": "loop",
        "jointOrder": list(driven_order),
        "frames": [
            {
                "timeMs": 0.0,
                "localRotations": [list(value) for value in authored_start],
                "localTranslations": [[0.0, 0.0, 0.0] for _ in driven_order],
            },
            {
                "timeMs": 500.0,
                "localRotations": [list(value) for value in authored_end],
                "localTranslations": [[0.25, 0.0, 0.0] for _ in driven_order],
            },
            {
                "timeMs": 1000.0,
                "localRotations": [list(value) for value in authored_start],
                "localTranslations": [[0.0, 0.0, 0.0] for _ in driven_order],
            },
        ],
        "source": {
            "uri": "synthetic.gltf",
            "sha256": "b" * 64,
            "animationIndex": 0,
            "animationName": "Synthetic Motion",
        },
    }
    fingerprint = rig_fingerprint(unsigned)
    payload = {
        "schema": unsigned["schema"],
        "characterId": unsigned["characterId"],
        "rigId": unsigned["rigId"],
        "rigFingerprint": unsigned["rigFingerprint"],
        "clipFingerprint": {
            "algorithm": RIG_FINGERPRINT_ALGORITHM,
            "canonicalization": RIG_FINGERPRINT_CANONICALIZATION,
            "value": fingerprint,
        },
        "clipId": unsigned["clipId"],
        "name": unsigned["name"],
        "durationMs": unsigned["durationMs"],
        "sampleRateHz": unsigned["sampleRateHz"],
        "playbackMode": unsigned["playbackMode"],
        "jointOrder": unsigned["jointOrder"],
        "frames": unsigned["frames"],
        "source": unsigned["source"],
    }
    clip_path = directory / "clip.json"
    clip_path.write_text(json.dumps(payload), encoding="utf-8")
    manifest_path = directory / "manifest.json"
    manifest_path.write_text("{}", encoding="utf-8")
    raw = {
        "rig": {
            "motionRoot": "motion_root",
            "jointOrder": ["motion_root", *joint_ids],
            "joints": joints,
            "drivenJointOrder": list(driven_order),
        },
        "trainingClips": [
            {
                "clipId": "motion",
                "path": "clip.json",
                "fingerprint": fingerprint,
                "playbackMode": "loop",
            }
        ],
    }
    return SelectedCharacterRig(
        character_id="synthetic",
        rig_id=f"synthetic-{joint_count}",
        fingerprint="a" * 64,
        driven_joint_order=driven_order,
        checkpoint=CharacterCheckpointContract(
            format=CHECKPOINT_FORMAT,
            metadata_schema=CHECKPOINT_METADATA_SCHEMA,
            path=f"checkpoints/characters/synthetic/{'a' * 64}/motion.pt",
            character_id="synthetic",
            rig_fingerprint="a" * 64,
            driven_joint_order=driven_order,
            manifest_declared=True,
        ),
        path=manifest_path,
        source="character_manifest",
        raw=raw,
    )


def rewrite_clip(rig: SelectedCharacterRig, mutate) -> None:
    clip_path = rig.path.parent / "clip.json"
    payload = json.loads(clip_path.read_text(encoding="utf-8"))
    mutate(payload)
    unsigned = {key: value for key, value in payload.items() if key != "clipFingerprint"}
    fingerprint = rig_fingerprint(unsigned)
    payload["clipFingerprint"]["value"] = fingerprint
    clip_path.write_text(json.dumps(payload), encoding="utf-8")
    rig.raw["trainingClips"][0]["fingerprint"] = fingerprint


class CharacterAnimationTeacherTests(unittest.TestCase):
    def test_production_data_gen_emits_real_non_identity_cat_pose(self) -> None:
        with tempfile.TemporaryDirectory() as temporary, redirect_stdout(io.StringIO()):
            result = data_gen_main([
                "--episodes", "1",
                "--duration-ms", "1000",
                "--output", temporary,
                "--seed", "42",
                "--context-steps", "2",
                "--horizon-steps", "4",
                "--dt-ms", "33",
            ])
            output = Path(temporary)
            manifest = json.loads(
                (output / "dataset-manifest.json").read_text(encoding="utf-8")
            )
            records = [
                json.loads(line)
                for line in (output / "episode-00000.ndjson")
                .read_text(encoding="utf-8")
                .splitlines()
            ]

        self.assertEqual(result, 0)
        self.assertEqual(len(manifest["conditionFeatureOrder"]), 48)
        self.assertEqual(manifest["checkpointTarget"]["characterId"], "cat")
        self.assertEqual(
            manifest["checkpointTarget"]["drivenJointOrder"],
            manifest["drivenJointOrder"],
        )
        self.assertTrue(records)
        self.assertEqual(
            list(records[0]["condition"][0]),
            manifest["conditionFeatureOrder"],
        )
        self.assertEqual(
            list(records[0]["target"][0]),
            manifest["targetRecordFieldOrder"],
        )
        self.assertEqual(records[0]["metadata"]["pose_source"], "character_animation")
        self.assertEqual(records[0]["metadata"]["animation_clip_id"], "being-cute")
        self.assertEqual(
            records[0]["metadata"]["animation_clip_fingerprint"],
            manifest["clipFingerprints"][0]["fingerprint"],
        )
        local_quaternions = [
            quaternion
            for record in records
            for target in record["target"]
            for quaternion in zip(*[iter(target["quaternions"][4:])] * 4)
        ]
        self.assertTrue(
            any(not same_rotation(quaternion, IDENTITY) for quaternion in local_quaternions)
        )
        neutral_facial = [1.0, 0.0, 0.0, 0.0, 0.0]
        self.assertTrue(
            any(
                target["facial"] != neutral_facial
                for record in records
                for target in record["target"]
            )
        )

    def test_variable_topology_is_non_identity_exact_order_and_preserves_motion(self) -> None:
        for joint_count in (2, 5):
            with self.subTest(joint_count=joint_count), tempfile.TemporaryDirectory() as temporary:
                rig = make_fixture(Path(temporary), joint_count)
                teacher = CharacterAnimationTeacher(
                    StubBackend(), rig, workspace_root=Path(temporary) / "unrelated-root"
                )
                base = teacher.base.generate(world(timestamp_ms=250), 11, 250)
                decorated = teacher.generate(world(timestamp_ms=250), 11, 250)

                self.assertEqual(decorated.behavior, base.behavior)
                self.assertEqual(decorated.target, base.target)
                self.assertEqual(
                    [(point.dx, point.dy) for point in decorated.points],
                    [(point.dx, point.dy) for point in base.points],
                )
                self.assertTrue(all(point.root_translation == (0.0, 0.0, 0.0) for point in decorated.points))
                self.assertTrue(all(point.root_rotation == IDENTITY for point in decorated.points))
                self.assertTrue(all(point.bone_rotations is None for point in decorated.points))
                self.assertTrue(
                    all(len(point.local_rotation_deltas or ()) == joint_count for point in decorated.points)
                )
                first = decorated.points[0].local_rotation_deltas
                assert first is not None
                self.assertTrue(any(not same_rotation(value, IDENTITY) for value in first))

                authored = teacher.clips[0].sample_absolute_rotations(250.0)
                rest_by_id = {
                    joint["id"]: tuple(joint["restLocal"]["rotation"])
                    for joint in rig.raw["rig"]["joints"]
                }
                for output_index, joint_id in enumerate(rig.driven_joint_order):
                    rest = rest_by_id[joint_id]
                    inverse = (-rest[0], -rest[1], -rest[2], rest[3])
                    expected = multiply(inverse, authored[output_index])
                    self.assertTrue(same_rotation(first[output_index], expected))

    def test_selection_is_seed_independent_and_adjacent_plans_are_continuous(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            rig = make_fixture(Path(temporary), 3)
            first_teacher = CharacterAnimationTeacher(
                StubBackend(dt_ms=250, point_count=3),
                rig,
                workspace_root=Path(temporary) / "unrelated-root",
            )
            second_teacher = CharacterAnimationTeacher(
                StubBackend(dt_ms=250, point_count=3),
                rig,
                workspace_root=Path(temporary) / "unrelated-root",
            )
            first = first_teacher.generate(world(timestamp_ms=0, seq=1), 10, 0)
            same_clock_different_seed = second_teacher.generate(
                world(timestamp_ms=0, seq=1), 999, 0
            )
            self.assertEqual(
                [point.local_rotation_deltas for point in first.points],
                [point.local_rotation_deltas for point in same_clock_different_seed.points],
            )

            adjacent = second_teacher.generate(world(timestamp_ms=500, seq=2), 1234, 500)
            self.assertEqual(
                first.points[-1].local_rotation_deltas,
                adjacent.points[0].local_rotation_deltas,
            )

    def test_q_and_negative_q_interpolate_on_the_shortest_arc(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            rig = make_fixture(Path(temporary), 3)
            teacher = CharacterAnimationTeacher(
                StubBackend(), rig, workspace_root=Path(temporary) / "unrelated-root"
            )
            clip = teacher.clips[0]
            start = clip.sample_absolute_rotations(0.0)
            middle = clip.sample_absolute_rotations(500.0)
            end = clip.frames[-1].local_rotations
            self.assertTrue(same_rotation(start[1], middle[1]))
            self.assertTrue(same_rotation(middle[1], end[1]))
            self.assertAlmostEqual(sum(value * value for value in middle[1]), 1.0, places=8)
            self.assertFalse(same_rotation(start[0], middle[0]))
            self.assertFalse(same_rotation(middle[0], end[0]))

            near_wrap = clip.sample_absolute_rotations(clip.duration_ms - 0.001)
            at_wrap = clip.sample_absolute_rotations(clip.duration_ms)
            after_wrap = clip.sample_absolute_rotations(clip.duration_ms + 0.001)
            self.assertTrue(same_rotation(clip.frames[-1].local_rotations[0], start[0]))
            self.assertTrue(same_rotation(near_wrap[0], at_wrap[0], tolerance=1e-6))
            self.assertTrue(same_rotation(at_wrap[0], start[0]))
            self.assertTrue(same_rotation(after_wrap[0], start[0], tolerance=1e-6))

    def test_non_closed_loop_clip_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            rig = make_fixture(directory, 3)
            rewrite_clip(
                rig,
                lambda payload: payload["frames"][-1]["localRotations"].__setitem__(
                    0,
                    list(quat((1, 0, 0), 90.0)),
                ),
            )
            with self.assertRaisesRegex(ValueError, "loop seam"):
                CharacterAnimationTeacher(
                    StubBackend(),
                    rig,
                    workspace_root=directory / "unrelated-root",
                )

    def test_animation_schema_limits_fail_closed(self) -> None:
        cases = (
            (
                "sample rate",
                lambda payload: payload.__setitem__("sampleRateHz", 1001.0),
                "sampleRateHz",
            ),
            (
                "name length",
                lambda payload: payload.__setitem__("name", "x" * 257),
                "at most 256",
            ),
        )
        for label, mutate, message in cases:
            with self.subTest(case=label), tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary)
                rig = make_fixture(directory, 3)
                rewrite_clip(rig, mutate)
                with self.assertRaisesRegex(ValueError, message):
                    CharacterAnimationTeacher(
                        StubBackend(),
                        rig,
                        workspace_root=directory / "unrelated-root",
                    )

    def test_mismatches_unsafe_paths_and_missing_clips_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            rig = make_fixture(directory, 3)
            rewrite_clip(rig, lambda payload: payload["jointOrder"].reverse())
            with self.assertRaisesRegex(ValueError, "jointOrder"):
                CharacterAnimationTeacher(
                    StubBackend(), rig, workspace_root=directory / "unrelated-root"
                )

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            rig = make_fixture(directory, 3)
            rig.raw["trainingClips"][0]["fingerprint"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "manifest entry"):
                CharacterAnimationTeacher(
                    StubBackend(), rig, workspace_root=directory / "unrelated-root"
                )

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            rig = make_fixture(directory, 3)
            rig.raw["trainingClips"][0]["path"] = "../clip.json"
            with self.assertRaisesRegex(ValueError, "safe relative"):
                CharacterAnimationTeacher(
                    StubBackend(), rig, workspace_root=directory / "unrelated-root"
                )

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            rig = make_fixture(directory, 3)
            no_clips = replace(rig, raw={"rig": rig.raw["rig"], "trainingClips": []})
            with self.assertRaisesRegex(ValueError, "no trainingClips"):
                CharacterAnimationTeacher(
                    StubBackend(), no_clips, workspace_root=directory / "unrelated-root"
                )

    def test_dataset_and_sample_provenance_are_bound_to_clip(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            rig = make_fixture(directory, 3)
            teacher = CharacterAnimationTeacher(
                StubBackend(), rig, workspace_root=directory / "unrelated-root"
            )
            plan = teacher.generate(world(timestamp_ms=125), 8, 125)
            sample = TrainingSample([], [], {"plan_id": plan.plan_id})
            attach_teacher_provenance([sample], teacher)
            self.assertEqual(sample.metadata["pose_source"], "character_animation")
            self.assertEqual(
                sample.metadata["animation_clip_fingerprint"],
                teacher.clips[0].fingerprint,
            )

            output = directory / "dataset"
            manifest_path = ensure_dataset_manifest(
                output,
                rig,
                context_steps=8,
                horizon_steps=12,
                dt_ms=33,
                teacher_provenance=teacher.dataset_provenance(),
            )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["teacherBackend"], "stub-motion")
            self.assertEqual(manifest["poseSource"], "character_animation")
            self.assertEqual(
                manifest["checkpointTarget"],
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
                manifest["clipFingerprints"],
                [{"clipId": "motion", "fingerprint": teacher.clips[0].fingerprint}],
            )
            self.assertEqual(teacher.retained_plan_provenance_count, 0)

    def test_plan_provenance_is_bounded_and_consumed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            rig = make_fixture(directory, 2)
            teacher = CharacterAnimationTeacher(
                StubBackend(point_count=2),
                rig,
                workspace_root=directory / "unrelated-root",
            )
            for index in range(MAX_PLAN_PROVENANCE + 17):
                teacher.generate(world(timestamp_ms=index, seq=index), index, index)

            self.assertEqual(
                teacher.retained_plan_provenance_count,
                MAX_PLAN_PROVENANCE,
            )
            with self.assertRaisesRegex(ValueError, "No animation-teacher provenance"):
                teacher.consume_plan_provenance("stub-0-0")
            newest_id = f"stub-{MAX_PLAN_PROVENANCE + 16}-{MAX_PLAN_PROVENANCE + 16}"
            teacher.consume_plan_provenance(newest_id)
            self.assertEqual(
                teacher.retained_plan_provenance_count,
                MAX_PLAN_PROVENANCE - 1,
            )

    def test_long_episode_consumes_provenance_when_each_sample_is_created(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            rig = make_fixture(directory, 2)
            teacher = CharacterAnimationTeacher(
                StubBackend(point_count=2),
                rig,
                workspace_root=directory / "unrelated-root",
            )
            simulator = DesktopSimulator(
                context_steps=1,
                horizon_steps=2,
                dt_ms=33,
                character_rig=rig,
            )
            samples = simulator.generate_episode(
                teacher,
                ScenarioConfig(
                    num_displays=(1, 1),
                    num_windows=(0, 0),
                    duration_ms=(MAX_PLAN_PROVENANCE + 25) * WORLD_STATE_DT_MS,
                    window_move_probability=0.0,
                    click_probability=1.0,
                ),
                episode_seed=123,
                sample_metadata_provider=teacher.consume_plan_provenance,
            )

            self.assertGreater(len(samples), MAX_PLAN_PROVENANCE)
            self.assertTrue(
                all(sample.metadata["pose_source"] == "character_animation" for sample in samples)
            )
            self.assertEqual(teacher.retained_plan_provenance_count, 0)


if __name__ == "__main__":
    unittest.main()
