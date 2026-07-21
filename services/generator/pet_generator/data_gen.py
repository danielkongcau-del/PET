"""Batch training data generation using the simulator with teacher backend.

Usage:
  python -m pet_generator.data_gen --episodes 1000 --output data/train
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any, Mapping

from .character_animation import CharacterAnimationTeacher
from .character_rig import SelectedCharacterRig, load_selected_character_rig
from .planner import AutoregressiveMotionBackend, PlannerConfig
from .simulator import (
    CONDITION_FEATURE_ORDER,
    FACIAL_PARAMETER_ORDER,
    QUATERNION_COMPONENT_ORDER,
    MOTION_TICK_MS,
    PLAN_SAMPLING_SEMANTICS,
    ROOT_TRANSLATION_ORDER,
    SCENARIO_EVENT_TYPES,
    TARGET_MOTION_FIELD_ORDER,
    TARGET_RECORD_FIELD_ORDER,
    WORLD_STATE_DT_MS,
    DesktopSimulator,
    ScenarioConfig,
    TrainingSample,
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generate PET training data via teacher simulation")
    p.add_argument("--episodes", type=int, default=1000, help="Number of episodes to generate")
    p.add_argument("--duration-ms", type=int, default=30_000, help="Episode duration in ms")
    p.add_argument("--output", type=str, default="data/train", help="Output directory")
    p.add_argument("--seed", type=int, default=42, help="Base random seed")
    p.add_argument("--horizon-steps", type=int, default=12)
    p.add_argument("--context-steps", type=int, default=8)
    p.add_argument(
        "--dt-ms",
        type=int,
        default=33,
        help="plan keyframe interval in milliseconds",
    )
    p.add_argument(
        "--world-state-dt-ms",
        type=int,
        default=WORLD_STATE_DT_MS,
        help="World-state publication cadence; the live host default is 50 ms (20 Hz)",
    )
    return p


def save_samples(samples: list[TrainingSample], output_dir: Path, episode_id: int) -> int:
    """Save one episode's samples as NDJSON. Returns number of samples written."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"episode-{episode_id:05d}.ndjson"
    temporary = output_dir / f"episode-{episode_id:05d}.ndjson.tmp"
    count = 0
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            for sample in samples:
                stream.write(json.dumps({
                    "condition": sample.condition_frames,
                    "target": sample.target_poses,
                    "metadata": sample.metadata,
                }, ensure_ascii=False, allow_nan=False) + "\n")
                count += 1
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return count


def ensure_dataset_manifest(
    output_dir: Path,
    rig: SelectedCharacterRig,
    *,
    context_steps: int,
    horizon_steps: int,
    dt_ms: int,
    teacher_provenance: Mapping[str, Any],
    world_state_dt_ms: int = WORLD_STATE_DT_MS,
    scenario_config: ScenarioConfig | None = None,
) -> Path:
    """Bind one output directory to an exact character/checkpoint ABI."""
    if (
        not isinstance(world_state_dt_ms, int)
        or isinstance(world_state_dt_ms, bool)
        or not 8 <= world_state_dt_ms <= 1_000
    ):
        raise ValueError("world_state_dt_ms must be an integer between 8 and 1000")
    if horizon_steps * dt_ms < world_state_dt_ms:
        raise ValueError("plan horizon must cover one world-state interval")
    teacher = _validated_teacher_provenance(teacher_provenance)
    scenario = scenario_config or ScenarioConfig()
    payload = {
        "schema": "pet-training-dataset-v1",
        "format": "episode-ndjson-v1",
        "characterId": rig.character_id,
        "rigId": rig.rig_id,
        "rigFingerprint": rig.fingerprint,
        "rigSource": rig.source,
        "drivenJointOrder": list(rig.driven_joint_order),
        "checkpointTarget": {
            "format": rig.checkpoint.format,
            "metadataSchema": rig.checkpoint.metadata_schema,
            "path": rig.checkpoint.path,
            "characterId": rig.checkpoint.character_id,
            "rigFingerprint": rig.checkpoint.rig_fingerprint,
            "drivenJointOrder": list(rig.checkpoint.driven_joint_order),
            "manifestDeclared": rig.checkpoint.manifest_declared,
        },
        "targetQuaternionOrder": ["__root_rotation__", *rig.driven_joint_order],
        "conditionFeatureOrder": list(CONDITION_FEATURE_ORDER),
        "targetRecordFieldOrder": list(TARGET_RECORD_FIELD_ORDER),
        "targetMotionFieldOrder": list(TARGET_MOTION_FIELD_ORDER),
        "quaternionComponentOrder": list(QUATERNION_COMPONENT_ORDER),
        "rootTranslationOrder": list(ROOT_TRANSLATION_ORDER),
        "facialOrder": list(FACIAL_PARAMETER_ORDER),
        "contextSteps": context_steps,
        "horizonSteps": horizon_steps,
        "worldStateDtMs": world_state_dt_ms,
        "planDtMs": dt_ms,
        "executionClock": {
            "sampling": PLAN_SAMPLING_SEMANTICS,
            "origin": "plan.generated_at_ms",
            "motionTickMs": MOTION_TICK_MS,
            "motionTimerPhase": "absolute-from-simulator-reset",
            "coincidentTimerOrder": "motion-before-world-state",
            "advanceToNextWorldStateMs": world_state_dt_ms,
        },
        "scenarioEvents": {
            "supported": list(SCENARIO_EVENT_TYPES),
            "cadenceMs": world_state_dt_ms,
            "probabilityPerTick": {
                "window_move": scenario.window_move_probability,
                "window_minimize": scenario.window_minimize_probability,
                "window_maximize": scenario.window_maximize_probability,
                "window_restore": scenario.window_restore_probability,
                "fullscreen_enter": scenario.fullscreen_probability,
                "fullscreen_restore": scenario.fullscreen_restore_probability,
                "pet_click": scenario.click_probability,
            },
        },
        "teacherBackend": teacher["teacherBackend"],
        "poseSource": teacher["poseSource"],
        "poseSpace": teacher["poseSpace"],
        "poseSelector": teacher["poseSelector"],
        "clipFingerprints": teacher["clipFingerprints"],
        "unsupportedPoseChannels": teacher["unsupportedPoseChannels"],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "dataset-manifest.json"
    existing_episodes = sorted(output_dir.glob("episode-*.ndjson"))
    if existing_episodes:
        raise ValueError(
            f"Output directory {output_dir} already contains episode data; "
            "use a new empty directory for each generation run"
        )
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != payload:
            raise ValueError(
                f"Output directory {output_dir} is already bound to a different dataset ABI"
            )
        return path
    temporary = output_dir / "dataset-manifest.json.tmp"
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    return path


def _validated_teacher_provenance(value: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "teacherBackend",
        "poseSource",
        "poseSpace",
        "poseSelector",
        "clipFingerprints",
        "unsupportedPoseChannels",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise ValueError("teacher_provenance does not match the dataset provenance ABI")
    result: dict[str, Any] = {}
    for key in ("teacherBackend", "poseSource", "poseSpace", "poseSelector"):
        item = value[key]
        if not isinstance(item, str) or not item:
            raise ValueError(f"teacher_provenance.{key} must be a non-empty string")
        result[key] = item

    clips = value["clipFingerprints"]
    if not isinstance(clips, list) or not clips:
        raise ValueError("teacher_provenance.clipFingerprints must be non-empty")
    normalized_clips: list[dict[str, str]] = []
    clip_ids: set[str] = set()
    for index, clip in enumerate(clips):
        if not isinstance(clip, Mapping) or set(clip) != {"clipId", "fingerprint"}:
            raise ValueError(
                f"teacher_provenance.clipFingerprints[{index}] has invalid fields"
            )
        clip_id = clip["clipId"]
        fingerprint = clip["fingerprint"]
        if not isinstance(clip_id, str) or not clip_id or clip_id in clip_ids:
            raise ValueError("teacher clip ids must be non-empty and unique")
        if (
            not isinstance(fingerprint, str)
            or len(fingerprint) != 64
            or any(character not in "0123456789abcdef" for character in fingerprint)
        ):
            raise ValueError("teacher clip fingerprints must be lowercase SHA-256 values")
        clip_ids.add(clip_id)
        normalized_clips.append({"clipId": clip_id, "fingerprint": fingerprint})
    result["clipFingerprints"] = normalized_clips

    unsupported = value["unsupportedPoseChannels"]
    if not isinstance(unsupported, list) or any(
        not isinstance(channel, str) or not channel for channel in unsupported
    ):
        raise ValueError("teacher_provenance.unsupportedPoseChannels must be a string array")
    result["unsupportedPoseChannels"] = list(unsupported)
    return result


def attach_teacher_provenance(
    samples: list[TrainingSample],
    teacher: CharacterAnimationTeacher,
) -> None:
    """Bind every sample to the exact authored clip/phase that supplied pose."""

    for index, sample in enumerate(samples):
        plan_id = sample.metadata.get("plan_id")
        if not isinstance(plan_id, str) or not plan_id:
            raise ValueError(f"Training sample {index} has no plan_id provenance")
        sample.metadata.update(teacher.consume_plan_provenance(plan_id))


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = ScenarioConfig(duration_ms=args.duration_ms)
    planner_config = PlannerConfig(horizon_steps=args.horizon_steps, dt_ms=args.dt_ms)
    character_rig = load_selected_character_rig()
    base_backend = AutoregressiveMotionBackend(planner_config)
    backend = CharacterAnimationTeacher(base_backend, character_rig)
    teacher_provenance = backend.dataset_provenance()
    sim = DesktopSimulator(
        context_steps=args.context_steps,
        horizon_steps=args.horizon_steps,
        dt_ms=args.dt_ms,
        world_state_dt_ms=args.world_state_dt_ms,
        character_rig=character_rig,
    )
    ensure_dataset_manifest(
        output_dir,
        character_rig,
        context_steps=args.context_steps,
        horizon_steps=args.horizon_steps,
        dt_ms=args.dt_ms,
        teacher_provenance=teacher_provenance,
        world_state_dt_ms=args.world_state_dt_ms,
        scenario_config=config,
    )

    total_samples = 0
    start_time = time.monotonic()

    for ep in range(args.episodes):
        episode_seed = args.seed + ep
        samples = sim.generate_episode(
            backend,
            config,
            episode_seed,
            sample_metadata_provider=backend.consume_plan_provenance,
        )
        n = save_samples(samples, output_dir, ep)
        total_samples += n

        if (ep + 1) % 100 == 0:
            elapsed = time.monotonic() - start_time
            rate = total_samples / max(elapsed, 0.001)
            print(f"[{ep + 1}/{args.episodes}] {total_samples} samples, {rate:.0f} samples/s")

    elapsed = time.monotonic() - start_time
    print(f"Done: {total_samples} samples in {elapsed:.1f}s ({total_samples / max(elapsed, 0.001):.0f} samples/s)")
    print(f"Output: {output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
