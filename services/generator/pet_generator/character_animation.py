"""Authored character-animation teacher for offline dataset generation.

The procedural backend remains responsible for root motion, behavior and
targets.  This decorator replaces only the 3D pose with deterministic samples
from character-authored clips.  Clip rotations are absolute joint-local
rotations, so they are converted to the protocol's rest-local delta convention
before being attached to a plan.

Per-joint translations are deliberately validated but not consumed.  The
current motion protocol has no matching per-joint translation channel.
"""

from __future__ import annotations

import bisect
from collections import OrderedDict
from dataclasses import dataclass, replace
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from .backend import MotionBackend, MotionPlan
from .character_rig import (
    RIG_FINGERPRINT_ALGORITHM,
    RIG_FINGERPRINT_CANONICALIZATION,
    SelectedCharacterRig,
    project_root,
    rig_fingerprint,
)
from .state import WorldState


CHARACTER_ANIMATION_SCHEMA = "pet-character-animation-v1"
POSE_SOURCE = "character_animation"
POSE_SPACE = "rest_local_delta"
SELECTOR_VERSION = "character-fixed-clip-world-time-playback-v2"
_IDENTITY_QUAT = (0.0, 0.0, 0.0, 1.0)
_ZERO_TRANSLATION = (0.0, 0.0, 0.0)
_MAX_CLIP_BYTES = 64 * 1024 * 1024
MAX_PLAN_PROVENANCE = 1024


Quat = tuple[float, float, float, float]
Vec3 = tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class AnimationFrame:
    time_ms: float
    local_rotations: tuple[Quat, ...]


@dataclass(frozen=True, slots=True)
class CharacterAnimationClip:
    clip_id: str
    name: str
    fingerprint: str
    path: Path
    duration_ms: float
    sample_rate_hz: float
    playback_mode: str
    joint_order: tuple[str, ...]
    frames: tuple[AnimationFrame, ...]

    def sample_absolute_rotations(self, time_ms: float) -> tuple[Quat, ...]:
        """Sample the looping authored clip with shortest-arc quaternion SLERP."""

        if not math.isfinite(time_ms):
            raise ValueError("Animation sample time must be finite")
        wrapped = (
            time_ms % self.duration_ms
            if self.playback_mode == "loop"
            else max(0.0, min(time_ms, self.duration_ms))
        )
        times = [frame.time_ms for frame in self.frames]
        if wrapped <= times[0]:
            return self.frames[0].local_rotations
        right = bisect.bisect_right(times, wrapped)
        if right >= len(self.frames):
            return self.frames[-1].local_rotations
        left = right - 1
        left_frame = self.frames[left]
        right_frame = self.frames[right]
        span = right_frame.time_ms - left_frame.time_ms
        factor = (wrapped - left_frame.time_ms) / span
        return tuple(
            _slerp(left_rotation, right_rotation, factor)
            for left_rotation, right_rotation in zip(
                left_frame.local_rotations,
                right_frame.local_rotations,
            )
        )


@dataclass(frozen=True, slots=True)
class PlanPoseProvenance:
    clip_id: str
    clip_fingerprint: str
    phase_ms: float

    def to_metadata(self, teacher_backend: str) -> dict[str, Any]:
        return {
            "teacher_backend": teacher_backend,
            "pose_source": POSE_SOURCE,
            "pose_space": POSE_SPACE,
            "animation_clip_id": self.clip_id,
            "animation_clip_fingerprint": self.clip_fingerprint,
            "animation_phase_ms": self.phase_ms,
            "unsupported_pose_channels": ["localTranslations"],
            "pose_selector": SELECTOR_VERSION,
        }


class CharacterAnimationTeacher(MotionBackend):
    """Training-only backend decorator that injects authored skeletal poses."""

    name = "character-animation-teacher"

    def __init__(
        self,
        base: MotionBackend,
        character_rig: SelectedCharacterRig,
        *,
        workspace_root: Path | None = None,
    ) -> None:
        self.base = base
        self.character_rig = character_rig
        self.workspace_root = (workspace_root or project_root()).resolve()
        self._rest_rotations = _rest_rotations_for_driven(character_rig)
        self._clips = _load_training_clips(
            character_rig,
            workspace_root=self.workspace_root,
        )
        if not self._clips:
            raise ValueError(
                "Selected character has no valid trainingClips; refusing to generate "
                "an identity-pose training dataset"
            )
        self._plan_provenance: OrderedDict[str, PlanPoseProvenance] = OrderedDict()

    @property
    def clips(self) -> tuple[CharacterAnimationClip, ...]:
        return self._clips

    def dataset_provenance(self) -> dict[str, Any]:
        return {
            "teacherBackend": self.base.name,
            "poseSource": POSE_SOURCE,
            "poseSpace": POSE_SPACE,
            "poseSelector": SELECTOR_VERSION,
            "clipFingerprints": [
                {"clipId": clip.clip_id, "fingerprint": clip.fingerprint}
                for clip in self._clips
            ],
            "unsupportedPoseChannels": ["localTranslations"],
        }

    @property
    def retained_plan_provenance_count(self) -> int:
        return len(self._plan_provenance)

    def consume_plan_provenance(self, plan_id: str) -> dict[str, Any]:
        """Return and remove per-plan metadata so long runs cannot accumulate it."""

        provenance = self._plan_provenance.pop(plan_id, None)
        if provenance is None:
            raise ValueError(f"No animation-teacher provenance exists for plan {plan_id!r}")
        return provenance.to_metadata(self.base.name)

    def generate(self, world: WorldState, seed: int, generated_at_ms: int) -> MotionPlan:
        plan = self.base.generate(world, seed, generated_at_ms)
        clip, phase_ms = self._select_clip_and_phase(world)
        points = []
        for point in plan.points:
            authored = clip.sample_absolute_rotations(phase_ms + point.t_ms)
            deltas = tuple(
                _canonical_quaternion(
                    _quat_multiply(_quat_inverse(rest), absolute)
                )
                for rest, absolute in zip(self._rest_rotations, authored)
            )
            points.append(
                replace(
                    point,
                    bone_rotations=None,
                    root_translation=_ZERO_TRANSLATION,
                    root_rotation=_IDENTITY_QUAT,
                    local_rotation_deltas=deltas,
                )
            )
        decorated = replace(plan, points=tuple(points))
        self._plan_provenance.pop(decorated.plan_id, None)
        self._plan_provenance[decorated.plan_id] = PlanPoseProvenance(
            clip_id=clip.clip_id,
            clip_fingerprint=clip.fingerprint,
            phase_ms=phase_ms,
        )
        while len(self._plan_provenance) > MAX_PLAN_PROVENANCE:
            self._plan_provenance.popitem(last=False)
        return decorated

    def _select_clip_and_phase(
        self,
        world: WorldState,
    ) -> tuple[CharacterAnimationClip, float]:
        # Clip identity is stable for the selected character.  A future manifest
        # can add behavior-tagged clip sets without changing the absolute clock.
        material = "\0".join(
            [
                self.character_rig.fingerprint,
                SELECTOR_VERSION,
                *(f"{clip.clip_id}:{clip.fingerprint}" for clip in self._clips),
            ]
        ).encode("utf-8")
        digest = hashlib.sha256(material).digest()
        clip = self._clips[int.from_bytes(digest[:8], "big") % len(self._clips)]
        # The world timestamp is the episode-stable animation clock.  Rolling
        # replans therefore continue from the prior plan instead of jumping to a
        # seed-derived phase.
        timestamp = float(world.timestamp_ms)
        phase_ms = (
            timestamp % clip.duration_ms
            if clip.playback_mode == "loop"
            else max(0.0, min(timestamp, clip.duration_ms))
        )
        return clip, phase_ms

    def cancel(self, plan_id: str | None = None) -> bool:
        if plan_id is None:
            self._plan_provenance.clear()
        else:
            self._plan_provenance.pop(plan_id, None)
        return self.base.cancel(plan_id)

    def configure_timing(self, plan_horizon_ms: int, plan_dt_ms: int) -> None:
        self.base.configure_timing(plan_horizon_ms, plan_dt_ms)

    def close(self) -> None:
        self.base.close()

    def prepare(self) -> None:
        self.base.prepare()

    def set_skeletal_enabled(self, enabled: bool) -> None:
        self.base.set_skeletal_enabled(enabled)

    def set_skeletal_3d(self, enabled_3d: bool) -> None:
        self.base.set_skeletal_3d(enabled_3d)

    def metrics(self) -> Mapping[str, Any]:
        return {
            **self.base.metrics(),
            "backend": self.name,
            "teacher_backend": self.base.name,
            "pose_source": POSE_SOURCE,
            "animation_clip_count": len(self._clips),
        }


def _load_training_clips(
    rig: SelectedCharacterRig,
    *,
    workspace_root: Path,
) -> tuple[CharacterAnimationClip, ...]:
    entries = rig.raw.get("trainingClips")
    if not isinstance(entries, list) or not entries:
        raise ValueError(
            "Selected character manifest has no trainingClips; refusing identity targets"
        )
    clips: list[CharacterAnimationClip] = []
    clip_ids: set[str] = set()
    for index, raw_entry in enumerate(entries):
        entry = _strict_object(
            raw_entry,
            {"clipId", "path", "fingerprint", "playbackMode"},
            f"trainingClips[{index}]",
        )
        clip_id = _schema_id(entry.get("clipId"), f"trainingClips[{index}].clipId")
        if clip_id in clip_ids:
            raise ValueError(f"Duplicate training clip id: {clip_id}")
        clip_ids.add(clip_id)
        expected_fingerprint = _sha256(
            entry.get("fingerprint"),
            f"trainingClips[{index}].fingerprint",
        )
        reference = _bounded_string(
            entry.get("path"),
            f"trainingClips[{index}].path",
            maximum=512,
        )
        playback_mode = _playback_mode(
            entry.get("playbackMode"),
            f"trainingClips[{index}].playbackMode",
        )
        path = _resolve_clip_path(
            reference,
            rig=rig,
            workspace_root=workspace_root,
        )
        clips.append(
            _load_clip_file(
                path,
                rig=rig,
                manifest_clip_id=clip_id,
                manifest_fingerprint=expected_fingerprint,
                manifest_playback_mode=playback_mode,
            )
        )
    return tuple(clips)


def _resolve_clip_path(
    reference: str,
    *,
    rig: SelectedCharacterRig,
    workspace_root: Path,
) -> Path:
    relative = Path(reference)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("trainingClips paths must be safe relative paths")
    manifest_path = rig.path.resolve()
    try:
        manifest_path.relative_to(workspace_root)
    except ValueError:
        base = manifest_path.parent
    else:
        # Checked-in manifests use repository-root-relative training clip paths.
        base = workspace_root
    candidate = (base / relative).resolve()
    try:
        candidate.relative_to(base)
    except ValueError as exc:
        raise ValueError("trainingClips path escapes its allowed asset root") from exc
    if not candidate.is_file():
        raise ValueError(f"Training clip is unavailable: {candidate}")
    return candidate


def _load_clip_file(
    path: Path,
    *,
    rig: SelectedCharacterRig,
    manifest_clip_id: str,
    manifest_fingerprint: str,
    manifest_playback_mode: str,
) -> CharacterAnimationClip:
    if path.stat().st_size > _MAX_CLIP_BYTES:
        raise ValueError(f"Training clip exceeds {_MAX_CLIP_BYTES} bytes: {path.name}")
    payload = _strict_object(
        json.loads(path.read_text(encoding="utf-8")),
        {
            "schema",
            "characterId",
            "rigId",
            "rigFingerprint",
            "clipFingerprint",
            "clipId",
            "name",
            "durationMs",
            "sampleRateHz",
            "playbackMode",
            "jointOrder",
            "frames",
            "source",
        },
        "animation clip",
    )
    if payload.get("schema") != CHARACTER_ANIMATION_SCHEMA:
        raise ValueError(f"Unsupported character animation schema: {payload.get('schema')!r}")
    character_id = _schema_id(payload.get("characterId"), "animation.characterId")
    if character_id != rig.character_id:
        raise ValueError("Animation characterId does not match selected character")
    rig_id = _schema_id(payload.get("rigId"), "animation.rigId")
    if rig_id != rig.rig_id:
        raise ValueError("Animation rigId does not match selected character")
    if payload.get("rigFingerprint") != rig.fingerprint:
        raise ValueError("Animation rigFingerprint does not match selected character")
    clip_id = _schema_id(payload.get("clipId"), "animation.clipId")
    if clip_id != manifest_clip_id:
        raise ValueError("Animation clipId does not match its manifest entry")
    name = _bounded_string(payload.get("name"), "animation.name", maximum=256)
    joint_order = _string_tuple(payload.get("jointOrder"), "animation.jointOrder")
    if joint_order != rig.driven_joint_order:
        raise ValueError("Animation jointOrder must exactly match rig.drivenJointOrder")
    duration_ms = _positive_finite(payload.get("durationMs"), "animation.durationMs")
    sample_rate_hz = _positive_finite(payload.get("sampleRateHz"), "animation.sampleRateHz")
    if sample_rate_hz > 1000.0:
        raise ValueError("animation.sampleRateHz must not exceed 1000")
    playback_mode = _playback_mode(payload.get("playbackMode"), "animation.playbackMode")
    if playback_mode != manifest_playback_mode:
        raise ValueError("Animation playbackMode does not match its manifest entry")

    fingerprint_record = _strict_object(
        payload.get("clipFingerprint"),
        {"algorithm", "canonicalization", "value"},
        "animation.clipFingerprint",
    )
    if fingerprint_record.get("algorithm") != RIG_FINGERPRINT_ALGORITHM:
        raise ValueError("Unsupported animation fingerprint algorithm")
    if fingerprint_record.get("canonicalization") != RIG_FINGERPRINT_CANONICALIZATION:
        raise ValueError("Unsupported animation fingerprint canonicalization")
    declared_fingerprint = _sha256(
        fingerprint_record.get("value"),
        "animation.clipFingerprint.value",
    )
    unsigned = {key: value for key, value in payload.items() if key != "clipFingerprint"}
    if rig_fingerprint(unsigned) != declared_fingerprint:
        raise ValueError("Animation clipFingerprint does not match its payload")
    if declared_fingerprint != manifest_fingerprint:
        raise ValueError("Animation fingerprint does not match its manifest entry")

    source = _strict_object(
        payload.get("source"),
        {"uri", "sha256", "animationIndex", "animationName"},
        "animation.source",
    )
    _bounded_string(source.get("uri"), "animation.source.uri", maximum=1024)
    _sha256(source.get("sha256"), "animation.source.sha256")
    if type(source.get("animationIndex")) is not int or source["animationIndex"] < 0:
        raise ValueError("animation.source.animationIndex must be a non-negative integer")
    _bounded_string(
        source.get("animationName"),
        "animation.source.animationName",
        maximum=256,
    )

    raw_frames = payload.get("frames")
    if not isinstance(raw_frames, list) or len(raw_frames) < 2:
        raise ValueError("Animation must contain at least two frames")
    frames: list[AnimationFrame] = []
    previous_time = -math.inf
    for frame_index, raw_frame in enumerate(raw_frames):
        frame = _strict_object(
            raw_frame,
            {"timeMs", "localRotations", "localTranslations"},
            f"animation.frames[{frame_index}]",
        )
        time_ms = _nonnegative_finite(
            frame.get("timeMs"),
            f"animation.frames[{frame_index}].timeMs",
        )
        if time_ms <= previous_time:
            raise ValueError("Animation frame times must be strictly increasing")
        previous_time = time_ms
        rotations = _quaternion_array(
            frame.get("localRotations"),
            len(joint_order),
            f"animation.frames[{frame_index}].localRotations",
        )
        # This channel is intentionally unsupported by MotionPoint today, but
        # it is still validated so malformed authored data cannot be hidden.
        _translation_array(
            frame.get("localTranslations"),
            len(joint_order),
            f"animation.frames[{frame_index}].localTranslations",
        )
        frames.append(AnimationFrame(time_ms, rotations))
    if abs(frames[0].time_ms) > 1e-6:
        raise ValueError("Animation first frame must be at time 0")
    if abs(frames[-1].time_ms - duration_ms) > 1e-5:
        raise ValueError("Animation last frame must match durationMs")
    if playback_mode == "loop":
        for joint_index, (first, last) in enumerate(zip(
            frames[0].local_rotations,
            frames[-1].local_rotations,
        )):
            if abs(abs(sum(left * right for left, right in zip(first, last))) - 1.0) > 1e-6:
                raise ValueError(
                    "Animation loop seam rotations must be equivalent for every joint; "
                    f"joint index {joint_index} differs"
                )
    return CharacterAnimationClip(
        clip_id=clip_id,
        name=name,
        fingerprint=declared_fingerprint,
        path=path,
        duration_ms=duration_ms,
        sample_rate_hz=sample_rate_hz,
        playback_mode=playback_mode,
        joint_order=joint_order,
        frames=tuple(frames),
    )


def _rest_rotations_for_driven(rig: SelectedCharacterRig) -> tuple[Quat, ...]:
    raw_rig = rig.raw.get("rig")
    if not isinstance(raw_rig, Mapping):
        raise ValueError("Character animation teacher requires a manifest rig object")
    raw_joints = raw_rig.get("joints")
    if not isinstance(raw_joints, list):
        raise ValueError("Character animation teacher requires rig.joints")
    by_id: dict[str, Mapping[str, Any]] = {}
    for index, raw_joint in enumerate(raw_joints):
        if not isinstance(raw_joint, Mapping):
            raise ValueError(f"rig.joints[{index}] must be an object")
        joint_id = raw_joint.get("id")
        if isinstance(joint_id, str):
            by_id[joint_id] = raw_joint
    rotations: list[Quat] = []
    for joint_id in rig.driven_joint_order:
        joint = by_id.get(joint_id)
        if joint is None:
            raise ValueError(f"Driven joint {joint_id!r} has no rig.joints entry")
        rest = joint.get("restLocal")
        if not isinstance(rest, Mapping):
            raise ValueError(f"Driven joint {joint_id!r} has no restLocal")
        rotations.append(
            _unit_quaternion(rest.get("rotation"), f"Driven joint {joint_id!r} rest rotation")
        )
    return tuple(rotations)


def _slerp(left: Quat, right: Quat, factor: float) -> Quat:
    a = _canonical_quaternion(left)
    b = _canonical_quaternion(right)
    dot = sum(x * y for x, y in zip(a, b))
    if dot < 0.0:
        b = tuple(-value for value in b)  # type: ignore[assignment]
        dot = -dot
    dot = max(-1.0, min(1.0, dot))
    if dot > 0.9995:
        return _canonical_quaternion(
            tuple(a[index] + (b[index] - a[index]) * factor for index in range(4))
        )
    theta = math.acos(dot)
    sin_theta = math.sin(theta)
    left_weight = math.sin((1.0 - factor) * theta) / sin_theta
    right_weight = math.sin(factor * theta) / sin_theta
    return _canonical_quaternion(
        tuple(
            left_weight * a[index] + right_weight * b[index]
            for index in range(4)
        )
    )


def _quat_multiply(left: Quat, right: Quat) -> Quat:
    lx, ly, lz, lw = left
    rx, ry, rz, rw = right
    return (
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
        lw * rw - lx * rx - ly * ry - lz * rz,
    )


def _quat_inverse(value: Quat) -> Quat:
    x, y, z, w = value
    return (-x, -y, -z, w)


def _canonical_quaternion(value: Sequence[float]) -> Quat:
    if len(value) != 4:
        raise ValueError("Quaternion must contain four components")
    norm = math.sqrt(sum(float(component) ** 2 for component in value))
    if not math.isfinite(norm) or norm <= 1e-12:
        raise ValueError("Quaternion must have finite non-zero length")
    result = tuple(float(component) / norm for component in value)
    sign_probe = (result[3], result[0], result[1], result[2])
    for component in sign_probe:
        if abs(component) <= 1e-12:
            continue
        if component < 0:
            result = tuple(-item for item in result)
        break
    return result  # type: ignore[return-value]


def _unit_quaternion(value: Any, label: str) -> Quat:
    components = _finite_vector(value, 4, label)
    if any(component < -1.0 or component > 1.0 for component in components):
        raise ValueError(f"{label} components must be between -1 and 1")
    norm = math.sqrt(sum(component * component for component in components))
    if abs(norm - 1.0) > 1e-5:
        raise ValueError(f"{label} must be a unit quaternion")
    return _canonical_quaternion(components)


def _quaternion_array(value: Any, count: int, label: str) -> tuple[Quat, ...]:
    if not isinstance(value, list) or len(value) != count:
        raise ValueError(f"{label} must align with animation.jointOrder")
    return tuple(
        _unit_quaternion(quaternion, f"{label}[{index}]")
        for index, quaternion in enumerate(value)
    )


def _translation_array(value: Any, count: int, label: str) -> tuple[Vec3, ...]:
    if not isinstance(value, list) or len(value) != count:
        raise ValueError(f"{label} must align with animation.jointOrder")
    return tuple(
        _finite_vector(translation, 3, f"{label}[{index}]")  # type: ignore[arg-type]
        for index, translation in enumerate(value)
    )


def _finite_vector(value: Any, length: int, label: str) -> tuple[float, ...]:
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"{label} must contain {length} numeric components")
    result: list[float] = []
    for component in value:
        if not isinstance(component, (int, float)) or isinstance(component, bool):
            raise ValueError(f"{label} must contain only numbers")
        number = float(component)
        if not math.isfinite(number):
            raise ValueError(f"{label} must contain only finite numbers")
        result.append(number)
    return tuple(result)


def _strict_object(value: Any, keys: set[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    actual = set(value)
    if actual != keys:
        missing = sorted(keys - actual)
        extra = sorted(actual - keys)
        raise ValueError(f"{label} fields mismatch; missing={missing}, extra={extra}")
    return value


def _nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or len(value) > 1024:
        raise ValueError(f"{label} must be a non-empty string")
    return value


_SCHEMA_ID = re.compile(r"^[a-z0-9][a-z0-9._-]*$")


def _bounded_string(value: Any, label: str, *, maximum: int) -> str:
    text = _nonempty_string(value, label)
    if len(text) > maximum:
        raise ValueError(f"{label} must contain at most {maximum} characters")
    return text


def _schema_id(value: Any, label: str) -> str:
    text = _bounded_string(value, label, maximum=128)
    if _SCHEMA_ID.fullmatch(text) is None:
        raise ValueError(f"{label} does not match the animation schema id pattern")
    return text


def _playback_mode(value: Any, label: str) -> str:
    if value not in {"loop", "once"}:
        raise ValueError(f"{label} must be 'loop' or 'once'")
    return value


def _string_tuple(value: Any, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value or any(
        not isinstance(item, str) or not item for item in value
    ):
        raise ValueError(f"{label} must be a non-empty string array")
    if len(value) > 128 or any(len(item) > 128 for item in value):
        raise ValueError(f"{label} exceeds the animation schema limits")
    if len(set(value)) != len(value):
        raise ValueError(f"{label} must not contain duplicates")
    return tuple(value)


def _sha256(value: Any, label: str) -> str:
    text = _nonempty_string(value, label)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return text


def _positive_finite(value: Any, label: str) -> float:
    number = _nonnegative_finite(value, label)
    if number <= 0:
        raise ValueError(f"{label} must be positive")
    return number


def _nonnegative_finite(value: Any, label: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{label} must be numeric")
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"{label} must be finite and non-negative")
    return number
