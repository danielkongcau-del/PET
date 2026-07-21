"""Shared, character-agnostic rig and animation contract helpers.

The runtime manifest deliberately keeps character identity, joint count and
checkpoint location in data.  Nothing in this module assumes a cat rig or a
fixed number of model-driven joints.
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
from pathlib import Path
from typing import Any, Mapping, Sequence


RIG_MANIFEST_SCHEMA = "pet-character-rig-manifest-v1"
ANIMATION_SCHEMA = "pet-character-animation-v1"
FINGERPRINT_ALGORITHM = "pet-canonical-json-f64-v1+sha256"
FINGERPRINT_CANONICALIZATION = "pet-canonical-json-f64-v1"
CHECKPOINT_FORMAT = "pet-character-motion-checkpoint-v1"
CHECKPOINT_METADATA_SCHEMA = "pet-character-motion-checkpoint-metadata-v1"
CHECKPOINT_ROOT = "checkpoints/characters"
CHECKPOINT_FILENAME = "motion.pt"
F64_ACCUMULATION = "pet-left-to-right-f64-sum-v1"
CHECKPOINT_DATASET_SCHEMA = "pet-training-dataset-v1"
CHECKPOINT_DATASET_FORMAT = "episode-ndjson-v1"
CHECKPOINT_NORMALIZATION_SCHEMA = "pet-feature-normalization-v1"
CHECKPOINT_NORMALIZATION_TRANSFORM = "(value-offset)/scale"
CHECKPOINT_MODEL_CONFIG_SCHEMA = "pet-motion-model-config-v1"
WINDOWS_RESERVED_CHARACTER_BASENAMES = {
    "con",
    "prn",
    "aux",
    "nul",
    *(f"com{index}" for index in range(1, 10)),
    *(f"lpt{index}" for index in range(1, 10)),
}


def checkpoint_target_path(character_id: str, rig_fingerprint: str) -> str:
    """Return the only portable checkpoint target for one character/Rig ABI."""

    safe_character_id = validate_character_identifier(character_id, "characterId")
    fingerprint = _require_sha256(rig_fingerprint, "rigFingerprint")
    return f"{CHECKPOINT_ROOT}/{safe_character_id}/{fingerprint}/{CHECKPOINT_FILENAME}"


def validate_manifest_identifier(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or len(value) > 128:
        raise ValueError(f"{label} must be a non-empty manifest identifier")
    allowed = "abcdefghijklmnopqrstuvwxyz0123456789._-"
    if value[0] not in "abcdefghijklmnopqrstuvwxyz0123456789" or any(
        character not in allowed for character in value
    ):
        raise ValueError(f"{label} must be a lowercase manifest identifier")
    return value


def validate_character_identifier(value: Any, label: str) -> str:
    identifier = validate_manifest_identifier(value, label)
    basename = identifier.split(".", 1)[0]
    if identifier.endswith(".") or basename in WINDOWS_RESERVED_CHARACTER_BASENAMES:
        raise ValueError(f"{label} must be a Windows-safe character identifier")
    return identifier


def validate_joint_identifier(value: Any, label: str) -> str:
    if not isinstance(value, str) or not 1 <= len(value) <= 128:
        raise ValueError(f"{label} must be a lowercase joint identifier")
    if value[0] not in "abcdefghijklmnopqrstuvwxyz_" or any(
        character not in "abcdefghijklmnopqrstuvwxyz0123456789_"
        for character in value
    ):
        raise ValueError(f"{label} must be a lowercase joint identifier")
    return value


def _canonical_number(value: int | float) -> str:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("Rig fingerprints reject NaN and infinite numbers")
    if number == 0.0:
        number = 0.0  # canonicalise negative zero
    return "f64:" + struct.pack(">d", number).hex()


def canonicalize_for_fingerprint(value: Any) -> Any:
    """Return the JSON-compatible canonical projection used by fingerprints.

    Object keys are sorted by the JSON encoder, arrays retain their declared
    order, and every number becomes its IEEE-754 binary64 representation.  The
    numeric projection avoids Python/JavaScript spelling differences such as
    ``1`` versus ``1.0`` while remaining straightforward to reproduce with a
    JavaScript ``DataView``.
    """

    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, str):
        return _require_unicode_scalar_string(value)
    if isinstance(value, (int, float)):
        return _canonical_number(value)
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            text_key = _require_unicode_scalar_string(str(key))
            result[text_key] = canonicalize_for_fingerprint(item)
        return result
    if isinstance(value, (list, tuple)):
        return [canonicalize_for_fingerprint(item) for item in value]
    raise TypeError(f"Unsupported fingerprint value: {type(value).__name__}")


def _require_unicode_scalar_string(value: str) -> str:
    if any(0xD800 <= ord(character) <= 0xDFFF for character in value):
        raise ValueError(
            "Rig fingerprints reject strings that are not Unicode scalar sequences"
        )
    return value


def canonical_fingerprint_bytes(value: Any) -> bytes:
    projected = canonicalize_for_fingerprint(value)
    return json.dumps(
        projected,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def fingerprint_value(value: Any) -> str:
    return hashlib.sha256(canonical_fingerprint_bytes(value)).hexdigest()


def fingerprint_record(value: Any) -> dict[str, str]:
    return {
        "algorithm": FINGERPRINT_ALGORITHM,
        "canonicalization": FINGERPRINT_CANONICALIZATION,
        "value": fingerprint_value(value),
    }


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def nearest_driven_parent_indices(
    joint_order: Sequence[str],
    parent_indices: Sequence[int],
    driven_joint_order: Sequence[str],
) -> tuple[list[int], list[int]]:
    """Map driven joints to the full rig and their nearest driven ancestors."""

    if len(joint_order) != len(parent_indices):
        raise ValueError("joint_order and parent_indices must have equal lengths")
    full_index = {joint_id: index for index, joint_id in enumerate(joint_order)}
    if len(full_index) != len(joint_order):
        raise ValueError("joint_order contains duplicate ids")
    driven_index = {joint_id: index for index, joint_id in enumerate(driven_joint_order)}
    if len(driven_index) != len(driven_joint_order):
        raise ValueError("driven_joint_order contains duplicate ids")

    driven_joint_indices: list[int] = []
    driven_parent_indices: list[int] = []
    for joint_id in driven_joint_order:
        if joint_id not in full_index:
            raise ValueError(f'Driven joint "{joint_id}" is absent from joint_order')
        index = full_index[joint_id]
        driven_joint_indices.append(index)
        parent = parent_indices[index]
        visited: set[int] = set()
        nearest = -1
        while parent >= 0:
            if parent >= len(joint_order):
                raise ValueError(f"Parent index {parent} is outside joint_order")
            if parent in visited:
                raise ValueError("Cycle detected while resolving driven parent indices")
            visited.add(parent)
            parent_id = joint_order[parent]
            if parent_id in driven_index:
                nearest = driven_index[parent_id]
                break
            parent = parent_indices[parent]
        driven_parent_indices.append(nearest)
    return driven_joint_indices, driven_parent_indices


def validate_rig_contract(rig: Mapping[str, Any]) -> None:
    coordinate_system = rig.get("coordinateSystem")
    if not isinstance(coordinate_system, Mapping):
        raise ValueError("rig.coordinateSystem must be an object")
    if coordinate_system.get("handedness") not in ("right", "left"):
        raise ValueError("rig.coordinateSystem.handedness is unsupported")
    up = coordinate_system.get("up")
    forward = coordinate_system.get("forward")
    axes = {"+X", "-X", "+Y", "-Y", "+Z", "-Z"}
    if up not in axes or forward not in axes:
        raise ValueError("rig coordinate axes are unsupported")
    if up[-1] == forward[-1]:
        raise ValueError("rig coordinate up and forward axes must be orthogonal")
    units = coordinate_system.get("units")
    if not isinstance(units, str) or not units:
        raise ValueError("rig.coordinateSystem.units must be a non-empty string")

    joint_order = rig.get("jointOrder")
    joints = rig.get("joints")
    driven_order = rig.get("drivenJointOrder")
    driven_indices = rig.get("drivenJointIndices")
    driven_parents = rig.get("drivenParentIndices")
    mask_groups = rig.get("maskGroups")
    motion_root = rig.get("motionRoot")

    if not isinstance(joint_order, list) or not joint_order:
        raise ValueError("rig.jointOrder must be a non-empty list")
    for index, joint_id in enumerate(joint_order):
        validate_joint_identifier(joint_id, f"rig.jointOrder[{index}]")
    if len(joint_order) > 512:
        raise ValueError("rig.jointOrder exceeds the manifest limit of 512 joints")
    if not isinstance(joints, list) or len(joints) != len(joint_order):
        raise ValueError("rig.joints must align one-to-one with rig.jointOrder")
    if len(set(joint_order)) != len(joint_order):
        raise ValueError("rig.jointOrder contains duplicate ids")
    validate_joint_identifier(motion_root, "rig.motionRoot")
    if motion_root not in joint_order:
        raise ValueError("rig.motionRoot must reference a joint")

    parent_indices: list[int] = []
    root_count = 0
    for index, (joint_id, joint) in enumerate(zip(joint_order, joints)):
        if isinstance(joint, Mapping):
            validate_joint_identifier(joint.get("id"), f"rig.joints[{index}].id")
        if not isinstance(joint, Mapping) or joint.get("id") != joint_id:
            raise ValueError(f"rig.joints[{index}] does not match jointOrder")
        parent_index = joint.get("parentIndex")
        if not isinstance(parent_index, int) or isinstance(parent_index, bool):
            raise ValueError(f"Joint {joint_id} has a non-integer parentIndex")
        if parent_index == -1:
            root_count += 1
        elif parent_index < 0 or parent_index >= index:
            raise ValueError(
                f"Joint {joint_id} parentIndex must reference an earlier joint"
            )
        parent_indices.append(parent_index)
        rest = joint.get("restLocal")
        if not isinstance(rest, Mapping):
            raise ValueError(f"Joint {joint_id} has no restLocal transform")
        _finite_vector(rest.get("translation"), 3, f"Joint {joint_id} translation")
        rotation = _finite_vector(rest.get("rotation"), 4, f"Joint {joint_id} rotation")
        scale = _finite_vector(rest.get("scale"), 3, f"Joint {joint_id} scale")
        rotation_norm = math.sqrt(sum(component * component for component in rotation))
        if abs(rotation_norm - 1.0) > 1e-5:
            raise ValueError(f"Joint {joint_id} rest rotation must be a unit quaternion")
        if any(abs(component) <= 1e-12 for component in scale):
            raise ValueError(f"Joint {joint_id} rest scale cannot contain zero")
        dof_mask = joint.get("dofMask")
        if not isinstance(dof_mask, Mapping):
            raise ValueError(f"Joint {joint_id} has no dofMask")
        for field in ("translation", "rotation", "scale"):
            mask = dof_mask.get(field)
            if not isinstance(mask, list) or len(mask) != 3 or any(type(item) is not bool for item in mask):
                raise ValueError(f"Joint {joint_id} {field} DOF mask must contain three booleans")
    if root_count != 1:
        raise ValueError("rig must contain exactly one root joint")
    root_index = joint_order.index(motion_root)
    if root_index != 0 or parent_indices[root_index] != -1:
        raise ValueError("motionRoot must be joint index 0 with parentIndex -1")

    if not isinstance(driven_order, list) or not driven_order:
        raise ValueError("rig.drivenJointOrder must be a non-empty list")
    for index, joint_id in enumerate(driven_order):
        validate_joint_identifier(joint_id, f"rig.drivenJointOrder[{index}]")
    if len(driven_order) > 128:
        raise ValueError(
            "rig.drivenJointOrder exceeds the motion protocol limit of 128 joints"
        )
    expected_indices, expected_parents = nearest_driven_parent_indices(
        joint_order, parent_indices, driven_order
    )
    if driven_indices != expected_indices:
        raise ValueError("rig.drivenJointIndices does not match drivenJointOrder")
    if driven_parents != expected_parents:
        raise ValueError("rig.drivenParentIndices is not the nearest-driven hierarchy")
    if not isinstance(mask_groups, Mapping) or not mask_groups:
        raise ValueError("rig.maskGroups must be a non-empty object")
    for name, mask in mask_groups.items():
        validate_manifest_identifier(name, "rig.maskGroups key")
        if not isinstance(mask, list) or len(mask) != len(driven_order):
            raise ValueError(f"Mask group {name} must match drivenJointOrder length")
        if any(type(item) is not bool for item in mask):
            raise ValueError(f"Mask group {name} must contain booleans")


def _finite_vector(value: Any, length: int, label: str) -> list[float]:
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"{label} must contain {length} numbers")
    result: list[float] = []
    for component in value:
        if not isinstance(component, (int, float)) or isinstance(component, bool):
            raise ValueError(f"{label} must contain only numbers")
        number = float(component)
        if not math.isfinite(number):
            raise ValueError(f"{label} must contain only finite numbers")
        result.append(number)
    return result


def validate_manifest(manifest: Mapping[str, Any]) -> None:
    if manifest.get("schema") != RIG_MANIFEST_SCHEMA:
        raise ValueError(f"Unsupported rig manifest schema: {manifest.get('schema')!r}")
    character_id = validate_character_identifier(
        manifest.get("characterId"), "characterId"
    )
    validate_manifest_identifier(manifest.get("rigId"), "rigId")
    rig = manifest.get("rig")
    if not isinstance(rig, Mapping):
        raise ValueError("manifest.rig must be an object")
    validate_rig_contract(rig)
    render = manifest.get("render")
    if not isinstance(render, Mapping):
        raise ValueError("manifest.render must be an object")
    if render.get("canvas") != [48, 48]:
        raise ValueError("render.canvas must be the normalized 48x48 runtime canvas")
    if render.get("displayScale") != 2:
        raise ValueError("render.displayScale must be 2 for the normalized runtime window")
    render_modes = {"sprite", "skinned_mesh", "debug_skeleton"}
    mode = render.get("mode")
    fallback_modes = render.get("fallbackModes")
    if mode not in render_modes:
        raise ValueError("render.mode is unsupported")
    if not isinstance(fallback_modes, list) or any(
        item not in render_modes for item in fallback_modes
    ):
        raise ValueError("render.fallbackModes contains an unsupported mode")
    if len(set(fallback_modes)) != len(fallback_modes) or mode in fallback_modes:
        raise ValueError("render modes must be unique and omit the preferred mode from fallbacks")
    sprite = render.get("sprite")
    mesh = render.get("mesh")
    if (mode == "sprite" or "sprite" in fallback_modes) and not isinstance(sprite, Mapping):
        raise ValueError("sprite rendering requires render.sprite")
    if (mode == "skinned_mesh" or "skinned_mesh" in fallback_modes) and not isinstance(mesh, Mapping):
        raise ValueError("skinned mesh rendering requires render.mesh")
    source = manifest.get("source")
    if not isinstance(source, Mapping):
        raise ValueError("manifest.source must be an object")
    if isinstance(mesh, Mapping) and mesh.get("skinIndex") != source.get("selectedSkinIndex"):
        raise ValueError("render.mesh.skinIndex must match source.selectedSkinIndex")
    fingerprint = manifest.get("rigFingerprint")
    if not isinstance(fingerprint, Mapping):
        raise ValueError("manifest.rigFingerprint must be an object")
    if fingerprint.get("algorithm") != FINGERPRINT_ALGORITHM:
        raise ValueError("Unsupported rig fingerprint algorithm")
    if fingerprint.get("canonicalization") != FINGERPRINT_CANONICALIZATION:
        raise ValueError("Unsupported rig fingerprint canonicalization")
    expected = fingerprint_value(rig)
    if fingerprint.get("value") != expected:
        raise ValueError("rigFingerprint does not match manifest.rig")
    checkpoint = manifest.get("checkpoint")
    if not isinstance(checkpoint, Mapping):
        raise ValueError("manifest.checkpoint must be an object")
    expected_checkpoint_fields = {
        "format",
        "metadataSchema",
        "path",
        "characterId",
        "rigFingerprint",
        "drivenJointOrder",
    }
    if set(checkpoint) != expected_checkpoint_fields:
        raise ValueError("checkpoint fields do not match the v1 checkpoint binding ABI")
    if checkpoint.get("format") != CHECKPOINT_FORMAT:
        raise ValueError("checkpoint.format is unsupported")
    if checkpoint.get("metadataSchema") != CHECKPOINT_METADATA_SCHEMA:
        raise ValueError("checkpoint.metadataSchema is unsupported")
    if checkpoint.get("characterId") != character_id:
        raise ValueError("checkpoint.characterId must match manifest.characterId")
    if checkpoint.get("rigFingerprint") != expected:
        raise ValueError("checkpoint.rigFingerprint must match rigFingerprint")
    driven_order = rig.get("drivenJointOrder")
    if checkpoint.get("drivenJointOrder") != driven_order:
        raise ValueError("checkpoint.drivenJointOrder must exactly match rig.drivenJointOrder")
    expected_path = checkpoint_target_path(character_id, expected)
    if checkpoint.get("path") != expected_path:
        raise ValueError(
            "checkpoint.path must be the character-isolated canonical checkpoint target"
        )
    provenance = manifest.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("manifest.provenance must be an object")
    if provenance.get("generatorVersion") != 2:
        raise ValueError("manifest.provenance.generatorVersion must be 2")
    if provenance.get("f64Accumulation") != F64_ACCUMULATION:
        raise ValueError("manifest.provenance.f64Accumulation is unsupported")


def validate_checkpoint_metadata(
    metadata: Mapping[str, Any],
    expected_checkpoint: Mapping[str, Any] | None = None,
) -> None:
    """Validate self-describing metadata stored in a checkpoint bundle."""

    if not isinstance(metadata, Mapping):
        raise ValueError("checkpoint metadata must be an object")
    _require_exact_fields(
        metadata,
        {
            "schema",
            "characterId",
            "rigFingerprint",
            "drivenJointOrder",
            "dataset",
            "normalization",
            "model",
        },
        "checkpoint metadata",
    )
    if metadata.get("schema") != CHECKPOINT_METADATA_SCHEMA:
        raise ValueError("checkpoint metadata schema is unsupported")
    character_id = validate_character_identifier(
        metadata.get("characterId"), "checkpoint metadata.characterId"
    )
    rig_fingerprint = _require_sha256(
        metadata.get("rigFingerprint"), "checkpoint metadata.rigFingerprint"
    )
    driven_order = metadata.get("drivenJointOrder")
    if (
        not isinstance(driven_order, list)
        or not 1 <= len(driven_order) <= 128
        or len(set(driven_order)) != len(driven_order)
    ):
        raise ValueError(
            "checkpoint metadata.drivenJointOrder must contain 1..128 unique joints"
        )
    for index, joint_id in enumerate(driven_order):
        validate_joint_identifier(
            joint_id, f"checkpoint metadata.drivenJointOrder[{index}]"
        )
    if expected_checkpoint is not None and (
        character_id != expected_checkpoint.get("characterId")
        or rig_fingerprint != expected_checkpoint.get("rigFingerprint")
        or driven_order != expected_checkpoint.get("drivenJointOrder")
    ):
        raise ValueError("checkpoint metadata identity does not match the selected character")

    dataset = metadata.get("dataset")
    if not isinstance(dataset, Mapping):
        raise ValueError("checkpoint metadata.dataset must be an object")
    _require_exact_fields(
        dataset, {"schema", "format", "manifestSha256"}, "checkpoint metadata.dataset"
    )
    if dataset.get("schema") != CHECKPOINT_DATASET_SCHEMA:
        raise ValueError("checkpoint metadata.dataset.schema is unsupported")
    if dataset.get("format") != CHECKPOINT_DATASET_FORMAT:
        raise ValueError("checkpoint metadata.dataset.format is unsupported")
    _require_sha256(
        dataset.get("manifestSha256"), "checkpoint metadata.dataset.manifestSha256"
    )

    normalization = metadata.get("normalization")
    if not isinstance(normalization, Mapping):
        raise ValueError("checkpoint metadata.normalization must be an object")
    _require_exact_fields(
        normalization,
        {"schema", "transform", "condition", "target"},
        "checkpoint metadata.normalization",
    )
    if normalization.get("schema") != CHECKPOINT_NORMALIZATION_SCHEMA:
        raise ValueError("checkpoint metadata.normalization.schema is unsupported")
    if normalization.get("transform") != CHECKPOINT_NORMALIZATION_TRANSFORM:
        raise ValueError("checkpoint metadata.normalization.transform is unsupported")
    _validate_normalization_group(
        normalization.get("condition"), "checkpoint metadata.normalization.condition"
    )
    _validate_normalization_group(
        normalization.get("target"), "checkpoint metadata.normalization.target"
    )

    model = metadata.get("model")
    if not isinstance(model, Mapping):
        raise ValueError("checkpoint metadata.model must be an object")
    _require_exact_fields(
        model,
        {"schema", "architecture", "implementationVersion", "config"},
        "checkpoint metadata.model",
    )
    if model.get("schema") != CHECKPOINT_MODEL_CONFIG_SCHEMA:
        raise ValueError("checkpoint metadata.model.schema is unsupported")
    validate_manifest_identifier(
        model.get("architecture"), "checkpoint metadata.model.architecture"
    )
    implementation_version = model.get("implementationVersion")
    if not isinstance(implementation_version, str) or not 1 <= len(implementation_version) <= 128:
        raise ValueError(
            "checkpoint metadata.model.implementationVersion must be a non-empty string"
        )
    config = model.get("config")
    if not isinstance(config, Mapping):
        raise ValueError("checkpoint metadata.model.config must be an object")
    _validate_json_value(config, "checkpoint metadata.model.config")


def _validate_normalization_group(value: Any, label: str) -> None:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    _require_exact_fields(value, {"featureOrder", "offset", "scale"}, label)
    feature_order = value.get("featureOrder")
    if (
        not isinstance(feature_order, list)
        or not 1 <= len(feature_order) <= 4096
        or any(
            not isinstance(item, str) or not item or len(item) > 128
            for item in feature_order
        )
        or len(set(feature_order)) != len(feature_order)
    ):
        raise ValueError(f"{label}.featureOrder must contain 1..4096 unique features")
    _finite_vector(value.get("offset"), len(feature_order), f"{label}.offset")
    scale = _finite_vector(value.get("scale"), len(feature_order), f"{label}.scale")
    if any(component <= 0 for component in scale):
        raise ValueError(f"{label}.scale must contain only positive values")


def _require_exact_fields(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise ValueError(f"{label} fields do not match the v1 ABI")


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _validate_json_value(value: Any, label: str, depth: int = 0) -> None:
    if depth > 32:
        raise ValueError(f"{label} exceeds the maximum JSON nesting depth")
    if value is None or isinstance(value, (str, bool)):
        return
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if not math.isfinite(float(value)):
            raise ValueError(f"{label} contains a non-finite number")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_json_value(item, f"{label}[{index}]", depth + 1)
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{label} contains a non-string object key")
            _validate_json_value(item, f"{label}.{key}", depth + 1)
        return
    raise ValueError(f"{label} contains a non-JSON value")


def validate_animation(animation: Mapping[str, Any]) -> None:
    """Validate cross-field invariants JSON Schema cannot express."""

    if animation.get("schema") != ANIMATION_SCHEMA:
        raise ValueError(f"Unsupported animation schema: {animation.get('schema')!r}")
    validate_character_identifier(animation.get("characterId"), "animation.characterId")
    validate_manifest_identifier(animation.get("rigId"), "animation.rigId")
    validate_manifest_identifier(animation.get("clipId"), "animation.clipId")
    playback_mode = animation.get("playbackMode")
    if playback_mode not in {"loop", "once"}:
        raise ValueError("animation.playbackMode must be 'loop' or 'once'")
    joint_order = animation.get("jointOrder")
    if not isinstance(joint_order, list) or not joint_order:
        raise ValueError("animation.jointOrder must be a non-empty list")
    for index, joint_id in enumerate(joint_order):
        validate_joint_identifier(joint_id, f"animation.jointOrder[{index}]")
    if len(joint_order) > 128:
        raise ValueError("animation.jointOrder exceeds the motion protocol limit of 128 joints")
    if len(set(joint_order)) != len(joint_order):
        raise ValueError("animation.jointOrder contains duplicate ids")
    frames = animation.get("frames")
    if not isinstance(frames, list) or len(frames) < 2:
        raise ValueError("animation.frames must contain at least two frames")
    previous_time = -math.inf
    previous_rotations: list[list[float]] | None = None
    for index, frame in enumerate(frames):
        if not isinstance(frame, Mapping):
            raise ValueError(f"animation.frames[{index}] must be an object")
        time_ms = frame.get("timeMs")
        if not isinstance(time_ms, (int, float)) or isinstance(time_ms, bool) or not math.isfinite(time_ms):
            raise ValueError(f"animation.frames[{index}].timeMs must be finite")
        if time_ms <= previous_time:
            raise ValueError("animation frame times must be strictly increasing")
        previous_time = float(time_ms)
        for field in ("localRotations", "localTranslations"):
            values = frame.get(field)
            if not isinstance(values, list) or len(values) != len(joint_order):
                raise ValueError(
                    f"animation.frames[{index}].{field} must align with jointOrder"
                )
        rotations = frame.get("localRotations")
        assert isinstance(rotations, list)
        normalized_rotations: list[list[float]] = []
        for joint_index, quaternion in enumerate(rotations):
            values = _finite_vector(
                quaternion,
                4,
                f"animation.frames[{index}].localRotations[{joint_index}]",
            )
            norm = math.sqrt(sum(component * component for component in values))
            if abs(norm - 1.0) > 1e-5:
                raise ValueError("animation local rotations must be unit quaternions")
            if (
                previous_rotations is not None
                and sum(
                    left * right
                    for left, right in zip(previous_rotations[joint_index], values)
                ) < -1e-8
            ):
                raise ValueError("animation local rotations must be hemisphere-continuous")
            normalized_rotations.append(values)
        translations = frame.get("localTranslations")
        assert isinstance(translations, list)
        for joint_index, translation in enumerate(translations):
            _finite_vector(
                translation,
                3,
                f"animation.frames[{index}].localTranslations[{joint_index}]",
            )
        previous_rotations = normalized_rotations
    duration_ms = animation.get("durationMs")
    if (
        not isinstance(duration_ms, (int, float))
        or isinstance(duration_ms, bool)
        or not math.isfinite(duration_ms)
        or duration_ms <= 0
    ):
        raise ValueError("animation.durationMs must be finite and positive")
    if abs(float(frames[0]["timeMs"])) > 1e-6:
        raise ValueError("animation first frame must be at time 0")
    if abs(float(frames[-1]["timeMs"]) - float(duration_ms)) > 1e-5:
        raise ValueError("animation last frame must match durationMs")
    if playback_mode == "loop":
        first_rotations = frames[0]["localRotations"]
        last_rotations = frames[-1]["localRotations"]
        for joint_index, (first, last) in enumerate(zip(first_rotations, last_rotations)):
            if abs(abs(sum(left * right for left, right in zip(first, last))) - 1.0) > 1e-6:
                raise ValueError(
                    "animation loop seam rotations must be equivalent for every joint; "
                    f"joint index {joint_index} differs"
                )
        first_translations = frames[0]["localTranslations"]
        last_translations = frames[-1]["localTranslations"]
        for joint_index, (first, last) in enumerate(zip(first_translations, last_translations)):
            if any(abs(left - right) > 1e-6 for left, right in zip(first, last)):
                raise ValueError(
                    "animation loop seam translations must match for every joint; "
                    f"joint index {joint_index} differs"
                )
    fingerprint = animation.get("clipFingerprint")
    if not isinstance(fingerprint, Mapping):
        raise ValueError("animation.clipFingerprint must be an object")
    if fingerprint.get("algorithm") != FINGERPRINT_ALGORITHM:
        raise ValueError("Unsupported clip fingerprint algorithm")
    if fingerprint.get("canonicalization") != FINGERPRINT_CANONICALIZATION:
        raise ValueError("Unsupported clip fingerprint canonicalization")
    unsigned = {key: value for key, value in animation.items() if key != "clipFingerprint"}
    if fingerprint.get("value") != fingerprint_value(unsigned):
        raise ValueError("clipFingerprint does not match the animation payload")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
        encoding="utf-8",
    )
