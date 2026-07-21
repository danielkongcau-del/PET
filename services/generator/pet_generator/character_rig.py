"""Runtime selection for character-specific rigs and checkpoints.

The protocol and model code are character-agnostic.  Each selected character
provides an exact driven-joint ABI and rig fingerprint through a manifest;
legacy skeleton JSON remains available only as a compatibility fallback.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import struct
from typing import Any, Mapping


CHARACTER_RIG_SCHEMA = "pet-character-rig-manifest-v1"
MAX_DRIVEN_JOINTS = 128
_SHA256_LENGTH = 64
RIG_FINGERPRINT_ALGORITHM = "pet-canonical-json-f64-v1+sha256"
RIG_FINGERPRINT_CANONICALIZATION = "pet-canonical-json-f64-v1"
CHECKPOINT_FORMAT = "pet-character-motion-checkpoint-v1"
CHECKPOINT_METADATA_SCHEMA = "pet-character-motion-checkpoint-metadata-v1"
CHECKPOINT_ROOT = "checkpoints/characters"
CHECKPOINT_FILENAME = "motion.pt"
CHECKPOINT_DATASET_SCHEMA = "pet-training-dataset-v1"
CHECKPOINT_DATASET_FORMAT = "episode-ndjson-v1"
CHECKPOINT_NORMALIZATION_SCHEMA = "pet-feature-normalization-v1"
CHECKPOINT_NORMALIZATION_TRANSFORM = "(value-offset)/scale"
CHECKPOINT_MODEL_CONFIG_SCHEMA = "pet-motion-model-config-v1"
F64_ACCUMULATION = "pet-left-to-right-f64-sum-v1"


@dataclass(frozen=True)
class CharacterCheckpointContract:
    """Expected metadata and isolated target for one character checkpoint.

    ``manifest_declared`` describes where the target came from; it is never an
    assertion that an artifact exists or has been loaded.
    """

    format: str
    metadata_schema: str
    path: str
    character_id: str
    rig_fingerprint: str
    driven_joint_order: tuple[str, ...]
    manifest_declared: bool


@dataclass(frozen=True)
class SelectedCharacterRig:
    character_id: str
    rig_id: str
    fingerprint: str
    driven_joint_order: tuple[str, ...]
    checkpoint: CharacterCheckpointContract
    path: Path
    source: str
    raw: Mapping[str, Any]


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def load_selected_character_rig(*, root: Path | None = None) -> SelectedCharacterRig:
    workspace = (root or project_root()).resolve()
    configured_manifest = os.environ.get("PET_CHARACTER_MANIFEST")
    manifest_path = _resolve_configured(
        workspace,
        configured_manifest,
        Path("assets/pet/runtime/cat-character-rig.manifest.json"),
    )
    if manifest_path.is_file():
        return _load_manifest(manifest_path)
    if configured_manifest:
        raise ValueError(
            f"Configured character manifest is unavailable: {manifest_path.name}"
        )

    configured_legacy = os.environ.get("PET_CHARACTER_RIG") or os.environ.get(
        "PET_SKELETON_3D"
    )
    legacy_path = _resolve_legacy(workspace, configured_legacy)
    return _load_legacy(legacy_path)


def _resolve_configured(workspace: Path, configured: str | None, fallback: Path) -> Path:
    selected = Path(configured) if configured else fallback
    if selected.is_absolute():
        return selected.resolve()
    return (workspace / selected).resolve()


def _resolve_legacy(workspace: Path, configured: str | None) -> Path:
    if not configured:
        return (workspace / "assets/pet/runtime/cat-skeleton-3d.json").resolve()
    selected = Path(configured)
    if selected.is_absolute():
        return selected.resolve()
    project_relative = (workspace / selected).resolve()
    if project_relative.is_file():
        return project_relative
    # Backwards compatibility: PET_SKELETON_3D historically accepted only a
    # file name relative to assets/pet/runtime.
    return (workspace / "assets/pet/runtime" / selected).resolve()


def _load_manifest(path: Path) -> SelectedCharacterRig:
    data = _read_object(path)
    if data.get("schema") != CHARACTER_RIG_SCHEMA:
        raise ValueError(f"Unsupported character manifest schema: {data.get('schema')!r}")
    character_id = _character_identifier(data.get("characterId"), "characterId")
    rig_id = _manifest_identifier(data.get("rigId"), "rigId")
    rig = _mapping(data.get("rig"), "rig")
    fingerprint_record = _mapping(data.get("rigFingerprint"), "rigFingerprint")
    if fingerprint_record.get("algorithm") != RIG_FINGERPRINT_ALGORITHM:
        raise ValueError("Unsupported rig fingerprint algorithm")
    if fingerprint_record.get("canonicalization") != RIG_FINGERPRINT_CANONICALIZATION:
        raise ValueError("Unsupported rig fingerprint canonicalization")
    fingerprint = _sha256(fingerprint_record.get("value"), "rigFingerprint.value")
    if rig_fingerprint(rig) != fingerprint:
        raise ValueError("rigFingerprint does not match manifest.rig")
    driven = _validate_manifest_rig(rig)
    checkpoint = _validate_checkpoint_contract(
        data.get("checkpoint"),
        character_id=character_id,
        rig_fingerprint=fingerprint,
        driven_joint_order=driven,
        manifest_declared=True,
    )
    _validate_manifest_render(data)
    _validate_manifest_provenance(data)
    return SelectedCharacterRig(
        character_id=character_id,
        rig_id=rig_id,
        fingerprint=fingerprint,
        driven_joint_order=driven,
        checkpoint=checkpoint,
        path=path,
        source="character_manifest",
        raw=data,
    )


def _validate_manifest_render(manifest: Mapping[str, Any]) -> None:
    render = _mapping(manifest.get("render"), "render")
    if render.get("canvas") != [48, 48]:
        raise ValueError("render.canvas must be the normalized 48x48 runtime canvas")
    if render.get("displayScale") != 2:
        raise ValueError("render.displayScale must be 2 for the normalized 96 DIP pet window")
    foot_anchor = _finite_vector(render.get("footAnchor"), 2, "render.footAnchor")
    if not 0 <= foot_anchor[0] <= 48 or not 0 <= foot_anchor[1] <= 48:
        raise ValueError("render.footAnchor must lie inside render.canvas")
    if render.get("sourceFacing") not in (-1, 1):
        raise ValueError("render.sourceFacing must be -1 or 1")

    supported_modes = {"sprite", "skinned_mesh", "debug_skeleton"}
    mode = render.get("mode")
    if mode not in supported_modes:
        raise ValueError("render.mode is unsupported")
    fallback_modes = render.get("fallbackModes")
    if not isinstance(fallback_modes, list) or any(
        candidate not in supported_modes for candidate in fallback_modes
    ):
        raise ValueError("render.fallbackModes contains an unsupported mode")
    if len(set(fallback_modes)) != len(fallback_modes) or mode in fallback_modes:
        raise ValueError("render modes must be unique and omit the preferred mode from fallbacks")

    sprite = render.get("sprite")
    if mode == "sprite" or "sprite" in fallback_modes:
        sprite = _mapping(sprite, "render.sprite")
        _relative_asset_reference(sprite.get("image"), "render.sprite.image")
        _relative_asset_reference(sprite.get("metadata"), "render.sprite.metadata")

    mesh = render.get("mesh")
    if mode == "skinned_mesh" or "skinned_mesh" in fallback_modes:
        mesh = _mapping(mesh, "render.mesh")
        if mesh.get("source") != "source":
            raise ValueError("render.mesh.source must be 'source'")
        skin_index = mesh.get("skinIndex")
        if type(skin_index) is not int or skin_index < 0:
            raise ValueError("render.mesh.skinIndex must be a non-negative integer")
        source = _mapping(manifest.get("source"), "source")
        if skin_index != source.get("selectedSkinIndex"):
            raise ValueError("render.mesh.skinIndex must match source.selectedSkinIndex")


def _validate_manifest_provenance(manifest: Mapping[str, Any]) -> None:
    provenance = _mapping(manifest.get("provenance"), "provenance")
    _exact_fields(
        provenance,
        {"generator", "generatorVersion", "profile", "f64Accumulation"},
        "provenance",
    )
    generator = provenance.get("generator")
    if not isinstance(generator, str) or not 1 <= len(generator) <= 512:
        raise ValueError("provenance.generator must be a non-empty string")
    if provenance.get("generatorVersion") != 2:
        raise ValueError("provenance.generatorVersion must be 2")
    profile = provenance.get("profile")
    if profile is not None and (not isinstance(profile, str) or len(profile) > 512):
        raise ValueError("provenance.profile must be null or a string")
    if provenance.get("f64Accumulation") != F64_ACCUMULATION:
        raise ValueError("provenance.f64Accumulation is unsupported")


def _validate_manifest_rig(rig: Mapping[str, Any]) -> tuple[str, ...]:
    coordinate_system = _mapping(rig.get("coordinateSystem"), "rig.coordinateSystem")
    if coordinate_system.get("handedness") not in ("right", "left"):
        raise ValueError("rig.coordinateSystem.handedness is unsupported")
    axes = {"+X", "-X", "+Y", "-Y", "+Z", "-Z"}
    up = coordinate_system.get("up")
    forward = coordinate_system.get("forward")
    if up not in axes or forward not in axes:
        raise ValueError("rig coordinate axes are unsupported")
    if str(up)[-1] == str(forward)[-1]:
        raise ValueError("rig coordinate up and forward axes must be orthogonal")
    units = coordinate_system.get("units")
    if not isinstance(units, str) or not units:
        raise ValueError("rig.coordinateSystem.units must be a non-empty string")

    motion_root = _joint_identifier(rig.get("motionRoot"), "rig.motionRoot")
    joint_order = _string_tuple(rig.get("jointOrder"), "rig.jointOrder")
    for index, joint_id in enumerate(joint_order):
        _joint_identifier(joint_id, f"rig.jointOrder[{index}]")
    joints = rig.get("joints")
    if not isinstance(joints, list) or len(joints) != len(joint_order):
        raise ValueError("rig.joints must align one-to-one with rig.jointOrder")
    if len(joint_order) < 2 or len(set(joint_order)) != len(joint_order):
        raise ValueError("rig.jointOrder must contain unique joints")
    if len(joint_order) > 512:
        raise ValueError("rig.jointOrder exceeds the manifest limit of 512 joints")
    if joint_order[0] != motion_root:
        raise ValueError("rig.motionRoot must be joint index 0")
    parent_indices: list[int] = []
    root_count = 0
    for index, joint in enumerate(joints):
        entry = _mapping(joint, f"rig.joints[{index}]")
        _joint_identifier(entry.get("id"), f"rig.joints[{index}].id")
        if entry.get("id") != joint_order[index]:
            raise ValueError(f"rig.joints[{index}] does not match jointOrder")
        parent = entry.get("parentIndex")
        if type(parent) is not int or parent < -1:
            raise ValueError(f"rig.joints[{index}] has an invalid parentIndex")
        if index == 0:
            if parent != -1:
                raise ValueError("rig motion root parentIndex must be -1")
        elif parent >= index:
            raise ValueError(f"rig.joints[{index}] parent must precede the joint")
        parent_indices.append(parent)
        if parent == -1:
            root_count += 1

        rest = _mapping(entry.get("restLocal"), f"rig.joints[{index}].restLocal")
        _finite_vector(
            rest.get("translation"), 3, f"rig.joints[{index}].restLocal.translation"
        )
        rest_rotation = _finite_vector(
            rest.get("rotation"), 4, f"rig.joints[{index}].restLocal.rotation"
        )
        rest_scale = _finite_vector(
            rest.get("scale"), 3, f"rig.joints[{index}].restLocal.scale"
        )
        rotation_norm = math.sqrt(sum(value * value for value in rest_rotation))
        if abs(rotation_norm - 1.0) > 1e-5:
            raise ValueError(
                f"rig.joints[{index}].restLocal.rotation must be a unit quaternion"
            )
        if any(abs(value) <= 1e-12 for value in rest_scale):
            raise ValueError(f"rig.joints[{index}].restLocal.scale cannot contain zero")

        dof_mask = _mapping(entry.get("dofMask"), f"rig.joints[{index}].dofMask")
        for field in ("translation", "rotation", "scale"):
            _boolean_vector(
                dof_mask.get(field),
                3,
                f"rig.joints[{index}].dofMask.{field}",
            )
    if root_count != 1:
        raise ValueError("rig must contain exactly one root and motionRoot must be joint index 0")

    driven = _string_tuple(rig.get("drivenJointOrder"), "rig.drivenJointOrder")
    for index, joint_id in enumerate(driven):
        _joint_identifier(joint_id, f"rig.drivenJointOrder[{index}]")
    if not 1 <= len(driven) <= MAX_DRIVEN_JOINTS:
        raise ValueError(
            f"Driven joint count must be between 1 and {MAX_DRIVEN_JOINTS}"
        )
    if len(set(driven)) != len(driven):
        raise ValueError("rig.drivenJointOrder contains duplicate ids")
    if motion_root in driven:
        raise ValueError("rig.motionRoot cannot be model-driven")
    indices = _integer_tuple(rig.get("drivenJointIndices"), "rig.drivenJointIndices")
    if len(indices) != len(driven):
        raise ValueError("rig.drivenJointIndices length mismatch")
    lookup = {joint_id: index for index, joint_id in enumerate(joint_order)}
    for output_index, joint_id in enumerate(driven):
        if lookup.get(joint_id) != indices[output_index]:
            raise ValueError("rig.drivenJointIndices does not match drivenJointOrder")
    parents = _integer_tuple(
        rig.get("drivenParentIndices"), "rig.drivenParentIndices"
    )
    driven_lookup = {joint_id: index for index, joint_id in enumerate(driven)}
    expected_parents: list[int] = []
    for joint_index in indices:
        parent_index = parent_indices[joint_index]
        nearest = -1
        while parent_index >= 0:
            parent_id = joint_order[parent_index]
            if parent_id in driven_lookup:
                nearest = driven_lookup[parent_id]
                break
            parent_index = parent_indices[parent_index]
        expected_parents.append(nearest)
    if tuple(expected_parents) != parents:
        raise ValueError("rig.drivenParentIndices is not the nearest-driven hierarchy")

    mask_groups = _mapping(rig.get("maskGroups"), "rig.maskGroups")
    if not mask_groups:
        raise ValueError("rig.maskGroups must be a non-empty object")
    for name, mask in mask_groups.items():
        _manifest_identifier(name, "rig.maskGroups key")
        _boolean_vector(mask, len(driven), f"rig.maskGroups.{name}")
    return driven


def _load_legacy(path: Path) -> SelectedCharacterRig:
    raw_bytes = path.read_bytes()
    data = _object(json.loads(raw_bytes.decode("utf-8")), "legacy rig")
    if data.get("schema") != "pet-rig-v2":
        raise ValueError(f"Unsupported legacy rig schema: {data.get('schema')!r}")
    motion_root = _joint_identifier(data.get("motionRoot"), "motionRoot")
    joints = data.get("joints")
    if not isinstance(joints, list) or len(joints) < 2:
        raise ValueError("Legacy rig must contain at least two joints")
    driven: list[str] = []
    ids: set[str] = set()
    entries: list[Mapping[str, Any]] = []
    for index, joint in enumerate(joints):
        entry = _mapping(joint, f"joints[{index}]")
        entries.append(entry)
        joint_id = _joint_identifier(entry.get("id"), f"joints[{index}].id")
        if joint_id in ids:
            raise ValueError(f"Duplicate legacy joint id: {joint_id}")
        ids.add(joint_id)
        physics = entry.get("physics")
        dofs = entry.get("poseDofs")
        physics_mode = physics.get("mode") if isinstance(physics, Mapping) else None
        rotation = dofs.get("rotation") if isinstance(dofs, Mapping) else False
        if (
            joint_id != motion_root
            and physics_mode not in ("secondary", "static")
            and rotation is True
        ):
            driven.append(joint_id)
    root = next((entry for entry in entries if entry.get("id") == motion_root), None)
    if root is None or root.get("parent") is not None or root.get("deform") is not False:
        raise ValueError("Legacy motionRoot is invalid")
    for index, entry in enumerate(entries):
        parent = entry.get("parent")
        if parent is not None and (not isinstance(parent, str) or parent not in ids):
            raise ValueError(f"joints[{index}] references a missing parent")
    draw_order = data.get("drawOrder", [])
    if isinstance(draw_order, list):
        if any(not isinstance(joint_id, str) or joint_id not in ids for joint_id in draw_order):
            raise ValueError("Legacy drawOrder references an unknown joint")
    if not 1 <= len(driven) <= MAX_DRIVEN_JOINTS:
        raise ValueError("Legacy rig driven joint count is outside protocol bounds")
    configured_character_id = os.environ.get("PET_CHARACTER_ID")
    normalized_character_id = (
        configured_character_id.strip() if configured_character_id is not None else ""
    )
    character_id = _character_identifier(
        normalized_character_id or "legacy-default", "PET_CHARACTER_ID"
    )
    fingerprint = hashlib.sha256(raw_bytes).hexdigest()
    driven_order = tuple(driven)
    return SelectedCharacterRig(
        character_id=character_id,
        rig_id="pet-rig-v2",
        fingerprint=fingerprint,
        driven_joint_order=driven_order,
        checkpoint=_derived_checkpoint_contract(
            character_id,
            fingerprint,
            driven_order,
        ),
        path=path,
        source="legacy_rig",
        raw=data,
    )


def rig_fingerprint(value: Any) -> str:
    projected = _canonical_projection(value)
    encoded = json.dumps(
        projected,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _canonical_projection(value: Any) -> Any:
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, str):
        return _require_unicode_scalar_string(value)
    if isinstance(value, (int, float)):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("Rig fingerprints reject non-finite numbers")
        if number == 0.0:
            number = 0.0
        return "f64:" + struct.pack(">d", number).hex()
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            text_key = _require_unicode_scalar_string(str(key))
            result[text_key] = _canonical_projection(item)
        return result
    if isinstance(value, (list, tuple)):
        return [_canonical_projection(item) for item in value]
    raise ValueError(f"Unsupported rig fingerprint value: {type(value).__name__}")


def _require_unicode_scalar_string(value: str) -> str:
    if any(0xD800 <= ord(character) <= 0xDFFF for character in value):
        raise ValueError(
            "Rig fingerprints reject strings that are not Unicode scalar sequences"
        )
    return value


def _read_object(path: Path) -> Mapping[str, Any]:
    if path.stat().st_size > 16 * 1024 * 1024:
        raise ValueError("Character manifest exceeds 16 MiB")
    return _object(json.loads(path.read_text(encoding="utf-8")), "manifest")


def _object(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    return _object(value, name)


def _nonempty_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value or len(value) > 128:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _manifest_identifier(value: Any, name: str) -> str:
    text = _nonempty_string(value, name)
    if not (
        text[0].islower()
        or text[0].isdigit()
    ) or any(character not in "abcdefghijklmnopqrstuvwxyz0123456789._-" for character in text):
        raise ValueError(f"{name} must be a lowercase manifest identifier")
    return text


_WINDOWS_RESERVED_CHARACTER_BASENAMES = {
    "con",
    "prn",
    "aux",
    "nul",
    *(f"com{index}" for index in range(1, 10)),
    *(f"lpt{index}" for index in range(1, 10)),
}


def _character_identifier(value: Any, name: str) -> str:
    text = _manifest_identifier(value, name)
    basename = text.split(".", 1)[0]
    if text.endswith(".") or basename in _WINDOWS_RESERVED_CHARACTER_BASENAMES:
        raise ValueError(f"{name} must be a Windows-safe character identifier")
    return text


def _is_joint_identifier(value: str) -> bool:
    return (
        1 <= len(value) <= 128
        and value[0] in "abcdefghijklmnopqrstuvwxyz_"
        and all(
            character in "abcdefghijklmnopqrstuvwxyz0123456789_"
            for character in value
        )
    )


def _joint_identifier(value: Any, name: str) -> str:
    text = _nonempty_string(value, name)
    if not _is_joint_identifier(text):
        raise ValueError(f"{name} must be a lowercase joint identifier")
    return text


def _sha256(value: Any, name: str) -> str:
    text = _nonempty_string(value, name)
    if len(text) != _SHA256_LENGTH or any(c not in "0123456789abcdef" for c in text):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return text


def _string_tuple(value: Any, name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value):
        raise ValueError(f"{name} must be an array of non-empty strings")
    return tuple(value)


def _integer_tuple(value: Any, name: str) -> tuple[int, ...]:
    if not isinstance(value, list) or any(type(item) is not int for item in value):
        raise ValueError(f"{name} must be an integer array")
    return tuple(value)


def _finite_vector(value: Any, length: int, name: str) -> tuple[float, ...]:
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"{name} must contain {length} numeric components")
    result: list[float] = []
    for component in value:
        if (
            not isinstance(component, (int, float))
            or isinstance(component, bool)
            or not math.isfinite(component)
        ):
            raise ValueError(f"{name} must contain only finite numbers")
        result.append(float(component))
    return tuple(result)


def _boolean_vector(value: Any, length: int, name: str) -> tuple[bool, ...]:
    if (
        not isinstance(value, list)
        or len(value) != length
        or any(type(component) is not bool for component in value)
    ):
        raise ValueError(f"{name} must contain {length} booleans")
    return tuple(value)


def _relative_asset_reference(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value or len(value) > 512 or "\0" in value:
        raise ValueError(f"{name} must be a non-empty path")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or ":" in value.split("/")[0]:
        raise ValueError(f"{name} must be a safe relative path")
    return value


def checkpoint_target_path(character_id: str, rig_fingerprint: str) -> str:
    safe_character_id = _character_identifier(character_id, "characterId")
    fingerprint = _sha256(rig_fingerprint, "rigFingerprint")
    return f"{CHECKPOINT_ROOT}/{safe_character_id}/{fingerprint}/{CHECKPOINT_FILENAME}"


def validate_checkpoint_metadata(
    value: Any,
    expected: CharacterCheckpointContract | None = None,
) -> Mapping[str, Any]:
    """Validate metadata carried inside a v1 character checkpoint bundle."""

    metadata = _mapping(value, "checkpoint metadata")
    _exact_fields(
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
    character_id = _character_identifier(
        metadata.get("characterId"), "checkpoint metadata.characterId"
    )
    rig_fingerprint = _sha256(
        metadata.get("rigFingerprint"), "checkpoint metadata.rigFingerprint"
    )
    driven_order = _string_tuple(
        metadata.get("drivenJointOrder"),
        "checkpoint metadata.drivenJointOrder",
    )
    if (
        not 1 <= len(driven_order) <= MAX_DRIVEN_JOINTS
        or len(set(driven_order)) != len(driven_order)
        or any(not _is_joint_identifier(joint_id) for joint_id in driven_order)
    ):
        raise ValueError(
            "checkpoint metadata.drivenJointOrder must contain 1..128 unique joints"
        )
    if expected is not None and (
        character_id != expected.character_id
        or rig_fingerprint != expected.rig_fingerprint
        or driven_order != expected.driven_joint_order
    ):
        raise ValueError("checkpoint metadata identity does not match the selected character")

    dataset = _mapping(metadata.get("dataset"), "checkpoint metadata.dataset")
    _exact_fields(dataset, {"schema", "format", "manifestSha256"}, "checkpoint metadata.dataset")
    if dataset.get("schema") != CHECKPOINT_DATASET_SCHEMA:
        raise ValueError("checkpoint metadata.dataset.schema is unsupported")
    if dataset.get("format") != CHECKPOINT_DATASET_FORMAT:
        raise ValueError("checkpoint metadata.dataset.format is unsupported")
    _sha256(dataset.get("manifestSha256"), "checkpoint metadata.dataset.manifestSha256")

    normalization = _mapping(
        metadata.get("normalization"), "checkpoint metadata.normalization"
    )
    _exact_fields(
        normalization,
        {"schema", "transform", "condition", "target"},
        "checkpoint metadata.normalization",
    )
    if normalization.get("schema") != CHECKPOINT_NORMALIZATION_SCHEMA:
        raise ValueError("checkpoint metadata.normalization.schema is unsupported")
    if normalization.get("transform") != CHECKPOINT_NORMALIZATION_TRANSFORM:
        raise ValueError("checkpoint metadata.normalization.transform is unsupported")
    for group_name in ("condition", "target"):
        _validate_normalization_group(
            normalization.get(group_name),
            f"checkpoint metadata.normalization.{group_name}",
        )

    model = _mapping(metadata.get("model"), "checkpoint metadata.model")
    _exact_fields(
        model,
        {"schema", "architecture", "implementationVersion", "config"},
        "checkpoint metadata.model",
    )
    if model.get("schema") != CHECKPOINT_MODEL_CONFIG_SCHEMA:
        raise ValueError("checkpoint metadata.model.schema is unsupported")
    _manifest_identifier(model.get("architecture"), "checkpoint metadata.model.architecture")
    implementation_version = model.get("implementationVersion")
    if not isinstance(implementation_version, str) or not 1 <= len(implementation_version) <= 128:
        raise ValueError(
            "checkpoint metadata.model.implementationVersion must be a non-empty string"
        )
    config = _mapping(model.get("config"), "checkpoint metadata.model.config")
    _validate_json_value(config, "checkpoint metadata.model.config")
    return metadata


def _derived_checkpoint_contract(
    character_id: str,
    rig_fingerprint: str,
    driven_joint_order: tuple[str, ...],
) -> CharacterCheckpointContract:
    return CharacterCheckpointContract(
        format=CHECKPOINT_FORMAT,
        metadata_schema=CHECKPOINT_METADATA_SCHEMA,
        path=checkpoint_target_path(character_id, rig_fingerprint),
        character_id=character_id,
        rig_fingerprint=rig_fingerprint,
        driven_joint_order=driven_joint_order,
        manifest_declared=False,
    )


def _validate_checkpoint_contract(
    value: Any,
    *,
    character_id: str,
    rig_fingerprint: str,
    driven_joint_order: tuple[str, ...],
    manifest_declared: bool,
) -> CharacterCheckpointContract:
    checkpoint = _mapping(value, "checkpoint")
    expected_fields = {
        "format",
        "metadataSchema",
        "path",
        "characterId",
        "rigFingerprint",
        "drivenJointOrder",
    }
    if set(checkpoint) != expected_fields:
        raise ValueError("checkpoint fields do not match the v1 checkpoint binding ABI")
    if checkpoint.get("format") != CHECKPOINT_FORMAT:
        raise ValueError("checkpoint.format is unsupported")
    if checkpoint.get("metadataSchema") != CHECKPOINT_METADATA_SCHEMA:
        raise ValueError("checkpoint.metadataSchema is unsupported")
    if checkpoint.get("characterId") != character_id:
        raise ValueError("checkpoint.characterId does not match characterId")
    if checkpoint.get("rigFingerprint") != rig_fingerprint:
        raise ValueError("checkpoint.rigFingerprint does not match rigFingerprint")
    checkpoint_order = _string_tuple(
        checkpoint.get("drivenJointOrder"),
        "checkpoint.drivenJointOrder",
    )
    if checkpoint_order != driven_joint_order:
        raise ValueError(
            "checkpoint.drivenJointOrder does not exactly match rig.drivenJointOrder"
        )
    expected_path = checkpoint_target_path(character_id, rig_fingerprint)
    if checkpoint.get("path") != expected_path:
        raise ValueError(
            "checkpoint.path must be the character-isolated canonical checkpoint target"
        )
    return CharacterCheckpointContract(
        format=CHECKPOINT_FORMAT,
        metadata_schema=CHECKPOINT_METADATA_SCHEMA,
        path=expected_path,
        character_id=character_id,
        rig_fingerprint=rig_fingerprint,
        driven_joint_order=driven_joint_order,
        manifest_declared=manifest_declared,
    )


def _validate_normalization_group(value: Any, name: str) -> None:
    group = _mapping(value, name)
    _exact_fields(group, {"featureOrder", "offset", "scale"}, name)
    feature_order = _string_tuple(group.get("featureOrder"), f"{name}.featureOrder")
    if (
        not 1 <= len(feature_order) <= 4096
        or len(set(feature_order)) != len(feature_order)
        or any(len(feature) > 128 for feature in feature_order)
    ):
        raise ValueError(f"{name}.featureOrder must contain 1..4096 unique features")
    offset = _finite_vector(group.get("offset"), len(feature_order), f"{name}.offset")
    scale = _finite_vector(group.get("scale"), len(feature_order), f"{name}.scale")
    if any(component <= 0 for component in scale):
        raise ValueError(f"{name}.scale must contain only positive values")
    if len(offset) != len(scale):  # Defensive clarity if vector validation changes.
        raise ValueError(f"{name} normalization vectors must have equal lengths")


def _exact_fields(value: Mapping[str, Any], expected: set[str], name: str) -> None:
    if set(value) != expected:
        raise ValueError(f"{name} fields do not match the v1 ABI")


def _validate_json_value(value: Any, name: str, depth: int = 0) -> None:
    if depth > 32:
        raise ValueError(f"{name} exceeds the maximum JSON nesting depth")
    if value is None or isinstance(value, (str, bool)):
        return
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if not math.isfinite(float(value)):
            raise ValueError(f"{name} contains a non-finite number")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_json_value(item, f"{name}[{index}]", depth + 1)
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{name} contains a non-string object key")
            _validate_json_value(item, f"{name}.{key}", depth + 1)
        return
    raise ValueError(f"{name} contains a non-JSON value")
