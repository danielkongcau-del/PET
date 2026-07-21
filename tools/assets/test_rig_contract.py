from __future__ import annotations

import json
import math
import struct
import sys
import tempfile
import unittest
from pathlib import Path

import jsonschema


ROOT = Path(__file__).resolve().parents[2]
RIG_TOOLS = ROOT / "tools" / "rig"
sys.path.insert(0, str(RIG_TOOLS))

from gltf_character import (  # noqa: E402
    F64_ACCUMULATION,
    _f64_sum,
    build_rig_manifest,
    clip_fingerprint_is_valid,
    extract_animation_payload,
    load_gltf,
    require_source,
)
from rig_contract import (  # noqa: E402
    CHECKPOINT_FORMAT,
    CHECKPOINT_METADATA_SCHEMA,
    canonical_fingerprint_bytes,
    checkpoint_target_path,
    fingerprint_record,
    fingerprint_value,
    nearest_driven_parent_indices,
    validate_animation,
    validate_checkpoint_metadata,
    validate_manifest,
    validate_rig_contract,
)


def _joint(joint_id: str, parent_index: int) -> dict:
    return {
        "id": joint_id,
        "parentIndex": parent_index,
        "restLocal": {
            "translation": [0.0, 0.0, 0.0],
            "rotation": [0.0, 0.0, 0.0, 1.0],
            "scale": [1.0, 1.0, 1.0],
        },
        "dofMask": {
            "translation": [False, False, False],
            "rotation": [parent_index >= 0] * 3,
            "scale": [False, False, False],
        },
        "semanticRole": "motion_root" if parent_index < 0 else "joint",
        "source": None,
    }


def _chain_rig(driven_count: int) -> dict:
    joint_order = ["root"] + [f"j_{index}" for index in range(driven_count)]
    joints = [_joint("root", -1)] + [
        _joint(joint_id, index) for index, joint_id in enumerate(joint_order[1:])
    ]
    driven_order = joint_order[1:]
    driven_indices, driven_parents = nearest_driven_parent_indices(
        joint_order, [joint["parentIndex"] for joint in joints], driven_order
    )
    return {
        "coordinateSystem": {
            "handedness": "right",
            "up": "+Y",
            "forward": "+Z",
            "units": "meter",
        },
        "motionRoot": "root",
        "jointOrder": joint_order,
        "joints": joints,
        "drivenJointOrder": driven_order,
        "drivenJointIndices": driven_indices,
        "drivenParentIndices": driven_parents,
        "maskGroups": {"all": [True] * driven_count},
    }


def _append(blob: bytearray, values: list[float]) -> tuple[int, int]:
    while len(blob) % 4:
        blob.append(0)
    offset = len(blob)
    blob.extend(struct.pack("<" + "f" * len(values), *values))
    return offset, len(values) * 4


def _write_synthetic_gltf(directory: Path) -> Path:
    blob = bytearray()
    identity = [
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1,
    ]
    views: list[dict] = []
    accessors: list[dict] = []

    def add(values: list[float], accessor_type: str, count: int) -> int:
        offset, length = _append(blob, values)
        views.append({"buffer": 0, "byteOffset": offset, "byteLength": length})
        accessors.append(
            {
                "bufferView": len(views) - 1,
                "componentType": 5126,
                "count": count,
                "type": accessor_type,
            }
        )
        return len(accessors) - 1

    inverse_accessor = add(identity + identity, "MAT4", 2)
    time_accessor = add([0.0, 1.0], "SCALAR", 2)
    rotation_accessor = add(
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, math.sin(math.pi / 4), math.cos(math.pi / 4)],
        "VEC4",
        2,
    )
    translation_accessor = add([0.0, 1.0, 0.0, 0.0, 2.0, 0.0], "VEC3", 2)
    # Mesh accessor references are provenance only for this fixture.
    position_accessor = add([0.0, 0.0, 0.0], "VEC3", 1)

    gltf = {
        "asset": {"version": "2.0"},
        "scene": 0,
        "scenes": [{"nodes": [0, 3]}],
        "nodes": [
            {"name": "Outside", "translation": [10.0, 0.0, 0.0], "children": [1]},
            {
                "name": "RootBone",
                "matrix": [
                    1, 0, 0, 0,
                    0, 1, 0, 0,
                    0, 0, 1, 0,
                    1, 2, 0, 1,
                ],
                "children": [2],
            },
            {"name": "ChildBone", "translation": [0.0, 1.0, 0.0]},
            {"name": "MeshNode", "mesh": 0, "skin": 0},
        ],
        "buffers": [{"uri": "fixture.bin", "byteLength": len(blob)}],
        "bufferViews": views,
        "accessors": accessors,
        "meshes": [
            {
                "primitives": [
                    {
                        "attributes": {"POSITION": position_accessor},
                        "mode": 4,
                    }
                ]
            }
        ],
        "skins": [
            {
                "name": "FixtureSkin",
                "skeleton": 1,
                "joints": [1, 2],
                "inverseBindMatrices": inverse_accessor,
            }
        ],
        "animations": [
            {
                "name": "Move",
                "samplers": [
                    {"input": time_accessor, "output": rotation_accessor, "interpolation": "LINEAR"},
                    {"input": time_accessor, "output": translation_accessor, "interpolation": "LINEAR"},
                ],
                "channels": [
                    {"sampler": 0, "target": {"node": 1, "path": "rotation"}},
                    {"sampler": 1, "target": {"node": 2, "path": "translation"}},
                ],
            }
        ],
    }
    (directory / "fixture.bin").write_bytes(blob)
    source = directory / "fixture.gltf"
    source.write_text(json.dumps(gltf), encoding="utf-8")
    return source


def _fixture_profile() -> dict:
    return {
        "characterId": "fixture",
        "rigId": "fixture-rig-v1",
        "drivenSourceNames": ["RootBone", "ChildBone"],
        "semanticRoles": {"RootBone": "root_bone", "ChildBone": "tip"},
        "maskGroups": {"all": ["RootBone", "ChildBone"], "tip": ["ChildBone"]},
        "sourceAvailableInCleanClone": True,
        "render": {
            "canvas": [48, 48],
            "displayScale": 2,
            "footAnchor": [24, 46],
            "sourceFacing": 1,
            "mode": "debug_skeleton",
            "fallbackModes": [],
            "sprite": None,
            "mesh": None,
        },
        "checkpoint": {
            "format": CHECKPOINT_FORMAT,
            "metadataSchema": CHECKPOINT_METADATA_SCHEMA,
        },
    }


def _checkpoint_metadata(checkpoint: dict) -> dict:
    return {
        "schema": CHECKPOINT_METADATA_SCHEMA,
        "characterId": checkpoint["characterId"],
        "rigFingerprint": checkpoint["rigFingerprint"],
        "drivenJointOrder": list(checkpoint["drivenJointOrder"]),
        "dataset": {
            "schema": "pet-training-dataset-v1",
            "format": "episode-ndjson-v1",
            "manifestSha256": "d" * 64,
        },
        "normalization": {
            "schema": "pet-feature-normalization-v1",
            "transform": "(value-offset)/scale",
            "condition": {
                "featureOrder": ["condition-x", "condition-y"],
                "offset": [1.0, 2.0],
                "scale": [0.5, 4.0],
            },
            "target": {
                "featureOrder": ["target-x"],
                "offset": [0.0],
                "scale": [1.0],
            },
        },
        "model": {
            "schema": "pet-motion-model-config-v1",
            "architecture": "test-transformer",
            "implementationVersion": "1.0.0",
            "config": {"hiddenSize": 64, "dropout": 0.1},
        },
    }


class RigFingerprintTests(unittest.TestCase):
    def test_transform_accumulation_is_versioned_and_left_to_right(self) -> None:
        self.assertEqual(F64_ACCUMULATION, "pet-left-to-right-f64-sum-v1")
        self.assertEqual(_f64_sum([1e16, 1.0, -1e16]), 0.0)

    def test_known_vector_is_order_numeric_and_negative_zero_stable(self) -> None:
        left = {"z": -0.0, "a": [1, 1.0, "猫", True]}
        right = {"a": [1.0, 1, "猫", True], "z": 0}
        self.assertEqual(canonical_fingerprint_bytes(left), canonical_fingerprint_bytes(right))
        self.assertEqual(
            "b21bd2bf16a62a6fa551caf656b1beabb0052a76d57d4a281f313c76bde87273",
            fingerprint_value(left),
        )

    def test_non_bmp_object_keys_use_unicode_scalar_order(self) -> None:
        self.assertEqual(
            "779163f1807236f11dc08265441105732b027b00503f465a8de2602c65c4364c",
            fingerprint_value({"maskGroups": {"\ue000": [True], "😀": [False]}}),
        )

    def test_fingerprint_rejects_non_scalar_unicode_in_nested_keys_and_values(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unicode scalar sequences"):
            fingerprint_value({"nested": {"value": "\ud800"}})
        with self.assertRaisesRegex(ValueError, "Unicode scalar sequences"):
            fingerprint_value({"nested": {"\udc00": True}})
        self.assertIsInstance(fingerprint_value({"nested": {"😀": "猫"}}), str)

    def test_variable_joint_counts_and_protocol_limit(self) -> None:
        for count in (1, 17, 128):
            with self.subTest(count=count):
                rig = _chain_rig(count)
                validate_rig_contract(rig)
                self.assertEqual(count, len(rig["drivenJointIndices"]))
                self.assertEqual([-1] + list(range(count - 1)), rig["drivenParentIndices"])
        with self.assertRaisesRegex(ValueError, "128"):
            validate_rig_contract(_chain_rig(129))

    def test_invalid_rest_quaternions_and_coordinate_axes_are_rejected(self) -> None:
        zero = _chain_rig(1)
        zero["joints"][1]["restLocal"]["rotation"] = [0.0, 0.0, 0.0, 0.0]
        with self.assertRaisesRegex(ValueError, "unit quaternion"):
            validate_rig_contract(zero)

        non_unit = _chain_rig(1)
        non_unit["joints"][1]["restLocal"]["rotation"] = [0.0, 0.0, 0.0, 0.5]
        with self.assertRaisesRegex(ValueError, "unit quaternion"):
            validate_rig_contract(non_unit)

        parallel = _chain_rig(1)
        parallel["coordinateSystem"]["forward"] = "-Y"
        with self.assertRaisesRegex(ValueError, "orthogonal"):
            validate_rig_contract(parallel)

    def test_joint_mask_and_character_identifiers_share_the_portable_contract(self) -> None:
        invalid_joint = _chain_rig(1)
        invalid_joint["jointOrder"][1] = "UpperJoint"
        invalid_joint["joints"][1]["id"] = "UpperJoint"
        invalid_joint["drivenJointOrder"][0] = "UpperJoint"
        with self.assertRaisesRegex(ValueError, "lowercase joint identifier"):
            validate_rig_contract(invalid_joint)

        invalid_mask = _chain_rig(1)
        invalid_mask["maskGroups"] = {"UpperMask": [True]}
        with self.assertRaisesRegex(ValueError, "lowercase manifest identifier"):
            validate_rig_contract(invalid_mask)

        for character_id in ("fox.", "nul", "nul.variant", ".", ".."):
            with self.subTest(character_id=character_id):
                with self.assertRaises(ValueError):
                    checkpoint_target_path(character_id, "0" * 64)


class GltfCharacterExtractionTests(unittest.TestCase):
    def test_trs_matrix_skin_mesh_ibm_and_animation_are_exported(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            source = _write_synthetic_gltf(directory)
            document = load_gltf(source)
            manifest = build_rig_manifest(document, _fixture_profile(), repository_root=directory)
            clip = extract_animation_payload(
                document, manifest, 0, clip_id="move", sample_rate_hz=2.0
            )
            loop_clip = extract_animation_payload(
                document,
                manifest,
                0,
                clip_id="move-loop",
                sample_rate_hz=2.0,
                playback_mode="loop",
            )

        validate_manifest(manifest)
        self.assertEqual(3, len(manifest["rig"]["jointOrder"]))
        self.assertEqual([1, 2], manifest["rig"]["drivenJointIndices"])
        self.assertEqual([-1, 0], manifest["rig"]["drivenParentIndices"])
        # The selected root's excluded parent transform is baked into its rest TRS.
        self.assertEqual([11.0, 2.0, 0.0], manifest["rig"]["joints"][1]["restLocal"]["translation"])
        self.assertEqual(2, len(manifest["source"]["skins"][0]["inverseBindMatrices"]))
        self.assertEqual(1, len(manifest["source"]["meshSkinReferences"]))
        self.assertEqual(1, len(manifest["source"]["animations"]))
        fingerprint = manifest["rigFingerprint"]["value"]
        self.assertEqual(
            manifest["checkpoint"],
            {
                "format": CHECKPOINT_FORMAT,
                "metadataSchema": CHECKPOINT_METADATA_SCHEMA,
                "path": checkpoint_target_path("fixture", fingerprint),
                "characterId": "fixture",
                "rigFingerprint": fingerprint,
                "drivenJointOrder": manifest["rig"]["drivenJointOrder"],
            },
        )
        self.assertEqual(3, len(clip["frames"]))
        self.assertTrue(clip_fingerprint_is_valid(clip))
        validate_animation(clip)
        self.assertNotEqual(
            clip["frames"][0]["localRotations"],
            clip["frames"][-1]["localRotations"],
        )
        self.assertNotEqual(
            clip["frames"][0]["localTranslations"],
            clip["frames"][-1]["localTranslations"],
        )
        self.assertEqual(loop_clip["playbackMode"], "loop")
        self.assertTrue(
            all(
                abs(abs(sum(left * right for left, right in zip(first, last))) - 1.0)
                <= 1e-8
                for first, last in zip(
                    loop_clip["frames"][0]["localRotations"],
                    loop_clip["frames"][-1]["localRotations"],
                )
            )
        )
        self.assertEqual(
            loop_clip["frames"][0]["localTranslations"],
            loop_clip["frames"][-1]["localTranslations"],
        )
        validate_animation(loop_clip)

    def test_checkpoint_binding_rejects_shared_or_mismatched_targets(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            source = _write_synthetic_gltf(directory)
            manifest = build_rig_manifest(
                load_gltf(source), _fixture_profile(), repository_root=directory
            )

        cases = [
            ("format", "pytorch-state-dict-v1", "format is unsupported"),
            (
                "metadataSchema",
                "pet-character-motion-checkpoint-metadata-v2",
                "metadataSchema is unsupported",
            ),
            ("characterId", "other", "characterId must match"),
            ("rigFingerprint", "f" * 64, "rigFingerprint must match"),
            ("path", "checkpoints/shared/motion.pt", "character-isolated"),
            ("drivenJointOrder", list(reversed(manifest["rig"]["drivenJointOrder"])), "exactly match"),
        ]
        for field, value, error in cases:
            with self.subTest(field=field):
                invalid = json.loads(json.dumps(manifest))
                invalid["checkpoint"][field] = value
                with self.assertRaisesRegex(ValueError, error):
                    validate_manifest(invalid)

        extra = json.loads(json.dumps(manifest))
        extra["checkpoint"]["sharedAcrossCharacters"] = True
        with self.assertRaisesRegex(ValueError, "fields do not match"):
            validate_manifest(extra)

    def test_profile_identifiers_are_validated_before_export(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            document = load_gltf(_write_synthetic_gltf(directory))

            invalid_rig_id = _fixture_profile()
            invalid_rig_id["rigId"] = "UpperRig"
            with self.assertRaisesRegex(ValueError, "lowercase manifest identifier"):
                build_rig_manifest(document, invalid_rig_id, repository_root=directory)

            invalid_character_id = _fixture_profile()
            invalid_character_id["characterId"] = "nul.variant"
            with self.assertRaisesRegex(ValueError, "Windows-safe"):
                build_rig_manifest(document, invalid_character_id, repository_root=directory)

            invalid_mask = _fixture_profile()
            invalid_mask["maskGroups"] = {"UpperMask": ["RootBone"]}
            with self.assertRaisesRegex(ValueError, "lowercase manifest identifier"):
                build_rig_manifest(document, invalid_mask, repository_root=directory)

    def test_checkpoint_metadata_validator_is_bound_and_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            source = _write_synthetic_gltf(directory)
            manifest = build_rig_manifest(
                load_gltf(source), _fixture_profile(), repository_root=directory
            )
        metadata = _checkpoint_metadata(manifest["checkpoint"])
        validate_checkpoint_metadata(metadata, manifest["checkpoint"])

        wrong_identity = json.loads(json.dumps(metadata))
        wrong_identity["characterId"] = "another-character"
        with self.assertRaisesRegex(ValueError, "identity does not match"):
            validate_checkpoint_metadata(wrong_identity, manifest["checkpoint"])

        ragged = json.loads(json.dumps(metadata))
        ragged["normalization"]["condition"]["scale"].pop()
        with self.assertRaisesRegex(ValueError, "must contain 2 numbers"):
            validate_checkpoint_metadata(ragged, manifest["checkpoint"])

        non_positive = json.loads(json.dumps(metadata))
        non_positive["normalization"]["target"]["scale"] = [0.0]
        with self.assertRaisesRegex(ValueError, "only positive"):
            validate_checkpoint_metadata(non_positive, manifest["checkpoint"])

    def test_multiple_skins_require_explicit_selection(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            source = _write_synthetic_gltf(directory)
            data = json.loads(source.read_text(encoding="utf-8"))
            data["skins"].append(dict(data["skins"][0]))
            source.write_text(json.dumps(data), encoding="utf-8")
            document = load_gltf(source)
            with self.assertRaisesRegex(ValueError, "2 skins.*skin-index"):
                build_rig_manifest(document, _fixture_profile(), repository_root=directory)
            selected = build_rig_manifest(
                document, _fixture_profile(), repository_root=directory, skin_index=1
            )
        self.assertEqual(1, selected["source"]["selectedSkinIndex"])

    def test_missing_source_error_explains_clean_clone_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            missing = Path(temporary_directory) / "assets" / "Cat" / "scene.gltf"
            with self.assertRaisesRegex(FileNotFoundError, "not tracked.*checked-in"):
                require_source(missing)

    def test_accessor_cannot_read_past_its_buffer_view(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            source = _write_synthetic_gltf(directory)
            data = json.loads(source.read_text(encoding="utf-8"))
            data["bufferViews"][0]["byteLength"] -= 4
            source.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "beyond its bufferView"):
                build_rig_manifest(load_gltf(source), _fixture_profile(), repository_root=directory)

    def test_animation_requires_strict_float_accessors(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            source = _write_synthetic_gltf(directory)
            data = json.loads(source.read_text(encoding="utf-8"))
            time_view = data["bufferViews"][data["accessors"][1]["bufferView"]]
            binary_path = directory / data["buffers"][0]["uri"]
            blob = bytearray(binary_path.read_bytes())
            struct.pack_into("<f", blob, time_view["byteOffset"] + 4, 0.0)
            binary_path.write_bytes(blob)
            with self.assertRaisesRegex(ValueError, "strictly increasing"):
                build_rig_manifest(load_gltf(source), _fixture_profile(), repository_root=directory)

        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            source = _write_synthetic_gltf(directory)
            data = json.loads(source.read_text(encoding="utf-8"))
            data["accessors"][2]["componentType"] = 5123
            source.write_text(json.dumps(data), encoding="utf-8")
            document = load_gltf(source)
            manifest = build_rig_manifest(document, _fixture_profile(), repository_root=directory)
            with self.assertRaisesRegex(ValueError, "FLOAT VEC4"):
                extract_animation_payload(
                    document, manifest, 0, clip_id="move", sample_rate_hz=2.0
                )

    def test_inverse_bind_matrices_require_float_mat4(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            source = _write_synthetic_gltf(directory)
            data = json.loads(source.read_text(encoding="utf-8"))
            data["accessors"][0]["componentType"] = 5123
            source.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "FLOAT MAT4"):
                build_rig_manifest(load_gltf(source), _fixture_profile(), repository_root=directory)


class CheckedCharacterAssetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        runtime = ROOT / "assets" / "pet" / "runtime"
        cls.manifest_path = runtime / "cat-character-rig.manifest.json"
        cls.clip_path = runtime / "cat-being-cute.animation.json"
        cls.manifest = json.loads(cls.manifest_path.read_text(encoding="utf-8"))
        cls.clip = json.loads(cls.clip_path.read_text(encoding="utf-8"))
        cls.manifest_schema = json.loads(
            (runtime / "character-rig-manifest.schema.json").read_text(encoding="utf-8")
        )
        cls.animation_schema = json.loads(
            (runtime / "character-animation.schema.json").read_text(encoding="utf-8")
        )
        cls.checkpoint_metadata_schema = json.loads(
            (runtime / "character-motion-checkpoint-metadata.schema.json").read_text(
                encoding="utf-8"
            )
        )

    def test_schemas_and_checked_assets_validate(self) -> None:
        jsonschema.Draft202012Validator.check_schema(self.manifest_schema)
        jsonschema.Draft202012Validator.check_schema(self.animation_schema)
        jsonschema.Draft202012Validator.check_schema(self.checkpoint_metadata_schema)
        jsonschema.Draft202012Validator(self.manifest_schema).validate(self.manifest)
        jsonschema.Draft202012Validator(self.animation_schema).validate(self.clip)
        validate_manifest(self.manifest)
        validate_animation(self.clip)
        self.assertTrue(clip_fingerprint_is_valid(self.clip))

    def test_checkpoint_metadata_schema_accepts_only_self_describing_metadata(self) -> None:
        metadata = _checkpoint_metadata(self.manifest["checkpoint"])
        jsonschema.Draft202012Validator(self.checkpoint_metadata_schema).validate(metadata)
        validate_checkpoint_metadata(metadata, self.manifest["checkpoint"])

        raw_state_dict = {
            "schema": CHECKPOINT_METADATA_SCHEMA,
            "characterId": "cat",
            "rigFingerprint": self.manifest["rigFingerprint"]["value"],
            "drivenJointOrder": self.manifest["rig"]["drivenJointOrder"],
        }
        with self.assertRaises(jsonschema.ValidationError):
            jsonschema.Draft202012Validator(self.checkpoint_metadata_schema).validate(
                raw_state_dict
            )

    def test_official_schemas_share_character_joint_rig_and_mask_identifier_rules(self) -> None:
        manifest_validator = jsonschema.Draft202012Validator(self.manifest_schema)
        animation_validator = jsonschema.Draft202012Validator(self.animation_schema)
        metadata_validator = jsonschema.Draft202012Validator(self.checkpoint_metadata_schema)

        for character_id in ("fox.", "nul", "nul.variant", ".", ".."):
            with self.subTest(contract="character", character_id=character_id):
                invalid_manifest = json.loads(json.dumps(self.manifest))
                invalid_manifest["characterId"] = character_id
                invalid_manifest["checkpoint"]["characterId"] = character_id
                invalid_manifest["checkpoint"]["path"] = (
                    f"checkpoints/characters/{character_id}/"
                    f"{invalid_manifest['rigFingerprint']['value']}/motion.pt"
                )
                self.assertFalse(manifest_validator.is_valid(invalid_manifest))

                invalid_animation = json.loads(json.dumps(self.clip))
                invalid_animation["characterId"] = character_id
                self.assertFalse(animation_validator.is_valid(invalid_animation))

                invalid_metadata = _checkpoint_metadata(self.manifest["checkpoint"])
                invalid_metadata["characterId"] = character_id
                self.assertFalse(metadata_validator.is_valid(invalid_metadata))

        invalid_joint_manifest = json.loads(json.dumps(self.manifest))
        invalid_joint_manifest["rig"]["jointOrder"][1] = "UpperJoint"
        invalid_joint_manifest["rig"]["joints"][1]["id"] = "UpperJoint"
        self.assertFalse(manifest_validator.is_valid(invalid_joint_manifest))

        invalid_joint_animation = json.loads(json.dumps(self.clip))
        invalid_joint_animation["jointOrder"][0] = "UpperJoint"
        self.assertFalse(animation_validator.is_valid(invalid_joint_animation))

        invalid_joint_metadata = _checkpoint_metadata(self.manifest["checkpoint"])
        invalid_joint_metadata["drivenJointOrder"][0] = "UpperJoint"
        self.assertFalse(metadata_validator.is_valid(invalid_joint_metadata))

        invalid_rig_id = json.loads(json.dumps(self.manifest))
        invalid_rig_id["rigId"] = "UpperRig"
        self.assertFalse(manifest_validator.is_valid(invalid_rig_id))

        invalid_mask = json.loads(json.dumps(self.manifest))
        mask = next(iter(invalid_mask["rig"]["maskGroups"].values()))
        invalid_mask["rig"]["maskGroups"] = {"UpperMask": mask}
        self.assertFalse(manifest_validator.is_valid(invalid_mask))

    def test_animation_validator_rejects_joint_frame_mismatch(self) -> None:
        invalid = json.loads(json.dumps(self.clip))
        invalid["frames"][0]["localRotations"].pop()
        invalid["clipFingerprint"] = fingerprint_record(
            {key: value for key, value in invalid.items() if key != "clipFingerprint"}
        )
        with self.assertRaisesRegex(ValueError, "align with jointOrder"):
            validate_animation(invalid)

    def test_checked_cat_contract_is_generic_non_identity_and_provenanced(self) -> None:
        rig = self.manifest["rig"]
        driven_count = len(rig["drivenJointOrder"])
        self.assertGreater(driven_count, 1)
        self.assertLessEqual(driven_count, 128)
        self.assertEqual(driven_count, len(rig["drivenJointIndices"]))
        self.assertEqual(driven_count, len(rig["drivenParentIndices"]))
        self.assertTrue(all(len(mask) == driven_count for mask in rig["maskGroups"].values()))
        self.assertEqual(rig["drivenJointOrder"], self.clip["jointOrder"])
        self.assertFalse(self.manifest["source"]["availableInCleanClone"])
        self.assertTrue(self.manifest["source"]["requiredAtBuild"])
        self.assertGreater(len(self.manifest["source"]["skins"][0]["inverseBindMatrices"]), 1)
        self.assertGreater(len(self.manifest["source"]["meshSkinReferences"]), 0)
        self.assertGreater(len(self.manifest["source"]["animations"]), 0)
        fingerprint = self.manifest["rigFingerprint"]["value"]
        self.assertEqual(
            fingerprint,
            "518cf8f9fc29e5c8286f16cfe930ad022fbd2153dd8c3ef68121c8d7c58df0d7",
        )
        self.assertEqual(
            self.clip["clipFingerprint"]["value"],
            "ca95578105deccb5c06d0b631b8b9921b8bb82f03392b0ff3d533f85725dd6de",
        )
        self.assertEqual(self.manifest["provenance"]["generatorVersion"], 2)
        self.assertEqual(
            self.manifest["provenance"]["f64Accumulation"],
            F64_ACCUMULATION,
        )
        self.assertEqual(self.manifest["checkpoint"]["format"], CHECKPOINT_FORMAT)
        self.assertEqual(
            self.manifest["checkpoint"]["metadataSchema"],
            CHECKPOINT_METADATA_SCHEMA,
        )
        self.assertEqual(
            self.manifest["checkpoint"]["path"],
            checkpoint_target_path("cat", fingerprint),
        )
        self.assertEqual(
            self.manifest["checkpoint"]["drivenJointOrder"],
            rig["drivenJointOrder"],
        )
        self.assertEqual(
            self.clip["clipFingerprint"]["value"],
            self.manifest["trainingClips"][0]["fingerprint"],
        )
        self.assertEqual(self.clip["playbackMode"], "loop")
        self.assertEqual(
            self.clip["playbackMode"],
            self.manifest["trainingClips"][0]["playbackMode"],
        )
        self.assertTrue(
            all(
                abs(abs(sum(left * right for left, right in zip(first, last))) - 1.0)
                <= 1e-8
                for first, last in zip(
                    self.clip["frames"][0]["localRotations"],
                    self.clip["frames"][-1]["localRotations"],
                )
            ),
            "Loop training clips must have an equivalent duration seam",
        )
        first = self.clip["frames"][0]
        self.assertTrue(
            any(
                frame["localRotations"] != first["localRotations"]
                or frame["localTranslations"] != first["localTranslations"]
                for frame in self.clip["frames"][1:]
            ),
            "The checked training clip must contain real, non-identity motion",
        )
        previous = None
        for frame in self.clip["frames"]:
            rotations = frame["localRotations"]
            if previous is not None:
                for before, after in zip(previous, rotations):
                    self.assertGreaterEqual(
                        sum(left * right for left, right in zip(before, after)),
                        -1e-8,
                        "Adjacent authored quaternions must stay in one hemisphere",
                    )
            previous = rotations


if __name__ == "__main__":
    unittest.main()
