from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from pet_generator.character_rig import (
    CHARACTER_RIG_SCHEMA,
    CHECKPOINT_FORMAT,
    CHECKPOINT_METADATA_SCHEMA,
    CHECKPOINT_DATASET_FORMAT,
    CHECKPOINT_DATASET_SCHEMA,
    CHECKPOINT_MODEL_CONFIG_SCHEMA,
    CHECKPOINT_NORMALIZATION_SCHEMA,
    CHECKPOINT_NORMALIZATION_TRANSFORM,
    RIG_FINGERPRINT_ALGORITHM,
    RIG_FINGERPRINT_CANONICALIZATION,
    checkpoint_target_path,
    load_selected_character_rig,
    rig_fingerprint,
    validate_checkpoint_metadata,
)


def make_rig(driven_count: int = 2) -> dict:
    joints = [
        {
            "id": "motion_root",
            "parentIndex": -1,
            "restLocal": {
                "translation": [0, 0, 0],
                "rotation": [0, 0, 0, 1],
                "scale": [1, 1, 1],
            },
            "dofMask": {
                "translation": [True, True, True],
                "rotation": [True, True, True],
                "scale": [False, False, False],
            },
            "semanticRole": "motion_root",
            "source": None,
        }
    ]
    for index in range(driven_count):
        joints.append(
            {
                "id": f"joint_{index}",
                "parentIndex": index,
                "restLocal": {
                    "translation": [index + 0.5, 0, 0],
                    "rotation": [0, 0, 0, 1],
                    "scale": [1, 1, 1],
                },
                "dofMask": {
                    "translation": [False, False, False],
                    "rotation": [True, True, True],
                    "scale": [False, False, False],
                },
                "semanticRole": "appendage",
                "source": None,
            }
        )
    order = [joint["id"] for joint in joints]
    driven = order[1:]
    return {
        "coordinateSystem": {
            "handedness": "right",
            "up": "+Y",
            "forward": "+X",
            "units": "meter",
        },
        "motionRoot": "motion_root",
        "jointOrder": order,
        "joints": joints,
        "drivenJointOrder": driven,
        "drivenJointIndices": list(range(1, len(order))),
        "drivenParentIndices": list(range(-1, len(driven) - 1)),
        "maskGroups": {"model": [True] * len(driven)},
    }


def make_manifest(rig: dict, character_id: str = "test-character") -> dict:
    fingerprint = rig_fingerprint(rig)
    return {
        "schema": CHARACTER_RIG_SCHEMA,
        "characterId": character_id,
        "rigId": "test-rig",
        "rigFingerprint": {
            "algorithm": RIG_FINGERPRINT_ALGORITHM,
            "canonicalization": RIG_FINGERPRINT_CANONICALIZATION,
            "value": fingerprint,
        },
        "rig": rig,
        "checkpoint": {
            "format": CHECKPOINT_FORMAT,
            "metadataSchema": CHECKPOINT_METADATA_SCHEMA,
            "path": checkpoint_target_path(character_id, fingerprint),
            "characterId": character_id,
            "rigFingerprint": fingerprint,
            "drivenJointOrder": list(rig["drivenJointOrder"]),
        },
        "render": {
            "canvas": [48, 48],
            "displayScale": 2,
            "footAnchor": [24, 46],
            "sourceFacing": -1,
            "mode": "debug_skeleton",
            "fallbackModes": [],
            "sprite": None,
            "mesh": None,
        },
        "source": {"selectedSkinIndex": 0},
        "trainingClips": [],
        "provenance": {
            "generator": "test",
            "generatorVersion": 2,
            "profile": None,
            "f64Accumulation": "pet-left-to-right-f64-sum-v1",
        },
    }


def resign_manifest(manifest: dict) -> None:
    fingerprint = rig_fingerprint(manifest["rig"])
    manifest["rigFingerprint"]["value"] = fingerprint
    manifest["checkpoint"]["rigFingerprint"] = fingerprint
    manifest["checkpoint"]["drivenJointOrder"] = list(
        manifest["rig"]["drivenJointOrder"]
    )
    manifest["checkpoint"]["path"] = checkpoint_target_path(
        manifest["characterId"], fingerprint
    )


class CharacterRigTests(unittest.TestCase):
    def load_manifest(self, manifest: dict):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "character.manifest.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            with patch.dict(
                os.environ,
                {"PET_CHARACTER_MANIFEST": str(path)},
                clear=False,
            ):
                return load_selected_character_rig()

    def test_variable_joint_manifest_preserves_exact_output_order(self) -> None:
        selected = self.load_manifest(make_manifest(make_rig(3)))
        self.assertEqual(selected.character_id, "test-character")
        self.assertEqual(
            selected.driven_joint_order,
            ("joint_0", "joint_1", "joint_2"),
        )
        self.assertEqual(selected.checkpoint.format, CHECKPOINT_FORMAT)
        self.assertEqual(
            selected.checkpoint.metadata_schema,
            CHECKPOINT_METADATA_SCHEMA,
        )
        self.assertEqual(
            selected.checkpoint.driven_joint_order,
            selected.driven_joint_order,
        )
        self.assertTrue(selected.checkpoint.manifest_declared)

    def test_checkpoint_binding_rejects_invalid_format_identity_order_and_path(self) -> None:
        cases = [
            (
                "format",
                lambda checkpoint: checkpoint.update(format="pytorch-state-dict-v1"),
                "checkpoint.format is unsupported",
            ),
            (
                "metadata schema",
                lambda checkpoint: checkpoint.update(
                    metadataSchema="pet-character-motion-checkpoint-metadata-v2"
                ),
                "metadataSchema is unsupported",
            ),
            (
                "character identity",
                lambda checkpoint: checkpoint.update(characterId="another-character"),
                "characterId does not match",
            ),
            (
                "rig identity",
                lambda checkpoint: checkpoint.update(rigFingerprint="f" * 64),
                "rigFingerprint does not match",
            ),
            (
                "joint order",
                lambda checkpoint: checkpoint.update(
                    drivenJointOrder=list(reversed(checkpoint["drivenJointOrder"]))
                ),
                "drivenJointOrder does not exactly match",
            ),
            (
                "path",
                lambda checkpoint: checkpoint.update(
                    path="checkpoints/shared/motion.pt"
                ),
                "character-isolated canonical checkpoint target",
            ),
            (
                "unknown field",
                lambda checkpoint: checkpoint.update(sharedAcrossCharacters=True),
                "fields do not match",
            ),
        ]
        for name, mutate, error in cases:
            with self.subTest(name=name):
                manifest = make_manifest(make_rig())
                mutate(manifest["checkpoint"])
                with self.assertRaisesRegex(ValueError, error):
                    self.load_manifest(manifest)

    def test_checkpoint_metadata_binds_dataset_normalization_model_and_identity(self) -> None:
        selected = self.load_manifest(make_manifest(make_rig()))
        metadata = {
            "schema": CHECKPOINT_METADATA_SCHEMA,
            "characterId": selected.character_id,
            "rigFingerprint": selected.fingerprint,
            "drivenJointOrder": list(selected.driven_joint_order),
            "dataset": {
                "schema": CHECKPOINT_DATASET_SCHEMA,
                "format": CHECKPOINT_DATASET_FORMAT,
                "manifestSha256": "d" * 64,
            },
            "normalization": {
                "schema": CHECKPOINT_NORMALIZATION_SCHEMA,
                "transform": CHECKPOINT_NORMALIZATION_TRANSFORM,
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
                "schema": CHECKPOINT_MODEL_CONFIG_SCHEMA,
                "architecture": "test-transformer",
                "implementationVersion": "1.0.0",
                "config": {"hiddenSize": 64, "dropout": 0.1},
            },
        }
        self.assertIs(validate_checkpoint_metadata(metadata, selected.checkpoint), metadata)

        wrong_identity = json.loads(json.dumps(metadata))
        wrong_identity["characterId"] = "another-character"
        with self.assertRaisesRegex(ValueError, "identity does not match"):
            validate_checkpoint_metadata(wrong_identity, selected.checkpoint)

        ragged = json.loads(json.dumps(metadata))
        ragged["normalization"]["condition"]["scale"].pop()
        with self.assertRaisesRegex(ValueError, "must contain 2 numeric components"):
            validate_checkpoint_metadata(ragged, selected.checkpoint)

        non_positive = json.loads(json.dumps(metadata))
        non_positive["normalization"]["target"]["scale"] = [0.0]
        with self.assertRaisesRegex(ValueError, "only positive"):
            validate_checkpoint_metadata(non_positive, selected.checkpoint)

        non_finite_config = json.loads(json.dumps(metadata))
        non_finite_config["model"]["config"]["dropout"] = float("nan")
        with self.assertRaisesRegex(ValueError, "non-finite"):
            validate_checkpoint_metadata(non_finite_config, selected.checkpoint)

    def test_characters_with_the_same_rig_have_distinct_checkpoint_targets(self) -> None:
        rig = make_rig()
        first = self.load_manifest(make_manifest(rig, "character-a"))
        second = self.load_manifest(make_manifest(rig, "character-b"))
        self.assertEqual(first.fingerprint, second.fingerprint)
        self.assertNotEqual(first.checkpoint.path, second.checkpoint.path)
        self.assertIn("/character-a/", first.checkpoint.path)
        self.assertIn("/character-b/", second.checkpoint.path)

    def test_windows_unsafe_character_ids_are_rejected_before_path_derivation(self) -> None:
        for character_id in ("fox.", "nul", "nul.variant"):
            with self.subTest(character_id=character_id):
                manifest = make_manifest(make_rig())
                fingerprint = manifest["rigFingerprint"]["value"]
                manifest["characterId"] = character_id
                manifest["checkpoint"]["characterId"] = character_id
                manifest["checkpoint"]["path"] = (
                    f"checkpoints/characters/{character_id}/{fingerprint}/motion.pt"
                )
                with self.assertRaisesRegex(ValueError, "Windows-safe"):
                    self.load_manifest(manifest)
                with self.assertRaisesRegex(ValueError, "Windows-safe"):
                    checkpoint_target_path(character_id, fingerprint)

    def test_rig_ids_mask_names_and_joint_ids_use_the_shared_identifier_rules(self) -> None:
        invalid_rig_id = make_manifest(make_rig())
        invalid_rig_id["rigId"] = "UpperRig"
        with self.assertRaisesRegex(ValueError, "lowercase manifest identifier"):
            self.load_manifest(invalid_rig_id)

        invalid_mask = make_manifest(make_rig())
        invalid_mask["rig"]["maskGroups"]["😀"] = invalid_mask["rig"]["maskGroups"].pop("model")
        resign_manifest(invalid_mask)
        with self.assertRaisesRegex(ValueError, "lowercase manifest identifier"):
            self.load_manifest(invalid_mask)

        invalid_joint = make_manifest(make_rig())
        old_joint = invalid_joint["rig"]["drivenJointOrder"][0]
        invalid_joint["rig"]["jointOrder"][1] = "UPPER_JOINT"
        invalid_joint["rig"]["joints"][1]["id"] = "UPPER_JOINT"
        invalid_joint["rig"]["drivenJointOrder"][0] = "UPPER_JOINT"
        self.assertEqual(old_joint, "joint_0")
        resign_manifest(invalid_joint)
        with self.assertRaisesRegex(ValueError, "lowercase joint identifier"):
            self.load_manifest(invalid_joint)

    def test_legacy_rig_derives_target_without_declaring_an_artifact(self) -> None:
        legacy = {
            "schema": "pet-rig-v2",
            "motionRoot": "motion_root",
            "joints": [
                {
                    "id": "motion_root",
                    "parent": None,
                    "deform": False,
                    "physics": {"mode": "kinematic"},
                    "poseDofs": {"rotation": False},
                },
                {
                    "id": "body",
                    "parent": "motion_root",
                    "deform": True,
                    "physics": {"mode": "kinematic"},
                    "poseDofs": {"rotation": True},
                },
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "assets" / "pet" / "runtime" / "cat-skeleton-3d.json"
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps(legacy), encoding="utf-8")
            with patch.dict(
                os.environ,
                {"PET_CHARACTER_ID": "  legacy-fox  "},
                clear=True,
            ):
                selected = load_selected_character_rig(root=Path(directory))
        self.assertEqual(selected.source, "legacy_rig")
        self.assertFalse(selected.checkpoint.manifest_declared)
        self.assertEqual(
            selected.checkpoint.path,
            checkpoint_target_path("legacy-fox", selected.fingerprint),
        )
        self.assertEqual(selected.checkpoint.driven_joint_order, ("body",))

    def test_legacy_default_character_identity_is_stable_without_environment(self) -> None:
        legacy = {
            "schema": "pet-rig-v2",
            "motionRoot": "motion_root",
            "joints": [
                {
                    "id": "motion_root",
                    "parent": None,
                    "deform": False,
                    "physics": {"mode": "kinematic"},
                    "poseDofs": {"rotation": False},
                },
                {
                    "id": "body",
                    "parent": "motion_root",
                    "deform": True,
                    "physics": {"mode": "kinematic"},
                    "poseDofs": {"rotation": True},
                },
            ],
            "drawOrder": ["body"],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "assets" / "pet" / "runtime" / "cat-skeleton-3d.json"
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps(legacy), encoding="utf-8")
            with patch.dict(os.environ, {}, clear=True):
                selected = load_selected_character_rig(root=Path(directory))
                legacy["joints"][1]["parent"] = "missing"
                path.write_text(json.dumps(legacy), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "missing parent"):
                    load_selected_character_rig(root=Path(directory))
        self.assertEqual(selected.character_id, "legacy-default")
        self.assertIn("/legacy-default/", selected.checkpoint.path)

    def test_stale_fingerprint_is_rejected(self) -> None:
        manifest = make_manifest(make_rig())
        manifest["rig"]["coordinateSystem"]["units"] = "centimeter"
        with self.assertRaisesRegex(ValueError, "rigFingerprint does not match"):
            self.load_manifest(manifest)

    def test_protocol_joint_limit_is_enforced(self) -> None:
        with self.assertRaisesRegex(ValueError, "between 1 and 128"):
            self.load_manifest(make_manifest(make_rig(129)))

    def test_nearest_driven_parent_is_exact(self) -> None:
        manifest = make_manifest(make_rig(3))
        manifest["rig"]["drivenParentIndices"] = [-1, -1, -1]
        resign_manifest(manifest)
        with self.assertRaisesRegex(ValueError, "nearest-driven hierarchy"):
            self.load_manifest(manifest)

    def test_parallel_projection_axes_are_rejected(self) -> None:
        manifest = make_manifest(make_rig())
        manifest["rig"]["coordinateSystem"]["forward"] = "-Y"
        resign_manifest(manifest)
        with self.assertRaisesRegex(ValueError, "must be orthogonal"):
            self.load_manifest(manifest)

    def test_non_unit_rest_quaternion_is_rejected(self) -> None:
        manifest = make_manifest(make_rig())
        manifest["rig"]["joints"][1]["restLocal"]["rotation"] = [0, 0, 0, 2]
        resign_manifest(manifest)
        with self.assertRaisesRegex(ValueError, "unit quaternion"):
            self.load_manifest(manifest)

    def test_zero_rest_scale_is_rejected(self) -> None:
        manifest = make_manifest(make_rig())
        manifest["rig"]["joints"][1]["restLocal"]["scale"] = [1, 0, 1]
        resign_manifest(manifest)
        with self.assertRaisesRegex(ValueError, "cannot contain zero"):
            self.load_manifest(manifest)

    def test_mask_group_must_match_driven_order(self) -> None:
        manifest = make_manifest(make_rig())
        manifest["rig"]["maskGroups"]["model"] = [True]
        resign_manifest(manifest)
        with self.assertRaisesRegex(ValueError, "must contain 2 booleans"):
            self.load_manifest(manifest)

    def test_runtime_canvas_is_normalized(self) -> None:
        manifest = make_manifest(make_rig())
        manifest["render"]["canvas"] = [64, 64]
        with self.assertRaisesRegex(ValueError, "normalized 48x48"):
            self.load_manifest(manifest)

    def test_runtime_display_scale_is_fixed(self) -> None:
        manifest = make_manifest(make_rig())
        manifest["render"]["displayScale"] = 3
        with self.assertRaisesRegex(ValueError, "must be 2"):
            self.load_manifest(manifest)

    def test_fingerprint_matches_typescript_fixed_vector(self) -> None:
        self.assertEqual(
            rig_fingerprint(
                {
                    "unicode": "猫",
                    "integer": 1,
                    "negativeZero": -0.0,
                    "array": [1.5, True],
                }
            ),
            "e9aa44fdcd0ffd83017e623bc861bc5cf3ccdcaf60ebf751ef1e017fd67ef4f9",
        )
        self.assertEqual(
            rig_fingerprint({"maskGroups": {"\ue000": [True], "😀": [False]}}),
            "779163f1807236f11dc08265441105732b027b00503f465a8de2602c65c4364c",
        )

    def test_fingerprint_rejects_non_scalar_unicode_in_nested_keys_and_values(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unicode scalar sequences"):
            rig_fingerprint({"nested": {"value": "\ud800"}})
        with self.assertRaisesRegex(ValueError, "Unicode scalar sequences"):
            rig_fingerprint({"nested": {"\udc00": True}})
        self.assertIsInstance(rig_fingerprint({"nested": {"😀": "猫"}}), str)


if __name__ == "__main__":
    unittest.main()
