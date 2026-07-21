import assert from "node:assert/strict";
import { mkdir, mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";

import {
  CHARACTER_RIG_SCHEMA,
  CHECKPOINT_FORMAT,
  CHECKPOINT_METADATA_SCHEMA,
  CHECKPOINT_DATASET_FORMAT,
  CHECKPOINT_DATASET_SCHEMA,
  CHECKPOINT_MODEL_CONFIG_SCHEMA,
  CHECKPOINT_NORMALIZATION_SCHEMA,
  CHECKPOINT_NORMALIZATION_TRANSFORM,
  checkpointTargetPath,
  loadCharacterRigConfig,
  rigFingerprint,
  RIG_FINGERPRINT_ALGORITHM,
  RIG_FINGERPRINT_CANONICALIZATION,
  validateCharacterCheckpointMetadata,
} from "../src/character-rig.js";

function makeRig(drivenCount = 2): Record<string, unknown> {
  const joints: Array<Record<string, unknown>> = [{
    id: "motion_root",
    parentIndex: -1,
    restLocal: { translation: [0, 0, 0], rotation: [0, 0, 0, 1], scale: [1, 1, 1] },
    dofMask: { translation: [true, true, true], rotation: [true, true, true], scale: [false, false, false] },
    semanticRole: "motion_root",
    source: null,
  }];
  for (let index = 0; index < drivenCount; index += 1) {
    joints.push({
      id: `joint_${index}`,
      parentIndex: index,
      restLocal: { translation: [index + 0.5, 0, 0], rotation: [0, 0, 0, 1], scale: [1, 1, 1] },
      dofMask: { translation: [false, false, false], rotation: [true, true, true], scale: [false, false, false] },
      semanticRole: index % 2 === 0 ? "spine" : "appendage",
      source: null,
    });
  }
  const jointOrder = joints.map((joint) => String(joint.id));
  const drivenJointOrder = jointOrder.slice(1);
  return {
    coordinateSystem: { handedness: "right", up: "+Y", forward: "+X", units: "meter" },
    motionRoot: "motion_root",
    jointOrder,
    joints,
    drivenJointOrder,
    drivenJointIndices: drivenJointOrder.map((_id, index) => index + 1),
    drivenParentIndices: drivenJointOrder.map((_id, index) => index - 1),
    maskGroups: { model: drivenJointOrder.map(() => true) },
  };
}

function makeManifest(rig: Record<string, unknown>, characterId = "test-character"): Record<string, unknown> {
  const fingerprint = rigFingerprint(rig);
  const drivenJointOrder = rig.drivenJointOrder as string[];
  return {
    schema: CHARACTER_RIG_SCHEMA,
    characterId,
    rigId: "test-rig",
    rigFingerprint: {
      algorithm: RIG_FINGERPRINT_ALGORITHM,
      canonicalization: RIG_FINGERPRINT_CANONICALIZATION,
      value: fingerprint,
    },
    rig,
    render: {
      canvas: [48, 48],
      displayScale: 2,
      footAnchor: [24, 46],
      sourceFacing: -1,
      mode: "debug_skeleton",
      fallbackModes: ["sprite"],
      sprite: { image: "character.png", metadata: "character-parts.json" },
      mesh: null,
    },
    checkpoint: {
      format: CHECKPOINT_FORMAT,
      metadataSchema: CHECKPOINT_METADATA_SCHEMA,
      path: checkpointTargetPath(characterId, fingerprint),
      characterId,
      rigFingerprint: fingerprint,
      drivenJointOrder: [...drivenJointOrder],
    },
    source: {},
    trainingClips: [],
    provenance: {
      generator: "test",
      generatorVersion: 2,
      profile: null,
      f64Accumulation: "pet-left-to-right-f64-sum-v1",
    },
  };
}

async function withManifest(manifest: Record<string, unknown>, run: (path: string) => Promise<void>): Promise<void> {
  const directory = await mkdtemp(join(tmpdir(), "pet-character-rig-"));
  const path = join(directory, "character.manifest.json");
  try {
    await writeFile(path, `${JSON.stringify(manifest)}\n`, "utf8");
    await run(path);
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
}

test("character manifest drives arbitrary joint count and exact output order", async () => {
  const rig = makeRig(3);
  await withManifest(makeManifest(rig), async (path) => {
    const loaded = await loadCharacterRigConfig("C:/unused", { PET_CHARACTER_MANIFEST: path });
    assert.equal(loaded.characterId, "test-character");
    assert.equal(loaded.source, "character_manifest");
    assert.deepEqual(loaded.modelDrivenBoneIds, ["joint_0", "joint_1", "joint_2"]);
    assert.equal(loaded.modelDrivenCount, 3);
    assert.equal(loaded.sha256, rigFingerprint(rig));
    assert.equal(loaded.checkpoint.format, CHECKPOINT_FORMAT);
    assert.equal(loaded.checkpoint.metadataSchema, CHECKPOINT_METADATA_SCHEMA);
    assert.deepEqual(loaded.checkpoint.drivenJointOrder, loaded.modelDrivenBoneIds);
    assert.equal(loaded.checkpoint.manifestDeclared, true);
  });
});

test("checkpoint binding rejects format, identity, order, path, and field errors", async (t) => {
  const cases: Array<{
    readonly name: string;
    readonly mutate: (checkpoint: Record<string, unknown>) => void;
    readonly error: RegExp;
  }> = [
    {
      name: "raw state dict format",
      mutate: (checkpoint) => { checkpoint.format = "pytorch-state-dict-v1"; },
      error: /checkpoint\.format is unsupported/,
    },
    {
      name: "metadata schema",
      mutate: (checkpoint) => { checkpoint.metadataSchema = "pet-character-motion-checkpoint-metadata-v2"; },
      error: /metadataSchema is unsupported/,
    },
    {
      name: "character identity",
      mutate: (checkpoint) => { checkpoint.characterId = "another-character"; },
      error: /characterId does not match/,
    },
    {
      name: "rig identity",
      mutate: (checkpoint) => { checkpoint.rigFingerprint = "f".repeat(64); },
      error: /rigFingerprint does not match/,
    },
    {
      name: "joint order",
      mutate: (checkpoint) => {
        checkpoint.drivenJointOrder = [...(checkpoint.drivenJointOrder as string[])].reverse();
      },
      error: /drivenJointOrder does not exactly match/,
    },
    {
      name: "non-isolated path",
      mutate: (checkpoint) => { checkpoint.path = "checkpoints/shared/motion.pt"; },
      error: /character-isolated canonical checkpoint target/,
    },
    {
      name: "unknown field",
      mutate: (checkpoint) => { checkpoint.sharedAcrossCharacters = true; },
      error: /fields do not match/,
    },
  ];
  for (const invalidCase of cases) {
    await t.test(invalidCase.name, async () => {
      const manifest = makeManifest(makeRig());
      invalidCase.mutate(manifest.checkpoint as Record<string, unknown>);
      await withManifest(manifest, async (path) => {
        await assert.rejects(
          loadCharacterRigConfig("C:/unused", { PET_CHARACTER_MANIFEST: path }),
          invalidCase.error,
        );
      });
    });
  }
});

test("two characters with the same rig still have isolated checkpoint targets", async () => {
  const rig = makeRig();
  const first = makeManifest(rig, "character-a");
  const second = makeManifest(rig, "character-b");
  await withManifest(first, async (firstPath) => {
    const firstLoaded = await loadCharacterRigConfig("C:/unused", { PET_CHARACTER_MANIFEST: firstPath });
    await withManifest(second, async (secondPath) => {
      const secondLoaded = await loadCharacterRigConfig("C:/unused", { PET_CHARACTER_MANIFEST: secondPath });
      assert.equal(firstLoaded.sha256, secondLoaded.sha256);
      assert.notEqual(firstLoaded.checkpoint.path, secondLoaded.checkpoint.path);
      assert.match(firstLoaded.checkpoint.path, /\/character-a\//);
      assert.match(secondLoaded.checkpoint.path, /\/character-b\//);
    });
  });
});

test("Windows-unsafe character ids are rejected before checkpoint path derivation", async (t) => {
  for (const characterId of ["fox.", "nul", "nul.variant"]) {
    await t.test(characterId, async () => {
      const manifest = makeManifest(makeRig());
      const fingerprint = (manifest.rigFingerprint as Record<string, unknown>).value as string;
      manifest.characterId = characterId;
      const checkpoint = manifest.checkpoint as Record<string, unknown>;
      checkpoint.characterId = characterId;
      checkpoint.path = `checkpoints/characters/${characterId}/${fingerprint}/motion.pt`;
      await withManifest(manifest, async (path) => {
        await assert.rejects(
          loadCharacterRigConfig("C:/unused", { PET_CHARACTER_MANIFEST: path }),
          /Windows-safe/,
        );
      });
      assert.throws(() => checkpointTargetPath(characterId, fingerprint), /Windows-safe/);
    });
  }
});

test("rig ids, mask names, and joint ids use the shared identifier rules", async (t) => {
  const cases: Array<{ readonly name: string; readonly manifest: Record<string, unknown>; readonly error: RegExp }> = [];

  const invalidRigId = makeManifest(makeRig());
  invalidRigId.rigId = "UpperRig";
  cases.push({ name: "rig id", manifest: invalidRigId, error: /lowercase manifest identifier/ });

  const maskRig = makeRig();
  const masks = maskRig.maskGroups as Record<string, unknown>;
  masks["😀"] = masks.model;
  delete masks.model;
  cases.push({ name: "mask name", manifest: makeManifest(maskRig), error: /lowercase manifest identifier/ });

  const jointRig = makeRig();
  (jointRig.jointOrder as string[])[1] = "UPPER_JOINT";
  ((jointRig.joints as Array<Record<string, unknown>>)[1]!).id = "UPPER_JOINT";
  (jointRig.drivenJointOrder as string[])[0] = "UPPER_JOINT";
  cases.push({ name: "joint id", manifest: makeManifest(jointRig), error: /lowercase joint identifier/ });

  for (const invalidCase of cases) {
    await t.test(invalidCase.name, async () => {
      await withManifest(invalidCase.manifest, async (path) => {
        await assert.rejects(
          loadCharacterRigConfig("C:/unused", { PET_CHARACTER_MANIFEST: path }),
          invalidCase.error,
        );
      });
    });
  }
});

test("checkpoint metadata binds dataset, normalization, model config, and character identity", async () => {
  await withManifest(makeManifest(makeRig()), async (path) => {
    const selected = await loadCharacterRigConfig("C:/unused", { PET_CHARACTER_MANIFEST: path });
    const metadata: Record<string, unknown> = {
      schema: CHECKPOINT_METADATA_SCHEMA,
      characterId: selected.characterId,
      rigFingerprint: selected.sha256,
      drivenJointOrder: [...selected.modelDrivenBoneIds],
      dataset: {
        schema: CHECKPOINT_DATASET_SCHEMA,
        format: CHECKPOINT_DATASET_FORMAT,
        manifestSha256: "d".repeat(64),
      },
      normalization: {
        schema: CHECKPOINT_NORMALIZATION_SCHEMA,
        transform: CHECKPOINT_NORMALIZATION_TRANSFORM,
        condition: {
          featureOrder: ["condition-x", "condition-y"],
          offset: [1, 2],
          scale: [0.5, 4],
        },
        target: {
          featureOrder: ["target-x"],
          offset: [0],
          scale: [1],
        },
      },
      model: {
        schema: CHECKPOINT_MODEL_CONFIG_SCHEMA,
        architecture: "test-transformer",
        implementationVersion: "1.0.0",
        config: { hiddenSize: 64, dropout: 0.1 },
      },
    };
    const validated = validateCharacterCheckpointMetadata(metadata, selected.checkpoint);
    assert.deepEqual(validated.drivenJointOrder, selected.modelDrivenBoneIds);
    assert.equal(validated.model.config.hiddenSize, 64);

    const clone = (): Record<string, unknown> => JSON.parse(JSON.stringify(metadata)) as Record<string, unknown>;
    const wrongIdentity = clone();
    wrongIdentity.characterId = "another-character";
    assert.throws(
      () => validateCharacterCheckpointMetadata(wrongIdentity, selected.checkpoint),
      /identity does not match/,
    );
    const ragged = clone();
    const raggedNormalization = ragged.normalization as Record<string, unknown>;
    const raggedCondition = raggedNormalization.condition as Record<string, unknown>;
    raggedCondition.scale = [1];
    assert.throws(
      () => validateCharacterCheckpointMetadata(ragged, selected.checkpoint),
      /must contain 2 finite numbers/,
    );
    const nonPositive = clone();
    const nonPositiveNormalization = nonPositive.normalization as Record<string, unknown>;
    const nonPositiveTarget = nonPositiveNormalization.target as Record<string, unknown>;
    nonPositiveTarget.scale = [0];
    assert.throws(
      () => validateCharacterCheckpointMetadata(nonPositive, selected.checkpoint),
      /only positive/,
    );
    const nonFinite = clone();
    const nonFiniteModel = nonFinite.model as Record<string, unknown>;
    const nonFiniteConfig = nonFiniteModel.config as Record<string, unknown>;
    nonFiniteConfig.dropout = Number.NaN;
    assert.throws(
      () => validateCharacterCheckpointMetadata(nonFinite, selected.checkpoint),
      /non-finite/,
    );
  });
});

test("legacy rigs derive an isolated target without declaring an existing checkpoint", async () => {
  const directory = await mkdtemp(join(tmpdir(), "pet-legacy-rig-"));
  const runtime = join(directory, "assets", "pet", "runtime");
  try {
    await mkdir(runtime, { recursive: true });
    await writeFile(join(runtime, "cat-skeleton-3d.json"), JSON.stringify({
      schema: "pet-rig-v2",
      motionRoot: "motion_root",
      canvas: [48, 48],
      displayScale: 2,
      sourceFacing: -1,
      joints: [
        { id: "motion_root", parent: null, deform: false, physics: { mode: "kinematic" }, poseDofs: { rotation: false } },
        { id: "body", parent: "motion_root", deform: true, physics: { mode: "kinematic" }, poseDofs: { rotation: true } },
      ],
      drawOrder: ["body"],
    }), "utf8");
    const loaded = await loadCharacterRigConfig(directory, { PET_CHARACTER_ID: "  legacy-fox  " });
    assert.equal(loaded.source, "legacy_rig");
    assert.equal(loaded.checkpoint.manifestDeclared, false);
    assert.equal(
      loaded.checkpoint.path,
      checkpointTargetPath("legacy-fox", loaded.sha256),
    );
    assert.deepEqual(loaded.checkpoint.drivenJointOrder, ["body"]);
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});

test("legacy default character identity is stable without environment overrides", async () => {
  const directory = await mkdtemp(join(tmpdir(), "pet-legacy-default-"));
  const runtime = join(directory, "assets", "pet", "runtime");
  const path = join(runtime, "cat-skeleton-3d.json");
  try {
    await mkdir(runtime, { recursive: true });
    const legacy = {
      schema: "pet-rig-v2",
      motionRoot: "motion_root",
      joints: [
        { id: "motion_root", parent: null, deform: false, physics: { mode: "kinematic" }, poseDofs: { rotation: false } },
        { id: "body", parent: "motion_root", deform: true, physics: { mode: "kinematic" }, poseDofs: { rotation: true } },
      ],
      drawOrder: ["body"],
    };
    await writeFile(path, JSON.stringify(legacy), "utf8");
    const loaded = await loadCharacterRigConfig(directory, {});
    assert.equal(loaded.characterId, "legacy-default");
    assert.match(loaded.checkpoint.path, /\/legacy-default\//);

    legacy.joints[1]!.parent = "missing";
    await writeFile(path, JSON.stringify(legacy), "utf8");
    await assert.rejects(loadCharacterRigConfig(directory, {}), /missing parent/);
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});

test("rig fingerprint canonicalization matches the Python implementation", () => {
  assert.equal(
    rigFingerprint({ unicode: "猫", integer: 1, negativeZero: -0, array: [1.5, true] }),
    "e9aa44fdcd0ffd83017e623bc861bc5cf3ccdcaf60ebf751ef1e017fd67ef4f9",
  );
  assert.equal(
    rigFingerprint({ maskGroups: { "\uE000": [true], "😀": [false] } }),
    "779163f1807236f11dc08265441105732b027b00503f465a8de2602c65c4364c",
  );
});

test("rig fingerprint rejects non-scalar Unicode in nested keys and values", () => {
  const unpairedHigh = String.fromCharCode(0xD800);
  const unpairedLow = String.fromCharCode(0xDC00);
  assert.throws(
    () => rigFingerprint({ nested: { value: unpairedHigh } }),
    /Unicode scalar sequences/,
  );
  assert.throws(
    () => rigFingerprint({ nested: { [unpairedLow]: true } }),
    /Unicode scalar sequences/,
  );
  assert.doesNotThrow(() => rigFingerprint({ nested: { "😀": "猫" } }));
});

test("character manifest rejects a stale rig fingerprint", async () => {
  const manifest = makeManifest(makeRig());
  const rig = manifest.rig as Record<string, unknown>;
  (rig.coordinateSystem as Record<string, unknown>).units = "centimeter";
  await withManifest(manifest, async (path) => {
    await assert.rejects(
      loadCharacterRigConfig("C:/unused", { PET_CHARACTER_MANIFEST: path }),
      /rigFingerprint does not match/,
    );
  });
});

test("full rigs may be large but protocol-driven joints are capped at 128", async () => {
  const rig = makeRig(129);
  await withManifest(makeManifest(rig), async (path) => {
    await assert.rejects(
      loadCharacterRigConfig("C:/unused", { PET_CHARACTER_MANIFEST: path }),
      /between 1 and 128/,
    );
  });
});

test("drivenParentIndices must name each joint's nearest driven ancestor", async () => {
  const rig = makeRig(3);
  rig.drivenParentIndices = [-1, -1, 1];
  await withManifest(makeManifest(rig), async (path) => {
    await assert.rejects(
      loadCharacterRigConfig("C:/unused", { PET_CHARACTER_MANIFEST: path }),
      /nearest-driven hierarchy/,
    );
  });
});

test("manifest rig validation rejects malformed transforms, axes, roots, DOFs, and masks", async (t) => {
  const cases: Array<{
    readonly name: string;
    readonly mutate: (rig: Record<string, unknown>) => void;
    readonly error: RegExp;
  }> = [
    {
      name: "parallel coordinate axes",
      mutate: (rig) => { (rig.coordinateSystem as Record<string, unknown>).forward = "-Y"; },
      error: /must be orthogonal/,
    },
    {
      name: "non-unit rest quaternion",
      mutate: (rig) => {
        const joints = rig.joints as Array<Record<string, unknown>>;
        (joints[1]!.restLocal as Record<string, unknown>).rotation = [0, 0, 0, 2];
      },
      error: /unit quaternion/,
    },
    {
      name: "zero rest scale",
      mutate: (rig) => {
        const joints = rig.joints as Array<Record<string, unknown>>;
        (joints[1]!.restLocal as Record<string, unknown>).scale = [1, 0, 1];
      },
      error: /cannot contain zero/,
    },
    {
      name: "ragged DOF mask",
      mutate: (rig) => {
        const joints = rig.joints as Array<Record<string, unknown>>;
        (joints[1]!.dofMask as Record<string, unknown>).rotation = [true, false];
      },
      error: /must contain 3 booleans/,
    },
    {
      name: "second hierarchy root",
      mutate: (rig) => {
        const joints = rig.joints as Array<Record<string, unknown>>;
        joints[1]!.parentIndex = -1;
      },
      error: /exactly one root/,
    },
    {
      name: "mask length mismatch",
      mutate: (rig) => { (rig.maskGroups as Record<string, unknown>).model = [true]; },
      error: /must contain 2 booleans/,
    },
  ];

  for (const invalidCase of cases) {
    await t.test(invalidCase.name, async () => {
      const rig = makeRig();
      invalidCase.mutate(rig);
      await withManifest(makeManifest(rig), async (path) => {
        await assert.rejects(
          loadCharacterRigConfig("C:/unused", { PET_CHARACTER_MANIFEST: path }),
          invalidCase.error,
        );
      });
    });
  }
});

test("manifest sprite assets cannot be absolute or traverse above the manifest", async (t) => {
  const cases = [
    { reference: "C:\\outside\\cat.png", error: /must be relative/ },
    { reference: "..\\outside.png", error: /cannot traverse parent directories/ },
    { reference: "images/../outside.png", error: /cannot traverse parent directories/ },
  ] as const;

  for (const invalidCase of cases) {
    await t.test(invalidCase.reference, async () => {
      const manifest = makeManifest(makeRig());
      const render = manifest.render as Record<string, unknown>;
      (render.sprite as Record<string, unknown>).image = invalidCase.reference;
      await withManifest(manifest, async (path) => {
        await assert.rejects(
          loadCharacterRigConfig("C:/unused", { PET_CHARACTER_MANIFEST: path }),
          invalidCase.error,
        );
      });
    });
  }
});

test("debug-only rendering does not require unused sprite or mesh records", async () => {
  const manifest = makeManifest(makeRig());
  manifest.render = {
    canvas: [48, 48],
    displayScale: 2,
    footAnchor: [24, 46],
    sourceFacing: -1,
    mode: "debug_skeleton",
    fallbackModes: [],
  };
  await withManifest(manifest, async (path) => {
    const loaded = await loadCharacterRigConfig("C:/unused", { PET_CHARACTER_MANIFEST: path });
    assert.equal(loaded.render.mode, "debug_skeleton");
    assert.equal(loaded.render.spriteImagePath, null);
    assert.equal(loaded.render.spriteMetadataPath, null);
  });
});

test("render mode preference and fallback entries must be unique", async (t) => {
  const cases = [
    ["sprite", "sprite"],
    ["debug_skeleton"],
  ] as const;
  for (const fallbackModes of cases) {
    await t.test(fallbackModes.join(","), async () => {
      const manifest = makeManifest(makeRig());
      (manifest.render as Record<string, unknown>).fallbackModes = [...fallbackModes];
      await withManifest(manifest, async (path) => {
        await assert.rejects(
          loadCharacterRigConfig("C:/unused", { PET_CHARACTER_MANIFEST: path }),
          /modes must be unique/,
        );
      });
    });
  }
});

test("skinned mesh declarations must match source skin identity", async () => {
  const manifest = makeManifest(makeRig());
  const render = manifest.render as Record<string, unknown>;
  render.fallbackModes = ["sprite", "skinned_mesh"];
  render.mesh = { source: "source", skinIndex: 1 };
  manifest.source = { selectedSkinIndex: 0 };
  await withManifest(manifest, async (path) => {
    await assert.rejects(
      loadCharacterRigConfig("C:/unused", { PET_CHARACTER_MANIFEST: path }),
      /must match source.selectedSkinIndex/,
    );
  });
});

test("runtime canvas and display scale are normalized across all characters", async (t) => {
  const cases = [
    { field: "canvas", value: [64, 64], error: /normalized 48x48/ },
    { field: "displayScale", value: 3, error: /must be 2/ },
  ] as const;
  for (const invalidCase of cases) {
    await t.test(invalidCase.field, async () => {
      const manifest = makeManifest(makeRig());
      (manifest.render as Record<string, unknown>)[invalidCase.field] = invalidCase.value;
      await withManifest(manifest, async (path) => {
        await assert.rejects(
          loadCharacterRigConfig("C:/unused", { PET_CHARACTER_MANIFEST: path }),
          invalidCase.error,
        );
      });
    });
  }
});
