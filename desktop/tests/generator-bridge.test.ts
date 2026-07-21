import assert from "node:assert/strict";
import { resolve } from "node:path";
import test from "node:test";

import {
  advertisedHostCapabilities,
  GeneratorBridge,
  isCurrentChildEvent,
  modeUsesCharacterRig,
  modeUsesQuaternionRig,
  selectSkeletalMode,
  SkeletalNegotiationGate,
  type SkeletalMode,
  type SkeletalNegotiationToken,
} from "../src/generator-bridge.js";
import { releaseSentClicks } from "../src/click-handoff.js";
import {
  fallbackSkeletalPoseForMode,
  MotionController,
  planMatchesSkeletalMode,
  selectSkeletalPoseForMode,
} from "../src/motion-controller.js";
import type { CharacterRigConfig } from "../src/character-rig.js";
import type { PetVisualState, PetWindow } from "../src/pet-window.js";
import type { ClickEvent, HorizonPlanPayload, PlanPoint } from "../src/protocol.js";

const HASH = "a".repeat(64);

const POINT_BASE = {
  t_ms: 0,
  dx: 0,
  dy: 0,
  vx: 0,
  vy: 0,
  facing: 1,
  lean: 0,
  squash: 1,
  bob: 0,
  expression: "neutral",
} as const;

test("host capability advertisement fails closed for oversized legacy rigs", () => {
  assert.equal(advertisedHostCapabilities(30).includes("skeletal_motion"), true);
  assert.equal(advertisedHostCapabilities(33).includes("skeletal_motion"), false);
  assert.equal(advertisedHostCapabilities(33).includes("skeletal_motion_3d_local_quat"), true);
  assert.equal(advertisedHostCapabilities(null).includes("skeletal_motion"), false);
});

test("motion pose and fallback stay inside the negotiated encoding", () => {
  const facial = {
    eye_scale: 1.1,
    eye_squint: 0.2,
    mouth_open: 0.3,
    ear_angle: -0.1,
    brow_tilt: 0.4,
  };
  const planar: PlanPoint = { ...POINT_BASE, bone_rotations: [0.25], facial_params: facial };
  const quaternion: PlanPoint = {
    ...POINT_BASE,
    root_translation: [0, 0, 0],
    root_rotation: [0, 0, 0, 1],
    local_rotation_deltas: [[0, 0, 0, 1]],
  };

  assert.deepEqual(selectSkeletalPoseForMode("full_3d", 1, planar), { facialParams: facial });
  assert.deepEqual(selectSkeletalPoseForMode("full_2d", 1, quaternion), {});
  assert.deepEqual(selectSkeletalPoseForMode("full_2d", 1, planar), {
    boneRotations: [0.25],
    facialParams: facial,
  });
  assert.deepEqual(selectSkeletalPoseForMode("none", 0, planar), { facialParams: facial },
    "facial parameters are independent of skeletal capability negotiation");
  assert.deepEqual(selectSkeletalPoseForMode("full_3d", 1, quaternion), {
    rootTranslation: [0, 0, 0],
    rootRotation: [0, 0, 0, 1],
    localRotationDeltas: [[0, 0, 0, 1]],
  });
  assert.deepEqual(fallbackSkeletalPoseForMode("full_2d", 2), { boneRotations: [0, 0] });
  assert.deepEqual(fallbackSkeletalPoseForMode("full_3d", 1), {
    rootTranslation: [0, 0, 0],
    rootRotation: [0, 0, 0, 1],
    localRotationDeltas: [[0, 0, 0, 1]],
  });
  assert.deepEqual(fallbackSkeletalPoseForMode("full_2d", 33), {});
});

test("host click feedback clears sampled facial parameters before overriding expression", () => {
  const originalNow = Date.now;
  const originalSetInterval = globalThis.setInterval;
  const originalClearInterval = globalThis.clearInterval;
  const states: PetVisualState[] = [];
  let now = 1_800_000_000_000;
  let tick: (() => void) | undefined;
  const timer = {} as NodeJS.Timeout;

  Date.now = () => now;
  globalThis.setInterval = ((callback: () => void) => {
    tick = callback;
    return timer;
  }) as typeof setInterval;
  globalThis.clearInterval = (() => undefined) as typeof clearInterval;

  const petWindow = {
    footAnchorDip: { x: 48, y: 92 },
    visible: true,
    setSceneAllowed: () => undefined,
    setFootDip: () => undefined,
    setFootPhysical: () => undefined,
    sendVisualState: (state: PetVisualState) => states.push(state),
  } as unknown as PetWindow;
  const controller = new MotionController({
    petWindow,
    initialFootPhysical: { x: 200, y: 300 },
    onPlanCancelled: () => undefined,
  });

  try {
    controller.start();
    controller.setGeneratorStatus("ready");
    controller.setSurfaceSnapshot({
      displays: [{
        id: "display-test",
        bounds: { x: 0, y: 0, width: 1_000, height: 800 },
        work_area: { x: 0, y: 0, width: 1_000, height: 760 },
        scale_factor: 1,
        is_primary: true,
      }],
      windows: [],
      surfaces: [],
      scene: { fullscreen_active: false, pet_allowed: true },
      capturedAtMs: now,
    });
    assert.equal(controller.setSkeletalConfig({
      raw: {},
      sha256: "a".repeat(64),
      modelDrivenBoneIds: ["root"],
      modelDrivenCount: 1,
      characterId: "test-character",
      rigId: "test-rig",
      source: "character_manifest",
    }, "full_3d"), true);
    controller.rememberWorldState(1);
    const point = {
      ...POINT_BASE,
      expression: "happy",
      facial_params: {
        eye_scale: 1.15,
        eye_squint: 0.3,
        mouth_open: 0.4,
        ear_angle: 0,
        brow_tilt: 0.1,
      },
      root_translation: [0, 0, 0],
      root_rotation: [0, 0, 0, 1],
      local_rotation_deltas: [[0, 0, 0, 1]],
    } satisfies PlanPoint;
    const plan: HorizonPlanPayload = {
      plan_id: "click-facial-override",
      based_on_seq: 1,
      behavior: "walk",
      generated_at_ms: now,
      valid_until_ms: now + 500,
      dt_ms: 33,
      confidence: 1,
      seed: 1,
      points: [point, { ...point, t_ms: 33 }],
    };
    assert.equal(controller.offerPlan(plan), "accepted");

    controller.registerClickFeedback();
    now += 16;
    assert.ok(tick);
    tick();

    const rendered = states.at(-1);
    assert.ok(rendered);
    assert.equal(rendered.expression, "surprised");
    assert.equal("facialParams" in rendered, false,
      "the renderer must fall back to surprised defaults instead of stale sampled parameters");
  } finally {
    controller.dispose();
    Date.now = originalNow;
    globalThis.setInterval = originalSetInterval;
    globalThis.clearInterval = originalClearInterval;
  }
});

test("an entire horizon must match its negotiated pose encoding", () => {
  const planar: PlanPoint = { ...POINT_BASE, bone_rotations: [0.25] };
  const quaternion: PlanPoint = {
    ...POINT_BASE,
    root_translation: [0, 0, 0],
    root_rotation: [0, 0, 0, 1],
    local_rotation_deltas: [[0, 0, 0, 1]],
  };
  const plan = (points: PlanPoint[]): HorizonPlanPayload => ({
    plan_id: "pose-plan",
    based_on_seq: 1,
    generated_at_ms: 1,
    valid_until_ms: 1_000,
    dt_ms: 33,
    behavior: "idle",
    confidence: 1,
    seed: 7,
    points,
  });

  assert.equal(planMatchesSkeletalMode("full_2d", 1, plan([planar, planar])), true);
  assert.equal(planMatchesSkeletalMode("full_3d", 1, plan([quaternion, quaternion])), true);
  assert.equal(planMatchesSkeletalMode("full_3d", 1, plan([quaternion, planar])), false,
    "mixed encodings cannot reach the renderer");
  assert.equal(planMatchesSkeletalMode("full_3d", 1, plan([{ ...POINT_BASE }])), false,
    "a negotiated 3D plan cannot omit its pose branch");
  assert.equal(planMatchesSkeletalMode("none", 0, plan([planar])), true,
    "non-rendering modes may ignore pose fields without dropping root motion");
});

test("click ownership transfers once and stale child drain callbacks are ignored", () => {
  const click = (id: string): ClickEvent => ({
    id,
    button: "left",
    x: 1,
    y: 2,
    target: "pet",
    timestamp_ms: 3,
  });
  const sent = [click("sent")];
  assert.deepEqual(releaseSentClicks([...sent], sent), []);
  assert.deepEqual(
    releaseSentClicks([...sent, click("new")], sent).map((item) => item.id),
    ["new"],
  );
  assert.equal(isCurrentChildEvent(1, 2, false, false), false);
  assert.equal(isCurrentChildEvent(2, 2, false, true), true);
  assert.equal(isCurrentChildEvent(2, 2, true, true), false);
});

test("skeletal negotiation keeps 3D and legacy 2D as distinct states", () => {
  const common = {
    host3D: true,
    host2D: true,
    generatorSkeletonHash: HASH,
    hostSkeletonHash: HASH,
  };
  const transitions: SkeletalMode[] = [
    selectSkeletalMode({ ...common, generator3D: true, generator2D: true }),
    selectSkeletalMode({ ...common, generator3D: false, generator2D: true }),
    selectSkeletalMode({ ...common, generator3D: false, generator2D: false }),
    selectSkeletalMode({ ...common, generator3D: true, generator2D: true }),
  ];

  assert.deepEqual(transitions, ["full_3d", "full_2d", "host_only", "full_3d"]);
  assert.deepEqual(transitions.map(modeUsesQuaternionRig), [true, false, false, true]);
  assert.deepEqual(transitions.map(modeUsesCharacterRig), [true, true, false, true]);
});

test("3D negotiation fails closed for absent or mismatched rig fingerprints", () => {
  const state = {
    host3D: true,
    generator3D: true,
    host2D: true,
    generator2D: true,
    generatorSkeletonHash: HASH,
    hostSkeletonHash: HASH,
  };
  assert.equal(selectSkeletalMode({ ...state, generatorSkeletonHash: null }), "host_only");
  assert.equal(selectSkeletalMode({ ...state, hostSkeletonHash: null }), "generator_only");
  assert.equal(selectSkeletalMode({ ...state, hostSkeletonHash: "b".repeat(64) }), "none");
});

test("legacy 2D negotiation is also bound to the exact rig fingerprint", () => {
  const state = {
    host3D: true,
    generator3D: false,
    host2D: true,
    generator2D: true,
    generatorSkeletonHash: HASH,
    hostSkeletonHash: HASH,
  };
  assert.equal(selectSkeletalMode(state), "full_2d");
  assert.equal(selectSkeletalMode({ ...state, generatorSkeletonHash: null }), "host_only");
  assert.equal(selectSkeletalMode({ ...state, hostSkeletonHash: null }), "generator_only");
  assert.equal(selectSkeletalMode({ ...state, hostSkeletonHash: "b".repeat(64) }), "none");
});

test("a later ready handshake prevents an older async result from committing", async () => {
  const gate = new SkeletalNegotiationGate();
  const commits: SkeletalMode[] = [];
  let resolveOld!: (mode: SkeletalMode) => void;
  let resolveNew!: (mode: SkeletalMode) => void;
  const oldResult = new Promise<SkeletalMode>((resolve) => { resolveOld = resolve; });
  const newResult = new Promise<SkeletalMode>((resolve) => { resolveNew = resolve; });

  const finish = async (token: SkeletalNegotiationToken, result: Promise<SkeletalMode>): Promise<void> => {
    const mode = await result;
    if (gate.isCurrent(token, 7)) commits.push(mode);
  };

  const oldToken = gate.begin(7);
  const oldCompletion = finish(oldToken, oldResult);
  const newToken = gate.begin(7);
  const newCompletion = finish(newToken, newResult);
  resolveNew("full_2d");
  await newCompletion;
  resolveOld("full_3d");
  await oldCompletion;

  assert.deepEqual(commits, ["full_2d"]);
  assert.equal(gate.isCurrent(newToken, 8), false, "a child-generation change also invalidates the result");
  gate.invalidate();
  assert.equal(gate.isCurrent(newToken, 7), false, "explicit restart/dispose invalidates the result");
});

test("GeneratorBridge ignores a superseded ready result before accepting plans", async () => {
  const environmentKeys = [
    "PET_DISABLE_GENERATOR",
    "PET_PYTHON",
    "PET_GENERATOR_ENTRY",
    "PET_GENERATOR_CWD",
    "PET_TEST_SKELETON_SHA",
  ] as const;
  const saved = new Map(environmentKeys.map((key) => [key, process.env[key]] as const));
  const characterRig: CharacterRigConfig = {
    raw: { schema: "test-rig" },
    sha256: HASH,
    characterId: "test-character",
    rigId: "test-rig",
    modelDrivenBoneIds: ["joint"],
    modelDrivenCount: 1,
    checkpoint: {
      format: "pet-character-motion-checkpoint-v1",
      metadataSchema: "pet-character-motion-checkpoint-metadata-v1",
      path: `checkpoints/characters/test-character/${HASH}/motion.pt`,
      characterId: "test-character",
      rigFingerprint: HASH,
      drivenJointOrder: ["joint"],
      manifestDeclared: true,
    },
    source: "character_manifest",
    path: "test.manifest.json",
    render: {
      canvas: [48, 48],
      displayScale: 2,
      footAnchor: [24, 46],
      sourceFacing: -1,
      mode: "debug_skeleton",
      fallbackModes: [],
      spriteImagePath: null,
      spriteMetadataPath: null,
    },
  };
  const modes: SkeletalMode[] = [];
  let bridge: GeneratorBridge | undefined;
  try {
    delete process.env.PET_DISABLE_GENERATOR;
    process.env.PET_PYTHON = process.execPath;
    process.env.PET_GENERATOR_ENTRY = resolve(process.cwd(), "tests", "fixtures", "generator-double-ready.cjs");
    process.env.PET_GENERATOR_CWD = process.cwd();
    process.env.PET_TEST_SKELETON_SHA = HASH;

    let resolveReady!: () => void;
    let rejectReady!: (error: Error) => void;
    const ready = new Promise<void>((resolvePromise, rejectPromise) => {
      resolveReady = resolvePromise;
      rejectReady = rejectPromise;
    });
    const timeout = setTimeout(() => rejectReady(new Error("fixture generator did not become ready")), 3_000);
    bridge = new GeneratorBridge({
      projectRoot: resolve(process.cwd(), ".."),
      sessionId: "double-ready-session",
      petWidthPhysical: 96,
      petHeightPhysical: 96,
      characterRig,
      onStatus: (status) => {
        if (status === "ready") {
          clearTimeout(timeout);
          resolveReady();
        }
      },
      onPlan: () => { throw new Error("fixture must not emit plans"); },
      onSkeletalMode: (mode) => modes.push(mode),
    });
    bridge.start();
    await ready;

    assert.equal(bridge.status, "ready");
    assert.equal(bridge.skeletalMode, "full_2d");
    assert.deepEqual(modes, ["full_2d"], "the superseded 3D negotiation must never commit");
  } finally {
    bridge?.dispose();
    for (const key of environmentKeys) {
      const value = saved.get(key);
      if (value === undefined) delete process.env[key];
      else process.env[key] = value;
    }
  }
});
