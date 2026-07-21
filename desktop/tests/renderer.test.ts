import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import vm from "node:vm";

type Vec3 = [number, number, number];
type Quat = [number, number, number, number];

interface TestJointWorld {
  readonly position: Vec3;
  readonly rotation: Quat;
  readonly scale: Vec3;
}

interface ProjectedJoint {
  readonly idx: number;
  readonly sx: number;
  readonly sy: number;
  readonly sz: number;
  readonly rotation: Quat;
}

interface SpriteWarpControl {
  readonly x: number;
  readonly y: number;
  readonly angle: number;
}

interface SpriteWarpBindings {
  readonly width: number;
  readonly height: number;
  readonly influenceCount: number;
  readonly indices: Int32Array;
  readonly weights: Float64Array;
}

interface RendererTestApi {
  loadSkeleton(config: unknown): boolean;
  computeFK(rootTranslation: unknown, rootRotation: unknown, deltas: unknown): boolean;
  project(width?: number, height?: number, padding?: number): ProjectedJoint[];
  draw(): void;
  buildSpriteWarpBindings(
    width: number,
    height: number,
    restControls: readonly SpriteWarpControl[],
    maximumInfluences?: number,
  ): SpriteWarpBindings | null;
  warpSpritePixels(
    sourcePixels: Uint8ClampedArray,
    width: number,
    height: number,
    restControls: readonly SpriteWarpControl[],
    posedControls: readonly SpriteWarpControl[],
    bindings: SpriteWarpBindings,
  ): Uint8ClampedArray | null;
  getWarpedPixels(): Uint8ClampedArray | null;
  getJointWorld(): TestJointWorld[];
  getRig(): {
    readonly motionRoot: string;
    readonly jointIds: readonly string[];
    readonly parentIndices: readonly number[];
    readonly drivenJointIds: readonly string[];
    readonly projectionBasis: {
      readonly horizontal: Vec3;
      readonly vertical: Vec3;
      readonly depth: Vec3;
    };
    readonly renderMode: "sprite" | "debug_skeleton";
  } | null;
}

interface RigJoint {
  readonly id: string;
  readonly parentIndex: number;
  readonly restLocal: {
    readonly translation: Vec3;
    readonly rotation: Quat;
    readonly scale: Vec3;
  };
  readonly dofMask: {
    readonly translation: readonly boolean[];
    readonly rotation: readonly boolean[];
    readonly scale: readonly boolean[];
  };
  readonly semanticRole: string;
  readonly sprite?: string | null;
}

const rendererSourcePromise = readFile(
  new URL("../../src/renderer/renderer.js", import.meta.url),
  "utf8",
);

function rigJoint(
  id: string,
  parentIndex: number,
  translation: Vec3,
  semanticRole = "bone",
  rotation: Quat = [0, 0, 0, 1],
  rotationEnabled = false,
): RigJoint {
  return {
    id,
    parentIndex,
    restLocal: {
      translation,
      rotation,
      scale: [1, 1, 1],
    },
    dofMask: {
      translation: [false, false, false],
      rotation: [rotationEnabled, rotationEnabled, rotationEnabled],
      scale: [false, false, false],
    },
    semanticRole,
    sprite: null,
  };
}

function genericManifest(
  joints: readonly RigJoint[],
  drivenJointIndices: readonly number[],
  coordinateSystem = { handedness: "right", up: "+Y", forward: "+X" },
  render?: Record<string, unknown>,
) {
  return {
    schema: "pet-character-rig-manifest-v1",
    characterId: "test",
    ...(render ? { render } : {}),
    rig: {
      coordinateSystem,
      motionRoot: "__motion_root__",
      joints,
      drivenJointIndices,
      drivenJointOrder: drivenJointIndices.map((index) => joints[index]?.id),
    },
  };
}

function plain<T>(value: T): T {
  return JSON.parse(JSON.stringify(value)) as T;
}

function assertVecClose(actual: readonly number[], expected: readonly number[], epsilon = 1e-9): void {
  assert.equal(actual.length, expected.length);
  for (let i = 0; i < expected.length; i++) {
    assert.ok(Math.abs((actual[i] ?? Number.NaN) - (expected[i] ?? Number.NaN)) <= epsilon,
      `component ${i}: expected ${expected[i]}, received ${actual[i]}`);
  }
}

async function createRendererHarness(initialSkeleton: unknown = null) {
  const rendererSource = await rendererSourcePromise;
  const drawCalls: unknown[][] = [];
  const translations: Array<[number, number]> = [];
  const scales: Array<[number, number]> = [];
  const rotations: number[] = [];
  const fillRects: Array<[number, number, number, number]> = [];
  const lineSegments: Array<[number, number, number, number]> = [];
  let pathStart: [number, number] | null = null;
  let pathEnd: [number, number] | null = null;
  let strokeCount = 0;
  const rasterSurfaces: Array<{
    width: number;
    height: number;
    pixels: Uint8ClampedArray;
  }> = [];

  const context = {
    imageSmoothingEnabled: true,
    fillStyle: "",
    strokeStyle: "",
    lineWidth: 1,
    lineCap: "butt",
    clearRect() {},
    save() {},
    restore() {},
    translate(x: number, y: number) { translations.push([x, y]); },
    rotate(radians: number) { rotations.push(radians); },
    scale(x: number, y: number) { scales.push([x, y]); },
    fillRect(x: number, y: number, width: number, height: number) {
      fillRects.push([x, y, width, height]);
    },
    drawImage(...args: unknown[]) { drawCalls.push(args); },
    getImageData() { return { data: new Uint8ClampedArray([0, 0, 0, 255]) }; },
    beginPath() { pathStart = null; pathEnd = null; },
    moveTo(x: number, y: number) {
      pathStart = [x, y];
      pathEnd = [x, y];
    },
    lineTo(x: number, y: number) { pathEnd = [x, y]; },
    quadraticCurveTo(_cx: number, _cy: number, x: number, y: number) { pathEnd = [x, y]; },
    stroke() {
      strokeCount += 1;
      if (pathStart && pathEnd) lineSegments.push([...pathStart, ...pathEnd]);
    },
    fill() {},
    ellipse() {},
    arc() {},
    arcTo() {},
    closePath() {},
  };
  const canvas = {
    width: 48,
    height: 48,
    getContext: () => context,
    addEventListener() {},
    getBoundingClientRect: () => ({
      left: 0, top: 0, right: 96, bottom: 96, width: 96, height: 96,
    }),
  };
  let stateHandler: ((state: Record<string, unknown>) => void) | undefined;
  let skeletalConfigHandler: ((config: unknown) => void) | undefined;
  let loadedImage: FakeImage | undefined;
  class FakeImage {
    onload: (() => void) | null = null;
    onerror: (() => void) | null = null;
    #source = "";
    pixels = new Uint8ClampedArray(48 * 48 * 4);
    get src(): string { return this.#source; }
    set src(value: string) {
      this.#source = value;
      loadedImage = this;
    }
  }
  const petHost = {
    onState(handler: (state: Record<string, unknown>) => void) { stateHandler = handler; },
    onProbe() {},
    reportHit() {},
    click() {},
    ready() {},
    getSkeletalConfig: () => initialSkeleton,
    onSkeletalConfig(handler: (config: unknown) => void) {
      skeletalConfigHandler = handler;
      return () => { skeletalConfigHandler = undefined; };
    },
  };
  const rendererWindow: {
    petHost: typeof petHost;
    __PET_RENDERER_TEST__: true;
    __petRendererTest?: RendererTestApi;
  } = { petHost, __PET_RENDERER_TEST__: true };
  const document = {
    documentElement: { dataset: {} as Record<string, string> },
    getElementById: () => canvas,
    createElement: (tagName: string) => {
      assert.equal(tagName, "canvas");
      const surface = {
        width: 48,
        height: 48,
        pixels: new Uint8ClampedArray(48 * 48 * 4),
      };
      const rasterContext = {
        imageSmoothingEnabled: true,
        clearRect() { surface.pixels.fill(0); },
        drawImage(image: { pixels?: Uint8ClampedArray }) {
          if (image?.pixels?.length === surface.width * surface.height * 4) {
            surface.pixels = new Uint8ClampedArray(image.pixels);
          }
        },
        getImageData() { return { data: new Uint8ClampedArray(surface.pixels) }; },
        createImageData(width: number, height: number) {
          return { data: new Uint8ClampedArray(width * height * 4), width, height };
        },
        putImageData(imageData: { data: Uint8ClampedArray }) {
          surface.pixels = new Uint8ClampedArray(imageData.data);
        },
      };
      rasterSurfaces.push(surface);
      return Object.assign(surface, { getContext: () => rasterContext });
    },
    addEventListener() {},
  };

  vm.runInNewContext(rendererSource, {
    document,
    window: rendererWindow,
    Image: FakeImage,
    Uint8ClampedArray,
  });
  assert.ok(stateHandler);
  assert.ok(rendererWindow.__petRendererTest);

  return {
    api: rendererWindow.__petRendererTest,
    context,
    drawCalls,
    translations,
    scales,
    rotations,
    fillRects,
    lineSegments,
    get strokeCount() { return strokeCount; },
    resetStrokeCount() { strokeCount = 0; },
    stateHandler,
    getLoadedImage: () => loadedImage,
    setLoadedImagePixels: (pixels: Uint8ClampedArray) => {
      assert.ok(loadedImage);
      assert.equal(pixels.length, 48 * 48 * 4);
      loadedImage.pixels = new Uint8ClampedArray(pixels);
    },
    rasterSurfaces,
    getSkeletalConfigHandler: () => skeletalConfigHandler,
  };
}

test("the flattened cat is drawn once as a complete sprite", async () => {
  const metadata = JSON.parse(await readFile(
    new URL("../../../assets/pet/runtime/cat-parts.json", import.meta.url),
    "utf8",
  )) as {
    readonly footAnchor: readonly [number, number];
    readonly sourceFacing: -1 | 1;
  };
  const harness = await createRendererHarness();

  harness.stateHandler({
    assetParts: metadata,
    assetDataUrl: "data:image/png;base64,AA==",
    facing: 1,
    squash: 1,
    lean: 0.25,
  });
  const loadedImage = harness.getLoadedImage();
  assert.ok(loadedImage?.onload);
  harness.drawCalls.length = 0;
  harness.translations.length = 0;
  harness.scales.length = 0;
  harness.rotations.length = 0;
  loadedImage.onload();

  assert.equal(harness.context.imageSmoothingEnabled, false);
  assert.equal(harness.drawCalls.length, 1);
  assert.strictEqual(harness.drawCalls[0]?.[0], loadedImage);
  assert.deepEqual(harness.drawCalls[0]?.slice(1), [0, 0, 48, 48]);
  assert.deepEqual(harness.translations, [
    [metadata.footAnchor[0], metadata.footAnchor[1]],
    [-metadata.footAnchor[0], -metadata.footAnchor[1]],
  ]);
  assert.equal(metadata.sourceFacing, -1);
  assert.equal(Math.sign(harness.scales[0]?.[0] ?? 0), -1,
    "logical right mirrors the left-facing master");
  assert.ok((harness.rotations[0] ?? 0) > 0,
    "rightward screen-space lean remains positive");

  harness.scales.length = 0;
  harness.rotations.length = 0;
  harness.stateHandler({ facing: -1, lean: -0.25 });
  assert.equal(Math.sign(harness.scales[0]?.[0] ?? 0), 1,
    "logical left keeps the master unmirrored");
  assert.ok((harness.rotations[0] ?? 0) < 0,
    "leftward screen-space lean remains negative");
});

test("every sparse facial channel changes the final whole-sprite transform and absent state resets it", async () => {
  const harness = await createRendererHarness();
  harness.stateHandler({
    assetDataUrl: "data:image/png;base64,AA==",
    expression: "neutral",
    facing: 1,
    squash: 1,
    lean: 0,
  });
  const loadedImage = harness.getLoadedImage();
  assert.ok(loadedImage?.onload);
  loadedImage.onload();

  const renderTransform = (next: Record<string, unknown>) => {
    harness.drawCalls.length = 0;
    harness.translations.length = 0;
    harness.scales.length = 0;
    harness.rotations.length = 0;
    harness.stateHandler(next);
    assert.equal(harness.drawCalls.length, 1);
    assert.strictEqual(harness.drawCalls[0]?.[0], loadedImage);
    return {
      translations: harness.translations.map((pair) => [...pair]),
      scales: harness.scales.map((pair) => [...pair]),
      rotations: [...harness.rotations],
    };
  };

  const neutral = renderTransform({ expression: "neutral" });
  const channels = [
    ["eye_scale", 1.25],
    ["eye_squint", 0.5],
    ["mouth_open", 0.5],
    ["ear_angle", 0.25],
    ["brow_tilt", 0.5],
  ] as const;
  for (const [channel, value] of channels) {
    const transformed = renderTransform({
      expression: "neutral",
      facialParams: { [channel]: value },
    });
    assert.notDeepEqual(transformed, neutral,
      `${channel} must reach the final canvas transform`);
  }

  assert.deepEqual(
    renderTransform({ expression: "neutral" }),
    neutral,
    "a state without facialParams clears the previous sparse override",
  );
  assert.notDeepEqual(
    renderTransform({ expression: "surprised" }),
    neutral,
    "expression supplies defaults when the facialParams object is absent",
  );
  assert.deepEqual(
    renderTransform({ expression: "surprised", facialParams: {} }),
    neutral,
    "missing channels inside an explicit sparse object use neutral values",
  );
});

test("the pure sprite warp preserves identity and consumes a leaf rotation", async () => {
  const { api } = await createRendererHarness();
  const width = 5;
  const height = 5;
  const source = new Uint8ClampedArray(width * height * 4);
  for (let pixel = 0; pixel < width * height; pixel++) {
    source[pixel * 4] = pixel * 7;
    source[pixel * 4 + 1] = 255 - pixel * 5;
    source[pixel * 4 + 2] = pixel * 3;
    source[pixel * 4 + 3] = 255;
  }
  const rest = [{ x: 2.5, y: 2.5, angle: 0 }];
  const bindings = api.buildSpriteWarpBindings(width, height, rest);
  assert.ok(bindings);

  const identity = api.warpSpritePixels(source, width, height, rest, rest, bindings);
  assert.ok(identity);
  assert.deepEqual([...identity], [...source], "identity pose is pixel-exact");

  const rotated = api.warpSpritePixels(source, width, height, rest, [
    { x: 2.5, y: 2.5, angle: Math.PI / 2 },
  ], bindings);
  assert.ok(rotated);
  assert.notDeepEqual([...rotated], [...source],
    "a stationary leaf origin still rotates its bound pixels");
});

test("normal sprite rendering changes final raster pixels for different 3D deltas", async () => {
  const manifest = genericManifest([
    rigJoint("__motion_root__", -1, [0, 0, 0], "motion_root"),
    rigJoint("leaf", 0, [5, 0, 0], "bone", [0, 0, 0, 1], true),
  ], [1], undefined, {
    canvas: [48, 48],
    displayScale: 2,
    footAnchor: [24, 46],
    sourceFacing: -1,
    mode: "sprite",
    fallbackModes: ["debug_skeleton"],
    sprite: { image: "character.png", metadata: "character.json" },
    mesh: null,
  });
  const harness = await createRendererHarness(manifest);
  const source = new Uint8ClampedArray(48 * 48 * 4);
  for (let y = 0; y < 48; y++) {
    for (let x = 0; x < 48; x++) {
      const offset = (y * 48 + x) * 4;
      source[offset] = (x * 5) % 256;
      source[offset + 1] = (y * 7) % 256;
      source[offset + 2] = (x * 3 + y * 11) % 256;
      source[offset + 3] = 255;
    }
  }
  harness.stateHandler({
    debug: false,
    assetDataUrl: "data:image/png;base64,AA==",
    rootTranslation: [0, 0, 0],
    rootRotation: [0, 0, 0, 1],
    localRotationDeltas: [[0, 0, 0, 1]],
  });
  harness.setLoadedImagePixels(source);
  const loadedImage = harness.getLoadedImage();
  assert.ok(loadedImage?.onload);
  loadedImage.onload();
  const identityPixels = harness.api.getWarpedPixels();
  assert.ok(identityPixels);
  assert.deepEqual([...identityPixels], [...source]);

  harness.resetStrokeCount();
  const half = Math.sqrt(0.5);
  harness.stateHandler({
    debug: false,
    rootTranslation: [0, 0, 0],
    rootRotation: [0, 0, 0, 1],
    localRotationDeltas: [[0, 0, half, half]],
  });
  const rotatedPixels = harness.api.getWarpedPixels();
  assert.ok(rotatedPixels);
  assert.notDeepEqual([...rotatedPixels], [...identityPixels],
    "the non-debug sprite path consumes the driven quaternion");
  assert.equal(harness.strokeCount, 0, "the visible difference is not a debug skeleton overlay");
  const finalSurface = harness.rasterSurfaces.at(-1);
  assert.ok(finalSurface);
  assert.deepEqual([...finalSurface.pixels], [...rotatedPixels],
    "the changed raster is uploaded to the canvas drawn by the normal path");

  harness.stateHandler({
    debug: false,
    boneRotations: [Math.PI / 2],
  });
  const legacyPixels = harness.api.getWarpedPixels();
  assert.ok(legacyPixels);
  assert.notDeepEqual([...legacyPixels], [...identityPixels],
    "the negotiated planar angle also reaches the normal sprite pixels");
  assert.deepEqual([...legacyPixels], [...rotatedPixels],
    "a planar angle is converted around the projected depth axis before FK");
});

test("generic facial fallback wraps both pose-aware sprites and debug skeletons", async () => {
  const manifest = genericManifest([
    rigJoint("__motion_root__", -1, [0, 0, 0], "motion_root"),
    rigJoint("leaf", 0, [5, 0, 0], "bone", [0, 0, 0, 1], true),
  ], [1], undefined, {
    canvas: [48, 48], displayScale: 2, footAnchor: [24, 46], sourceFacing: -1,
    mode: "sprite", fallbackModes: ["debug_skeleton"],
    sprite: { image: "character.png", metadata: "character.json" }, mesh: null,
  });
  const pose = {
    rootTranslation: [0, 0, 0],
    rootRotation: [0, 0, 0, 1],
    localRotationDeltas: [[0, 0, 0, 1]],
  };
  const poseHarness = await createRendererHarness(manifest);
  poseHarness.stateHandler({
    assetDataUrl: "data:image/png;base64,AA==",
    expression: "neutral",
    ...pose,
  });
  const loadedImage = poseHarness.getLoadedImage();
  assert.ok(loadedImage?.onload);
  loadedImage.onload();

  const poseSignature = (facialParams?: Record<string, number>) => {
    poseHarness.drawCalls.length = 0;
    poseHarness.translations.length = 0;
    poseHarness.scales.length = 0;
    poseHarness.rotations.length = 0;
    poseHarness.stateHandler({
      expression: "neutral",
      ...(facialParams ? { facialParams } : {}),
      ...pose,
    });
    assert.equal(poseHarness.drawCalls.length, 1);
    assert.notStrictEqual(poseHarness.drawCalls[0]?.[0], loadedImage,
      "the pose-aware path draws the warped raster surface");
    return {
      translations: poseHarness.translations.map((pair) => [...pair]),
      scales: poseHarness.scales.map((pair) => [...pair]),
      rotations: [...poseHarness.rotations],
    };
  };
  const neutralPose = poseSignature();
  assert.notDeepEqual(poseSignature({ mouth_open: 0.75 }), neutralPose,
    "the generic facial transform wraps the pose-aware sprite output");

  const debugHarness = await createRendererHarness(genericManifest([
    rigJoint("__motion_root__", -1, [0, 0, 0], "motion_root"),
    rigJoint("leaf", 0, [5, 0, 0], "bone", [0, 0, 0, 1], true),
  ], [1]));
  debugHarness.stateHandler({ expression: "neutral", ...pose });
  const neutralDebug = {
    translations: debugHarness.translations.map((pair) => [...pair]),
    scales: debugHarness.scales.map((pair) => [...pair]),
    rotations: [...debugHarness.rotations],
  };
  debugHarness.translations.length = 0;
  debugHarness.scales.length = 0;
  debugHarness.rotations.length = 0;
  debugHarness.resetStrokeCount();
  debugHarness.stateHandler({
    expression: "neutral",
    facialParams: { ear_angle: 0.4 },
    ...pose,
  });
  assert.ok(debugHarness.strokeCount > 0, "the debug skeleton remains visibly rendered");
  assert.notDeepEqual({
    translations: debugHarness.translations.map((pair) => [...pair]),
    scales: debugHarness.scales.map((pair) => [...pair]),
    rotations: [...debugHarness.rotations],
  }, neutralDebug, "the generic facial transform wraps the debug-skeleton output");
});

test("sprite warp retains leaf twist when local X points into camera depth", async () => {
  const half = Math.sqrt(0.5);
  const manifest = genericManifest([
    rigJoint("__motion_root__", -1, [0, 0, 0], "motion_root"),
    // With +X character-forward, this rest rotation maps local +X onto camera Z.
    rigJoint("depth_axis_leaf", 0, [5, 0, 0], "bone", [0, half, 0, half], true),
  ], [1], undefined, {
    canvas: [48, 48], displayScale: 2, footAnchor: [24, 46], sourceFacing: -1,
    mode: "sprite", fallbackModes: ["debug_skeleton"],
    sprite: { image: "character.png", metadata: "character.json" }, mesh: null,
  });
  const harness = await createRendererHarness(manifest);
  const source = new Uint8ClampedArray(48 * 48 * 4);
  for (let pixel = 0; pixel < 48 * 48; pixel++) {
    source[pixel * 4] = (pixel * 17) % 256;
    source[pixel * 4 + 1] = (pixel * 29) % 256;
    source[pixel * 4 + 2] = (pixel * 43) % 256;
    source[pixel * 4 + 3] = 255;
  }
  harness.stateHandler({
    assetDataUrl: "data:image/png;base64,AA==",
    rootTranslation: [0, 0, 0], rootRotation: [0, 0, 0, 1],
    localRotationDeltas: [[0, 0, 0, 1]],
  });
  harness.setLoadedImagePixels(source);
  const loadedImage = harness.getLoadedImage();
  assert.ok(loadedImage?.onload);
  loadedImage.onload();
  const restPixels = harness.api.getWarpedPixels();
  assert.ok(restPixels);

  harness.stateHandler({
    rootTranslation: [0, 0, 0], rootRotation: [0, 0, 0, 1],
    localRotationDeltas: [[half, 0, 0, half]],
  });
  const twistedPixels = harness.api.getWarpedPixels();
  assert.ok(twistedPixels);
  assert.notDeepEqual([...twistedPixels], [...restPixels],
    "a stable rest-selected reference axis makes the depth-axis leaf twist visible");
});

test("sprite data URLs accept only host-supported raster MIME types and valid base64", async (t) => {
  for (const mime of ["image/png", "image/webp", "image/jpeg"] as const) {
    await t.test(`accepts ${mime}`, async () => {
      const harness = await createRendererHarness();
      const dataUrl = `data:${mime};base64,AA==`;
      harness.stateHandler({ assetDataUrl: dataUrl });
      assert.equal(harness.getLoadedImage()?.src, dataUrl);
    });
  }

  for (const dataUrl of [
    "data:image/jpg;base64,AA==",
    "data:image/svg+xml;base64,AA==",
    "data:text/html;base64,AA==",
    "data:image/png;charset=utf-8;base64,AA==",
    "data:image/png;base64,not base64",
    "data:image/png;base64,AAA",
    "data:image/png;base64,",
  ]) {
    await t.test(`rejects ${dataUrl}`, async () => {
      const harness = await createRendererHarness();
      harness.stateHandler({ assetDataUrl: dataUrl });
      assert.equal(harness.getLoadedImage(), undefined);
    });
  }
});

test("identity rest transforms and identity deltas preserve the authored chain", async () => {
  const manifest = genericManifest([
    rigJoint("__motion_root__", -1, [0, 0, 0], "motion_root"),
    rigJoint("body", 0, [2, 3, 4], "bone", [0, 0, 0, 1], true),
    rigJoint("tip", 1, [5, 0, 0]),
  ], [1]);
  const { api } = await createRendererHarness(manifest);

  assert.equal(api.computeFK([10, -1, 2], [0, 0, 0, 1], [[0, 0, 0, 1]]), true);
  const world = plain(api.getJointWorld());
  assertVecClose(world[0]?.position ?? [], [10, -1, 2]);
  assertVecClose(world[1]?.position ?? [], [12, 2, 6]);
  assertVecClose(world[2]?.position ?? [], [17, 2, 6]);
  assertVecClose(world[2]?.rotation ?? [], [0, 0, 0, 1]);
});

test("a single X-axis delta rotates descendants through 3D quaternion FK", async () => {
  const manifest = genericManifest([
    rigJoint("__motion_root__", -1, [0, 0, 0], "motion_root"),
    rigJoint("hinge", 0, [0, 0, 0], "bone", [0, 0, 0, 1], true),
    rigJoint("tip", 1, [0, 1, 0]),
  ], [1]);
  const { api } = await createRendererHarness(manifest);
  const half = Math.sqrt(0.5);

  assert.equal(api.computeFK([0, 0, 0], [0, 0, 0, 1], [[half, 0, 0, half]]), true);
  const world = plain(api.getJointWorld());
  assertVecClose(world[2]?.position ?? [], [0, 0, 1], 1e-8);
  assertVecClose(world[1]?.rotation ?? [], [half, 0, 0, half], 1e-8);
});

test("rest rotation composes before a unit local delta", async () => {
  const half = Math.sqrt(0.5);
  const manifest = genericManifest([
    rigJoint("__motion_root__", -1, [0, 0, 0], "motion_root"),
    rigJoint("hinge", 0, [0, 0, 0], "bone", [0, 0, half, half], true),
    rigJoint("tip", 1, [0, 1, 0]),
  ], [1]);
  const { api } = await createRendererHarness(manifest);

  assert.equal(api.computeFK([0, 0, 0], [0, 0, 0, 1], [[half, 0, 0, half]]), true);
  const world = plain(api.getJointWorld());
  assertVecClose(world[1]?.rotation ?? [], [0.5, 0.5, 0.5, 0.5], 1e-8);
  assertVecClose(world[2]?.position ?? [], [0, 0, 1], 1e-8);
});

test("FK propagates non-uniform rest scale with the full affine matrix order", async () => {
  const half = Math.sqrt(0.5);
  const root = rigJoint("__motion_root__", -1, [0, 0, 0], "motion_root");
  const scaledRoot: RigJoint = {
    ...root,
    restLocal: { ...root.restLocal, scale: [2, 1, 1] },
  };
  const manifest = genericManifest([
    scaledRoot,
    rigJoint("rotated", 0, [0, 0, 0], "bone", [0, 0, half, half]),
    rigJoint("tip", 1, [1, 0, 0]),
  ], []);
  const { api } = await createRendererHarness(manifest);

  assert.equal(api.computeFK([0, 0, 0], [0, 0, 0, 1], []), true);
  assertVecClose(plain(api.getJointWorld())[2]?.position ?? [], [0, 1, 0], 1e-8);
});

test("FK accepts arbitrary joint counts and parents that appear later in manifest order", async () => {
  const manifest = genericManifest([
    rigJoint("__motion_root__", -1, [0, 0, 0], "motion_root"),
    rigJoint("grandchild", 2, [0, 3, 0]),
    rigJoint("parent", 0, [2, 0, 0]),
    rigJoint("sibling", 0, [-1, 0, 0]),
    rigJoint("great_grandchild", 1, [0, 0, 4]),
  ], []);
  const { api } = await createRendererHarness(manifest);

  assert.equal(api.computeFK([0, 0, 0], [0, 0, 0, 1], []), true);
  const world = plain(api.getJointWorld());
  assertVecClose(world[1]?.position ?? [], [2, 3, 0]);
  assertVecClose(world[2]?.position ?? [], [2, 0, 0]);
  assertVecClose(world[3]?.position ?? [], [-1, 0, 0]);
  assertVecClose(world[4]?.position ?? [], [2, 3, 4]);
  assert.deepEqual(plain(api.getRig()?.parentIndices), [-1, 2, 0, 0, 1]);
});

test("an all-bone rig with no sprites or semantic parts draws a visible skeleton fallback", async () => {
  const manifest = genericManifest([
    rigJoint("__motion_root__", -1, [0, 0, 0], "motion_root"),
    rigJoint("bone_a", 0, [0, 2, 0], "bone", [0, 0, 0, 1], true),
    rigJoint("bone_b", 1, [8, 0, 0]),
  ], [1]);
  const harness = await createRendererHarness(manifest);
  harness.fillRects.length = 0;
  harness.lineSegments.length = 0;
  harness.resetStrokeCount();

  harness.stateHandler({
    rootTranslation: [0, 0, 0],
    rootRotation: [0, 0, 0, 1],
    localRotationDeltas: [[0, 0, 0, 1]],
  });

  assert.ok(harness.strokeCount >= 2, "outlined bones issue visible stroke calls");
  assert.ok(harness.lineSegments.length >= 2, "at least one parent-child edge is rendered");
  assert.ok(harness.fillRects.length >= 2, "joint markers remain visible for zero-length bones");
});

test("orthographic projection dynamically fits extreme poses inside the 48px canvas", async () => {
  const manifest = genericManifest([
    rigJoint("__motion_root__", -1, [0, 0, 0], "motion_root"),
    rigJoint("far_left", 0, [-100, -50, 2]),
    rigJoint("far_right", 0, [900, 450, -2]),
    rigJoint("depth_only", 0, [0, 0, 100]),
  ], []);
  const { api } = await createRendererHarness(manifest);
  assert.equal(api.computeFK([4000, -3000, 20], [0, 0, 0, 1], []), true);

  const projected = plain(api.project(48, 48, 3));
  assert.equal(projected.length, 4);
  for (const point of projected) {
    assert.ok(point.sx >= 3 - 1e-9 && point.sx <= 45 + 1e-9,
      `x=${point.sx} must fit the padded canvas`);
    assert.ok(point.sy >= 3 - 1e-9 && point.sy <= 45 + 1e-9,
      `y=${point.sy} must fit the padded canvas`);
  }
  assert.ok(Math.abs(Math.max(...projected.map((point) => point.sy)) - 45) <= 1e-9,
    "wide poses stay bottom-aligned with the pet's contact edge");
  assert.deepEqual(projected.map((point) => point.idx), [2, 0, 1, 3],
    "equal-depth joints are stable by manifest index and Z is painter-sorted");
});

test("side-view projection follows +X and +Z character-forward coordinate systems", async () => {
  const cases = [
    {
      forward: "+X",
      forwardOffset: [10, 0, 0] as Vec3,
      backOffset: [0, 0, -2] as Vec3,
      expectedDepth: [0, 0, 1] as Vec3,
    },
    {
      forward: "+Z",
      forwardOffset: [0, 0, 10] as Vec3,
      backOffset: [2, 0, 0] as Vec3,
      expectedDepth: [-1, 0, 0] as Vec3,
    },
  ] as const;

  for (const projectionCase of cases) {
    const manifest = genericManifest([
      rigJoint("__motion_root__", -1, [0, 0, 0], "motion_root"),
      rigJoint("forward", 0, projectionCase.forwardOffset),
      rigJoint("up", 0, [0, 10, 0]),
      rigJoint("back", 0, projectionCase.backOffset),
    ], [], { handedness: "right", up: "+Y", forward: projectionCase.forward });
    const harness = await createRendererHarness(manifest);
    assert.equal(harness.api.computeFK([0, 0, 0], [0, 0, 0, 1], []), true);
    const projected = plain(harness.api.project(48, 48, 3));
    const root = projected.find((point) => point.idx === 0);
    const forward = projected.find((point) => point.idx === 1);
    const up = projected.find((point) => point.idx === 2);
    assert.ok(root && forward && up);
    assert.ok(forward.sx > root.sx, `${projectionCase.forward} maps character-forward to screen-right`);
    assert.ok(up.sy < root.sy, "+Y maps up on the canvas");
    assert.equal(projected[0]?.idx, 3, "negative camera depth is painted first");
    assertVecClose(plain(harness.api.getRig()?.projectionBasis.depth ?? []),
      projectionCase.expectedDepth);

    harness.stateHandler({
      facing: -1,
      rootTranslation: [0, 0, 0],
      rootRotation: [0, 0, 0, 1],
      localRotationDeltas: [],
    });
    const mirrored = plain(harness.api.project(48, 48, 3));
    const mirroredRoot = mirrored.find((point) => point.idx === 0);
    const mirroredForward = mirrored.find((point) => point.idx === 1);
    assert.ok(mirroredRoot && mirroredForward && mirroredForward.sx < mirroredRoot.sx,
      "logical left mirrors character-forward without changing camera depth");
  }
});

test("manifest sprite mode is resource-aware and exposes skeleton only in debug mode", async () => {
  const manifest = genericManifest([
    rigJoint("__motion_root__", -1, [0, 0, 0], "motion_root"),
    rigJoint("bone", 0, [5, 0, 0], "bone", [0, 0, 0, 1], true),
  ], [1], undefined, {
    canvas: [48, 48],
    displayScale: 2,
    footAnchor: [24, 46],
    sourceFacing: -1,
    mode: "sprite",
    fallbackModes: ["skinned_mesh", "debug_skeleton"],
    sprite: { image: "cat.png", metadata: "cat.json" },
    mesh: null,
  });
  const harness = await createRendererHarness(manifest);
  harness.fillRects.length = 0;
  harness.lineSegments.length = 0;
  harness.resetStrokeCount();
  harness.stateHandler({
    rootTranslation: [0, 0, 0],
    rootRotation: [0, 0, 0, 1],
    localRotationDeltas: [[0, 0, 0, 1]],
  });

  assert.equal(harness.api.getRig()?.renderMode, "debug_skeleton");
  assert.ok(harness.strokeCount > 0,
    "a missing sprite follows the declared chain to a pose-aware skeleton");

  harness.stateHandler({ assetDataUrl: "data:image/png;base64,AA==" });
  const loadedImage = harness.getLoadedImage();
  assert.ok(loadedImage?.onload);
  harness.drawCalls.length = 0;
  harness.resetStrokeCount();
  loadedImage.onload();

  assert.equal(harness.api.getRig()?.renderMode, "sprite");
  assert.equal(harness.strokeCount, 0, "normal sprite mode does not expose debug bones");
  assert.equal(harness.drawCalls.length, 1, "the loaded character sprite remains visible");

  harness.resetStrokeCount();
  harness.stateHandler({
    debug: true,
    rootTranslation: [0, 0, 0],
    rootRotation: [0, 0, 0, 1],
    localRotationDeltas: [[0, 0, 0, 1]],
  });
  assert.ok(harness.strokeCount > 0, "debug mode overlays the computed skeleton");

  harness.resetStrokeCount();
  harness.stateHandler({
    debug: false,
    assetDataUrl: "data:image/svg+xml;base64,AA==",
    rootTranslation: [0, 0, 0],
    rootRotation: [0, 0, 0, 1],
    localRotationDeltas: [[0, 0, 0, 1]],
  });
  assert.equal(harness.api.getRig()?.renderMode, "debug_skeleton");
  assert.ok(harness.strokeCount > 0,
    "an invalid replacement clears the stale sprite and returns to a safe fallback");
});

test("a null skeletal config clears FK and pose-warp state before redrawing", async () => {
  const half = Math.sqrt(0.5);
  const manifest = genericManifest([
    rigJoint("__motion_root__", -1, [0, 0, 0], "motion_root"),
    rigJoint("leaf", 0, [5, 0, 0], "bone", [0, 0, 0, 1], true),
  ], [1], undefined, {
    canvas: [48, 48],
    displayScale: 2,
    footAnchor: [24, 46],
    sourceFacing: -1,
    mode: "sprite",
    fallbackModes: ["debug_skeleton"],
    sprite: { image: "character.png", metadata: "character.json" },
    mesh: null,
  });
  const harness = await createRendererHarness(manifest);
  harness.stateHandler({
    assetDataUrl: "data:image/png;base64,AA==",
    rootTranslation: [0, 0, 0],
    rootRotation: [0, 0, 0, 1],
    localRotationDeltas: [[0, 0, half, half]],
  });
  const loadedImage = harness.getLoadedImage();
  assert.ok(loadedImage?.onload);
  loadedImage.onload();
  assert.ok(harness.api.getWarpedPixels());
  assert.ok(harness.api.getRig());

  harness.drawCalls.length = 0;
  const configHandler = harness.getSkeletalConfigHandler();
  assert.ok(configHandler);
  configHandler(null);

  assert.equal(harness.api.getRig(), null);
  assert.equal(harness.api.getWarpedPixels(), null);
  assert.equal(harness.drawCalls.length, 1);
  assert.strictEqual(harness.drawCalls[0]?.[0], loadedImage,
    "a cleared rig redraws the legacy whole sprite");

  configHandler(manifest);
  assert.ok(harness.api.getRig());
  assert.equal(harness.api.getWarpedPixels(), null);
  assertVecClose(plain(harness.api.getJointWorld())[1]?.rotation ?? [], [0, 0, 0, 1]);
  harness.api.draw();
  assertVecClose(plain(harness.api.getJointWorld())[1]?.rotation ?? [], [0, 0, 0, 1], 1e-8);

  harness.stateHandler({
    rootTranslation: [0, 0, 0], rootRotation: [0, 0, 0, 1],
    localRotationDeltas: [[0, 0, half, half]],
  });
  assertVecClose(plain(harness.api.getJointWorld())[1]?.rotation ?? [], [0, 0, half, half], 1e-8);
});

test("an invalid replacement rig also clears the previous FK state", async () => {
  const manifest = genericManifest([
    rigJoint("__motion_root__", -1, [0, 0, 0], "motion_root"),
    rigJoint("leaf", 0, [5, 0, 0], "bone", [0, 0, 0, 1], true),
  ], [1]);
  const harness = await createRendererHarness(manifest);
  assert.ok(harness.api.getRig());

  const configHandler = harness.getSkeletalConfigHandler();
  assert.ok(configHandler);
  configHandler({ rig: { joints: [{ id: "broken", parentIndex: 0 }] } });

  assert.equal(harness.api.getRig(), null);
  assert.equal(harness.api.getWarpedPixels(), null);
});

test("renderer rig normalization rejects disconnected roots, invalid parents and rest TRS", async () => {
  const manifest = genericManifest([
    rigJoint("__motion_root__", -1, [0, 0, 0], "motion_root"),
    rigJoint("leaf", 0, [1, 0, 0], "bone", [0, 0, 0, 1], true),
  ], [1]);
  const { api } = await createRendererHarness(manifest);
  const mutate = (update: (rig: Record<string, unknown>) => void): unknown => {
    const candidate = JSON.parse(JSON.stringify(manifest)) as Record<string, unknown>;
    update(candidate.rig as Record<string, unknown>);
    return candidate;
  };
  const jointsOf = (rig: Record<string, unknown>): Array<Record<string, unknown>> =>
    rig.joints as Array<Record<string, unknown>>;

  assert.equal(api.loadSkeleton(mutate((rig) => {
    jointsOf(rig)[1]!.parentIndex = -1;
  })), false, "a second root is disconnected from motionRoot");
  assert.equal(api.loadSkeleton(mutate((rig) => {
    jointsOf(rig)[1]!.parentIndex = "bad";
  })), false, "an explicitly invalid parentIndex cannot become an implicit root");
  assert.equal(api.loadSkeleton(mutate((rig) => {
    const rest = jointsOf(rig)[1]!.restLocal as Record<string, unknown>;
    rest.rotation = [0, 0, 0, 0];
  })), false, "a zero rest quaternion is rejected");
  assert.equal(api.loadSkeleton(mutate((rig) => {
    const rest = jointsOf(rig)[1]!.restLocal as Record<string, unknown>;
    rest.scale = [1, 0, 1];
  })), false, "a zero rest scale is rejected");
  assert.equal(api.getRig(), null);
});

test("skinned_mesh explicitly resolves to its first renderer-supported fallback", async () => {
  const manifest = genericManifest([
    rigJoint("__motion_root__", -1, [0, 0, 0], "motion_root"),
    rigJoint("bone", 0, [5, 0, 0], "bone", [0, 0, 0, 1], true),
  ], [1], undefined, {
    canvas: [48, 48],
    displayScale: 2,
    footAnchor: [24, 46],
    sourceFacing: -1,
    mode: "skinned_mesh",
    fallbackModes: ["sprite", "debug_skeleton"],
    sprite: { image: "cat.png", metadata: "cat.json" },
    mesh: { source: "source", skinIndex: 0 },
  });
  const harness = await createRendererHarness(manifest);

  assert.equal(harness.api.getRig()?.renderMode, "debug_skeleton",
    "an unavailable sprite is skipped in favor of the next executable fallback");
  harness.stateHandler({ assetDataUrl: "data:image/png;base64,AA==" });
  const loadedImage = harness.getLoadedImage();
  assert.ok(loadedImage?.onload);
  loadedImage.onload();
  assert.equal(harness.api.getRig()?.renderMode, "sprite",
    "the canvas renderer does not claim to skin mesh buffers it has not loaded");
  harness.drawCalls.length = 0;
  harness.resetStrokeCount();
  harness.stateHandler({
    rootTranslation: [0, 0, 0],
    rootRotation: [0, 0, 0, 1],
    localRotationDeltas: [[0, 0, 0, 1]],
  });
  assert.equal(harness.strokeCount, 0);
  assert.equal(harness.drawCalls.length, 1, "sprite fallback remains visible");
});

test("a failed character sprite load renders the declared debug skeleton, not the built-in cat", async () => {
  const manifest = genericManifest([
    rigJoint("__motion_root__", -1, [0, 0, 0], "motion_root"),
    rigJoint("bone", 0, [5, 0, 0], "bone", [0, 0, 0, 1], true),
  ], [1], undefined, {
    canvas: [48, 48],
    displayScale: 2,
    footAnchor: [24, 46],
    sourceFacing: -1,
    mode: "sprite",
    fallbackModes: ["debug_skeleton"],
    sprite: { image: "character.webp", metadata: "character.json" },
    mesh: null,
  });
  const harness = await createRendererHarness(manifest);
  harness.stateHandler({
    assetDataUrl: "data:image/webp;base64,AA==",
    rootTranslation: [0, 0, 0],
    rootRotation: [0, 0, 0, 1],
    localRotationDeltas: [[0, 0, 0, 1]],
  });
  const loadedImage = harness.getLoadedImage();
  assert.ok(loadedImage?.onerror);
  harness.fillRects.length = 0;
  harness.resetStrokeCount();
  loadedImage.onerror();

  assert.equal(harness.api.getRig()?.renderMode, "debug_skeleton");
  assert.ok(harness.strokeCount > 0);
  assert.ok(harness.fillRects.length <= 4,
    "the generic joint markers render without the many rectangles of drawFallbackCat");
});

test("the legacy pet-rig-v2 asset keeps its arbitrary driven count and renders generically", async () => {
  const legacyRig = JSON.parse(await readFile(
    new URL("../../../assets/pet/runtime/cat-skeleton-3d.json", import.meta.url),
    "utf8",
  )) as {
    readonly joints: readonly unknown[];
  };
  const harness = await createRendererHarness(legacyRig);
  const normalized = plain(harness.api.getRig());
  assert.ok(normalized);
  assert.equal(normalized.jointIds.length, legacyRig.joints.length);
  assert.ok(normalized.drivenJointIds.length > 0);

  harness.fillRects.length = 0;
  harness.lineSegments.length = 0;
  harness.resetStrokeCount();
  const identityDeltas = normalized.drivenJointIds.map(() => [0, 0, 0, 1]);
  harness.stateHandler({
    rootTranslation: [0, 0, 0],
    rootRotation: [0, 0, 0, 1],
    localRotationDeltas: identityDeltas,
  });

  assert.ok(harness.strokeCount > 0);
  assert.ok(harness.fillRects.length > 0);
  assert.ok(plain(harness.api.getJointWorld())
    .flatMap((joint) => [...joint.position, ...joint.rotation, ...joint.scale])
    .every(Number.isFinite));
});

test("the checked-in character manifest uses a +Z basis and waits for its sprite resource", async () => {
  const manifest = JSON.parse(await readFile(
    new URL("../../../assets/pet/runtime/cat-character-rig.manifest.json", import.meta.url),
    "utf8",
  )) as {
    readonly rig: {
      readonly joints: readonly unknown[];
      readonly drivenJointIndices: readonly number[];
    };
  };
  const harness = await createRendererHarness(manifest);
  const { api } = harness;
  const normalized = plain(api.getRig());
  assert.ok(normalized);
  assert.equal(normalized.jointIds.length, manifest.rig.joints.length);
  assert.equal(normalized.drivenJointIds.length, manifest.rig.drivenJointIndices.length);
  assert.equal(normalized.renderMode, "debug_skeleton");
  assertVecClose(normalized.projectionBasis.horizontal, [0, 0, 1]);
  assertVecClose(normalized.projectionBasis.vertical, [0, 1, 0]);
  assertVecClose(normalized.projectionBasis.depth, [-1, 0, 0]);

  harness.stateHandler({ assetDataUrl: "data:image/png;base64,AA==" });
  const loadedImage = harness.getLoadedImage();
  assert.ok(loadedImage?.onload);
  loadedImage.onload();
  assert.equal(api.getRig()?.renderMode, "sprite");

  const identities = normalized.drivenJointIds.map(() => [0, 0, 0, 1]);
  assert.equal(api.computeFK([0, 0, 0], [0, 0, 0, 1], identities), true);
  assert.ok(plain(api.getJointWorld())
    .flatMap((joint) => [...joint.position, ...joint.rotation, ...joint.scale])
    .every(Number.isFinite));
});

test("invalid pose vectors, zero quaternions and non-unit quaternions fail closed", async () => {
  const manifest = genericManifest([
    rigJoint("__motion_root__", -1, [0, 0, 0], "motion_root"),
    rigJoint("bone", 0, [1, 0, 0], "bone", [0, 0, 0, 1], true),
  ], [1]);
  const { api } = await createRendererHarness(manifest);

  const rest = plain(api.getJointWorld());
  assert.equal(api.computeFK([Number.NaN, 2, Number.POSITIVE_INFINITY], [0, 0, 0, 0], [
    [Number.NaN, 0, 0, 1],
  ]), false);
  assert.equal(api.computeFK([0, 0, 0], [0, 0, 0, 2], [[0, 0, 0, 1]]), false);
  assert.equal(api.computeFK([0, 0, 0], [0, 0, 0, 1], [[0, 0, 0, 0]]), false);
  assert.deepEqual(plain(api.getJointWorld()), rest, "a rejected pose cannot partially mutate FK state");
});

test("FK rejects derived matrix overflow atomically", async () => {
  const manifest = genericManifest([
    rigJoint("__motion_root__", -1, [Number.MAX_VALUE, 0, 0], "motion_root"),
    rigJoint("bone", 0, [1, 0, 0], "bone", [0, 0, 0, 1], true),
  ], [1]);
  const { api } = await createRendererHarness(manifest);
  const rest = plain(api.getJointWorld());

  assert.equal(
    api.computeFK([Number.MAX_VALUE, 0, 0], [0, 0, 0, 1], [[0, 0, 0, 1]]),
    false,
  );
  assert.deepEqual(
    plain(api.getJointWorld()),
    rest,
    "an overflow in a derived world matrix cannot publish a partial pose",
  );

  const explosiveRoot = rigJoint("__motion_root__", -1, [0, 0, 0], "motion_root");
  const explosiveChild = rigJoint("bone", 0, [Number.MAX_VALUE, 0, 0], "bone");
  const explosiveManifest = genericManifest([
    {
      ...explosiveRoot,
      restLocal: {
        ...explosiveRoot.restLocal,
        scale: [Number.MAX_VALUE, 1, 1] as Vec3,
      },
    },
    explosiveChild,
  ], []);
  const overflowHarness = await createRendererHarness();
  assert.equal(overflowHarness.api.loadSkeleton(explosiveManifest), false);
  assert.equal(overflowHarness.api.getRig(), null);
});
