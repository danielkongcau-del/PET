import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import vm from "node:vm";

test("the flattened cat is drawn once as a complete sprite", async () => {
  const rendererSource = await readFile(new URL("../../src/renderer/renderer.js", import.meta.url), "utf8");
  const metadata = JSON.parse(await readFile(new URL("../../../assets/pet/runtime/cat-parts.json", import.meta.url), "utf8")) as {
    readonly footAnchor: readonly [number, number];
    readonly sourceFacing: -1 | 1;
  };
  const drawCalls: unknown[][] = [];
  const translations: Array<[number, number]> = [];
  const scales: Array<[number, number]> = [];
  const rotations: number[] = [];
  const context = {
    imageSmoothingEnabled: true,
    fillStyle: "",
    clearRect() {},
    save() {},
    restore() {},
    translate(x: number, y: number) { translations.push([x, y]); },
    rotate(radians: number) { rotations.push(radians); },
    scale(x: number, y: number) { scales.push([x, y]); },
    fillRect() {},
    drawImage(...args: unknown[]) { drawCalls.push(args); },
    getImageData() { return { data: new Uint8ClampedArray([0, 0, 0, 255]) }; },
  };
  const canvas = {
    getContext: () => context,
    addEventListener() {},
    getBoundingClientRect: () => ({ left: 0, top: 0, right: 96, bottom: 96, width: 96, height: 96 }),
  };
  let stateHandler: ((state: Record<string, unknown>) => void) | undefined;
  let loadedImage: FakeImage | undefined;
  class FakeImage {
    onload: (() => void) | null = null;
    set src(_value: string) { loadedImage = this; }
  }
  const petHost = {
    onState(handler: (state: Record<string, unknown>) => void) { stateHandler = handler; },
    onProbe() {},
    reportHit() {},
    click() {},
    ready() {},
  };
  const document = {
    documentElement: { dataset: {} as Record<string, string> },
    getElementById: () => canvas,
    addEventListener() {},
  };
  vm.runInNewContext(rendererSource, {
    document,
    window: { petHost },
    Image: FakeImage,
    Uint8ClampedArray,
  });

  assert.ok(stateHandler);
  stateHandler({
    assetParts: metadata,
    assetDataUrl: "data:image/png;base64,AA==",
    facing: 1,
    squash: 1,
    lean: 0.25,
  });
  assert.ok(loadedImage?.onload);
  drawCalls.length = 0;
  translations.length = 0;
  scales.length = 0;
  rotations.length = 0;
  loadedImage.onload();

  assert.equal(context.imageSmoothingEnabled, false);
  assert.equal(drawCalls.length, 1);
  assert.strictEqual(drawCalls[0]?.[0], loadedImage);
  assert.deepEqual(drawCalls[0]?.slice(1), [0, 0, 48, 48]);
  assert.deepEqual(translations, [
    [metadata.footAnchor[0], metadata.footAnchor[1]],
    [-metadata.footAnchor[0], -metadata.footAnchor[1]],
  ]);
  assert.equal(metadata.sourceFacing, -1);
  assert.equal(Math.sign(scales[0]?.[0] ?? 0), -1, "logical right mirrors the left-facing master");
  assert.ok((rotations[0] ?? 0) > 0, "rightward screen-space lean remains positive");

  scales.length = 0;
  rotations.length = 0;
  stateHandler({ facing: -1, lean: -0.25 });
  assert.equal(Math.sign(scales[0]?.[0] ?? 0), 1, "logical left keeps the master unmirrored");
  assert.ok((rotations[0] ?? 0) < 0, "leftward screen-space lean remains negative");
});
