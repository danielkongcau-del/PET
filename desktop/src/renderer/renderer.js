const canvas = document.getElementById("pet");
const context = canvas.getContext("2d", { alpha: true, willReadFrequently: true });
context.imageSmoothingEnabled = false;

let state = {
  facing: 1,
  lean: 0,
  squash: 1,
  bob: 0,
  expression: "neutral",
  behavior: "fallback",
  generatorStatus: "starting",
  debug: false,
};
let sourceImage = null;
let sourceFootAnchor = [24, 46];
let sourceFacing = -1;
let lastAssetDataUrl = "";
let lastReportedHit = null;

window.petHost.onState((next) => {
  if (!next || typeof next !== "object") return;
  state = { ...state, ...next };
  document.documentElement.dataset.generator = String(state.generatorStatus || "stopped");
  document.documentElement.dataset.debug = state.debug === true ? "true" : "false";
  const spriteMetadata = readSpriteMetadata(state.assetParts);
  if (spriteMetadata) {
    sourceFootAnchor = spriteMetadata.footAnchor;
    sourceFacing = spriteMetadata.sourceFacing;
  }
  if (typeof state.assetDataUrl === "string" && state.assetDataUrl.startsWith("data:image/png;base64,") && state.assetDataUrl !== lastAssetDataUrl) {
    lastAssetDataUrl = state.assetDataUrl;
    const image = new Image();
    image.onload = () => { sourceImage = image; draw(); };
    image.src = state.assetDataUrl;
  }
  draw();
});

window.petHost.onProbe((point) => {
  if (!point || !Number.isFinite(point.clientX) || !Number.isFinite(point.clientY)) return;
  reportHit(isOpaqueAt(point.clientX, point.clientY), true);
});

canvas.addEventListener("mousemove", (event) => {
  reportHit(isOpaqueAt(event.clientX, event.clientY));
}, { passive: true });

canvas.addEventListener("mouseleave", () => reportHit(false));
canvas.addEventListener("mousedown", (event) => {
  if (!isOpaqueAt(event.clientX, event.clientY)) return;
  event.preventDefault();
  event.stopPropagation();
  const button = event.button === 2 ? "right" : event.button === 1 ? "middle" : "left";
  window.petHost.click(button);
});
document.addEventListener("contextmenu", (event) => event.preventDefault());

function reportHit(opaque, force = false) {
  if (!force && lastReportedHit === opaque) return;
  lastReportedHit = opaque;
  window.petHost.reportHit(opaque);
}

function isOpaqueAt(clientX, clientY) {
  const rect = canvas.getBoundingClientRect();
  if (clientX < rect.left || clientX >= rect.right || clientY < rect.top || clientY >= rect.bottom) return false;
  const x = Math.max(0, Math.min(47, Math.floor((clientX - rect.left) * 48 / rect.width)));
  const y = Math.max(0, Math.min(47, Math.floor((clientY - rect.top) * 48 / rect.height)));
  return context.getImageData(x, y, 1, 1).data[3] >= 48;
}

function draw() {
  context.clearRect(0, 0, 48, 48);
  context.save();
  const [footX, footY] = sourceFootAnchor;
  context.translate(footX, footY);
  const squash = clamp(Number(state.squash) || 1, 0.5, 1.5);
  const facing = state.facing === -1 ? -1 : 1;
  // lean is already signed in screen coordinates by the motion planner.
  context.rotate(clamp(Number(state.lean) || 0, -1, 1) * 0.11);
  context.scale(sourceFacing * facing / Math.sqrt(squash), squash);
  context.translate(-footX, -footY);
  // cat-48.png is a flattened sprite. Rotating rectangular slices cut from it
  // exposes transparent cracks between the head, body and legs. Until the art
  // has genuine overlapping semantic layers, keep every source pixel in one
  // coherent draw and apply generated pose transforms to the whole sprite.
  if (sourceImage) context.drawImage(sourceImage, 0, 0, 48, 48);
  else drawFallbackCat(context);
  context.restore();
}

function readSpriteMetadata(value) {
  if (!value || typeof value !== "object" || value.canvas?.[0] !== 48 || value.canvas?.[1] !== 48 ||
    !Array.isArray(value.footAnchor) || value.footAnchor.length !== 2 || !value.footAnchor.every(Number.isFinite)) return null;
  const [x, y] = value.footAnchor.map(Number);
  if (x < 0 || x > 48 || y < 0 || y > 48) return null;
  const metadataFacing = value.sourceFacing === 1 || value.sourceFacing === -1 ? value.sourceFacing : -1;
  return { footAnchor: [x, y], sourceFacing: metadataFacing };
}

function drawFallbackCat(ctx) {
  // Pixel-step silhouette matching the supplied white, black-outline cat.
  ctx.fillStyle = "#171717";
  ctx.fillRect(7, 17, 3, 21);
  ctx.fillRect(9, 13, 3, 7);
  ctx.fillRect(11, 10, 3, 5);
  ctx.fillRect(14, 13, 5, 3);
  ctx.fillRect(18, 10, 3, 5);
  ctx.fillRect(21, 8, 3, 7);
  ctx.fillRect(23, 13, 10, 3);
  ctx.fillRect(31, 15, 4, 4);
  ctx.fillRect(34, 18, 3, 15);
  ctx.fillRect(36, 28, 4, 4);
  ctx.fillRect(39, 23, 3, 7);
  ctx.fillRect(41, 17, 3, 8);
  ctx.fillRect(39, 14, 4, 4);
  ctx.fillRect(36, 16, 3, 5);
  ctx.fillRect(38, 20, 3, 3);
  ctx.fillRect(10, 37, 5, 3);
  ctx.fillRect(14, 39, 5, 3);
  ctx.fillRect(18, 36, 3, 5);
  ctx.fillRect(29, 36, 3, 5);
  ctx.fillRect(31, 39, 6, 3);
  ctx.fillRect(35, 34, 3, 6);
  ctx.fillRect(13, 15, 20, 24);
  ctx.fillStyle = "#faf9f4";
  ctx.fillRect(10, 18, 24, 18);
  ctx.fillRect(12, 14, 6, 8);
  ctx.fillRect(14, 16, 13, 13);
  ctx.fillRect(20, 13, 4, 8);
  ctx.fillRect(15, 35, 5, 4);
  ctx.fillRect(31, 32, 4, 6);
  ctx.fillRect(38, 18, 3, 3);
  ctx.fillRect(39, 23, 2, 4);
  ctx.fillStyle = "#171717";
  ctx.fillRect(16, 21, 2, 2);
  ctx.fillRect(26, 21, 2, 2);
  ctx.fillRect(21, 24, 3, 2);
  if (state.expression === "surprised") ctx.fillRect(21, 27, 3, 3);
}

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

draw();
window.petHost.ready();
