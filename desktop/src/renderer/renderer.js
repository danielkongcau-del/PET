// ── Canvas setup ──
const canvas = document.getElementById("pet");
const context = canvas.getContext("2d", { alpha: true, willReadFrequently: true });
context.imageSmoothingEnabled = false;

// ── State (shared with legacy path) ──
let state = {
  facing: 1, lean: 0, squash: 1, bob: 0,
  expression: "neutral", behavior: "fallback",
  generatorStatus: "starting", debug: false,
};
let sourceImage = null;
let sourceFootAnchor = [24, 46];
let sourceFacing = -1;
let lastAssetDataUrl = "";
let lastReportedHit = null;

// ── Skeletal FK state ──
let skeleton = null;           // parsed cat-skeleton-3d.json
let jointIndex = {};           // id → array index
let jointWorld = [];           // [{x, y, z, angle}] per joint
let modelDrivenMap = [];       // joint indices in local_rotation_deltas order
let deltaIndexMap = new Map(); // joint index → delta array index (O(1) reverse lookup)
let fkOrder = [];              // topologically sorted indices (parents before children)

// ═══════════════════════════════════════════════
//  Quaternion helpers
// ═══════════════════════════════════════════════

function quatAngleZ(q) {
  // Return the Z-axis rotation angle (radians) of quaternion [x,y,z,w].
  return 2 * Math.atan2(q[2], q[3]);
}

function quatMul(a, b) {
  return [
    a[3]*b[0] + a[0]*b[3] + a[1]*b[2] - a[2]*b[1],
    a[3]*b[1] - a[0]*b[2] + a[1]*b[3] + a[2]*b[0],
    a[3]*b[2] + a[0]*b[1] - a[1]*b[0] + a[2]*b[3],
    a[3]*b[3] - a[0]*b[0] - a[1]*b[1] - a[2]*b[2],
  ];
}

// ═══════════════════════════════════════════════
//  Skeleton loading
// ═══════════════════════════════════════════════

function loadSkeleton(config) {
  if (!config || !Array.isArray(config.joints)) return false;
  skeleton = config;
  jointIndex = {};
  skeleton.joints.forEach((j, i) => { jointIndex[j.id] = i; });

  // Build model_driven → local_rotation_deltas index mapping.
  modelDrivenMap = [];
  const motionRoot = skeleton.motionRoot || "__motion_root__";
  for (let i = 0; i < skeleton.joints.length; i++) {
    const j = skeleton.joints[i];
    if (j.id === motionRoot) continue;
    const phys = (j.physics || {});
    const dofs = (j.poseDofs || {});
    if (phys.mode === "secondary" || phys.mode === "static") continue;
    if (dofs.rotation === true) modelDrivenMap.push(i);
  }
  // Build O(1) reverse lookup.
  deltaIndexMap = new Map();
  for (let d = 0; d < modelDrivenMap.length; d++) {
    deltaIndexMap.set(modelDrivenMap[d], d);
  }

  jointWorld = skeleton.joints.map(() => ({ x: 0, y: 0, z: 0, angle: 0 }));

  // Topological sort: parents before children.
  fkOrder = [];
  const visited = new Set();
  function visit(idx) {
    if (visited.has(idx)) return;
    visited.add(idx);
    const joint = skeleton.joints[idx];
    const parentIdx = jointIndex[joint.parent];
    if (parentIdx !== undefined && !visited.has(parentIdx)) visit(parentIdx);
    fkOrder.push(idx);
  }
  for (let i = 0; i < skeleton.joints.length; i++) visit(i);

  return true;
}

// ═══════════════════════════════════════════════
//  FK computation
// ═══════════════════════════════════════════════

function computeFK(rootTranslation, rootRotation, localRotationDeltas) {
  if (!skeleton) return;

  // 1. Set motion root
  const motionRootId = skeleton.motionRoot || "__motion_root__";
  const motionRootIdx = jointIndex[motionRootId];
  if (motionRootIdx === undefined) return;

  const rt = Array.isArray(rootTranslation) && rootTranslation.length === 3 ? rootTranslation : [0, 0, 0];
  const rr = Array.isArray(rootRotation) && rootRotation.length === 4 ? rootRotation : [0, 0, 0, 1];
  jointWorld[motionRootIdx] = { x: rt[0], y: rt[1], z: rt[2], angle: quatAngleZ(rr) };

  // 2. Walk joints in topological order (parents before children). Skip motion root.
  for (const i of fkOrder) {
    if (i === motionRootIdx) continue;
    const joint = skeleton.joints[i];
    const parentIdx = jointIndex[joint.parent];
    if (parentIdx === undefined) continue;

    const parent = jointWorld[parentIdx];
    const rl = joint.restLocal || {};
    const restTrans = Array.isArray(rl.translation) && rl.translation.length === 3 ? rl.translation : [0, 0, 0];
    const restRot = Array.isArray(rl.rotation) && rl.rotation.length === 4 ? rl.rotation : [0, 0, 0, 1];

    // Look up local_rotation_delta for this joint via O(1) reverse map.
    const deltaIdx = deltaIndexMap.get(i);
    const deltaRot = deltaIdx >= 0 && Array.isArray(localRotationDeltas) && deltaIdx < localRotationDeltas.length
      ? localRotationDeltas[deltaIdx] : null;
    const deltaAngle = deltaRot && deltaRot.length === 4 ? quatAngleZ(deltaRot) : 0;

    const parentAngle = parent.angle;
    const restAngle = quatAngleZ(restRot);
    const cosA = Math.cos(parentAngle);
    const sinA = Math.sin(parentAngle);

    // Rotate restLocal translation by parent world angle, then add restAngle + delta.
    jointWorld[i] = {
      x: parent.x + restTrans[0] * cosA - restTrans[1] * sinA,
      y: parent.y + restTrans[0] * sinA + restTrans[1] * cosA,
      z: parent.z + restTrans[2],
      angle: parentAngle + restAngle + deltaAngle,
    };
  }
}

// ═══════════════════════════════════════════════
//  Projection: world → canvas
// ═══════════════════════════════════════════════

function project(originX, originY, scale) {
  // Orthographic side-view (pitch = 0).
  // World: +X=right, +Y=up, +Z=out-of-screen.
  // Canvas: +X=right, +Y=down.
  const result = jointWorld.map((jw, i) => ({
    idx: i,
    sx: originX + jw.x * scale,
    sy: originY - jw.y * scale,
    sz: jw.z,
    angle: jw.angle,
    id: skeleton ? skeleton.joints[i].id : "",
    sprite: skeleton ? skeleton.joints[i].sprite : null,
    role: skeleton ? skeleton.joints[i].role : "",
  }));
  // Sort back-to-front by Z for painter's algorithm.
  result.sort((a, b) => a.sz - b.sz);
  return result;
}

// ═══════════════════════════════════════════════
//  Procedural cat drawing (2D geometric shapes)
// ═══════════════════════════════════════════════

const PART_COLORS = {
  body:       { fill: "#f5f0e8", stroke: "#2d2d2d" },
  head:       { fill: "#f5f0e8", stroke: "#2d2d2d" },
  upper_arm:  { fill: "#f5f0e8", stroke: "#2d2d2d" },
  upper_leg:  { fill: "#f5f0e8", stroke: "#2d2d2d" },
  tail:       { fill: "#2d2d2d", stroke: "#2d2d2d" },
};

function drawProceduralPart(ctx, pos, joint, nextPos) {
  const role = joint.role || "";
  const sprite = joint.sprite || "";
  const angle = pos.angle;
  const size = 5; // base thickness in canvas units

  ctx.save();
  ctx.translate(pos.sx, pos.sy);

  if (role === "head" || sprite === "head") {
    // Ellipse head + triangle ears
    ctx.rotate(angle);
    ctx.fillStyle = PART_COLORS.head.fill;
    ctx.strokeStyle = PART_COLORS.head.stroke;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.ellipse(0, 0, 7, 5.5, 0, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
    // Ears
    ctx.beginPath();
    ctx.moveTo(-5, -4); ctx.lineTo(-7, -11); ctx.lineTo(-1, -5); ctx.fill();
    ctx.moveTo(5, -4); ctx.lineTo(7, -11); ctx.lineTo(1, -5); ctx.fill();
    ctx.stroke();
    // Eyes
    ctx.fillStyle = "#2d2d2d";
    ctx.beginPath();
    ctx.arc(-3, -1, 1.2, 0, Math.PI * 2); ctx.fill();
    ctx.arc(3, -1, 1.2, 0, Math.PI * 2); ctx.fill();
    // Nose
    ctx.fillStyle = "#f5a0a0";
    ctx.beginPath();
    ctx.arc(0, 1.5, 0.8, 0, Math.PI * 2); ctx.fill();
  } else if (role === "spine" || sprite === "body") {
    ctx.rotate(angle);
    ctx.fillStyle = PART_COLORS.body.fill;
    ctx.strokeStyle = PART_COLORS.body.stroke;
    ctx.lineWidth = 1;
    roundRect(ctx, -5, -7, 10, 14, 3);
    ctx.fill();
    ctx.stroke();
  } else if (role === "body_root" || sprite === "body_root") {
    // Pelvis anchor — small circle at the hip.
    ctx.fillStyle = PART_COLORS.body.fill;
    ctx.strokeStyle = PART_COLORS.body.stroke;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.arc(0, 0, 3, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
  } else if (role === "ear" || sprite === "ear") {
    // Ears deform the head sprite; no standalone geometry.
    // Head drawProceduralPart handles the visual triangles.
  } else if (role === "arm" || sprite.startsWith("upper_arm")) {
    ctx.rotate(angle + Math.PI / 2);
    ctx.fillStyle = PART_COLORS.upper_arm.fill;
    ctx.strokeStyle = PART_COLORS.upper_arm.stroke;
    ctx.lineWidth = 1;
    roundRect(ctx, -2.5, 0, 5, 8, 2);
    ctx.fill();
    ctx.stroke();
  } else if (role === "leg" || sprite.startsWith("upper_leg")) {
    ctx.rotate(angle + Math.PI / 2);
    ctx.fillStyle = PART_COLORS.upper_leg.fill;
    ctx.strokeStyle = PART_COLORS.upper_leg.stroke;
    ctx.lineWidth = 1;
    roundRect(ctx, -3, 0, 6, 9, 2.5);
    ctx.fill();
    ctx.stroke();
  } else if (role === "tail" || sprite === "tail_base") {
    // Draw a curved tail segment from this pos toward nextPos
    if (nextPos) {
      ctx.strokeStyle = PART_COLORS.tail.stroke;
      ctx.lineWidth = 2.5;
      ctx.lineCap = "round";
      ctx.beginPath();
      const dx = nextPos.sx - pos.sx;
      const dy = nextPos.sy - pos.sy;
      const midX = dx * 0.5;
      const midY = dy * 0.5 - 4;
      ctx.moveTo(0, 0);
      ctx.quadraticCurveTo(midX, midY, dx, dy);
      ctx.stroke();
    }
  } else if (role === "tail_tip" || sprite === "tail_tip") {
    // Small tuft at the tip (drawn by tail_base's nextPos pass)
    ctx.fillStyle = PART_COLORS.tail.fill;
    ctx.beginPath();
    ctx.arc(0, 0, 2, 0, Math.PI * 2); ctx.fill();
  }

  ctx.restore();
}

function roundRect(ctx, x, y, w, h, r) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.lineTo(x + w - r, y);
  ctx.arcTo(x + w, y, x + w, y + r, r);
  ctx.lineTo(x + w, y + h - r);
  ctx.arcTo(x + w, y + h, x + w - r, y + h, r);
  ctx.lineTo(x + r, y + h);
  ctx.arcTo(x, y + h, x, y + h - r, r);
  ctx.lineTo(x, y + r);
  ctx.arcTo(x, y, x + r, y, r);
  ctx.closePath();
}

// ═══════════════════════════════════════════════
//  Main draw function
// ═══════════════════════════════════════════════

function draw() {
  context.clearRect(0, 0, 48, 48);
  const has3D = skeleton && state.rootTranslation && state.localRotationDeltas;

  if (has3D) {
    // ── FK path ──
    computeFK(state.rootTranslation, state.rootRotation, state.localRotationDeltas);
    const projected = project(24, 46, 1.0); // origin at foot anchor (24, 46)

    const nextByParent = {};
    for (const p of projected) {
      const joint = skeleton.joints[p.idx];
      if (joint && joint.parent) nextByParent[joint.parent] = p;
    }

    for (const p of projected) {
      const joint = skeleton.joints[p.idx];
      if (!joint || joint.id === (skeleton.motionRoot || "__motion_root__")) continue;
      const next = nextByParent[joint.id];
      drawProceduralPart(context, p, joint, next || undefined);
    }
  } else {
    // ── Legacy whole-sprite path ──
    context.save();
    const [footX, footY] = sourceFootAnchor;
    context.translate(footX, footY);
    const squash = clamp(Number(state.squash) || 1, 0.5, 1.5);
    const facing = state.facing === -1 ? -1 : 1;
    context.rotate(clamp(Number(state.lean) || 0, -1, 1) * 0.11);
    context.scale(sourceFacing * facing / Math.sqrt(squash), squash);
    context.translate(-footX, -footY);
    if (sourceImage) context.drawImage(sourceImage, 0, 0, 48, 48);
    else drawFallbackCat(context);
    context.restore();
  }
}

// ═══════════════════════════════════════════════
//  IPC listeners
// ═══════════════════════════════════════════════

window.petHost.onState((next) => {
  if (!next || typeof next !== "object") return;
  state = { ...state, ...next };
  if (!("boneRotations" in next)) state.boneRotations = undefined;
  if (!("rootTranslation" in next)) state.rootTranslation = undefined;
  if (!("rootRotation" in next)) state.rootRotation = undefined;
  if (!("localRotationDeltas" in next)) state.localRotationDeltas = undefined;
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

// Try to load skeleton config at startup (may already be available).
if (typeof window.petHost.getSkeletalConfig === "function") {
  const cfg = window.petHost.getSkeletalConfig();
  if (cfg) loadSkeleton(cfg);
}

// Listen for late-arriving skeleton config (sent after generator negotiation).
if (typeof window.petHost.onSkeletalConfig === "function") {
  window.petHost.onSkeletalConfig((config) => {
    if (loadSkeleton(config)) draw();
  });
}

// ── Legacy helpers ──
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

function readSpriteMetadata(value) {
  if (!value || typeof value !== "object" || value.canvas?.[0] !== 48 || value.canvas?.[1] !== 48 ||
    !Array.isArray(value.footAnchor) || value.footAnchor.length !== 2 || !value.footAnchor.every(Number.isFinite)) return null;
  const [x, y] = value.footAnchor.map(Number);
  if (x < 0 || x > 48 || y < 0 || y > 48) return null;
  const metadataFacing = value.sourceFacing === 1 || value.sourceFacing === -1 ? value.sourceFacing : -1;
  return { footAnchor: [x, y], sourceFacing: metadataFacing };
}

function drawFallbackCat(ctx) {
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
