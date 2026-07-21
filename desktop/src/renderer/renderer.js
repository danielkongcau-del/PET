const canvas = document.getElementById("pet");
const context = canvas.getContext("2d", { alpha: true, willReadFrequently: true });
context.imageSmoothingEnabled = false;

const CANVAS_WIDTH = Number.isFinite(canvas.width) && canvas.width > 0 ? canvas.width : 48;
const CANVAS_HEIGHT = Number.isFinite(canvas.height) && canvas.height > 0 ? canvas.height : 48;
const IDENTITY_QUAT = Object.freeze([0, 0, 0, 1]);
const EPSILON = 1e-10;
const MIN_SCALE_MAGNITUDE = 1e-12;
const REST_QUAT_NORM_TOLERANCE = 1e-5;
const POSE_QUAT_NORM_TOLERANCE = 1e-3;
const NEUTRAL_FACIAL_PARAMS = Object.freeze({
  eye_scale: 1,
  eye_squint: 0,
  mouth_open: 0,
  ear_angle: 0,
  brow_tilt: 0,
});
const EXPRESSION_DEFAULTS = Object.freeze({
  neutral: NEUTRAL_FACIAL_PARAMS,
  surprised: Object.freeze({ eye_scale: 1.35, eye_squint: 0, mouth_open: 0.7, ear_angle: -0.2, brow_tilt: 0.3 }),
  happy: Object.freeze({ eye_scale: 1.15, eye_squint: 0.3, mouth_open: 0.4, ear_angle: 0, brow_tilt: 0.1 }),
  annoyed: Object.freeze({ eye_scale: 0.9, eye_squint: 0.6, mouth_open: 0, ear_angle: 0.3, brow_tilt: -0.5 }),
  curious: Object.freeze({ eye_scale: 1.1, eye_squint: 0, mouth_open: 0.1, ear_angle: 0, brow_tilt: 0.2 }),
  focused: Object.freeze({ eye_scale: 1, eye_squint: 0.3, mouth_open: 0, ear_angle: 0, brow_tilt: -0.15 }),
  sleepy: Object.freeze({ eye_scale: 0.65, eye_squint: 0.8, mouth_open: 0, ear_angle: -0.4, brow_tilt: 0 }),
  relieved: Object.freeze({ eye_scale: 1.05, eye_squint: 0, mouth_open: 0.2, ear_angle: 0, brow_tilt: 0.15 }),
});
const IDENTITY_MATRIX = Object.freeze([
  1, 0, 0, 0,
  0, 1, 0, 0,
  0, 0, 1, 0,
  0, 0, 0, 1,
]);

let state = {
  facing: 1, lean: 0, squash: 1, bob: 0,
  expression: "neutral", behavior: "fallback",
  generatorStatus: "starting", debug: false,
};
let sourceImage = null;
let spriteLoadState = "unrequested";
let sourceFootAnchor = [24, 46];
let sourceFacing = -1;
let lastAssetDataUrl = "";
let lastReportedHit = null;
const SAFE_RASTER_DATA_URL = /^data:image\/(?:png|webp|jpeg);base64,(?=[A-Za-z0-9+/])(?:[A-Za-z0-9+/]{4})*(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?$/;

// Normalized runtime rig state. Both pet-rig-v2 and
// pet-character-rig-manifest-v1 are converted to this representation.
let skeleton = null;
let jointWorld = [];
let restJointWorld = [];
let modelDrivenMap = [];
let deltaIndexMap = new Map();
let fkOrder = [];
let jointReferenceAxes = [];
let sourceContentBounds = [3, 3, 42, 43];
let spriteSourcePixels = null;
let spriteWarpCanvas = null;
let spriteWarpContext = null;
let spriteWarpBindings = null;
let spriteWarpRestControls = null;
let lastWarpedPixels = null;

function finiteNumber(value, fallback = 0) {
  return Number.isFinite(value) ? Number(value) : fallback;
}

function resolvedFacialParams() {
  const explicit = state.facialParams;
  const hasExplicit = explicit !== null && typeof explicit === "object" && !Array.isArray(explicit);
  const source = hasExplicit
    ? explicit
    : EXPRESSION_DEFAULTS[String(state.expression)] ?? NEUTRAL_FACIAL_PARAMS;
  return {
    // An explicit facial_params object is sparse by contract. Missing channels
    // are neutral, rather than leaking a previous frame or the expression map.
    eye_scale: clamp(finiteNumber(source.eye_scale, 1), 0.5, 1.5),
    eye_squint: clamp(finiteNumber(source.eye_squint, 0), 0, 1),
    mouth_open: clamp(finiteNumber(source.mouth_open, 0), 0, 1),
    ear_angle: clamp(finiteNumber(source.ear_angle, 0), -0.5, 0.5),
    brow_tilt: clamp(finiteNumber(source.brow_tilt, 0), -1, 1),
  };
}

function genericFacialTransform() {
  const facial = resolvedFacialParams();
  const eyeDelta = facial.eye_scale - 1;
  const facing = state.facing === -1 ? -1 : 1;
  return {
    // This deliberately subtle whole-character transform is the deterministic
    // fallback for rigs without character-specific facial bindings. Every
    // channel contributes, while the final bounds prevent extreme deformation.
    scaleX: clamp(1 + eyeDelta * 0.04 + facial.mouth_open * 0.01, 0.97, 1.04),
    scaleY: clamp(
      1 + eyeDelta * 0.03 - facial.eye_squint * 0.025 + facial.mouth_open * 0.035,
      0.95,
      1.06,
    ),
    rotation: facing * clamp(
      facial.ear_angle * 0.05 + facial.brow_tilt * 0.012,
      -0.035,
      0.035,
    ),
    bob: clamp(
      facial.eye_squint * 0.18 - facial.mouth_open * 0.35 - facial.brow_tilt * 0.3,
      -0.75,
      0.75,
    ),
  };
}

function drawWithGenericFacialTransform(drawContent) {
  const transform = genericFacialTransform();
  const isIdentity = Math.abs(transform.scaleX - 1) <= EPSILON &&
    Math.abs(transform.scaleY - 1) <= EPSILON &&
    Math.abs(transform.rotation) <= EPSILON &&
    Math.abs(transform.bob) <= EPSILON;
  if (isIdentity) {
    drawContent();
    return;
  }

  const [footX, footY] = sourceFootAnchor;
  context.save();
  try {
    context.translate(0, transform.bob);
    context.translate(footX, footY);
    context.rotate(transform.rotation);
    context.scale(transform.scaleX, transform.scaleY);
    context.translate(-footX, -footY);
    drawContent();
  } finally {
    context.restore();
  }
}

function finiteVec3(value, fallback = [0, 0, 0]) {
  if (!Array.isArray(value) || value.length !== 3) return [...fallback];
  return [
    finiteNumber(value[0], fallback[0]),
    finiteNumber(value[1], fallback[1]),
    finiteNumber(value[2], fallback[2]),
  ];
}

function isFiniteVec3(value) {
  return Array.isArray(value) && value.length === 3 && value.every(Number.isFinite);
}

function isUnitQuat(value, tolerance) {
  if (!Array.isArray(value) || value.length !== 4 || !value.every(Number.isFinite)) return false;
  const norm = Math.hypot(value[0], value[1], value[2], value[3]);
  return Number.isFinite(norm) && norm > EPSILON && Math.abs(norm - 1) <= tolerance;
}

function normalizeQuat(value) {
  if (!Array.isArray(value) || value.length !== 4 || !value.every(Number.isFinite)) {
    return [...IDENTITY_QUAT];
  }
  const q = value.map(Number);
  const norm = Math.hypot(q[0], q[1], q[2], q[3]);
  if (!Number.isFinite(norm) || norm <= EPSILON) return [...IDENTITY_QUAT];
  const inverseNorm = 1 / norm;
  return q.map((component) => component * inverseNorm);
}

function quatMul(a, b) {
  return [
    a[3] * b[0] + a[0] * b[3] + a[1] * b[2] - a[2] * b[1],
    a[3] * b[1] - a[0] * b[2] + a[1] * b[3] + a[2] * b[0],
    a[3] * b[2] + a[0] * b[1] - a[1] * b[0] + a[2] * b[3],
    a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2],
  ];
}

function quatRotateVec3(quaternion, vector) {
  const q = normalizeQuat(quaternion);
  const v = finiteVec3(vector);
  const qx = q[0];
  const qy = q[1];
  const qz = q[2];
  const qw = q[3];
  const tx = 2 * (qy * v[2] - qz * v[1]);
  const ty = 2 * (qz * v[0] - qx * v[2]);
  const tz = 2 * (qx * v[1] - qy * v[0]);
  return finiteVec3([
    v[0] + qw * tx + (qy * tz - qz * ty),
    v[1] + qw * ty + (qz * tx - qx * tz),
    v[2] + qw * tz + (qx * ty - qy * tx),
  ]);
}

function trsMatrix(translation, rotation, scale) {
  const [x, y, z, w] = normalizeQuat(rotation);
  const [sx, sy, sz] = scale;
  const xx = x * x;
  const yy = y * y;
  const zz = z * z;
  const xy = x * y;
  const xz = x * z;
  const yz = y * z;
  const wx = w * x;
  const wy = w * y;
  const wz = w * z;
  return [
    (1 - 2 * (yy + zz)) * sx, 2 * (xy - wz) * sy, 2 * (xz + wy) * sz, translation[0],
    2 * (xy + wz) * sx, (1 - 2 * (xx + zz)) * sy, 2 * (yz - wx) * sz, translation[1],
    2 * (xz - wy) * sx, 2 * (yz + wx) * sy, (1 - 2 * (xx + yy)) * sz, translation[2],
    0, 0, 0, 1,
  ];
}

function multiplyMatrices(left, right) {
  const output = new Array(16).fill(0);
  for (let row = 0; row < 4; row++) {
    for (let column = 0; column < 4; column++) {
      let value = 0;
      for (let inner = 0; inner < 4; inner++) {
        value += left[row * 4 + inner] * right[inner * 4 + column];
      }
      output[row * 4 + column] = value;
    }
  }
  return output;
}

function isFiniteMatrix(matrix) {
  return Array.isArray(matrix) && matrix.length === 16 && matrix.every(Number.isFinite);
}

function matrixTranslation(matrix) {
  return finiteVec3([matrix[3], matrix[7], matrix[11]]);
}

function transformMatrixDirection(matrix, vector) {
  return finiteVec3([
    matrix[0] * vector[0] + matrix[1] * vector[1] + matrix[2] * vector[2],
    matrix[4] * vector[0] + matrix[5] * vector[1] + matrix[6] * vector[2],
    matrix[8] * vector[0] + matrix[9] * vector[1] + matrix[10] * vector[2],
  ]);
}

function quatConjugate(quaternion) {
  const q = normalizeQuat(quaternion);
  return [-q[0], -q[1], -q[2], q[3]];
}

function axisAngleQuat(axis, angle) {
  const normalizedAxis = finiteVec3(axis);
  const axisLength = Math.hypot(...normalizedAxis);
  if (!Number.isFinite(angle) || !Number.isFinite(axisLength) || axisLength <= EPSILON) {
    return [...IDENTITY_QUAT];
  }
  const halfAngle = angle / 2;
  const factor = Math.sin(halfAngle) / axisLength;
  return normalizeQuat([
    normalizedAxis[0] * factor,
    normalizedAxis[1] * factor,
    normalizedAxis[2] * factor,
    Math.cos(halfAngle),
  ]);
}

function multiplyVec3(a, b) {
  return finiteVec3([a[0] * b[0], a[1] * b[1], a[2] * b[2]]);
}

function dotVec3(a, b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

function crossVec3(a, b) {
  return [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
}

function axisVector(value) {
  const axes = {
    "+X": [1, 0, 0], "-X": [-1, 0, 0],
    "+Y": [0, 1, 0], "-Y": [0, -1, 0],
    "+Z": [0, 0, 1], "-Z": [0, 0, -1],
  };
  return axes[value] ? [...axes[value]] : null;
}

function projectionBasis(coordinateSystem) {
  const horizontal = axisVector(coordinateSystem?.forward) ?? [1, 0, 0];
  const vertical = axisVector(coordinateSystem?.up) ?? [0, 1, 0];
  if (Math.abs(dotVec3(horizontal, vertical)) > EPSILON) {
    return { horizontal: [1, 0, 0], vertical: [0, 1, 0], depth: [0, 0, 1] };
  }
  const handednessSign = coordinateSystem?.handedness === "left" ? -1 : 1;
  const depth = crossVec3(horizontal, vertical)
    .map((component) => component * handednessSign);
  return { horizontal, vertical, depth };
}

function matrixCanvasAngle(matrix, localAxis, basis, facingSign) {
  const axis = transformMatrixDirection(matrix, localAxis);
  const screenX = dotVec3(axis, basis.horizontal) * facingSign;
  const screenY = -dotVec3(axis, basis.vertical);
  if (Math.hypot(screenX, screenY) <= EPSILON) return 0;
  return Math.atan2(screenY, screenX);
}

function selectStableJointReferenceAxes(worldPose, basis) {
  const candidates = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
  return worldPose.map((world) => {
    let selected = candidates[0];
    let selectedMagnitude = -1;
    for (const candidate of candidates) {
      const direction = transformMatrixDirection(world.matrix, candidate);
      const horizontal = dotVec3(direction, basis.horizontal);
      const vertical = dotVec3(direction, basis.vertical);
      const magnitude = horizontal * horizontal + vertical * vertical;
      if (magnitude > selectedMagnitude) {
        selected = candidate;
        selectedMagnitude = magnitude;
      }
    }
    return [...selected];
  });
}

function rotationDofEnabled(joint) {
  if (joint.poseDofs?.rotation === true) return true;
  const rotation = joint.dofMask?.rotation;
  if (rotation === true) return true;
  if (typeof rotation === "number") return rotation !== 0;
  return Array.isArray(rotation) && rotation.some((enabled) => enabled === true || enabled === 1);
}

function normalizeRenderConfig(value) {
  if (!value || typeof value !== "object") return null;
  const supported = new Set(["sprite", "skinned_mesh", "debug_skeleton"]);
  const mode = supported.has(value.mode) ? value.mode : null;
  if (!mode) return null;
  const fallbackModes = Array.isArray(value.fallbackModes)
    ? value.fallbackModes.filter((candidate, index, all) =>
      supported.has(candidate) && all.indexOf(candidate) === index)
    : [];
  const canvasSize = Array.isArray(value.canvas) && value.canvas.length === 2 &&
    value.canvas.every((component) => Number.isFinite(component) && component > 0)
    ? value.canvas.map(Number) : null;
  const footAnchor = Array.isArray(value.footAnchor) && value.footAnchor.length === 2 &&
    value.footAnchor.every(Number.isFinite)
    ? value.footAnchor.map(Number) : null;
  return {
    mode,
    fallbackModes,
    canvas: canvasSize,
    footAnchor,
    sourceFacing: value.sourceFacing === 1 || value.sourceFacing === -1
      ? value.sourceFacing : null,
    sprite: value.sprite && typeof value.sprite === "object" ? value.sprite : null,
    mesh: value.mesh && typeof value.mesh === "object" ? value.mesh : null,
  };
}

function normalizeSkeletonConfig(config) {
  if (!config || typeof config !== "object") return null;
  const rig = config.rig && typeof config.rig === "object" ? config.rig : config;
  if (!Array.isArray(rig.joints) || rig.joints.length === 0) return null;
  const handedness = rig.coordinateSystem?.handedness;
  const upAxis = axisVector(rig.coordinateSystem?.up);
  const forwardAxis = axisVector(rig.coordinateSystem?.forward);
  if ((handedness !== "right" && handedness !== "left") || !upAxis || !forwardAxis ||
      Math.abs(dotVec3(upAxis, forwardAxis)) > EPSILON) return null;

  const ids = [];
  const indexById = new Map();
  for (let i = 0; i < rig.joints.length; i++) {
    const id = typeof rig.joints[i]?.id === "string" ? rig.joints[i].id : "";
    if (!id || indexById.has(id)) return null;
    indexById.set(id, i);
    ids.push(id);
  }

  const joints = [];
  for (let i = 0; i < rig.joints.length; i++) {
    const rawJoint = rig.joints[i];
    let parentIndex = -1;
    const hasParentIndex = Object.prototype.hasOwnProperty.call(rawJoint, "parentIndex");
    const hasParentId = Object.prototype.hasOwnProperty.call(rawJoint, "parent");
    if (hasParentIndex) {
      if (!Number.isInteger(rawJoint.parentIndex)) return null;
      parentIndex = Number(rawJoint.parentIndex);
      if (hasParentId) {
        const expectedParent = parentIndex >= 0 ? ids[parentIndex] : null;
        if (rawJoint.parent !== expectedParent) return null;
      }
    } else if (hasParentId) {
      if (rawJoint.parent === null) {
        parentIndex = -1;
      } else if (typeof rawJoint.parent === "string") {
        const resolved = indexById.get(rawJoint.parent);
        if (resolved === undefined) return null;
        parentIndex = resolved;
      } else {
        return null;
      }
    } else {
      return null;
    }
    if (parentIndex < -1 || parentIndex >= rig.joints.length || parentIndex === i) return null;

    const rawRest = rawJoint.restLocal;
    if (!rawRest || typeof rawRest !== "object" ||
        !isFiniteVec3(rawRest.translation) ||
        !isUnitQuat(rawRest.rotation, REST_QUAT_NORM_TOLERANCE) ||
        !isFiniteVec3(rawRest.scale) ||
        rawRest.scale.some((component) => Math.abs(component) <= MIN_SCALE_MAGNITUDE)) return null;

    joints.push({
      id: ids[i],
      parentIndex,
      parent: parentIndex >= 0 ? ids[parentIndex] : null,
      role: typeof rawJoint.semanticRole === "string"
        ? rawJoint.semanticRole
        : typeof rawJoint.role === "string" ? rawJoint.role : "",
      sprite: typeof rawJoint.sprite === "string" ? rawJoint.sprite : null,
      restLocal: {
        translation: rawRest.translation.map(Number),
        rotation: rawRest.rotation.map(Number),
        scale: rawRest.scale.map(Number),
      },
      poseDofs: rawJoint.poseDofs && typeof rawJoint.poseDofs === "object"
        ? rawJoint.poseDofs : {},
      dofMask: rawJoint.dofMask && typeof rawJoint.dofMask === "object"
        ? rawJoint.dofMask : {},
      physics: rawJoint.physics && typeof rawJoint.physics === "object"
        ? rawJoint.physics : {},
      source: rawJoint.source ?? null,
    });
  }
  const hasJointOrder = Object.prototype.hasOwnProperty.call(rig, "jointOrder");
  if (hasJointOrder && !Array.isArray(rig.jointOrder)) return null;
  if (Array.isArray(rig.jointOrder)) {
    if (rig.jointOrder.length !== joints.length) return null;
    for (let i = 0; i < joints.length; i++) {
      if (rig.jointOrder[i] !== joints[i].id) return null;
    }
  }

  // Kahn's algorithm rejects cycles while preserving manifest order among siblings.
  const children = joints.map(() => []);
  const indegree = joints.map((joint) => joint.parentIndex >= 0 ? 1 : 0);
  for (let i = 0; i < joints.length; i++) {
    const parentIndex = joints[i].parentIndex;
    if (parentIndex >= 0) children[parentIndex].push(i);
  }
  const queue = [];
  for (let i = 0; i < indegree.length; i++) {
    if (indegree[i] === 0) queue.push(i);
  }
  const order = [];
  for (let cursor = 0; cursor < queue.length; cursor++) {
    const current = queue[cursor];
    order.push(current);
    for (const child of children[current]) {
      indegree[child] -= 1;
      if (indegree[child] === 0) queue.push(child);
    }
  }
  if (order.length !== joints.length) return null;

  if (typeof rig.motionRoot !== "string" || rig.motionRoot.length === 0) return null;
  const motionRootIndex = indexById.get(rig.motionRoot) ?? -1;
  const rootIndices = joints.flatMap((joint, index) => joint.parentIndex < 0 ? [index] : []);
  if (motionRootIndex < 0 || rootIndices.length !== 1 || rootIndices[0] !== motionRootIndex) return null;

  let drivenIndices = null;
  const hasDrivenIndices = Object.prototype.hasOwnProperty.call(rig, "drivenJointIndices");
  const hasDrivenOrder = Object.prototype.hasOwnProperty.call(rig, "drivenJointOrder");
  if (hasDrivenIndices && !Array.isArray(rig.drivenJointIndices)) return null;
  if (hasDrivenOrder && !Array.isArray(rig.drivenJointOrder)) return null;
  if (Array.isArray(rig.drivenJointIndices)) {
    drivenIndices = [];
    const seen = new Set();
    for (const value of rig.drivenJointIndices) {
      if (!Number.isInteger(value) || value < 0 || value >= joints.length || seen.has(value)) return null;
      seen.add(value);
      drivenIndices.push(Number(value));
    }
  } else if (Array.isArray(rig.drivenJointOrder)) {
    drivenIndices = [];
    const seen = new Set();
    for (const value of rig.drivenJointOrder) {
      const index = typeof value === "string"
        ? indexById.get(value)
        : Number.isInteger(value) ? value : undefined;
      if (index === undefined || index < 0 || index >= joints.length || seen.has(index)) return null;
      seen.add(index);
      drivenIndices.push(index);
    }
  }

  if (drivenIndices && Array.isArray(rig.drivenJointOrder)) {
    if (rig.drivenJointOrder.length !== drivenIndices.length) return null;
    for (let i = 0; i < drivenIndices.length; i++) {
      const expected = rig.drivenJointOrder[i];
      if (typeof expected !== "string" || joints[drivenIndices[i]].id !== expected) return null;
    }
  }

  if (!drivenIndices) {
    drivenIndices = [];
    for (let i = 0; i < joints.length; i++) {
      const joint = joints[i];
      if (i === motionRootIndex) continue;
      if (joint.physics.mode === "secondary" || joint.physics.mode === "static") continue;
      if (rotationDofEnabled(joint)) drivenIndices.push(i);
    }
  }
  if (drivenIndices.includes(motionRootIndex)) return null;

  return {
    raw: config,
    rig,
    render: normalizeRenderConfig(config.render),
    projectionBasis: projectionBasis(rig.coordinateSystem),
    joints,
    motionRoot: joints[motionRootIndex].id,
    motionRootIndex,
    drivenIndices,
    order,
  };
}

function invalidateSpriteWarp() {
  spriteWarpBindings = null;
  spriteWarpRestControls = null;
  lastWarpedPixels = null;
}

function clearSkeletalPoseState() {
  state.boneRotations = undefined;
  state.rootTranslation = undefined;
  state.rootRotation = undefined;
  state.localRotationDeltas = undefined;
}

function clearSkeleton() {
  skeleton = null;
  jointWorld = [];
  restJointWorld = [];
  modelDrivenMap = [];
  deltaIndexMap = new Map();
  fkOrder = [];
  jointReferenceAxes = [];
  clearSkeletalPoseState();
  invalidateSpriteWarp();
}

function loadSkeleton(config) {
  // A pose belongs to exactly one rig generation. Even a valid replacement
  // must wait for the first subsequent render-state message.
  clearSkeletalPoseState();
  if (config === null || config === undefined) {
    clearSkeleton();
    return true;
  }
  const normalized = normalizeSkeletonConfig(config);
  if (!normalized) {
    // A malformed replacement must not leave the previous session's rig live.
    clearSkeleton();
    return false;
  }

  skeleton = normalized;
  modelDrivenMap = [...normalized.drivenIndices];
  deltaIndexMap = new Map();
  for (let deltaIndex = 0; deltaIndex < modelDrivenMap.length; deltaIndex++) {
    deltaIndexMap.set(modelDrivenMap[deltaIndex], deltaIndex);
  }
  fkOrder = [...normalized.order];
  jointWorld = normalized.joints.map(() => ({
    position: [0, 0, 0],
    rotation: [...IDENTITY_QUAT],
    scale: [1, 1, 1],
    matrix: [...IDENTITY_MATRIX],
  }));
  if (normalized.render?.canvas?.[0] === CANVAS_WIDTH &&
      normalized.render.canvas[1] === CANVAS_HEIGHT) {
    const anchor = normalized.render.footAnchor;
    if (anchor && anchor[0] >= 0 && anchor[0] <= CANVAS_WIDTH &&
        anchor[1] >= 0 && anchor[1] <= CANVAS_HEIGHT) {
      sourceFootAnchor = [...anchor];
    }
  }
  if (normalized.render?.sourceFacing === 1 || normalized.render?.sourceFacing === -1) {
    sourceFacing = normalized.render.sourceFacing;
  }
  const identityDeltas = modelDrivenMap.map(() => [...IDENTITY_QUAT]);
  if (!computeFK([0, 0, 0], IDENTITY_QUAT, identityDeltas)) {
    clearSkeleton();
    return false;
  }
  restJointWorld = jointWorld.map((joint) => ({
    position: [...joint.position],
    rotation: [...joint.rotation],
    scale: [...joint.scale],
    matrix: [...joint.matrix],
  }));
  jointReferenceAxes = selectStableJointReferenceAxes(
    restJointWorld,
    normalized.projectionBasis,
  );
  invalidateSpriteWarp();
  return true;
}

function computeFK(rootTranslation, rootRotation, localRotationDeltas) {
  if (!skeleton || !isFiniteVec3(rootTranslation) ||
      !isUnitQuat(rootRotation, POSE_QUAT_NORM_TOLERANCE) ||
      !Array.isArray(localRotationDeltas) ||
      localRotationDeltas.length !== modelDrivenMap.length ||
      !localRotationDeltas.every((quaternion) =>
        isUnitQuat(quaternion, POSE_QUAT_NORM_TOLERANCE))) return false;

  const globalTranslation = rootTranslation.map(Number);
  const globalRotation = normalizeQuat(rootRotation);
  const deltas = localRotationDeltas;
  const rootMatrix = trsMatrix(globalTranslation, globalRotation, [1, 1, 1]);
  if (!isFiniteMatrix(rootMatrix)) return false;

  // Build a complete candidate pose before publishing it. A late overflow in a
  // deep hierarchy must not leave the first joints on a new pose and the rest
  // on the previous one.
  const nextJointWorld = skeleton.joints.map(() => null);

  for (const index of fkOrder) {
    const joint = skeleton.joints[index];
    const restTranslation = joint.restLocal.translation;
    const restRotation = normalizeQuat(joint.restLocal.rotation);
    const restScale = joint.restLocal.scale;
    const deltaIndex = deltaIndexMap.get(index);
    const deltaRotation = deltaIndex === undefined
      ? [...IDENTITY_QUAT]
      : normalizeQuat(deltas[deltaIndex]);
    // Deltas are authored in the joint's rest-local frame.
    const localRotation = normalizeQuat(quatMul(restRotation, deltaRotation));
    const localMatrix = trsMatrix(restTranslation, localRotation, restScale);
    if (!isFiniteMatrix(localMatrix)) return false;

    if (joint.parentIndex < 0) {
      const worldMatrix = multiplyMatrices(rootMatrix, localMatrix);
      const worldRotation = normalizeQuat(quatMul(globalRotation, localRotation));
      const worldScale = [...restScale];
      if (!isFiniteMatrix(worldMatrix) || !isUnitQuat(worldRotation, POSE_QUAT_NORM_TOLERANCE) ||
          !isFiniteVec3(worldScale)) return false;
      nextJointWorld[index] = {
        position: [worldMatrix[3], worldMatrix[7], worldMatrix[11]],
        rotation: worldRotation,
        scale: worldScale,
        matrix: worldMatrix,
      };
      continue;
    }

    const parent = nextJointWorld[joint.parentIndex];
    if (!parent) return false;
    const worldMatrix = multiplyMatrices(parent.matrix, localMatrix);
    const worldRotation = normalizeQuat(quatMul(parent.rotation, localRotation));
    const worldScale = [
      parent.scale[0] * restScale[0],
      parent.scale[1] * restScale[1],
      parent.scale[2] * restScale[2],
    ];
    if (!isFiniteMatrix(worldMatrix) || !isUnitQuat(worldRotation, POSE_QUAT_NORM_TOLERANCE) ||
        !isFiniteVec3(worldScale)) return false;
    nextJointWorld[index] = {
      position: [worldMatrix[3], worldMatrix[7], worldMatrix[11]],
      rotation: worldRotation,
      scale: worldScale,
      matrix: worldMatrix,
    };
  }
  if (nextJointWorld.some((joint) => joint === null)) return false;
  jointWorld = nextJointWorld;
  return true;
}

function legacyBoneRotationsToLocalDeltas(boneRotations) {
  if (!skeleton || !Array.isArray(boneRotations) ||
      boneRotations.length !== modelDrivenMap.length ||
      !boneRotations.every(Number.isFinite) ||
      restJointWorld.length !== skeleton.joints.length) return null;
  const modelDepthAxis = skeleton.projectionBasis.depth;
  return modelDrivenMap.map((jointIndex, deltaIndex) => {
    // Legacy angles describe planar rotation around the side-view camera axis.
    // Convert that model-space axis into each joint's authored rest-local frame
    // before producing the same local delta quaternion consumed by 3D FK.
    const localAxis = quatRotateVec3(
      quatConjugate(restJointWorld[jointIndex].rotation),
      modelDepthAxis,
    );
    return axisAngleQuat(localAxis, Number(boneRotations[deltaIndex]));
  });
}

function hasProceduralVisual(joint) {
  const role = joint.role || "";
  const sprite = joint.sprite || "";
  return role === "head" || sprite === "head" ||
    role === "spine" || sprite === "body" ||
    role === "body_root" || sprite === "body_root" ||
    role === "arm" || sprite.startsWith("upper_arm") ||
    role === "leg" || sprite.startsWith("upper_leg") ||
    role === "tail" || sprite === "tail_base" ||
    role === "tail_tip" || sprite === "tail_tip";
}

function project(width = CANVAS_WIDTH, height = CANVAS_HEIGHT, requestedPadding) {
  if (!skeleton || jointWorld.length === 0) return [];
  const safeWidth = Math.max(1, finiteNumber(width, CANVAS_WIDTH));
  const safeHeight = Math.max(1, finiteNumber(height, CANVAS_HEIGHT));
  const hasLargeParts = skeleton.joints.some(hasProceduralVisual);
  const defaultPadding = hasLargeParts ? 8 : 3;
  const maximumPadding = Math.max(0, Math.min(safeWidth, safeHeight) / 2 - 0.5);
  const padding = clamp(finiteNumber(requestedPadding, defaultPadding), 0, maximumPadding);
  const basis = skeleton.projectionBasis;
  const facingSign = state.facing === -1 ? -1 : 1;

  const raw = jointWorld.map((world, index) => ({
    idx: index,
    x: finiteNumber(dotVec3(world.position, basis.horizontal)) * facingSign,
    y: -finiteNumber(dotVec3(world.position, basis.vertical)),
    z: finiteNumber(dotVec3(world.position, basis.depth)),
  }));
  let minX = Number.POSITIVE_INFINITY;
  let maxX = Number.NEGATIVE_INFINITY;
  let minY = Number.POSITIVE_INFINITY;
  let maxY = Number.NEGATIVE_INFINITY;
  for (const point of raw) {
    minX = Math.min(minX, point.x);
    maxX = Math.max(maxX, point.x);
    minY = Math.min(minY, point.y);
    maxY = Math.max(maxY, point.y);
  }
  const extentX = maxX - minX;
  const extentY = maxY - minY;
  const availableWidth = Math.max(1, safeWidth - 2 * padding);
  const bottomEdge = clamp(finiteNumber(sourceFootAnchor[1], safeHeight - padding),
    padding, safeHeight - padding);
  const availableHeight = Math.max(1, bottomEdge - padding);
  const scaleX = extentX > EPSILON ? availableWidth / extentX : Number.POSITIVE_INFINITY;
  const scaleY = extentY > EPSILON ? availableHeight / extentY : Number.POSITIVE_INFINITY;
  let scale = Math.min(scaleX, scaleY);
  if (!Number.isFinite(scale) || scale <= 0) scale = 1;
  const drawnWidth = extentX * scale;
  const offsetX = (safeWidth - drawnWidth) / 2 - minX * scale;
  // Bottom-align the fitted skeleton so its visual contact point stays close
  // to the host's foot anchor even when a wide pose leaves vertical slack.
  const offsetY = bottomEdge - maxY * scale;

  const result = raw.map((point) => {
    const joint = skeleton.joints[point.idx];
    const world = jointWorld[point.idx];
    return {
      idx: point.idx,
      parentIdx: joint.parentIndex,
      sx: offsetX + point.x * scale,
      sy: offsetY + point.y * scale,
      sz: point.z,
      angle: matrixCanvasAngle(
        world.matrix,
        jointReferenceAxes[point.idx] ?? [1, 0, 0],
        basis,
        facingSign,
      ),
      rotation: [...world.rotation],
      id: joint.id,
      sprite: joint.sprite,
      role: joint.role,
    };
  });
  result.sort((a, b) => a.sz - b.sz || a.idx - b.idx);
  return result;
}

// The sprite has no authored skin weights, so the renderer derives a small,
// deterministic 2D deformation field from the character's rest skeleton. The
// same rest-space fit is reused for every pose; unlike project(), it never
// refits a moving pose and therefore does not erase root or joint motion.
function buildStableSpriteProjectionFrame(worldPose = restJointWorld) {
  if (!skeleton || !Array.isArray(worldPose) || worldPose.length !== skeleton.joints.length) {
    return null;
  }
  const basis = skeleton.projectionBasis;
  const horizontalSign = sourceFacing === -1 ? -1 : 1;
  const raw = worldPose.map((world) => ({
    x: finiteNumber(dotVec3(world.position, basis.horizontal)) * horizontalSign,
    y: -finiteNumber(dotVec3(world.position, basis.vertical)),
  }));
  if (raw.length === 0) return null;

  let minX = Number.POSITIVE_INFINITY;
  let maxX = Number.NEGATIVE_INFINITY;
  let minY = Number.POSITIVE_INFINITY;
  let maxY = Number.NEGATIVE_INFINITY;
  for (const point of raw) {
    minX = Math.min(minX, point.x);
    maxX = Math.max(maxX, point.x);
    minY = Math.min(minY, point.y);
    maxY = Math.max(maxY, point.y);
  }

  const [boundsX, boundsY, boundsWidth, boundsHeight] = sourceContentBounds;
  const targetLeft = clamp(finiteNumber(boundsX, 0), 0, CANVAS_WIDTH);
  const targetTop = clamp(finiteNumber(boundsY, 0), 0, CANVAS_HEIGHT);
  const targetRight = clamp(targetLeft + Math.max(1, finiteNumber(boundsWidth, CANVAS_WIDTH)),
    targetLeft, CANVAS_WIDTH);
  const contentBottom = clamp(targetTop + Math.max(1, finiteNumber(boundsHeight, CANVAS_HEIGHT)),
    targetTop, CANVAS_HEIGHT);
  const targetBottom = clamp(Math.min(contentBottom, finiteNumber(sourceFootAnchor[1], contentBottom)),
    targetTop, CANVAS_HEIGHT);
  const availableWidth = Math.max(1, targetRight - targetLeft);
  const availableHeight = Math.max(1, targetBottom - targetTop);
  const extentX = maxX - minX;
  const extentY = maxY - minY;
  const scaleX = extentX > EPSILON ? availableWidth / extentX : Number.POSITIVE_INFINITY;
  const scaleY = extentY > EPSILON ? availableHeight / extentY : Number.POSITIVE_INFINITY;
  let scale = Math.min(scaleX, scaleY);
  if (!Number.isFinite(scale) || scale <= EPSILON) scale = 1;
  const drawnWidth = extentX * scale;
  return {
    basis,
    horizontalSign,
    scale,
    offsetX: targetLeft + (availableWidth - drawnWidth) / 2 - minX * scale,
    offsetY: targetBottom - maxY * scale,
  };
}

function buildSpritePoseControls(worldPose, frame) {
  if (!skeleton || !frame || !Array.isArray(worldPose) ||
      worldPose.length !== skeleton.joints.length ||
      jointReferenceAxes.length !== skeleton.joints.length) return null;
  return worldPose.map((world, index) => ({
    x: frame.offsetX + dotVec3(world.position, frame.basis.horizontal) *
      frame.horizontalSign * frame.scale,
    y: frame.offsetY - dotVec3(world.position, frame.basis.vertical) * frame.scale,
    // The local reference axis is selected once from the strongest rest-pose
    // screen projection. Reusing it avoids the depth-axis singularity of a
    // hard-coded local +X axis while retaining leaf-joint twist.
    angle: matrixCanvasAngle(
      world.matrix,
      jointReferenceAxes[index],
      frame.basis,
      frame.horizontalSign,
    ),
  }));
}

function buildSpriteWarpBindings(width, height, restControls, maximumInfluences = 4) {
  if (!Number.isInteger(width) || width <= 0 || !Number.isInteger(height) || height <= 0 ||
      !Array.isArray(restControls) || restControls.length === 0 ||
      !restControls.every((control) => Number.isFinite(control?.x) &&
        Number.isFinite(control?.y) && Number.isFinite(control?.angle))) return null;
  const influenceCount = Math.max(1, Math.min(
    restControls.length,
    Number.isInteger(maximumInfluences) ? maximumInfluences : 4,
  ));
  const pixelCount = width * height;
  const indices = new Int32Array(pixelCount * influenceCount);
  indices.fill(-1);
  const weights = new Float64Array(pixelCount * influenceCount);

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const pixelIndex = y * width + x;
      const px = x + 0.5;
      const py = y + 0.5;
      const nearestIndices = [];
      const nearestDistances = [];
      for (let controlIndex = 0; controlIndex < restControls.length; controlIndex++) {
        const control = restControls[controlIndex];
        const dx = px - control.x;
        const dy = py - control.y;
        const distanceSquared = dx * dx + dy * dy;
        let insertAt = nearestDistances.length;
        while (insertAt > 0 && (distanceSquared < nearestDistances[insertAt - 1] ||
          (distanceSquared === nearestDistances[insertAt - 1] &&
            controlIndex < nearestIndices[insertAt - 1]))) {
          insertAt -= 1;
        }
        if (insertAt >= influenceCount) continue;
        nearestDistances.splice(insertAt, 0, distanceSquared);
        nearestIndices.splice(insertAt, 0, controlIndex);
        if (nearestDistances.length > influenceCount) {
          nearestDistances.pop();
          nearestIndices.pop();
        }
      }

      const base = pixelIndex * influenceCount;
      if (nearestDistances[0] <= EPSILON) {
        indices[base] = nearestIndices[0];
        weights[base] = 1;
        continue;
      }
      let weightTotal = 0;
      for (let slot = 0; slot < nearestIndices.length; slot++) {
        const weight = 1 / (0.25 + nearestDistances[slot]);
        indices[base + slot] = nearestIndices[slot];
        weights[base + slot] = weight;
        weightTotal += weight;
      }
      if (!Number.isFinite(weightTotal) || weightTotal <= EPSILON) return null;
      for (let slot = 0; slot < nearestIndices.length; slot++) {
        weights[base + slot] /= weightTotal;
      }
    }
  }
  return { width, height, influenceCount, indices, weights };
}

function shortestAngleDelta(from, to) {
  let delta = finiteNumber(to) - finiteNumber(from);
  while (delta > Math.PI) delta -= Math.PI * 2;
  while (delta < -Math.PI) delta += Math.PI * 2;
  return delta;
}

function warpSpritePixels(sourcePixels, width, height, restControls, posedControls, bindings) {
  const expectedLength = width * height * 4;
  if (!sourcePixels || sourcePixels.length !== expectedLength ||
      !Array.isArray(restControls) || !Array.isArray(posedControls) ||
      restControls.length === 0 || restControls.length !== posedControls.length ||
      !bindings || bindings.width !== width || bindings.height !== height ||
      !Number.isInteger(bindings.influenceCount) || bindings.influenceCount <= 0 ||
      bindings.indices?.length !== width * height * bindings.influenceCount ||
      bindings.weights?.length !== bindings.indices.length ||
      !posedControls.every((control) => Number.isFinite(control?.x) &&
        Number.isFinite(control?.y) && Number.isFinite(control?.angle))) return null;

  const identityPose = restControls.every((rest, index) => {
    const posed = posedControls[index];
    return Math.abs(rest.x - posed.x) <= EPSILON &&
      Math.abs(rest.y - posed.y) <= EPSILON &&
      Math.abs(shortestAngleDelta(rest.angle, posed.angle)) <= EPSILON;
  });
  if (identityPose) return new Uint8ClampedArray(sourcePixels);

  const output = new Uint8ClampedArray(expectedLength);
  const angleCos = new Float64Array(restControls.length);
  const angleSin = new Float64Array(restControls.length);
  for (let index = 0; index < restControls.length; index++) {
    const angle = shortestAngleDelta(restControls[index].angle, posedControls[index].angle);
    angleCos[index] = Math.cos(angle);
    angleSin[index] = Math.sin(angle);
  }

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const pixelIndex = y * width + x;
      const px = x + 0.5;
      const py = y + 0.5;
      let sourcePointX = 0;
      let sourcePointY = 0;
      let weightTotal = 0;
      const base = pixelIndex * bindings.influenceCount;
      for (let slot = 0; slot < bindings.influenceCount; slot++) {
        const controlIndex = bindings.indices[base + slot];
        const weight = bindings.weights[base + slot];
        if (controlIndex < 0 || weight <= 0) continue;
        const rest = restControls[controlIndex];
        const posed = posedControls[controlIndex];
        const dx = px - posed.x;
        const dy = py - posed.y;
        // Invert each joint's 2D rigid transform, then blend the bound source
        // samples. This is exact for one control and for global rigid motion.
        sourcePointX += weight * (rest.x + angleCos[controlIndex] * dx +
          angleSin[controlIndex] * dy);
        sourcePointY += weight * (rest.y - angleSin[controlIndex] * dx +
          angleCos[controlIndex] * dy);
        weightTotal += weight;
      }
      if (!Number.isFinite(sourcePointX) || !Number.isFinite(sourcePointY) ||
          weightTotal <= EPSILON) {
        return null;
      }
      const sourceX = Math.floor(sourcePointX / weightTotal);
      const sourceY = Math.floor(sourcePointY / weightTotal);
      if (sourceX < 0 || sourceX >= width || sourceY < 0 || sourceY >= height) continue;
      const sourceOffset = (sourceY * width + sourceX) * 4;
      const outputOffset = pixelIndex * 4;
      output[outputOffset] = sourcePixels[sourceOffset];
      output[outputOffset + 1] = sourcePixels[sourceOffset + 1];
      output[outputOffset + 2] = sourcePixels[sourceOffset + 2];
      output[outputOffset + 3] = sourcePixels[sourceOffset + 3];
    }
  }
  return output;
}

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

  ctx.save();
  ctx.translate(pos.sx, pos.sy);

  if (role === "head" || sprite === "head") {
    ctx.rotate(angle);
    ctx.fillStyle = PART_COLORS.head.fill;
    ctx.strokeStyle = PART_COLORS.head.stroke;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.ellipse(0, 0, 7, 5.5, 0, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(-5, -4); ctx.lineTo(-7, -11); ctx.lineTo(-1, -5); ctx.fill();
    ctx.moveTo(5, -4); ctx.lineTo(7, -11); ctx.lineTo(1, -5); ctx.fill();
    ctx.stroke();
    ctx.fillStyle = "#2d2d2d";
    ctx.beginPath();
    ctx.arc(-3, -1, 1.2, 0, Math.PI * 2); ctx.fill();
    ctx.arc(3, -1, 1.2, 0, Math.PI * 2); ctx.fill();
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
    ctx.fillStyle = PART_COLORS.body.fill;
    ctx.strokeStyle = PART_COLORS.body.stroke;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.arc(0, 0, 3, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
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
    if (!nextPos) {
      ctx.restore();
      return false;
    }
    ctx.strokeStyle = PART_COLORS.tail.stroke;
    ctx.lineWidth = 2.5;
    ctx.lineCap = "round";
    ctx.beginPath();
    const dx = nextPos.sx - pos.sx;
    const dy = nextPos.sy - pos.sy;
    ctx.moveTo(0, 0);
    ctx.quadraticCurveTo(dx * 0.5, dy * 0.5 - 4, dx, dy);
    ctx.stroke();
  } else if (role === "tail_tip" || sprite === "tail_tip") {
    ctx.fillStyle = PART_COLORS.tail.fill;
    ctx.beginPath();
    ctx.arc(0, 0, 2, 0, Math.PI * 2);
    ctx.fill();
  } else {
    ctx.restore();
    return false;
  }

  ctx.restore();
  return true;
}

function drawGenericBone(ctx, pos, parentPos) {
  ctx.save();
  ctx.lineCap = "round";
  if (parentPos) {
    ctx.strokeStyle = "#242424";
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.moveTo(parentPos.sx, parentPos.sy);
    ctx.lineTo(pos.sx, pos.sy);
    ctx.stroke();
    ctx.strokeStyle = "#f5f0e8";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(parentPos.sx, parentPos.sy);
    ctx.lineTo(pos.sx, pos.sy);
    ctx.stroke();
  }
  ctx.fillStyle = "#242424";
  ctx.fillRect(Math.round(pos.sx) - 1.5, Math.round(pos.sy) - 1.5, 3, 3);
  ctx.fillStyle = "#f5f0e8";
  ctx.fillRect(Math.round(pos.sx) - 0.5, Math.round(pos.sy) - 0.5, 1, 1);
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

function buildDrawCommands(projected) {
  const byIndex = new Map(projected.map((point) => [point.idx, point]));
  const firstChild = new Map();
  for (let index = 0; index < skeleton.joints.length; index++) {
    const point = byIndex.get(index);
    if (!point) continue;
    if (point.parentIdx >= 0 && !firstChild.has(point.parentIdx)) {
      firstChild.set(point.parentIdx, point);
    }
  }

  const commands = [];
  for (const point of projected) {
    if (point.idx === skeleton.motionRootIndex) continue;
    const joint = skeleton.joints[point.idx];
    const next = firstChild.get(point.idx);
    if (hasProceduralVisual(joint) && !(joint.role === "tail" && !next)) {
      commands.push({ type: "part", depth: point.sz, index: point.idx, point, joint, next });
      if (state.debug === true) {
        const parent = point.parentIdx >= 0 ? byIndex.get(point.parentIdx) : undefined;
        commands.push({ type: "bone", depth: point.sz - EPSILON, index: point.idx, point, parent });
      }
    } else {
      const parent = point.parentIdx >= 0 ? byIndex.get(point.parentIdx) : undefined;
      const depth = parent ? (point.sz + parent.sz) / 2 : point.sz;
      commands.push({ type: "bone", depth, index: point.idx, point, parent });
    }
  }

  // A degenerate one-joint rig must still produce an opaque hit target.
  if (commands.length === 0 && projected.length > 0) {
    const point = projected[0];
    commands.push({ type: "bone", depth: point.sz, index: point.idx, point, parent: undefined });
  }
  commands.sort((a, b) => a.depth - b.depth || a.index - b.index ||
    (a.type === "bone" ? 0 : 1) - (b.type === "bone" ? 0 : 1));
  return commands;
}

function resolvedSkeletalRenderMode() {
  if (!skeleton?.render) return "debug_skeleton";
  const candidates = [skeleton.render.mode, ...skeleton.render.fallbackModes];
  for (const candidate of candidates) {
    if (candidate === "sprite" && sourceImage && spriteLoadState === "ready" &&
        spriteSourcePixels && spriteWarpContext) return candidate;
    if (candidate === "debug_skeleton") return candidate;
    // skinned_mesh is intentionally skipped until vertex/index/weight buffers
    // are available to the sandboxed renderer in a clean checkout.
  }
  // Keep an opaque, pose-aware result even if a manifest omitted an executable
  // fallback or a sprite failed to load.
  return "debug_skeleton";
}

function createRasterSurface() {
  if (typeof document.createElement !== "function") return null;
  const surface = document.createElement("canvas");
  if (!surface || typeof surface.getContext !== "function") return null;
  surface.width = CANVAS_WIDTH;
  surface.height = CANVAS_HEIGHT;
  const surfaceContext = surface.getContext("2d", { alpha: true, willReadFrequently: true });
  if (!surfaceContext || typeof surfaceContext.drawImage !== "function" ||
      typeof surfaceContext.getImageData !== "function" ||
      typeof surfaceContext.putImageData !== "function" ||
      typeof surfaceContext.createImageData !== "function") return null;
  surfaceContext.imageSmoothingEnabled = false;
  return { surface, context: surfaceContext };
}

function captureSpritePixels(image) {
  const capture = createRasterSurface();
  if (!capture) return false;
  try {
    capture.context.clearRect(0, 0, CANVAS_WIDTH, CANVAS_HEIGHT);
    capture.context.drawImage(image, 0, 0, CANVAS_WIDTH, CANVAS_HEIGHT);
    const pixels = capture.context.getImageData(0, 0, CANVAS_WIDTH, CANVAS_HEIGHT)?.data;
    if (!pixels || pixels.length !== CANVAS_WIDTH * CANVAS_HEIGHT * 4) return false;
    const warp = createRasterSurface();
    if (!warp) return false;
    spriteSourcePixels = new Uint8ClampedArray(pixels);
    spriteWarpCanvas = warp.surface;
    spriteWarpContext = warp.context;
    invalidateSpriteWarp();
    return true;
  } catch {
    spriteSourcePixels = null;
    spriteWarpCanvas = null;
    spriteWarpContext = null;
    invalidateSpriteWarp();
    return false;
  }
}

function drawSkeleton() {
  const projected = project(CANVAS_WIDTH, CANVAS_HEIGHT);
  for (const command of buildDrawCommands(projected)) {
    if (command.type === "part") {
      drawProceduralPart(context, command.point, command.joint, command.next);
    } else {
      drawGenericBone(context, command.point, command.parent);
    }
  }
}

function drawWholeSprite() {
  drawTransformedSprite(sourceImage);
}

function drawTransformedSprite(image) {
  context.save();
  const [footX, footY] = sourceFootAnchor;
  context.translate(footX, footY);
  const squash = clamp(Number(state.squash) || 1, 0.5, 1.5);
  const facing = state.facing === -1 ? -1 : 1;
  context.rotate(clamp(Number(state.lean) || 0, -1, 1) * 0.11);
  context.scale(sourceFacing * facing / Math.sqrt(squash), squash);
  context.translate(-footX, -footY);
  if (image) context.drawImage(image, 0, 0, CANVAS_WIDTH, CANVAS_HEIGHT);
  else drawFallbackCat(context);
  context.restore();
}

function drawPoseAwareSprite() {
  if (!skeleton || !spriteSourcePixels || !spriteWarpCanvas || !spriteWarpContext ||
      restJointWorld.length !== skeleton.joints.length ||
      jointWorld.length !== skeleton.joints.length) return false;
  const frame = buildStableSpriteProjectionFrame(restJointWorld);
  if (!frame) return false;
  const restControls = buildSpritePoseControls(restJointWorld, frame);
  const posedControls = buildSpritePoseControls(jointWorld, frame);
  if (!restControls || !posedControls) return false;
  if (!spriteWarpBindings || !spriteWarpRestControls) {
    spriteWarpBindings = buildSpriteWarpBindings(CANVAS_WIDTH, CANVAS_HEIGHT, restControls);
    spriteWarpRestControls = restControls;
  }
  if (!spriteWarpBindings) return false;
  const warped = warpSpritePixels(
    spriteSourcePixels,
    CANVAS_WIDTH,
    CANVAS_HEIGHT,
    spriteWarpRestControls,
    posedControls,
    spriteWarpBindings,
  );
  if (!warped) return false;
  try {
    const imageData = spriteWarpContext.createImageData(CANVAS_WIDTH, CANVAS_HEIGHT);
    imageData.data.set(warped);
    spriteWarpContext.putImageData(imageData, 0, 0);
  } catch {
    return false;
  }
  lastWarpedPixels = new Uint8ClampedArray(warped);
  drawTransformedSprite(spriteWarpCanvas);
  return true;
}

function drawScene() {
  const has3D = skeleton && Array.isArray(state.rootTranslation) &&
    Array.isArray(state.localRotationDeltas);
  const legacyDeltas = !has3D && skeleton
    ? legacyBoneRotationsToLocalDeltas(state.boneRotations)
    : null;
  const poseReady = has3D
    ? computeFK(state.rootTranslation, state.rootRotation, state.localRotationDeltas)
    : legacyDeltas
      ? computeFK([0, 0, 0], IDENTITY_QUAT, legacyDeltas)
      : false;

  if (poseReady) {
    if (resolvedSkeletalRenderMode() === "sprite") {
      if (drawPoseAwareSprite()) {
        if (state.debug === true) drawSkeleton();
      } else {
        // A skeletal sprite may never silently fall back to a static image:
        // the explicit debug skeleton proves that the pose is still consumed.
        drawSkeleton();
      }
    } else {
      drawSkeleton();
    }
    return;
  }

  // Legacy whole-sprite path remains the fallback whenever no complete 3D state exists.
  drawWholeSprite();
}

function draw() {
  context.clearRect(0, 0, CANVAS_WIDTH, CANVAS_HEIGHT);
  drawWithGenericFacialTransform(drawScene);
}

window.petHost.onState((next) => {
  if (!next || typeof next !== "object") return;
  state = { ...state, ...next };
  if (!("boneRotations" in next)) state.boneRotations = undefined;
  if (!("rootTranslation" in next)) state.rootTranslation = undefined;
  if (!("rootRotation" in next)) state.rootRotation = undefined;
  if (!("localRotationDeltas" in next)) state.localRotationDeltas = undefined;
  if (!("facialParams" in next)) state.facialParams = undefined;
  document.documentElement.dataset.generator = String(state.generatorStatus || "stopped");
  document.documentElement.dataset.debug = state.debug === true ? "true" : "false";
  const spriteMetadata = readSpriteMetadata(state.assetParts);
  if (spriteMetadata) {
    const metadataChanged = sourceFootAnchor[0] !== spriteMetadata.footAnchor[0] ||
      sourceFootAnchor[1] !== spriteMetadata.footAnchor[1] ||
      sourceFacing !== spriteMetadata.sourceFacing ||
      sourceContentBounds.some((value, index) => value !== spriteMetadata.contentBounds[index]);
    sourceFootAnchor = spriteMetadata.footAnchor;
    sourceFacing = spriteMetadata.sourceFacing;
    sourceContentBounds = spriteMetadata.contentBounds;
    if (metadataChanged) invalidateSpriteWarp();
  }
  if (typeof state.assetDataUrl === "string" && state.assetDataUrl !== lastAssetDataUrl) {
    const requestedDataUrl = state.assetDataUrl;
    lastAssetDataUrl = requestedDataUrl;
    sourceImage = null;
    spriteSourcePixels = null;
    spriteWarpCanvas = null;
    spriteWarpContext = null;
    invalidateSpriteWarp();
    if (!SAFE_RASTER_DATA_URL.test(requestedDataUrl)) {
      spriteLoadState = "failed";
    } else {
      spriteLoadState = "loading";
      const image = new Image();
      image.onload = () => {
        if (lastAssetDataUrl !== requestedDataUrl) return;
        sourceImage = image;
        spriteLoadState = captureSpritePixels(image) ? "ready" : "failed";
        draw();
      };
      image.onerror = () => {
        if (lastAssetDataUrl !== requestedDataUrl) return;
        sourceImage = null;
        spriteLoadState = "failed";
        draw();
      };
      image.src = requestedDataUrl;
    }
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

if (typeof window.petHost.getSkeletalConfig === "function") {
  const config = window.petHost.getSkeletalConfig();
  if (config) loadSkeleton(config);
}

if (typeof window.petHost.onSkeletalConfig === "function") {
  window.petHost.onSkeletalConfig((config) => {
    loadSkeleton(config);
    draw();
  });
}

function reportHit(opaque, force = false) {
  if (!force && lastReportedHit === opaque) return;
  lastReportedHit = opaque;
  window.petHost.reportHit(opaque);
}

function isOpaqueAt(clientX, clientY) {
  const rect = canvas.getBoundingClientRect();
  if (clientX < rect.left || clientX >= rect.right ||
      clientY < rect.top || clientY >= rect.bottom) return false;
  const x = Math.max(0, Math.min(CANVAS_WIDTH - 1,
    Math.floor((clientX - rect.left) * CANVAS_WIDTH / rect.width)));
  const y = Math.max(0, Math.min(CANVAS_HEIGHT - 1,
    Math.floor((clientY - rect.top) * CANVAS_HEIGHT / rect.height)));
  return context.getImageData(x, y, 1, 1).data[3] >= 48;
}

function readSpriteMetadata(value) {
  if (!value || typeof value !== "object" ||
      value.canvas?.[0] !== CANVAS_WIDTH || value.canvas?.[1] !== CANVAS_HEIGHT ||
      !Array.isArray(value.footAnchor) || value.footAnchor.length !== 2 ||
      !value.footAnchor.every(Number.isFinite)) return null;
  const [x, y] = value.footAnchor.map(Number);
  if (x < 0 || x > CANVAS_WIDTH || y < 0 || y > CANVAS_HEIGHT) return null;
  const metadataFacing = value.sourceFacing === 1 || value.sourceFacing === -1
    ? value.sourceFacing : -1;
  let contentBounds = [0, 0, CANVAS_WIDTH, CANVAS_HEIGHT];
  if (Array.isArray(value.contentBounds) && value.contentBounds.length === 4 &&
      value.contentBounds.every(Number.isFinite)) {
    const [boundsX, boundsY, boundsWidth, boundsHeight] = value.contentBounds.map(Number);
    if (boundsX >= 0 && boundsY >= 0 && boundsWidth > 0 && boundsHeight > 0 &&
        boundsX + boundsWidth <= CANVAS_WIDTH && boundsY + boundsHeight <= CANVAS_HEIGHT) {
      contentBounds = [boundsX, boundsY, boundsWidth, boundsHeight];
    }
  }
  return { footAnchor: [x, y], sourceFacing: metadataFacing, contentBounds };
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

if (window.__PET_RENDERER_TEST__ === true) {
  window.__petRendererTest = Object.freeze({
    loadSkeleton,
    computeFK,
    project,
    draw,
    normalizeQuat,
    quatMul,
    quatRotateVec3,
    buildSpriteWarpBindings,
    warpSpritePixels,
    getWarpedPixels: () => lastWarpedPixels
      ? new Uint8ClampedArray(lastWarpedPixels) : null,
    getJointWorld: () => jointWorld.map((joint) => ({
      position: [...joint.position],
      rotation: [...joint.rotation],
      scale: [...joint.scale],
    })),
    getRig: () => skeleton ? {
      motionRoot: skeleton.motionRoot,
      jointIds: skeleton.joints.map((joint) => joint.id),
      parentIndices: skeleton.joints.map((joint) => joint.parentIndex),
      drivenJointIds: modelDrivenMap.map((index) => skeleton.joints[index].id),
      projectionBasis: {
        horizontal: [...skeleton.projectionBasis.horizontal],
        vertical: [...skeleton.projectionBasis.vertical],
        depth: [...skeleton.projectionBasis.depth],
      },
      renderMode: resolvedSkeletalRenderMode(),
    } : null,
    buildDrawCommands,
  });
}

draw();
window.petHost.ready();
