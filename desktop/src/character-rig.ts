import { createHash } from "node:crypto";
import { existsSync } from "node:fs";
import { readFile } from "node:fs/promises";
import { basename, dirname, isAbsolute, join, resolve } from "node:path";

export const CHARACTER_RIG_SCHEMA = "pet-character-rig-manifest-v1" as const;
export const RIG_FINGERPRINT_ALGORITHM = "pet-canonical-json-f64-v1+sha256" as const;
export const RIG_FINGERPRINT_CANONICALIZATION = "pet-canonical-json-f64-v1" as const;
export const CHECKPOINT_FORMAT = "pet-character-motion-checkpoint-v1" as const;
export const CHECKPOINT_METADATA_SCHEMA = "pet-character-motion-checkpoint-metadata-v1" as const;
export const CHECKPOINT_ROOT = "checkpoints/characters" as const;
export const CHECKPOINT_FILENAME = "motion.pt" as const;
export const CHECKPOINT_DATASET_SCHEMA = "pet-training-dataset-v1" as const;
export const CHECKPOINT_DATASET_FORMAT = "episode-ndjson-v1" as const;
export const CHECKPOINT_NORMALIZATION_SCHEMA = "pet-feature-normalization-v1" as const;
export const CHECKPOINT_NORMALIZATION_TRANSFORM = "(value-offset)/scale" as const;
export const CHECKPOINT_MODEL_CONFIG_SCHEMA = "pet-motion-model-config-v1" as const;
export const F64_ACCUMULATION = "pet-left-to-right-f64-sum-v1" as const;
export const MAX_DRIVEN_JOINTS = 128;

export interface CharacterCheckpointContract {
  readonly format: typeof CHECKPOINT_FORMAT;
  readonly metadataSchema: typeof CHECKPOINT_METADATA_SCHEMA;
  /** Portable workspace-relative target; this does not assert that a file exists. */
  readonly path: string;
  readonly characterId: string;
  readonly rigFingerprint: string;
  readonly drivenJointOrder: readonly string[];
  readonly manifestDeclared: boolean;
}

export interface CharacterCheckpointMetadata {
  readonly schema: typeof CHECKPOINT_METADATA_SCHEMA;
  readonly characterId: string;
  readonly rigFingerprint: string;
  readonly drivenJointOrder: readonly string[];
  readonly dataset: {
    readonly schema: typeof CHECKPOINT_DATASET_SCHEMA;
    readonly format: typeof CHECKPOINT_DATASET_FORMAT;
    readonly manifestSha256: string;
  };
  readonly normalization: {
    readonly schema: typeof CHECKPOINT_NORMALIZATION_SCHEMA;
    readonly transform: typeof CHECKPOINT_NORMALIZATION_TRANSFORM;
    readonly condition: CharacterNormalizationGroup;
    readonly target: CharacterNormalizationGroup;
  };
  readonly model: {
    readonly schema: typeof CHECKPOINT_MODEL_CONFIG_SCHEMA;
    readonly architecture: string;
    readonly implementationVersion: string;
    readonly config: Readonly<Record<string, unknown>>;
  };
}

export interface CharacterNormalizationGroup {
  readonly featureOrder: readonly string[];
  readonly offset: readonly number[];
  readonly scale: readonly number[];
}

export interface CharacterRigConfig {
  /** Original manifest or legacy rig payload sent unchanged to the renderer. */
  readonly raw: Record<string, unknown>;
  /** Exact identity negotiated with the generator and bound to the checkpoint. */
  readonly sha256: string;
  readonly characterId: string;
  readonly rigId: string;
  readonly modelDrivenBoneIds: readonly string[];
  readonly modelDrivenCount: number;
  readonly checkpoint: CharacterCheckpointContract;
  readonly source: "character_manifest" | "legacy_rig";
  readonly path: string;
  readonly render: CharacterRenderConfig;
}

export interface CharacterRenderConfig {
  readonly canvas: readonly [number, number];
  readonly displayScale: number;
  readonly footAnchor: readonly [number, number];
  readonly sourceFacing: -1 | 1;
  readonly mode: "skinned_mesh" | "sprite" | "debug_skeleton";
  readonly fallbackModes: readonly ("skinned_mesh" | "sprite" | "debug_skeleton")[];
  readonly spriteImagePath: string | null;
  readonly spriteMetadataPath: string | null;
}

type Environment = Readonly<Record<string, string | undefined>>;

/** Load the selected character. A character manifest is preferred; legacy rigs remain a fallback. */
export async function loadCharacterRigConfig(
  projectRoot: string,
  environment: Environment = process.env,
): Promise<CharacterRigConfig> {
  const manifestPath = resolveConfiguredPath(
    projectRoot,
    environment.PET_CHARACTER_MANIFEST,
    join("assets", "pet", "runtime", "cat-character-rig.manifest.json"),
  );
  if (existsSync(manifestPath)) return loadCharacterManifest(manifestPath);
  if (environment.PET_CHARACTER_MANIFEST) {
    throw new Error(`Configured character manifest does not exist: ${basename(manifestPath)}`);
  }

  const legacyPath = resolveConfiguredPath(
    projectRoot,
    environment.PET_CHARACTER_RIG ?? environment.PET_SKELETON_3D,
    join("assets", "pet", "runtime", "cat-skeleton-3d.json"),
  );
  return loadLegacyRig(legacyPath, environment, projectRoot);
}

export function rigFingerprint(rig: unknown): string {
  return createHash("sha256").update(stableCanonicalJson(rig), "utf8").digest("hex");
}

function resolveConfiguredPath(projectRoot: string, configured: string | undefined, fallback: string): string {
  const selected = configured && configured.trim().length > 0 ? configured : fallback;
  return isAbsolute(selected) ? resolve(selected) : resolve(projectRoot, selected);
}

async function loadCharacterManifest(path: string): Promise<CharacterRigConfig> {
  const parsed = await readJsonObject(path);
  if (parsed.schema !== CHARACTER_RIG_SCHEMA) {
    throw new Error(`Unsupported character manifest schema: ${String(parsed.schema)}`);
  }
  const characterId = requireCharacterIdentifier(parsed.characterId, "characterId");
  const rigId = requireManifestIdentifier(parsed.rigId, "rigId");
  const rig = requireRecord(parsed.rig, "rig");
  const fingerprint = requireRecord(parsed.rigFingerprint, "rigFingerprint");
  if (fingerprint.algorithm !== RIG_FINGERPRINT_ALGORITHM ||
      fingerprint.canonicalization !== RIG_FINGERPRINT_CANONICALIZATION) {
    throw new Error("Unsupported rig fingerprint algorithm or canonicalization");
  }
  const declared = requireSha256(fingerprint.value, "rigFingerprint.value");
  const actual = rigFingerprint(rig);
  if (declared !== actual) throw new Error("rigFingerprint does not match manifest.rig");

  const driven = validateManifestRig(rig);
  const source = requireRecord(parsed.source, "source");
  const selectedSkin = source.selectedSkinIndex;
  if (selectedSkin !== undefined && (!Number.isInteger(selectedSkin) || (selectedSkin as number) < 0)) {
    throw new Error("source.selectedSkinIndex must be a non-negative integer");
  }
  const render = validateRenderConfig(parsed.render, dirname(path), source);
  validateManifestProvenance(parsed.provenance);
  const checkpoint = validateCheckpointContract(parsed.checkpoint, characterId, declared, driven, true);
  return {
    raw: parsed,
    sha256: declared,
    characterId,
    rigId,
    modelDrivenBoneIds: driven,
    modelDrivenCount: driven.length,
    checkpoint,
    source: "character_manifest",
    path,
    render,
  };
}

async function loadLegacyRig(path: string, environment: Environment, projectRoot: string): Promise<CharacterRigConfig> {
  const raw = await readFile(path);
  const parsed = JSON.parse(raw.toString("utf8")) as unknown;
  const rig = requireRecord(parsed, "legacy rig");
  if (rig.schema !== "pet-rig-v2") throw new Error(`Unsupported legacy rig schema: ${String(rig.schema)}`);
  const motionRoot = requireJointIdentifier(rig.motionRoot, "motionRoot");
  const joints = requireRecordArray(rig.joints, "joints");
  if (joints.length < 2) throw new Error("Legacy rig must contain at least two joints");

  const ids = validateLegacyHierarchy(joints, motionRoot);
  const driven = joints.flatMap((joint) => {
    const id = String(joint.id);
    if (id === motionRoot) return [];
    const physics = isRecord(joint.physics) ? joint.physics : {};
    const dofs = isRecord(joint.poseDofs) ? joint.poseDofs : {};
    return physics.mode !== "secondary" && physics.mode !== "static" && dofs.rotation === true ? [id] : [];
  });
  validateDrivenCount(driven.length);
  const drawOrder = Array.isArray(rig.drawOrder) ? rig.drawOrder : [];
  for (const entry of drawOrder) {
    if (typeof entry !== "string" || !ids.has(entry)) throw new Error("Legacy drawOrder references an unknown joint");
  }
  const characterId = requireCharacterIdentifier(
    environment.PET_CHARACTER_ID?.trim() || "legacy-default",
    "PET_CHARACTER_ID",
  );
  const fingerprint = createHash("sha256").update(raw).digest("hex");
  const checkpoint = derivedCheckpointContract(characterId, fingerprint, driven);
  const runtimeDirectory = dirname(path);
  const canvas = numericPair(rig.canvas, [48, 48], "legacy canvas");
  const sourceFacing = rig.sourceFacing === 1 ? 1 : -1;
  const configuredSprite = environment.PET_CHARACTER_ASSET ?? environment.PET_CAT_ASSET;
  const configuredMetadata = environment.PET_CHARACTER_PARTS ?? environment.PET_CAT_PARTS;
  return {
    raw: rig,
    sha256: fingerprint,
    characterId,
    rigId: String(rig.schema),
    modelDrivenBoneIds: driven,
    modelDrivenCount: driven.length,
    checkpoint,
    source: "legacy_rig",
    path,
    render: {
      canvas,
      displayScale: typeof rig.displayScale === "number" && rig.displayScale > 0 ? rig.displayScale : 2,
      footAnchor: [canvas[0] / 2, Math.max(0, canvas[1] - 2)],
      sourceFacing,
      mode: (environment.PET_RENDER_MODE as "skinned_mesh" | "sprite" | "debug_skeleton") ?? "sprite",
      fallbackModes: ["debug_skeleton"],
      spriteImagePath: configuredSprite
        ? (isAbsolute(configuredSprite) ? resolve(configuredSprite) : resolve(projectRoot, configuredSprite))
        : join(runtimeDirectory, "cat-48.png"),
      spriteMetadataPath: configuredMetadata
        ? (isAbsolute(configuredMetadata) ? resolve(configuredMetadata) : resolve(projectRoot, configuredMetadata))
        : join(runtimeDirectory, "cat-parts.json"),
    },
  };
}

function validateRenderConfig(
  value: unknown,
  manifestDirectory: string,
  source: Record<string, unknown>,
): CharacterRenderConfig {
  const render = requireRecord(value, "render");
  const canvas = numericPair(render.canvas, undefined, "render.canvas");
  if (canvas[0] !== 48 || canvas[1] !== 48) {
    throw new Error("render.canvas must be the normalized 48x48 runtime canvas");
  }
  const footAnchor = numericPair(render.footAnchor, undefined, "render.footAnchor");
  if (footAnchor[0] < 0 || footAnchor[0] > canvas[0] || footAnchor[1] < 0 || footAnchor[1] > canvas[1]) {
    throw new Error("render.footAnchor must lie inside render.canvas");
  }
  const displayScale = render.displayScale;
  if (displayScale !== 2) {
    throw new Error("render.displayScale must be 2 for the normalized 96 DIP pet window");
  }
  const sourceFacing = render.sourceFacing;
  if (sourceFacing !== -1 && sourceFacing !== 1) throw new Error("render.sourceFacing must be -1 or 1");
  const modes = new Set(["skinned_mesh", "sprite", "debug_skeleton"] as const);
  const mode = render.mode;
  if (typeof mode !== "string" || !modes.has(mode as "skinned_mesh" | "sprite" | "debug_skeleton")) {
    throw new Error("render.mode is unsupported");
  }
  if (!Array.isArray(render.fallbackModes)) {
    throw new Error("render.fallbackModes must be an array");
  }
  const fallbackModes = render.fallbackModes;
  if (fallbackModes.some((entry) => typeof entry !== "string" || !modes.has(entry as "skinned_mesh" | "sprite" | "debug_skeleton"))) {
    throw new Error("render.fallbackModes contains an unsupported mode");
  }
  if (new Set(fallbackModes).size !== fallbackModes.length || fallbackModes.includes(mode)) {
    throw new Error("render modes must be unique and omit the preferred mode from fallbacks");
  }
  const sprite = render.sprite === null || render.sprite === undefined
    ? null
    : requireRecord(render.sprite, "render.sprite");
  if ((mode === "sprite" || fallbackModes.includes("sprite")) && sprite === null) {
    throw new Error("sprite rendering requires render.sprite");
  }
  const image = sprite ? requireAssetReference(sprite.image, "render.sprite.image") : null;
  const metadata = sprite ? requireAssetReference(sprite.metadata, "render.sprite.metadata") : null;
  const mesh = render.mesh === null || render.mesh === undefined
    ? null
    : requireRecord(render.mesh, "render.mesh");
  if ((mode === "skinned_mesh" || fallbackModes.includes("skinned_mesh")) && mesh === null) {
    throw new Error("skinned mesh rendering requires render.mesh");
  }
  if (mesh !== null) {
    if (!Number.isInteger(mesh.skinIndex) || (mesh.skinIndex as number) < 0) {
      throw new Error("render.mesh.skinIndex must be a non-negative integer");
    }
    if (mesh.skinIndex !== source.selectedSkinIndex) {
      throw new Error("render.mesh.skinIndex must match source.selectedSkinIndex");
    }
  }
  return {
    canvas,
    displayScale,
    footAnchor,
    sourceFacing,
    mode: mode as CharacterRenderConfig["mode"],
    fallbackModes: [...fallbackModes] as CharacterRenderConfig["fallbackModes"],
    spriteImagePath: image ? resolveAssetPath(manifestDirectory, image) : null,
    spriteMetadataPath: metadata ? resolveAssetPath(manifestDirectory, metadata) : null,
  };
}

function validateManifestProvenance(value: unknown): void {
  const provenance = requireRecord(value, "provenance");
  requireExactFields(
    provenance,
    ["f64Accumulation", "generator", "generatorVersion", "profile"],
    "provenance",
  );
  if (typeof provenance.generator !== "string" ||
      provenance.generator.length < 1 || provenance.generator.length > 512) {
    throw new Error("provenance.generator must be a non-empty string");
  }
  if (provenance.generatorVersion !== 2) {
    throw new Error("provenance.generatorVersion must be 2");
  }
  if (provenance.profile !== null &&
      (typeof provenance.profile !== "string" || provenance.profile.length > 512)) {
    throw new Error("provenance.profile must be null or a string");
  }
  if (provenance.f64Accumulation !== F64_ACCUMULATION) {
    throw new Error("provenance.f64Accumulation is unsupported");
  }
}

function resolveAssetPath(baseDirectory: string, reference: string): string {
  if (isAbsolute(reference) || /^[\\/]/.test(reference) || /^[A-Za-z][A-Za-z0-9+.-]*:/.test(reference)) {
    throw new Error("Character manifest asset paths must be relative");
  }
  if (reference.split(/[\\/]+/).includes("..")) {
    throw new Error("Character manifest asset paths cannot traverse parent directories");
  }
  return resolve(baseDirectory, reference);
}

function requireAssetReference(value: unknown, path: string): string {
  if (typeof value !== "string" || value.length < 1 || value.length > 512 || value.includes("\0")) {
    throw new Error(`${path} must be a non-empty path`);
  }
  return value;
}

function numericPair(value: unknown, fallback: readonly [number, number] | undefined, path: string): [number, number] {
  if (value === undefined && fallback) return [...fallback];
  if (!Array.isArray(value) || value.length !== 2 || value.some((entry) => typeof entry !== "number" || !Number.isFinite(entry))) {
    throw new Error(`${path} must be a finite numeric pair`);
  }
  return [value[0] as number, value[1] as number];
}

function validateManifestRig(rig: Record<string, unknown>): readonly string[] {
  const coordinateSystem = requireRecord(rig.coordinateSystem, "rig.coordinateSystem");
  if (coordinateSystem.handedness !== "right" && coordinateSystem.handedness !== "left") {
    throw new Error("rig.coordinateSystem.handedness is unsupported");
  }
  const axes = new Set(["+X", "-X", "+Y", "-Y", "+Z", "-Z"]);
  const up = coordinateSystem.up;
  const forward = coordinateSystem.forward;
  if (typeof up !== "string" || typeof forward !== "string" || !axes.has(up) || !axes.has(forward)) {
    throw new Error("rig coordinate axes are unsupported");
  }
  if (up.slice(-1) === forward.slice(-1)) {
    throw new Error("rig coordinate up and forward axes must be orthogonal");
  }
  if (typeof coordinateSystem.units !== "string" || coordinateSystem.units.length === 0) {
    throw new Error("rig.coordinateSystem.units must be a non-empty string");
  }

  const motionRoot = requireJointIdentifier(rig.motionRoot, "rig.motionRoot");
  const jointOrder = requireStringArray(rig.jointOrder, "rig.jointOrder");
  jointOrder.forEach((jointId, index) => requireJointIdentifier(jointId, `rig.jointOrder[${index}]`));
  const joints = requireRecordArray(rig.joints, "rig.joints");
  if (jointOrder.length !== joints.length || jointOrder.length < 2) {
    throw new Error("rig.jointOrder and rig.joints must align one-to-one");
  }
  if (jointOrder.length > 512) throw new Error("rig.jointOrder exceeds the manifest limit of 512 joints");
  if (new Set(jointOrder).size !== jointOrder.length) throw new Error("rig.jointOrder contains duplicate ids");
  if (jointOrder[0] !== motionRoot) throw new Error("rig.motionRoot must be joint index 0");
  let rootCount = 0;
  for (let index = 0; index < joints.length; index += 1) {
    const joint = joints[index]!;
    requireJointIdentifier(joint.id, `rig.joints[${index}].id`);
    if (joint.id !== jointOrder[index]) throw new Error(`rig.joints[${index}] does not match jointOrder`);
    const parent = joint.parentIndex;
    if (!Number.isInteger(parent) || (parent as number) < -1 || (parent !== -1 && (parent as number) >= index)) {
      throw new Error(`rig.joints[${index}] has an invalid parentIndex`);
    }
    const parentIndex = parent as number;
    if (parentIndex === -1) rootCount += 1;

    const restLocal = requireRecord(joint.restLocal, `rig.joints[${index}].restLocal`);
    finiteNumericVector(restLocal.translation, 3, `rig.joints[${index}].restLocal.translation`);
    const restRotation = finiteNumericVector(restLocal.rotation, 4, `rig.joints[${index}].restLocal.rotation`);
    const restScale = finiteNumericVector(restLocal.scale, 3, `rig.joints[${index}].restLocal.scale`);
    const rotationNorm = Math.hypot(...restRotation);
    if (Math.abs(rotationNorm - 1) > 1e-5) {
      throw new Error(`rig.joints[${index}].restLocal.rotation must be a unit quaternion`);
    }
    if (restScale.some((component) => Math.abs(component) <= 1e-12)) {
      throw new Error(`rig.joints[${index}].restLocal.scale cannot contain zero`);
    }
    const dofMask = requireRecord(joint.dofMask, `rig.joints[${index}].dofMask`);
    for (const field of ["translation", "rotation", "scale"] as const) {
      requireBooleanVector(dofMask[field], 3, `rig.joints[${index}].dofMask.${field}`);
    }
  }
  if (rootCount !== 1) {
    throw new Error("rig must contain exactly one root joint");
  }

  const driven = requireStringArray(rig.drivenJointOrder, "rig.drivenJointOrder");
  driven.forEach((jointId, index) => requireJointIdentifier(jointId, `rig.drivenJointOrder[${index}]`));
  validateDrivenCount(driven.length);
  if (new Set(driven).size !== driven.length) throw new Error("rig.drivenJointOrder contains duplicate ids");
  if (driven.includes(motionRoot)) throw new Error("rig.motionRoot cannot be model-driven");
  const fullIndex = new Map(jointOrder.map((id, index) => [id, index] as const));
  const declaredIndices = requireIntegerArray(rig.drivenJointIndices, "rig.drivenJointIndices");
  if (declaredIndices.length !== driven.length) throw new Error("rig.drivenJointIndices length mismatch");
  driven.forEach((id, index) => {
    const expected = fullIndex.get(id);
    if (expected === undefined || declaredIndices[index] !== expected) {
      throw new Error("rig.drivenJointIndices does not match drivenJointOrder");
    }
  });
  const drivenParentIndices = requireIntegerArray(rig.drivenParentIndices, "rig.drivenParentIndices");
  const drivenIndex = new Map(driven.map((id, index) => [id, index] as const));
  const expectedParentIndices = declaredIndices.map((jointIndex) => {
    let parentIndex = (joints[jointIndex]?.parentIndex ?? -1) as number;
    while (parentIndex >= 0) {
      const nearest = drivenIndex.get(jointOrder[parentIndex]!);
      if (nearest !== undefined) return nearest;
      parentIndex = (joints[parentIndex]?.parentIndex ?? -1) as number;
    }
    return -1;
  });
  if (drivenParentIndices.length !== driven.length ||
      drivenParentIndices.some((value, index) => value !== expectedParentIndices[index])) {
    throw new Error("rig.drivenParentIndices is not the nearest-driven hierarchy");
  }

  const maskGroups = requireRecord(rig.maskGroups, "rig.maskGroups");
  const groups = Object.entries(maskGroups);
  if (groups.length === 0) throw new Error("rig.maskGroups must be a non-empty object");
  for (const [name, mask] of groups) {
    requireManifestIdentifier(name, "rig.maskGroups key");
    requireBooleanVector(mask, driven.length, `rig.maskGroups.${name}`);
  }
  return driven;
}

function validateLegacyHierarchy(joints: readonly Record<string, unknown>[], motionRoot: string): Set<string> {
  const ids = new Set<string>();
  for (const joint of joints) {
    const id = requireJointIdentifier(joint.id, "joint.id");
    if (ids.has(id)) throw new Error(`Duplicate joint id: ${id}`);
    ids.add(id);
  }
  const root = joints.find((joint) => joint.id === motionRoot);
  if (!root || root.parent !== null || root.deform !== false) throw new Error("Legacy motionRoot is invalid");
  for (const joint of joints) {
    if (joint.parent !== null && (typeof joint.parent !== "string" || !ids.has(joint.parent))) {
      throw new Error(`Joint ${String(joint.id)} references a missing parent`);
    }
  }
  return ids;
}

function validateDrivenCount(count: number): void {
  if (count < 1 || count > MAX_DRIVEN_JOINTS) {
    throw new Error(`Driven joint count must be between 1 and ${MAX_DRIVEN_JOINTS}; received ${count}`);
  }
}

function stableCanonicalJson(input: unknown): string {
  return stableStringify(projectCanonical(input));
}

function projectCanonical(value: unknown): unknown {
  if (value === null || typeof value === "boolean") return value;
  if (typeof value === "string") return requireUnicodeScalarString(value);
  if (typeof value === "number") {
    if (!Number.isFinite(value)) throw new Error("Rig fingerprints reject non-finite numbers");
    const bytes = Buffer.allocUnsafe(8);
    bytes.writeDoubleBE(Object.is(value, -0) ? 0 : value);
    return `f64:${bytes.toString("hex")}`;
  }
  if (Array.isArray(value)) return value.map(projectCanonical);
  if (isRecord(value)) {
    const keys = Object.keys(value);
    keys.forEach(requireUnicodeScalarString);
    return Object.fromEntries(
      keys.sort(compareUnicodeScalars).map((key) => [key, projectCanonical(value[key])]),
    );
  }
  throw new Error(`Unsupported rig fingerprint value: ${typeof value}`);
}

function requireUnicodeScalarString(value: string): string {
  for (const scalar of value) {
    const codePoint = scalar.codePointAt(0)!;
    if (codePoint >= 0xD800 && codePoint <= 0xDFFF) {
      throw new Error("Rig fingerprints reject strings that are not Unicode scalar sequences");
    }
  }
  return value;
}

function stableStringify(value: unknown): string {
  if (value === null || typeof value === "string" || typeof value === "boolean") return JSON.stringify(value);
  if (Array.isArray(value)) return `[${value.map(stableStringify).join(",")}]`;
  const record = requireRecord(value, "canonical value");
  return `{${Object.keys(record).sort(compareUnicodeScalars).map((key) => `${JSON.stringify(key)}:${stableStringify(record[key])}`).join(",")}}`;
}

function compareUnicodeScalars(left: string, right: string): number {
  const leftScalars = Array.from(left, (value) => value.codePointAt(0)!);
  const rightScalars = Array.from(right, (value) => value.codePointAt(0)!);
  const sharedLength = Math.min(leftScalars.length, rightScalars.length);
  for (let index = 0; index < sharedLength; index += 1) {
    const difference = leftScalars[index]! - rightScalars[index]!;
    if (difference !== 0) return difference;
  }
  return leftScalars.length - rightScalars.length;
}

async function readJsonObject(path: string): Promise<Record<string, unknown>> {
  const raw = await readFile(path, "utf8");
  if (raw.length > 16 * 1024 * 1024) throw new Error("Character manifest exceeds 16 MiB");
  return requireRecord(JSON.parse(raw) as unknown, "character manifest");
}

function requireIdentifier(value: unknown, path: string): string {
  if (typeof value !== "string" || value.length < 1 || value.length > 128) throw new Error(`${path} must be a non-empty identifier`);
  return value;
}

function requireManifestIdentifier(value: unknown, path: string): string {
  const identifier = requireIdentifier(value, path);
  if (!/^[a-z0-9][a-z0-9._-]*$/.test(identifier)) {
    throw new Error(`${path} must be a lowercase manifest identifier`);
  }
  return identifier;
}

const WINDOWS_RESERVED_CHARACTER_BASENAMES = new Set([
  "con", "prn", "aux", "nul",
  ...Array.from({ length: 9 }, (_value, index) => `com${index + 1}`),
  ...Array.from({ length: 9 }, (_value, index) => `lpt${index + 1}`),
]);

function requireCharacterIdentifier(value: unknown, path: string): string {
  const identifier = requireManifestIdentifier(value, path);
  const basename = identifier.split(".", 1)[0]!;
  if (identifier.endsWith(".") || WINDOWS_RESERVED_CHARACTER_BASENAMES.has(basename)) {
    throw new Error(`${path} must be a Windows-safe character identifier`);
  }
  return identifier;
}

function requireJointIdentifier(value: unknown, path: string): string {
  const identifier = requireIdentifier(value, path);
  if (!/^[a-z_][a-z0-9_]{0,127}$/.test(identifier)) {
    throw new Error(`${path} must be a lowercase joint identifier`);
  }
  return identifier;
}

export function checkpointTargetPath(characterId: string, rigFingerprintValue: string): string {
  const safeCharacterId = requireCharacterIdentifier(characterId, "characterId");
  const fingerprint = requireSha256(rigFingerprintValue, "rigFingerprint");
  return `${CHECKPOINT_ROOT}/${safeCharacterId}/${fingerprint}/${CHECKPOINT_FILENAME}`;
}

export function validateCharacterCheckpointMetadata(
  value: unknown,
  expected?: CharacterCheckpointContract,
): CharacterCheckpointMetadata {
  const metadata = requireRecord(value, "checkpoint metadata");
  requireExactFields(metadata, [
    "characterId",
    "dataset",
    "drivenJointOrder",
    "model",
    "normalization",
    "rigFingerprint",
    "schema",
  ], "checkpoint metadata");
  if (metadata.schema !== CHECKPOINT_METADATA_SCHEMA) {
    throw new Error("checkpoint metadata schema is unsupported");
  }
  const characterId = requireCharacterIdentifier(metadata.characterId, "checkpoint metadata.characterId");
  const rigFingerprintValue = requireSha256(metadata.rigFingerprint, "checkpoint metadata.rigFingerprint");
  const drivenJointOrder = requireStringArray(metadata.drivenJointOrder, "checkpoint metadata.drivenJointOrder");
  if (drivenJointOrder.length < 1 || drivenJointOrder.length > MAX_DRIVEN_JOINTS ||
      new Set(drivenJointOrder).size !== drivenJointOrder.length) {
    throw new Error("checkpoint metadata.drivenJointOrder must contain 1..128 unique joints");
  }
  drivenJointOrder.forEach((jointId, index) =>
    requireJointIdentifier(jointId, `checkpoint metadata.drivenJointOrder[${index}]`));
  if (expected && (characterId !== expected.characterId ||
      rigFingerprintValue !== expected.rigFingerprint ||
      drivenJointOrder.length !== expected.drivenJointOrder.length ||
      drivenJointOrder.some((jointId, index) => jointId !== expected.drivenJointOrder[index]))) {
    throw new Error("checkpoint metadata identity does not match the selected character");
  }

  const dataset = requireRecord(metadata.dataset, "checkpoint metadata.dataset");
  requireExactFields(dataset, ["format", "manifestSha256", "schema"], "checkpoint metadata.dataset");
  if (dataset.schema !== CHECKPOINT_DATASET_SCHEMA) {
    throw new Error("checkpoint metadata.dataset.schema is unsupported");
  }
  if (dataset.format !== CHECKPOINT_DATASET_FORMAT) {
    throw new Error("checkpoint metadata.dataset.format is unsupported");
  }
  const manifestSha256 = requireSha256(dataset.manifestSha256, "checkpoint metadata.dataset.manifestSha256");

  const normalization = requireRecord(metadata.normalization, "checkpoint metadata.normalization");
  requireExactFields(
    normalization,
    ["condition", "schema", "target", "transform"],
    "checkpoint metadata.normalization",
  );
  if (normalization.schema !== CHECKPOINT_NORMALIZATION_SCHEMA) {
    throw new Error("checkpoint metadata.normalization.schema is unsupported");
  }
  if (normalization.transform !== CHECKPOINT_NORMALIZATION_TRANSFORM) {
    throw new Error("checkpoint metadata.normalization.transform is unsupported");
  }
  const condition = validateNormalizationGroup(
    normalization.condition,
    "checkpoint metadata.normalization.condition",
  );
  const target = validateNormalizationGroup(
    normalization.target,
    "checkpoint metadata.normalization.target",
  );

  const model = requireRecord(metadata.model, "checkpoint metadata.model");
  requireExactFields(
    model,
    ["architecture", "config", "implementationVersion", "schema"],
    "checkpoint metadata.model",
  );
  if (model.schema !== CHECKPOINT_MODEL_CONFIG_SCHEMA) {
    throw new Error("checkpoint metadata.model.schema is unsupported");
  }
  const architecture = requireManifestIdentifier(model.architecture, "checkpoint metadata.model.architecture");
  if (typeof model.implementationVersion !== "string" ||
      model.implementationVersion.length < 1 || model.implementationVersion.length > 128) {
    throw new Error("checkpoint metadata.model.implementationVersion must be a non-empty string");
  }
  const config = requireRecord(model.config, "checkpoint metadata.model.config");
  validateJsonValue(config, "checkpoint metadata.model.config");

  return {
    schema: CHECKPOINT_METADATA_SCHEMA,
    characterId,
    rigFingerprint: rigFingerprintValue,
    drivenJointOrder,
    dataset: {
      schema: CHECKPOINT_DATASET_SCHEMA,
      format: CHECKPOINT_DATASET_FORMAT,
      manifestSha256,
    },
    normalization: {
      schema: CHECKPOINT_NORMALIZATION_SCHEMA,
      transform: CHECKPOINT_NORMALIZATION_TRANSFORM,
      condition,
      target,
    },
    model: {
      schema: CHECKPOINT_MODEL_CONFIG_SCHEMA,
      architecture,
      implementationVersion: model.implementationVersion,
      config,
    },
  };
}

function derivedCheckpointContract(
  characterId: string,
  rigFingerprintValue: string,
  drivenJointOrder: readonly string[],
): CharacterCheckpointContract {
  return {
    format: CHECKPOINT_FORMAT,
    metadataSchema: CHECKPOINT_METADATA_SCHEMA,
    path: checkpointTargetPath(characterId, rigFingerprintValue),
    characterId,
    rigFingerprint: rigFingerprintValue,
    drivenJointOrder: [...drivenJointOrder],
    manifestDeclared: false,
  };
}

function validateCheckpointContract(
  value: unknown,
  characterId: string,
  rigFingerprintValue: string,
  drivenJointOrder: readonly string[],
  manifestDeclared: boolean,
): CharacterCheckpointContract {
  const checkpoint = requireRecord(value, "checkpoint");
  const expectedFields = [
    "characterId",
    "drivenJointOrder",
    "format",
    "metadataSchema",
    "path",
    "rigFingerprint",
  ];
  if (Object.keys(checkpoint).sort().join("\0") !== expectedFields.join("\0")) {
    throw new Error("checkpoint fields do not match the v1 checkpoint binding ABI");
  }
  if (checkpoint.format !== CHECKPOINT_FORMAT) throw new Error("checkpoint.format is unsupported");
  if (checkpoint.metadataSchema !== CHECKPOINT_METADATA_SCHEMA) {
    throw new Error("checkpoint.metadataSchema is unsupported");
  }
  if (checkpoint.characterId !== characterId) {
    throw new Error("checkpoint.characterId does not match characterId");
  }
  if (checkpoint.rigFingerprint !== rigFingerprintValue) {
    throw new Error("checkpoint.rigFingerprint does not match rigFingerprint");
  }
  const checkpointOrder = requireStringArray(checkpoint.drivenJointOrder, "checkpoint.drivenJointOrder");
  if (checkpointOrder.length !== drivenJointOrder.length ||
      checkpointOrder.some((jointId, index) => jointId !== drivenJointOrder[index])) {
    throw new Error("checkpoint.drivenJointOrder does not exactly match rig.drivenJointOrder");
  }
  const expectedPath = checkpointTargetPath(characterId, rigFingerprintValue);
  if (checkpoint.path !== expectedPath) {
    throw new Error("checkpoint.path must be the character-isolated canonical checkpoint target");
  }
  return {
    format: CHECKPOINT_FORMAT,
    metadataSchema: CHECKPOINT_METADATA_SCHEMA,
    path: expectedPath,
    characterId,
    rigFingerprint: rigFingerprintValue,
    drivenJointOrder: [...drivenJointOrder],
    manifestDeclared,
  };
}

function validateNormalizationGroup(value: unknown, path: string): CharacterNormalizationGroup {
  const group = requireRecord(value, path);
  requireExactFields(group, ["featureOrder", "offset", "scale"], path);
  const featureOrder = requireStringArray(group.featureOrder, `${path}.featureOrder`);
  if (featureOrder.length < 1 || featureOrder.length > 4096 ||
      new Set(featureOrder).size !== featureOrder.length ||
      featureOrder.some((feature) => feature.length > 128)) {
    throw new Error(`${path}.featureOrder must contain 1..4096 unique features`);
  }
  const offset = finiteNumericVector(group.offset, featureOrder.length, `${path}.offset`);
  const scale = finiteNumericVector(group.scale, featureOrder.length, `${path}.scale`);
  if (scale.some((component) => component <= 0)) {
    throw new Error(`${path}.scale must contain only positive values`);
  }
  return { featureOrder, offset, scale };
}

function requireExactFields(value: Record<string, unknown>, expected: readonly string[], path: string): void {
  if (Object.keys(value).sort().join("\0") !== [...expected].sort().join("\0")) {
    throw new Error(`${path} fields do not match the v1 ABI`);
  }
}

function validateJsonValue(value: unknown, path: string, depth = 0): void {
  if (depth > 32) throw new Error(`${path} exceeds the maximum JSON nesting depth`);
  if (value === null || typeof value === "string" || typeof value === "boolean") return;
  if (typeof value === "number") {
    if (!Number.isFinite(value)) throw new Error(`${path} contains a non-finite number`);
    return;
  }
  if (Array.isArray(value)) {
    value.forEach((item, index) => validateJsonValue(item, `${path}[${index}]`, depth + 1));
    return;
  }
  if (isRecord(value)) {
    Object.entries(value).forEach(([key, item]) => validateJsonValue(item, `${path}.${key}`, depth + 1));
    return;
  }
  throw new Error(`${path} contains a non-JSON value`);
}

function requireSha256(value: unknown, path: string): string {
  if (typeof value !== "string" || !/^[0-9a-f]{64}$/.test(value)) throw new Error(`${path} must be a lowercase SHA-256`);
  return value;
}

function finiteNumericVector(value: unknown, length: number, path: string): number[] {
  if (!Array.isArray(value) || value.length !== length || value.some((entry) =>
    typeof entry !== "number" || !Number.isFinite(entry))) {
    throw new Error(`${path} must contain ${length} finite numbers`);
  }
  return [...value] as number[];
}

function requireBooleanVector(value: unknown, length: number, path: string): boolean[] {
  if (!Array.isArray(value) || value.length !== length || value.some((entry) => typeof entry !== "boolean")) {
    throw new Error(`${path} must contain ${length} booleans`);
  }
  return [...value] as boolean[];
}

function requireStringArray(value: unknown, path: string): string[] {
  if (!Array.isArray(value) || value.some((entry) => typeof entry !== "string" || entry.length < 1)) {
    throw new Error(`${path} must be an array of non-empty strings`);
  }
  return [...value];
}

function requireIntegerArray(value: unknown, path: string): number[] {
  if (!Array.isArray(value) || value.some((entry) => !Number.isInteger(entry))) throw new Error(`${path} must be an integer array`);
  return [...value] as number[];
}

function requireRecordArray(value: unknown, path: string): Record<string, unknown>[] {
  if (!Array.isArray(value) || value.some((entry) => !isRecord(entry))) throw new Error(`${path} must be an object array`);
  return value as Record<string, unknown>[];
}

function requireRecord(value: unknown, path: string): Record<string, unknown> {
  if (!isRecord(value)) throw new Error(`${path} must be an object`);
  return value;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}
