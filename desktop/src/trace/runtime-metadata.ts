import { createHash } from "node:crypto";
import { basename, relative, resolve } from "node:path";
import { readFile } from "node:fs/promises";

import { PROTOCOL_NAME, PROTOCOL_VERSION, type DisplayState } from "../protocol.js";
import type { JsonObject, JsonValue } from "./format.js";

const SOURCE_FINGERPRINT_FILES = [
  "desktop/src/motion-controller.ts",
  "desktop/src/motion-safety.ts",
  "desktop/src/surface-attachment.ts",
  "services/generator/pet_generator/backend.py",
  "services/generator/pet_generator/planner.py",
  "services/generator/pet_generator/service.py",
] as const;

export interface RuntimeTraceMetadataOptions {
  readonly projectRoot: string;
  readonly appVersion: string;
  readonly displays: readonly DisplayState[];
  readonly generator?: Readonly<Record<string, unknown>>;
  readonly character?: Readonly<{
    readonly characterId: string;
    readonly rigId: string;
    readonly rigFingerprint: string;
    readonly drivenJointOrder: readonly string[];
  }>;
}

/** Builds replay provenance without persisting usernames, hostnames, or absolute paths. */
export async function buildRuntimeTraceMetadata(options: RuntimeTraceMetadataOptions): Promise<JsonObject> {
  const checkpoint = await checkpointMetadata(process.env.PET_GENERATOR_CHECKPOINT);
  const source = await sourceFingerprint(options.projectRoot);
  return {
    runtime_schema: "pet-runtime-metadata-v1",
    app_version: options.appVersion,
    protocol: { name: PROTOCOL_NAME, version: PROTOCOL_VERSION },
    host: {
      platform: process.platform,
      arch: process.arch,
      node: process.versions.node,
      electron: process.versions.electron ?? "unknown",
      chrome: process.versions.chrome ?? "unknown",
    },
    generator: toJsonObject(options.generator ?? {}),
    character: options.character ? {
      character_id: options.character.characterId,
      rig_id: options.character.rigId,
      rig_fingerprint: options.character.rigFingerprint,
      driven_joint_order: [...options.character.drivenJointOrder],
    } : { configured: false },
    checkpoint,
    source,
    config: {
      coordinate_space: "physical_px",
      world_state_hz: 20,
      surface_poll_hz: 30,
      motion_sample_hz: 60,
      renderer_sprite_px: 48,
      host_window_dip: 96,
    },
    display_layout: options.displays.map((display) => ({
      id: display.id,
      bounds: { x: display.bounds.x, y: display.bounds.y, width: display.bounds.width, height: display.bounds.height },
      work_area: {
        x: display.work_area.x,
        y: display.work_area.y,
        width: display.work_area.width,
        height: display.work_area.height,
      },
      scale_factor: display.scale_factor,
      is_primary: display.is_primary,
    })),
  };
}

async function checkpointMetadata(checkpointPath: string | undefined): Promise<JsonObject> {
  if (!checkpointPath) return { configured: false };
  try {
    const bytes = await readFile(checkpointPath);
    return {
      configured: true,
      file: basename(checkpointPath),
      sha256: createHash("sha256").update(bytes).digest("hex"),
      bytes: bytes.byteLength,
    };
  } catch (error) {
    return {
      configured: true,
      file: basename(checkpointPath),
      unavailable: true,
      error: error instanceof Error ? error.name : "Error",
    };
  }
}

async function sourceFingerprint(projectRoot: string): Promise<JsonObject> {
  const digest = createHash("sha256");
  const included: string[] = [];
  const missing: string[] = [];
  for (const configuredPath of SOURCE_FINGERPRINT_FILES) {
    const fullPath = resolve(projectRoot, configuredPath);
    const safeName = relative(projectRoot, fullPath).replaceAll("\\", "/");
    try {
      const bytes = await readFile(fullPath);
      digest.update(safeName).update("\0").update(bytes).update("\0");
      included.push(safeName);
    } catch {
      missing.push(safeName);
    }
  }
  return { sha256: digest.digest("hex"), files: included, missing };
}

function toJsonObject(input: Readonly<Record<string, unknown>>): JsonObject {
  const output: JsonObject = {};
  for (const [key, value] of Object.entries(input)) {
    const converted = toJsonValue(value);
    if (converted !== undefined) output[key] = converted;
  }
  return output;
}

function toJsonValue(value: unknown): JsonValue | undefined {
  if (value === null || typeof value === "string" || typeof value === "boolean") return value;
  if (typeof value === "number") return Number.isFinite(value) ? value : undefined;
  if (Array.isArray(value)) return value.flatMap((entry) => {
    const converted = toJsonValue(entry);
    return converted === undefined ? [] : [converted];
  });
  if (typeof value === "object" && value !== null) return toJsonObject(value as Readonly<Record<string, unknown>>);
  return undefined;
}
