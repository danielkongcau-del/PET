import { appendFile, mkdir, rename, stat } from "node:fs/promises";
import { dirname } from "node:path";

type Level = "debug" | "info" | "warn" | "error";
type Fields = Readonly<Record<string, unknown>>;

const MAX_LOG_BYTES = 2 * 1024 * 1024;
let logPath: string | null = null;
let writeChain = Promise.resolve();

export function initializeLogger(path: string): void {
  logPath = path;
  void mkdir(dirname(path), { recursive: true });
}

export function debug(scope: string, message: string, fields: Fields = {}): void {
  if (process.env.PET_LOG_LEVEL === "debug") write("debug", scope, message, fields);
}

export function info(scope: string, message: string, fields: Fields = {}): void {
  write("info", scope, message, fields);
}

export function warn(scope: string, message: string, fields: Fields = {}): void {
  write("warn", scope, message, fields);
}

export function logError(scope: string, message: string, error: unknown, fields: Fields = {}): void {
  const safeError = error instanceof Error ? { error_name: error.name, error_message: safeString(error.message, 500) } : { error: safeString(String(error), 500) };
  write("error", scope, message, { ...fields, ...safeError });
}

function write(level: Level, scope: string, message: string, fields: Fields): void {
  const entry = JSON.stringify({
    timestamp_ms: Date.now(),
    level,
    scope: truncate(scope, 80),
    message: safeString(message, 300),
    ...sanitizeFields(fields),
  });
  if (level === "error" || level === "warn" || process.env.PET_LOG_CONSOLE === "1") {
    const method = level === "error" ? console.error : level === "warn" ? console.warn : console.log;
    method(`[pet:${scope}] ${message}`);
  }
  if (!logPath) return;
  const target = logPath;
  writeChain = writeChain.then(async () => {
    await mkdir(dirname(target), { recursive: true });
    await rotateIfNeeded(target);
    await appendFile(target, `${entry}\n`, "utf8");
  }).catch(() => undefined);
}

async function rotateIfNeeded(path: string): Promise<void> {
  try {
    const metadata = await stat(path);
    if (metadata.size < MAX_LOG_BYTES) return;
    await rename(path, `${path}.1`).catch(() => undefined);
  } catch {
    // The file does not exist yet.
  }
}

function sanitizeFields(fields: Fields): Record<string, unknown> {
  const safe: Record<string, unknown> = {};
  for (const [key, value] of Object.entries(fields)) {
    if (/title|key|screen|image|path|command|prompt|text/i.test(key)) continue;
    if (typeof value === "string") safe[key] = safeString(value, 200);
    else if (typeof value === "number" || typeof value === "boolean" || value === null) safe[key] = value;
    else if (Array.isArray(value)) safe[key] = { count: value.length };
    else if (value !== undefined) safe[key] = "[object]";
  }
  return safe;
}

function truncate(value: string, max: number): string {
  return value.length <= max ? value : `${value.slice(0, max)}...`;
}

function safeString(value: string, max: number): string {
  const redacted = value
    .replace(/[A-Za-z]:\\[^\s"']+/g, "[local-path]")
    .replace(/\\\\[^\s"']+/g, "[network-path]");
  return truncate(redacted, max);
}
