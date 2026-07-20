import { spawn } from "node:child_process";
import { existsSync } from "node:fs";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

import {
  DEFAULT_TRACE_THRESHOLDS,
  analyzeTraceEpisode,
  loadTraceEpisode,
  type TraceAnalysisReport,
  type TraceThresholds,
} from "./analysis.js";
import { replayTraceExecution } from "./execution-replay.js";
import { TraceRecorder } from "./recorder.js";
import { writeTraceReport } from "./report.js";

type ThresholdKey = Exclude<keyof TraceThresholds, "version">;

export interface TraceCliIo {
  readonly stdout?: (text: string) => void;
  readonly stderr?: (text: string) => void;
  readonly cwd?: string;
  readonly env?: NodeJS.ProcessEnv;
}

export interface RegenerationInvocation {
  readonly python: string;
  readonly script: string;
  readonly episode: string;
  readonly output: string;
}

export interface TraceCliDependencies {
  readonly runRegeneration?: (invocation: RegenerationInvocation) => Promise<{ readonly code: number; readonly stderr?: string }>;
  readonly recover?: (root: string) => Promise<readonly string[]>;
}

interface ReplayCommand {
  readonly command: "replay";
  readonly episode: string;
  readonly output?: string;
  readonly baseline?: string;
  readonly thresholds: Partial<TraceThresholds>;
  readonly regenerate: boolean;
  readonly python?: string;
}

interface RecoverCommand { readonly command: "recover"; readonly root: string }
type ParsedCommand = ReplayCommand | RecoverCommand;

const THRESHOLD_ALIASES: Readonly<Record<string, ThresholdKey>> = {
  plan_latency_p95_ms: "planLatencyP95Ms",
  click_response_p95_ms: "clickResponseP95Ms",
  carrier_error_p95_px: "carrierErrorP95Px",
  position_jump_max_px: "positionJumpMaxPx",
  landing_success_rate_min: "landingSuccessRateMin",
  fallback_ratio_max: "fallbackRatioMax",
  recording_gaps_max: "recordingGapsMax",
  invalid_values_max: "invalidValuesMax",
  screen_bounds_violations_max: "screenBoundsViolationsMax",
  surface_penetrations_max: "surfacePenetrationsMax",
  regression_ratio_max: "regressionRatioMax",
};

const THRESHOLD_KEYS = new Set<ThresholdKey>(Object.keys(DEFAULT_TRACE_THRESHOLDS)
  .filter((key) => key !== "version") as ThresholdKey[]);

export async function runTraceCli(
  argv: readonly string[],
  io: TraceCliIo = {},
  dependencies: TraceCliDependencies = {},
): Promise<number> {
  const stdout = io.stdout ?? ((text: string) => process.stdout.write(text));
  const stderr = io.stderr ?? ((text: string) => process.stderr.write(text));
  let parsed: ParsedCommand;
  try {
    parsed = parseTraceCliArguments(argv);
  } catch (error) {
    stderr(`${safeMessage(error)}\n`);
    return 1;
  }

  try {
    if (parsed.command === "recover") {
      const root = path.resolve(io.cwd ?? process.cwd(), parsed.root);
      const recovered = dependencies.recover
        ? await dependencies.recover(root)
        : await new TraceRecorder({ rootDir: root }).recoverIncompleteEpisodes();
      stdout(`${JSON.stringify({ command: "recover", recovered: recovered.length })}\n`);
      return 0;
    }

    const cwd = io.cwd ?? process.cwd();
    const episode = path.resolve(cwd, parsed.episode);
    const loaded = loadTraceEpisode(episode);
    const baseline = parsed.baseline ? await readBaseline(path.resolve(cwd, parsed.baseline)) : undefined;
    const report = analyzeTraceEpisode(loaded, {
      ...(Object.keys(parsed.thresholds).length > 0 ? { thresholds: parsed.thresholds } : {}),
      ...(baseline ? { baseline } : {}),
    });
    const output = parsed.output
      ? path.resolve(cwd, parsed.output)
      : path.join(workspaceRoot(), "data", "reports", path.basename(episode));
    await mkdir(output, { recursive: true });
    writeTraceReport(output, report);
    const execution = replayTraceExecution(loaded.records);
    await writeFile(path.join(output, "execution-replay.json"), `${JSON.stringify(execution, null, 2)}\n`, "utf8");

    let regenerated = false;
    if (parsed.regenerate) {
      const env = io.env ?? process.env;
      const python = parsed.python ?? env.PET_PYTHON ?? "D:/Anaconda/envs/pet-core/python.exe";
      const invocation: RegenerationInvocation = {
        python,
        script: path.join(workspaceRoot(), "tools", "trace", "regenerate.py"),
        episode,
        output: path.join(output, "regeneration.json"),
      };
      const result = dependencies.runRegeneration
        ? await dependencies.runRegeneration(invocation)
        : await runRegenerationProcess(invocation);
      if (result.code !== 0) throw new Error(result.stderr?.trim() || `regeneration exited with code ${result.code}`);
      regenerated = true;
    }

    stdout(`${JSON.stringify({
      command: "replay",
      verdict: report.verdict,
      records: report.source.recordCount,
      digest: report.deterministicDigest,
      execution_digest: execution.eventDigestSha256,
      regenerated,
      output,
    })}\n`);
    return report.verdict === "FAIL" ? 2 : 0;
  } catch (error) {
    stderr(`${safeMessage(error)}\n`);
    return 1;
  }
}

export function parseTraceCliArguments(argv: readonly string[]): ParsedCommand {
  const command = argv[0];
  if (command === "recover") {
    if (argv.length !== 2 || !argv[1]) throw usageError("recover requires exactly one ROOT argument");
    return { command, root: argv[1] };
  }
  if (command !== "replay") throw usageError("expected replay or recover command");
  const episode = argv[1];
  if (!episode || episode.startsWith("--")) throw usageError("replay requires an EPISODE argument");
  let output: string | undefined;
  let baseline: string | undefined;
  let python: string | undefined;
  let regenerate = false;
  const thresholds: Partial<Record<ThresholdKey, number>> = {};
  for (let index = 2; index < argv.length; index += 1) {
    const token = argv[index]!;
    if (token === "--regenerate") { regenerate = true; continue; }
    if (token === "--output" || token === "--baseline" || token === "--python" || token === "--threshold") {
      const value = argv[++index];
      if (!value || value.startsWith("--")) throw usageError(`${token} requires a value`);
      if (token === "--output") {
        if (output !== undefined) throw usageError("--output may only be specified once");
        output = value;
      } else if (token === "--baseline") {
        if (baseline !== undefined) throw usageError("--baseline may only be specified once");
        baseline = value;
      } else if (token === "--python") {
        if (python !== undefined) throw usageError("--python may only be specified once");
        python = value;
      } else {
        const equals = value.indexOf("=");
        if (equals <= 0 || equals === value.length - 1) throw usageError("--threshold expects key=value");
        const rawKey = value.slice(0, equals);
        const key = THRESHOLD_ALIASES[rawKey] ?? (THRESHOLD_KEYS.has(rawKey as ThresholdKey) ? rawKey as ThresholdKey : undefined);
        if (!key) throw usageError(`unknown threshold: ${rawKey}`);
        const numeric = Number(value.slice(equals + 1));
        validateThreshold(key, numeric);
        thresholds[key] = numeric;
      }
      continue;
    }
    throw usageError(`unknown argument: ${token}`);
  }
  return {
    command, episode, thresholds: thresholds as Partial<TraceThresholds>, regenerate,
    ...(output ? { output } : {}), ...(baseline ? { baseline } : {}), ...(python ? { python } : {}),
  };
}

async function readBaseline(file: string): Promise<TraceAnalysisReport> {
  const parsed = JSON.parse(await readFile(file, "utf8")) as unknown;
  if (!isRecord(parsed) || parsed.schema !== "pet-trace-report" || parsed.version !== 1) throw new Error("baseline is not a PET trace report v1");
  return parsed as unknown as TraceAnalysisReport;
}

async function runRegenerationProcess(invocation: RegenerationInvocation): Promise<{ readonly code: number; readonly stderr: string }> {
  if (!existsSync(invocation.script)) return { code: 1, stderr: "regeneration tool is missing" };
  return await new Promise((resolve) => {
    const child = spawn(invocation.python, [invocation.script, invocation.episode, "--output", invocation.output, "--compare-only"], {
      windowsHide: true,
      shell: false,
      stdio: ["ignore", "ignore", "pipe"],
    });
    let stderr = "";
    child.stderr.setEncoding("utf8");
    child.stderr.on("data", (chunk: string) => { if (stderr.length < 16_384) stderr += chunk.slice(0, 16_384 - stderr.length); });
    child.once("error", (error) => resolve({ code: 1, stderr: safeMessage(error) }));
    child.once("exit", (code) => resolve({ code: code ?? 1, stderr }));
  });
}

function validateThreshold(key: ThresholdKey, value: number): void {
  if (!Number.isFinite(value) || value < 0) throw usageError(`threshold ${key} must be a finite non-negative number`);
  if ((key === "landingSuccessRateMin" || key === "fallbackRatioMax") && value > 1) throw usageError(`threshold ${key} must be between 0 and 1`);
  if (key === "regressionRatioMax" && value < 1) throw usageError("threshold regressionRatioMax must be at least 1");
}

function workspaceRoot(): string {
  return path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..", "..", "..");
}

function usageError(message: string): Error {
  return new Error(`${message}. Usage: replay EPISODE [--output DIR] [--baseline report.json] [--threshold key=value ...] [--regenerate] [--python EXE] | recover ROOT`);
}

function safeMessage(error: unknown): string { return error instanceof Error ? error.message.replace(/[\r\n]+/g, " ") : "trace command failed"; }
function isRecord(value: unknown): value is Record<string, unknown> { return typeof value === "object" && value !== null && !Array.isArray(value); }

const entry = process.argv[1];
if (entry && pathToFileURL(path.resolve(entry)).href === import.meta.url) {
  void runTraceCli(process.argv.slice(2)).then((code) => { process.exitCode = code; });
}
