/** Tray-first shell reduced from OpenPets apps/desktop/src/tray.ts. */
import { app, Menu, nativeImage, shell, Tray, type NativeImage } from "electron";
import { readFile } from "node:fs/promises";
import { join } from "node:path";

import type { GeneratorStatus } from "./generator-bridge.js";
import { logError } from "./logger.js";

export interface TrayActions {
  readonly togglePaused: () => void;
  readonly toggleDebug: () => void;
  readonly toggleRecording: () => void;
  readonly restartGenerator: () => void;
  readonly quit: () => void;
}

export interface TrayState {
  readonly paused: boolean;
  readonly debug: boolean;
  readonly generatorStatus: GeneratorStatus;
  readonly recording: RecordingIndicatorState;
}

export interface RecordingIndicatorState {
  readonly active: boolean;
  readonly elapsedMs: number;
  readonly bytesWritten: number;
}

export class PetTray {
  readonly #tray: Tray;
  readonly #actions: TrayActions;
  readonly #logsDirectory: string;
  readonly #recordingsDirectory: string;
  #state: TrayState;

  private constructor(
    tray: Tray,
    actions: TrayActions,
    logsDirectory: string,
    recordingsDirectory: string,
    state: TrayState,
  ) {
    this.#tray = tray;
    this.#actions = actions;
    this.#logsDirectory = logsDirectory;
    this.#recordingsDirectory = recordingsDirectory;
    this.#state = state;
    this.#tray.setToolTip("PET - 生成式像素桌宠");
    this.refresh(state);
  }

  static async create(
    projectRoot: string,
    logsDirectory: string,
    recordingsDirectory: string,
    actions: TrayActions,
    state: TrayState,
    characterIconPath?: string | null,
  ): Promise<PetTray> {
    const icon = await loadTrayIcon(projectRoot, characterIconPath);
    return new PetTray(new Tray(icon), actions, logsDirectory, recordingsDirectory, state);
  }

  refresh(state: TrayState): void {
    this.#state = state;
    const statusLabel: Record<GeneratorStatus, string> = {
      disabled: "生成器：已禁用（安全待机）",
      starting: "生成器：启动中",
      ready: "生成器：就绪",
      degraded: "生成器：异常（安全待机/重试中）",
      stopped: "生成器：已停止",
    };
    this.#tray.setContextMenu(Menu.buildFromTemplate([
      { label: "PET", enabled: false },
      { label: statusLabel[state.generatorStatus], enabled: false },
      { type: "separator" },
      {
        label: state.paused ? "恢复桌宠" : "暂停桌宠",
        click: this.#actions.togglePaused,
      },
      {
        label: "显示调试层",
        type: "checkbox",
        checked: state.debug,
        click: this.#actions.toggleDebug,
      },
      {
        label: "重启动作生成器",
        enabled: state.generatorStatus !== "disabled",
        click: this.#actions.restartGenerator,
      },
      { type: "separator" },
      {
        label: state.recording.active
          ? `停止轨迹记录（${formatDuration(state.recording.elapsedMs)} · ${formatBytes(state.recording.bytesWritten)}）`
          : "开始轨迹记录",
        click: this.#actions.toggleRecording,
      },
      {
        label: "打开轨迹目录",
        click: () => { void shell.openPath(this.#recordingsDirectory).catch((error) => logError("tray", "open recordings failed", error)); },
      },
      { type: "separator" },
      {
        label: "打开日志目录",
        click: () => { void shell.openPath(this.#logsDirectory).catch((error) => logError("tray", "open logs failed", error)); },
      },
      { type: "separator" },
      { label: "退出", click: this.#actions.quit },
    ]));
  }

  dispose(): void {
    this.#tray.destroy();
  }
}

function formatDuration(elapsedMs: number): string {
  const totalSeconds = Math.max(0, Math.floor(elapsedMs / 1_000));
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  return `${minutes}:${String(seconds).padStart(2, "0")}`;
}

function formatBytes(bytes: number): string {
  const safe = Math.max(0, bytes);
  if (safe < 1_024) return `${safe} B`;
  if (safe < 1_048_576) return `${(safe / 1_024).toFixed(1)} KiB`;
  if (safe < 1_073_741_824) return `${(safe / 1_048_576).toFixed(1)} MiB`;
  return `${(safe / 1_073_741_824).toFixed(2)} GiB`;
}

async function loadTrayIcon(projectRoot: string, characterIconPath?: string | null): Promise<NativeImage> {
  try {
    const bytes = await readFile(characterIconPath ?? join(projectRoot, "assets", "pet", "runtime", "cat-48.png"));
    const icon = nativeImage.createFromBuffer(bytes).resize({ width: 16, height: 16, quality: "best" });
    if (!icon.isEmpty()) return icon;
  } catch {
    // Fall through to an embedded no-network icon.
  }
  const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" shape-rendering="crispEdges"><path fill="#111" d="M2 5h2V3h2v2h4V3h2v2h2v8h-2v2H4v-2H2z"/><path fill="#fff" d="M4 6h8v6H4z"/><path fill="#111" d="M5 8h1v1H5zm5 0h1v1h-1z"/></svg>`;
  return nativeImage.createFromDataURL(`data:image/svg+xml;base64,${Buffer.from(svg).toString("base64")}`);
}
