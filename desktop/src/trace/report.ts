import { mkdirSync, writeFileSync } from "node:fs";
import { join } from "node:path";

import type { ReportSeriesPoint, TraceAnalysisReport, Verdict } from "./analysis.js";

export interface WrittenReport {
  readonly jsonPath: string;
  readonly htmlPath: string;
}

export function writeTraceReport(directory: string, report: TraceAnalysisReport): WrittenReport {
  mkdirSync(directory, { recursive: true });
  const jsonPath = join(directory, "report.json");
  const htmlPath = join(directory, "report.html");
  writeFileSync(jsonPath, `${JSON.stringify(report, null, 2)}\n`, "utf8");
  writeFileSync(htmlPath, renderTraceReportHtml(report), "utf8");
  return { jsonPath, htmlPath };
}

export function renderTraceReportHtml(report: TraceAnalysisReport): string {
  const safeData = JSON.stringify(report).replace(/</g, "\\u003c").replace(/>/g, "\\u003e").replace(/&/g, "\\u0026").replace(/\u2028/g, "\\u2028").replace(/\u2029/g, "\\u2029");
  const metricRows = [
    ["Plan latency p95", format(report.plans.latencyMs.p95, " ms")],
    ["Click response p95", format(report.interaction.clickResponseMs.p95, " ms")],
    ["Carrier error p95", format(report.motion.carrierErrorPx.p95, " px")],
    ["Position discontinuity max", format(report.motion.positionJumpPx.max, " px")],
    ["Landing success", report.motion.landingSuccessRate === null ? "n/a" : `${(report.motion.landingSuccessRate * 100).toFixed(2)}%`],
    ["Fallback time", `${(report.runtime.fallbackRatio * 100).toFixed(2)}%`],
    ["Generator restarts", String(report.runtime.generatorRestarts)],
    ["Recording gaps", String(report.runtime.recordingGaps)],
    ["Screen bounds violations", String(report.runtime.screenBoundsViolations)],
    ["Surface penetrations", String(report.runtime.surfacePenetrations)],
  ].map(([label, value]) => `<div class="metric"><span>${escapeHtml(label!)}</span><strong>${escapeHtml(value!)}</strong></div>`).join("");
  const checkRows = [...report.checks, ...report.regressions].map((check) => `<tr><td>${badge(check.status)}</td><td>${escapeHtml(check.name)}</td><td>${escapeHtml(check.value === null ? "n/a" : format(check.value))}</td><td>${escapeHtml(`${check.operator} ${format(check.threshold)}`)}</td><td>${escapeHtml(check.reason ?? "")}</td></tr>`).join("");
  const reasonRows = Object.entries(report.plans.rejectionReasons).map(([reason, count]) => `<tr><td>${escapeHtml(reason)}</td><td>${count}</td></tr>`).join("") || "<tr><td colspan=\"2\">None</td></tr>";
  return `<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>PET trace report</title><style>
:root{color-scheme:dark;background:#101218;color:#e9edf5;font:14px system-ui,-apple-system,Segoe UI,sans-serif}body{max-width:1200px;margin:auto;padding:24px}h1{font-size:24px;margin:0 0 6px}.muted{color:#9ca8ba}.summary{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:10px;margin:20px 0}.metric,.panel{background:#191e28;border:1px solid #2b3443;border-radius:10px;padding:14px}.metric span{display:block;color:#9ca8ba;font-size:12px}.metric strong{display:block;font-size:20px;margin-top:6px}.badge{display:inline-block;padding:3px 8px;border-radius:99px;font-weight:700}.PASS{background:#174f32;color:#7ef0ae}.WARN{background:#5a4612;color:#ffd970}.FAIL{background:#5c2028;color:#ff9da9}table{border-collapse:collapse;width:100%}th,td{text-align:left;border-bottom:1px solid #2b3443;padding:8px}canvas{width:100%;height:210px;background:#11151d;border-radius:6px}.charts{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:12px}.panel{margin:12px 0}.panel h2{font-size:16px;margin:0 0 10px}@media(max-width:600px){.charts{grid-template-columns:1fr}body{padding:12px}}
</style></head><body>
<h1>PET trace report ${badge(report.verdict)}</h1><div class="muted">${report.source.recordCount} records · ${format(report.source.durationMs / 1000, " s")} · digest ${escapeHtml(report.deterministicDigest.slice(0, 16))}</div>
<div class="summary">${metricRows}</div>
<div class="charts"><section class="panel"><h2>Motion foot position</h2><canvas id="motion"></canvas></section><section class="panel"><h2>Plan latency</h2><canvas id="latency"></canvas></section><section class="panel"><h2>Carrier error</h2><canvas id="carrier"></canvas></section><section class="panel"><h2>Host resources</h2><canvas id="process"></canvas></section></div>
<section class="panel"><h2>Checks</h2><table><thead><tr><th>Status</th><th>Metric</th><th>Value</th><th>Threshold</th><th>Note</th></tr></thead><tbody>${checkRows}</tbody></table></section>
<section class="panel"><h2>Plan results</h2><p>${report.plans.accepted} accepted, ${report.plans.rejected} rejected, ${report.plans.cancelled} cancelled (${report.plans.normalTopologyCancellations} expected topology changes).</p><table><thead><tr><th>Rejection reason</th><th>Count</th></tr></thead><tbody>${reasonRows}</tbody></table></section>
<script>"use strict";const report=${safeData};
function plot(id,groups){const c=document.getElementById(id),d=devicePixelRatio||1,r=c.getBoundingClientRect();c.width=Math.max(1,Math.floor(r.width*d));c.height=Math.max(1,Math.floor(r.height*d));const x=c.getContext("2d"),sets=groups.filter(g=>g.p.length);if(!sets.length){x.fillStyle="#9ca8ba";x.fillText("No samples",16,24);return}const all=sets.flatMap(g=>g.p),xmin=Math.min(...all.map(p=>p.elapsed_ms)),xmax=Math.max(...all.map(p=>p.elapsed_ms)),ymin=Math.min(...all.map(p=>p.value)),ymax=Math.max(...all.map(p=>p.value)),pad=20,px=v=>pad+(v-xmin)/Math.max(1,xmax-xmin)*(c.width-2*pad),py=v=>c.height-pad-(v-ymin)/Math.max(1e-9,ymax-ymin)*(c.height-2*pad);x.strokeStyle="#354052";x.strokeRect(pad,pad,c.width-2*pad,c.height-2*pad);for(const g of sets){x.beginPath();g.p.forEach((p,i)=>{const a=px(p.elapsed_ms),b=py(p.value);i?x.lineTo(a,b):x.moveTo(a,b)});x.strokeStyle=g.c;x.lineWidth=1.5*d;x.stroke()}}
plot("motion",[{p:report.series.motionX,c:"#65b7ff"},{p:report.series.motionY,c:"#ff9f75"}]);plot("latency",[{p:report.series.planLatency,c:"#c58cff"}]);plot("carrier",[{p:report.series.carrierError,c:"#6ce5ac"}]);plot("process",[{p:report.series.hostCpu,c:"#ffd166"},{p:report.series.hostRss,c:"#56cfe1"}]);
</script></body></html>`;
}

function badge(verdict: Verdict): string { return `<span class="badge ${verdict}">${verdict}</span>`; }
function format(value: number | null, suffix = ""): string { return value === null ? "n/a" : `${Number(value.toFixed(3))}${suffix}`; }
function escapeHtml(value: string): string { return value.replace(/[&<>"']/g, (character) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", "\"": "&quot;", "'": "&#39;" })[character]!); }
