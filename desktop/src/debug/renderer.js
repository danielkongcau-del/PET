const canvas = document.getElementById("overlay");
const legend = document.getElementById("legend");
const recording = document.getElementById("recording");
const context = canvas.getContext("2d");

window.petDebug.onState((state) => {
  if (!state || !state.display) return;
  resize();
  context.clearRect(0, 0, canvas.width, canvas.height);
  const ratio = window.devicePixelRatio || 1;
  const scale = state.display.scale_factor || 1;
  const localX = (x) => ((x - state.display.bounds.x) / scale) * ratio;
  const localY = (y) => ((y - state.display.bounds.y) / scale) * ratio;

  context.lineWidth = 2 * ratio;
  for (const surface of state.surfaces || []) {
    context.strokeStyle = surface.enabled && !surface.occluded
      ? (surface.kind === "window_top" ? "#20e3b2" : "#4dabf7")
      : "#ff6b6b";
    context.setLineDash(surface.occluded ? [5 * ratio, 4 * ratio] : []);
    context.beginPath();
    context.moveTo(localX(surface.x1), localY(surface.y));
    context.lineTo(localX(surface.x2), localY(surface.y));
    context.stroke();
  }
  context.setLineDash([]);

  if (state.plan && Array.isArray(state.plan.points) && state.plan.points.length) {
    context.strokeStyle = "#e599f7";
    context.lineWidth = 2 * ratio;
    context.beginPath();
    state.plan.points.forEach((point, index) => {
      const x = localX(point.x);
      const y = localY(point.y);
      if (index === 0) context.moveTo(x, y);
      else context.lineTo(x, y);
    });
    context.stroke();
  }

  context.fillStyle = "#ffd43b";
  context.beginPath();
  context.arc(localX(state.foot.x), localY(state.foot.y), 4 * ratio, 0, Math.PI * 2);
  context.fill();
  const metrics = state.generatorMetrics;
  const gauges = metrics?.gauges || {};
  const counters = metrics?.counters || {};
  const formatMs = (value) => Number.isFinite(value) ? `${Number(value).toFixed(2)} ms` : "n/a";
  const recordingState = state.recording || {};
  recording.hidden = !recordingState.active;
  if (recordingState.active) {
    const totalSeconds = Math.max(0, Math.floor(Number(recordingState.elapsedMs || 0) / 1000));
    const minutes = Math.floor(totalSeconds / 60);
    const seconds = totalSeconds % 60;
    const bytes = Math.max(0, Number(recordingState.bytesWritten || 0));
    const size = bytes < 1048576 ? `${(bytes / 1024).toFixed(1)} KiB` : `${(bytes / 1048576).toFixed(1)} MiB`;
    recording.textContent = `● REC ${minutes}:${String(seconds).padStart(2, "0")}  ${size}`;
  }
  legend.textContent = [
    `behavior: ${state.behavior}`,
    `generator: ${state.generatorStatus}`,
    `plan: ${state.plan ? state.plan.planId : "none"}`,
    `based_on_seq: ${state.plan ? state.plan.basedOnSeq : "none"}`,
    `reject: ${state.lastPlanRejection || "none"}`,
    `surfaces: ${(state.surfaces || []).length}`,
    `plan latency: ${formatMs(gauges.last_plan_latency_ms)} (max ${formatMs(gauges.max_plan_latency_ms)})`,
    `state drops: ${Number.isFinite(counters.world_states_dropped) ? counters.world_states_dropped : 0}`,
    `generator restarts: ${Number.isFinite(state.generatorRestarts) ? state.generatorRestarts : 0}`,
  ].join("\n");
});

window.addEventListener("resize", resize);
function resize() {
  const ratio = window.devicePixelRatio || 1;
  const width = Math.max(1, Math.round(innerWidth * ratio));
  const height = Math.max(1, Math.round(innerHeight * ratio));
  if (canvas.width !== width) canvas.width = width;
  if (canvas.height !== height) canvas.height = height;
}
resize();
