const { contextBridge, ipcRenderer } = require("electron");

const listeners = new Set();
const probeListeners = new Set();
let skeletalConfig = null;

ipcRenderer.on("pet:render-state", (_event, state) => {
  for (const listener of listeners) listener(state);
});
ipcRenderer.on("pet:probe-hit-test", (_event, point) => {
  for (const listener of probeListeners) listener(point);
});
ipcRenderer.on("pet:skeletal-config", (_event, config) => {
  skeletalConfig = config;
});

contextBridge.exposeInMainWorld("petHost", {
  ready: () => ipcRenderer.send("pet:ready"),
  reportHit: (opaque) => ipcRenderer.send("pet:hit-test", opaque === true),
  click: (button) => ipcRenderer.send("pet:click", button),
  getSkeletalConfig: () => skeletalConfig,
  onState: (listener) => {
    if (typeof listener !== "function") return () => {};
    listeners.add(listener);
    return () => listeners.delete(listener);
  },
  onProbe: (listener) => {
    if (typeof listener !== "function") return () => {};
    probeListeners.add(listener);
    return () => probeListeners.delete(listener);
  },
});
