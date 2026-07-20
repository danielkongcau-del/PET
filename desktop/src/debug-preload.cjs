const { contextBridge, ipcRenderer } = require("electron");

const listeners = new Set();
ipcRenderer.on("pet:debug-state", (_event, state) => {
  for (const listener of listeners) listener(state);
});

contextBridge.exposeInMainWorld("petDebug", {
  onState: (listener) => {
    if (typeof listener !== "function") return () => {};
    listeners.add(listener);
    return () => listeners.delete(listener);
  },
});
