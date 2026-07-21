"use strict";

let buffer = "";
let emitted = false;

process.stdin.setEncoding("utf8");
process.stdin.on("data", (chunk) => {
  buffer += chunk;
  for (;;) {
    const newline = buffer.indexOf("\n");
    if (newline < 0) break;
    const line = buffer.slice(0, newline).replace(/\r$/, "");
    buffer = buffer.slice(newline + 1);
    if (!line || emitted) continue;
    const hello = JSON.parse(line);
    if (hello.type !== "hello") continue;
    emitted = true;
    const now = Date.now();
    const base = {
      protocol: "pet-motion",
      version: 1,
      type: "ready",
      timestamp_ms: now,
    };
    const generator = {
      name: "double-ready-fixture",
      version: "1.0.0",
      pid: process.pid,
    };
    const first = {
      ...base,
      seq: 0,
      payload: {
        session_id: hello.payload.session_id,
        generator: { ...generator, skeleton_sha256: process.env.PET_TEST_SKELETON_SHA },
        accepted_version: 1,
        capabilities: ["skeletal_motion", "skeletal_motion_3d_local_quat"],
        ready_at_ms: now,
      },
    };
    const second = {
      ...base,
      seq: 1,
      payload: {
        session_id: hello.payload.session_id,
        generator: { ...generator, skeleton_sha256: process.env.PET_TEST_SKELETON_SHA },
        accepted_version: 1,
        capabilities: ["skeletal_motion"],
        ready_at_ms: now,
      },
    };
    // One write forces both handshakes through the host parser before either
    // asynchronous rig negotiation resumes.
    process.stdout.write(`${JSON.stringify(first)}\n${JSON.stringify(second)}\n`);
  }
});
