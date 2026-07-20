# PET desktop host

This directory is the deliberately small Windows 11 desktop-host derivative of
[OpenPets](../third_party/desktop/openpets). It keeps the parts needed by the
generative-pet experiment: a transparent always-on-top Electron window,
pixel-aware mouse interception, visible-window top surfaces, a supervised
Python sidecar, deterministic motion safety, a tray menu, and an optional debug
overlay.

`third_party/desktop/openpets` remains an immutable reference checkout. See
[`UPSTREAM.md`](UPSTREAM.md) for the source boundary and reused design ideas.

## Run

From the workspace root:

```powershell
pnpm --dir desktop install
pnpm --dir desktop check
pnpm --dir desktop dev
```

The desktop scripts first build the authoritative `@pet/protocol` workspace
package, because its runtime encoder/decoder and declarations are exported from
`packages/protocol/dist`.

The host finds the workspace through the parent of this directory. It launches
`D:\Anaconda\envs\pet-core\python.exe services/generator/run.py` by default.
The following environment variables are supported for development:

- `PET_PROJECT_ROOT`: absolute workspace root.
- `PET_PYTHON`: absolute Python executable (or a command available on PATH).
- `PET_GENERATOR_ENTRY`: absolute generator script path.
- `PET_GENERATOR_CWD`: child-process working directory.
- `PET_DISABLE_GENERATOR=1`: run only the deterministic safe-idle host.
- `PET_DEBUG_OVERLAY=1`: show surfaces and accepted plan points at startup.

The child protocol is NDJSON over private stdio; the host never opens a network
port. Coordinates crossing that boundary are Windows physical pixels. Electron
window placement remains in device-independent pixels (DIP), with conversion at
the process boundary.

## Current milestone

- One 96x96 transparent white pixel cat, rendered from the workspace
  `assets/pet/runtime/cat-48.png` when present and with a built-in canvas
  fallback otherwise.
- Visible window-top and per-display work-area-floor surfaces.
- Plan execution with stale-plan rejection, speed limiting, screen clamping,
  collision landing, and safe idle/fall behavior.
- Non-transparent cat pixels intercept clicks. Transparent pixels pass through
  to the application below.
- Full-screen applications and shell/security surfaces hide the pet; it is
  restored after they leave.
- Pause/resume, debug overlay, generator restart, and exit tray actions.
- Generator readiness timeout, heartbeat watchdog, exponential restart, and
  immediate safe-idle fallback.

Keyboard and screenshot sensors are intentionally only future capability
boundaries in this milestone; the three selected interactions need only mouse
events and window geometry.
