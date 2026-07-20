# OpenPets derivation note

Upstream reference: [alvinunreal/openpets](https://github.com/alvinunreal/openpets)
at commit `806659a46ef5b6833a3fbf69d21801af736acbff`, MIT licensed. The exact
reference snapshot is preserved at `third_party/desktop/openpets`.

This host is a clean, reduced derivative rather than a modified copy of the
reference checkout. The implementation follows these OpenPets designs:

- `apps/desktop/src/pet-window.ts`: transparent frameless BrowserWindow,
  always-on-top reassertion on Windows, renderer hit testing, and a main-process
  cursor probe to recover forwarded mouse events.
- `apps/desktop/src/window-tracker.ts`: `get-windows` enumeration, minimized
  Win32 bounds filtering, and non-overlapping async polling.
- `apps/desktop/src/pet-motion-engine.ts`: one shared ticker and one continuous
  authority for BrowserWindow position writes.
- `apps/desktop/src/display.ts`: bottom-centre/foot anchoring and work-area
  clamping.
- `apps/desktop/src/tray.ts`: tray-first lifecycle and pause/exit actions.

Concrete derivative files in this directory are:

- `src/pet-window.ts` and `src/preload.cjs` from the upstream pet-window and
  preload interaction path.
- `src/window-tracker-latch.ts` and `src/surface-tracker.ts` from the upstream
  window tracker and poller.
- `src/motion-controller.ts`, `src/motion-safety.ts`, and `src/geometry.ts` from
  the upstream single-writer motion/display geometry split.
- `src/tray.ts` from the upstream tray-first shell.

OpenPets' catalog, plugins, agent integrations, control centre, networking and
pet package formats are not copied into the prototype.
