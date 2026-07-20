/**
 * Adapted from OpenPets
 * apps/desktop/src/window-tracker-latch.ts at
 * 806659a46ef5b6833a3fbf69d21801af736acbff.
 *
 * Collapse timer pressure into at most one follow-up run. Window enumeration
 * may be slower than its polling interval and must never stack concurrently.
 */
export function createLatchedTick(task: () => Promise<void>): () => void {
  let running = false;
  let pending = false;

  const run = async (): Promise<void> => {
    if (running) {
      pending = true;
      return;
    }
    running = true;
    try {
      do {
        pending = false;
        await task();
      } while (pending);
    } finally {
      running = false;
    }
  };

  return () => { void run(); };
}
