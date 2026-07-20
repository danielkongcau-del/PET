/**
 * Prevents a single stale/re-anchored motion sample from mirroring the sprite.
 * A real turn remains responsive, but the requested direction must persist for
 * a handful of render frames before it becomes visible.
 */
export const FACING_CHANGE_HOLD_MS = 96;

export class FacingStabilizer {
  #value: -1 | 1;
  #candidate: -1 | 1 | null = null;
  #candidateSinceMs = 0;

  constructor(initial: -1 | 1 = 1, readonly holdMs = FACING_CHANGE_HOLD_MS) {
    this.#value = initial;
  }

  get value(): -1 | 1 {
    return this.#value;
  }

  update(desired: -1 | 1, nowMs: number): -1 | 1 {
    if (desired === this.#value) {
      this.resetPending();
      return this.#value;
    }
    if (this.#candidate !== desired) {
      this.#candidate = desired;
      this.#candidateSinceMs = nowMs;
      return this.#value;
    }
    if (Math.max(0, nowMs - this.#candidateSinceMs) >= this.holdMs) {
      this.#value = desired;
      this.resetPending();
    }
    return this.#value;
  }

  resetPending(): void {
    this.#candidate = null;
    this.#candidateSinceMs = 0;
  }
}
