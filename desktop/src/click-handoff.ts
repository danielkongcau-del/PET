import type { ClickEvent } from "./protocol.js";

/** Relinquish only the click batch accepted by GeneratorBridge. */
export function releaseSentClicks(
  pending: readonly ClickEvent[],
  sent: readonly ClickEvent[],
): ClickEvent[] {
  const sentIds = new Set(sent.map((click) => click.id));
  return pending.filter((click) => !sentIds.has(click.id));
}
