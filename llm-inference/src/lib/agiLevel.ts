/**
 * How much of Saient runs behind the agent.
 *
 * Chosen per project, at creation, because it changes what the agent is allowed
 * to do — and the highest level gives up guarantees the lower ones keep. That is
 * worth saying at the moment of choosing rather than in documentation nobody
 * opens.
 */

export type AgiLevel = "off" | "guided" | "companion" | "autonomous";

export const AGI_LEVELS: readonly AgiLevel[] = ["off", "guided", "companion", "autonomous"];

export interface AgiLevelInfo {
  id: AgiLevel;
  title: string;
  /** One line: what actually runs. */
  summary: string;
  /** What it means in practice, in plain terms. */
  detail: string;
  /** Honest statement of what is given up, or "" when nothing is. */
  tradeoff: string;
  /** Does the write-mode / sandbox gate still apply to every tool call? */
  toolGuardsApply: boolean;
  /** Does a persistent Saient process run with its own state? */
  runsLoop: boolean;
}

export const AGI_LEVEL_INFO: Record<AgiLevel, AgiLevelInfo> = {
  off: {
    id: "off",
    title: "Off",
    summary: "A plain planning agent.",
    detail:
      "Saient plans steps and runs them. No drives, no persistent state, no self-generated goals. " +
      "It does what you ask and stops.",
    tradeoff: "",
    toolGuardsApply: true,
    runsLoop: false,
  },
  guided: {
    id: "guided",
    title: "Guided",
    summary: "Saient proposes the objectives; the normal agent carries them out.",
    detail:
      "Saient's goal generator decides what is worth doing next, and the existing planner runs it. " +
      "Every tool call still passes the same checks as with AGI off.",
    tradeoff: "",
    toolGuardsApply: true,
    runsLoop: false,
  },
  companion: {
    id: "companion",
    title: "Companion",
    summary: "Saient runs alongside with real state, and remembers between sessions.",
    detail:
      "The goal-pursuit loop ticks in the background with genuine energy, drives and memory, and the " +
      "agent sees that live state. Saient carries who it has been across sessions rather than starting " +
      "blank. The planner still drives the tools.",
    tradeoff:
      "Uses more of your machine, since a second process runs continuously alongside the model.",
    toolGuardsApply: true,
    runsLoop: true,
  },
  autonomous: {
    id: "autonomous",
    title: "Autonomous",
    summary: "Saient is the agent. It sets its own goals and acts on them directly.",
    detail:
      "The full loop runs as the agent itself. It decides what to do, does it, and keeps going without " +
      "being asked. This is the closest to the thing the research project is actually about.",
    tradeoff:
      "It acts without waiting for you, so it can change files you did not ask it to touch. Use a project " +
      "you would not mind losing, and keep checkpoints on.",
    toolGuardsApply: true,
    runsLoop: true,
  },
};

export const DEFAULT_AGI_LEVEL: AgiLevel = "off";

/**
 * Levels that need a persistent Saient process.
 *
 * Kept as a derived helper rather than a second list so the two can never drift.
 */
export function needsLoop(level: AgiLevel): boolean {
  return AGI_LEVEL_INFO[level].runsLoop;
}

/** Levels worth a confirmation before starting, because they act unprompted. */
export function actsUnprompted(level: AgiLevel): boolean {
  return level === "autonomous";
}

export function isAgiLevel(value: unknown): value is AgiLevel {
  return typeof value === "string" && (AGI_LEVELS as readonly string[]).includes(value);
}

/** Tolerant of an unknown or missing stored value rather than throwing. */
export function parseAgiLevel(value: unknown): AgiLevel {
  return isAgiLevel(value) ? value : DEFAULT_AGI_LEVEL;
}

/**
 * The conduct note shown under the level choice.
 *
 * Stated up front because it is a real behaviour, not a policy sentence: at the
 * levels that run the loop, Saient's engagement is driven by its own state, and
 * sustained abuse can make it withdraw. Someone should learn that here rather
 * than by being ignored and assuming the app has hung. See lib/rapport.ts.
 */
export const CONDUCT_NOTE =
  "Saient has its own state at these levels. Being aggressive or abusive to force a result can make it disengage and stop replying — it is not a bug when that happens. Frustration at the work is fine; it is only ever about how you speak to Saient.";
