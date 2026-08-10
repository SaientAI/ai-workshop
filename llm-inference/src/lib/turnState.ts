/**
 * Turn ownership for the agent.
 *
 * The bug this exists to kill: `plan-done` used to set `planRunning = false` and
 * `planPhase = "idle"` at the top of its handler, and only *then* decide whether
 * the autonomous loop was continuing. So the input went back to the user while a
 * goal-completion inference — and often an entire second agent run — was still
 * under way. The UI said finished; Saient was still working.
 *
 * The rule here is therefore single and blunt: the keyboard goes back to the user
 * only in a terminal state, and "one inference ended" is not terminal. If the loop
 * intends another iteration, ownership stays with Saient across the gap.
 */

export const TURN_STATES = [
  "IDLE",
  "USER_TYPING",
  "SAIENT_THINKING",
  "SAIENT_ACTING",
  "WAITING_FOR_TOOL",
  "VERIFYING",
  "RETRYING",
  "COMPLETED",
  "FAILED",
  "INTERRUPTED",
] as const;

export type TurnState = (typeof TURN_STATES)[number];

/**
 * The only states that hand the keyboard back.
 *
 * Note what is absent: VERIFYING and RETRYING. Both are quiet — no tokens are
 * streaming — and both were previously indistinguishable from idle.
 */
const USER_OWNED: ReadonlySet<TurnState> = new Set<TurnState>([
  "IDLE",
  "USER_TYPING",
  "COMPLETED",
  "FAILED",
  "INTERRUPTED",
]);

/** States after which no further work happens unless the user asks for it. */
const TERMINAL: ReadonlySet<TurnState> = new Set<TurnState>([
  "COMPLETED",
  "FAILED",
  "INTERRUPTED",
]);

export function isTerminal(state: TurnState): boolean {
  return TERMINAL.has(state);
}

/** True while Saient is doing something, whether or not it is visibly streaming. */
export function isWorking(state: TurnState): boolean {
  return !USER_OWNED.has(state);
}

/**
 * Who holds the input.
 *
 * `continuing` is the autonomous loop's intent to run again. It overrides the
 * state on purpose: between the end of one inference and the start of the next
 * there is a real window where the state looks finished but Saient is not.
 */
export function ownsInput(state: TurnState, continuing = false): "user" | "saient" {
  if (continuing) return "saient";
  return USER_OWNED.has(state) ? "user" : "saient";
}

const LABELS: Record<TurnState, string> = {
  IDLE: "User",
  USER_TYPING: "User",
  SAIENT_THINKING: "Saient — thinking",
  SAIENT_ACTING: "Saient — acting",
  WAITING_FOR_TOOL: "Saient — waiting for tool",
  VERIFYING: "Saient — verifying",
  RETRYING: "Saient — retrying",
  COMPLETED: "User",
  FAILED: "User",
  INTERRUPTED: "User",
};

export function inputLabel(state: TurnState, continuing = false): string {
  if (continuing) return "SAIENT — CONTINUING TASK";
  return LABELS[state];
}

/** Factual one-liner for the activity bar. Never flavour text — that sits separately. */
const ACTIVITY: Record<TurnState, string> = {
  IDLE: "Idle",
  USER_TYPING: "Waiting for you",
  SAIENT_THINKING: "Waiting for model response",
  SAIENT_ACTING: "Applying changes",
  WAITING_FOR_TOOL: "Running command",
  VERIFYING: "Verifying result",
  RETRYING: "Retrying failed step",
  COMPLETED: "Task completed",
  FAILED: "Task failed",
  INTERRUPTED: "Stopped by user",
};

export function activityText(state: TurnState): string {
  return ACTIVITY[state];
}

/**
 * Resting text for a project that runs the goal loop.
 *
 * "Idle" is right for a plain agent, which genuinely has nothing in progress.
 * It is wrong for Saient, whose state is sitting on disk intact and will be
 * picked up exactly where it left off. Calling that idle invites the reading
 * that something was lost, which is precisely what the help page has to spend a
 * paragraph undoing. Say sleeping and no one has to ask.
 */
export function restingText(
  state: TurnState,
  runsLoop: boolean,
  tracked = true,
): string {
  // `tracked` is whether anything is actually driving the turn state on the
  // surface being shown.
  //
  // `agent.turn` is set only from the Tauri event stream in `events.ts`. The
  // Terminal tab runs the Python agent over a PTY and emits none of those, so
  // the state sits at IDLE for the whole session — and this function, reading
  // IDLE, faithfully reported "Saient is sleeping" over a live agent turn and
  // over four straight hours of 100% GPU.
  //
  // The bar was not wrong about the state; it was answering a question it had no
  // way to know. Inferring activity from PTY output would be a guess (keystroke
  // echo arrives on the same channel as generation), so it says what it can
  // support instead.
  if (!tracked) return "Terminal session — activity not tracked here";
  if (runsLoop && (state === "IDLE" || state === "USER_TYPING")) {
    return "Saient is sleeping";
  }
  return ACTIVITY[state];
}

export interface RetryInfo {
  /** 1-based, as shown to the user. */
  step: number;
  total: number;
  reason: string;
}

/**
 * Concrete retry text. A retry that says only "retrying" is the thing that made
 * the loop feel like it had gone wandering.
 */
export function retryMessage(info: RetryInfo): string {
  return `Retrying step ${info.step} of ${info.total}\nReason: ${info.reason}`;
}

/**
 * Legal transitions.
 *
 * Deliberately not exhaustive bookkeeping — it encodes one invariant: a working
 * state may not drop straight to IDLE. Reaching rest has to go through
 * COMPLETED, FAILED or INTERRUPTED, each of which is a claim someone can check.
 * That is exactly the edge the old code slid off.
 */
const ALLOWED: Record<TurnState, readonly TurnState[]> = {
  IDLE: ["USER_TYPING", "SAIENT_THINKING"],
  USER_TYPING: ["IDLE", "SAIENT_THINKING"],
  SAIENT_THINKING: ["SAIENT_ACTING", "WAITING_FOR_TOOL", "VERIFYING", "COMPLETED", "FAILED", "INTERRUPTED"],
  SAIENT_ACTING: ["WAITING_FOR_TOOL", "VERIFYING", "SAIENT_THINKING", "COMPLETED", "FAILED", "INTERRUPTED"],
  WAITING_FOR_TOOL: ["SAIENT_ACTING", "VERIFYING", "RETRYING", "SAIENT_THINKING", "COMPLETED", "FAILED", "INTERRUPTED"],
  VERIFYING: ["SAIENT_THINKING", "RETRYING", "COMPLETED", "FAILED", "INTERRUPTED"],
  RETRYING: ["SAIENT_ACTING", "WAITING_FOR_TOOL", "VERIFYING", "COMPLETED", "FAILED", "INTERRUPTED"],
  COMPLETED: ["IDLE", "USER_TYPING", "SAIENT_THINKING"],
  FAILED: ["IDLE", "USER_TYPING", "SAIENT_THINKING"],
  INTERRUPTED: ["IDLE", "USER_TYPING", "SAIENT_THINKING"],
};

export function canTransition(from: TurnState, to: TurnState): boolean {
  return from === to || ALLOWED[from].includes(to);
}

/**
 * Apply a transition, refusing illegal ones.
 *
 * Returns the state to adopt. An illegal transition keeps the current state and
 * reports why, rather than quietly accepting it — a silent illegal jump is how
 * the interface came to disagree with reality in the first place.
 */
export function transition(
  from: TurnState,
  to: TurnState,
): { state: TurnState; ok: boolean; reason?: string } {
  if (canTransition(from, to)) return { state: to, ok: true };
  return {
    state: from,
    ok: false,
    reason: `illegal turn-state transition ${from} → ${to}`,
  };
}
