/**
 * Saient Pulse — what Saient is doing, right now, factually.
 *
 * Two rules hold this together:
 *
 *  1. The animation is a function of real events. Every frame the robot shows is
 *     derived from an actual turn state and tool call, so a moving robot always
 *     means work is happening. A spinner that spins regardless is how an
 *     interface starts lying.
 *  2. The main line is always factual and never flavour. "Writing src/runtime.rs"
 *     is the text; "Teaching the semicolons discipline" is a separate, clearly
 *     secondary line. Personality is fine as long as it sits beside the facts
 *     rather than in front of them.
 */

import type { TurnState } from "./turnState.js";

/** Which animation the robot plays. Each maps to a real class of work. */
export type PulseAnimation =
  | "idle"
  | "thinking"    // looking between notes
  | "typing"      // running a command
  | "reading"     // scanning pages
  | "scanning"    // security scan — shield
  | "saving"      // filing into a drawer
  | "verifying"   // magnifying glass
  | "failed"      // sparks, then stops
  | "completed";  // sits down / ticks

export interface PulseActivity {
  animation: PulseAnimation;
  /** Factual description of the work. Never flavour text. */
  text: string;
  /** When this activity began, for the elapsed clock. */
  startedAt: number;
}

/** Tools grouped by the kind of work they represent. */
const READING_TOOLS = new Set(["fs_read", "fs_list", "fs_tree", "fs_search", "mem_recall"]);
const WRITING_TOOLS = new Set([
  "fs_write", "fs_mkdir", "fs_delete", "fs_move", "fs_copy",
  "apply_patch", "apply_unified_diff", "mem_remember",
]);

/** Verb shown per tool. Present continuous, because it is happening now. */
const TOOL_VERBS: Record<string, string> = {
  fs_read: "Reading",
  fs_list: "Listing",
  fs_tree: "Mapping",
  fs_search: "Searching",
  fs_write: "Writing",
  fs_mkdir: "Creating",
  fs_delete: "Deleting",
  fs_move: "Moving",
  fs_copy: "Copying",
  apply_patch: "Patching",
  apply_unified_diff: "Applying diff to",
  diff_proposed: "Diffing",
  exec: "Running",
  exec_command: "Running",
  mem_remember: "Remembering",
  mem_recall: "Recalling",
};

export interface StepContext {
  tool?: string;
  /** The file, command or query the step operates on. */
  target?: string | null;
}

/**
 * Pick the animation for a state and, where relevant, the tool in flight.
 *
 * Tool detail only matters while a step is actually running; in every other
 * state the turn state alone decides, so the robot cannot be left mid-gesture
 * after the work that justified it has finished.
 */
export function animationFor(state: TurnState, step?: StepContext): PulseAnimation {
  switch (state) {
    case "IDLE":
    case "USER_TYPING":
      return "idle";
    case "SAIENT_THINKING":
      return "thinking";
    case "VERIFYING":
      return "verifying";
    case "RETRYING":
      return "typing";
    case "COMPLETED":
      return "completed";
    case "FAILED":
      return "failed";
    case "INTERRUPTED":
      return "idle";
    case "WAITING_FOR_TOOL":
      return "typing";
    case "SAIENT_ACTING": {
      const tool = step?.tool ?? "";
      if (tool === "security_scan") return "scanning";
      if (READING_TOOLS.has(tool)) return "reading";
      if (WRITING_TOOLS.has(tool)) return "saving";
      if (tool.startsWith("exec")) return "typing";
      return "thinking";
    }
  }
}

/** Trim a long subject so the bar stays one line, keeping the informative end. */
export function shorten(target: string, max = 44): string {
  const t = target.trim();
  if (t.length <= max) return t;
  return "…" + t.slice(t.length - (max - 1));
}

/**
 * The factual line.
 *
 * Falls back to progressively less specific text rather than inventing detail:
 * verb plus subject, then verb alone, then the state's own description.
 */
export function activityLine(
  state: TurnState,
  step: StepContext | undefined,
  stateText: string,
): string {
  if (state === "SAIENT_ACTING" || state === "WAITING_FOR_TOOL") {
    const tool = step?.tool ?? "";
    const verb = TOOL_VERBS[tool];
    const target = step?.target?.trim();
    if (verb && target) return `${verb} ${shorten(target)}`;
    if (verb) return verb;
  }
  return stateText;
}

/**
 * Flavour lines. Deliberately meaningless — they carry no information, which is
 * exactly why they are safe to be funny. They sit under the factual line, never
 * replace it.
 */
const FLAVOUR: Record<PulseAnimation, readonly string[]> = {
  idle: ["Waiting, patiently, for once…", "Polishing the spanners…"],
  thinking: [
    "Interrogating the stack trace…",
    "Convincing electrons to cooperate…",
    "Considering three bad ideas…",
    "Checking it didn't confidently lie…",
  ],
  typing: [
    "Teaching the semicolons discipline…",
    "Negotiating with the compiler…",
    "Typing with conviction…",
  ],
  reading: ["Reading the small print…", "Squinting at someone's naming scheme…"],
  scanning: ["Frisking the dependencies…", "Checking nothing bites…"],
  saving: ["Filing this somewhere sensible…", "Putting things back tidily…"],
  verifying: [
    "Checking it didn't confidently lie…",
    "Marking its own homework, sceptically…",
  ],
  failed: ["That went poorly…", "Reassembling dignity…"],
  completed: ["Sitting down for a moment…", "Quietly pleased…"],
};

/**
 * Pick flavour deterministically from the activity's start time, so it stays put
 * while one activity runs instead of flickering on every re-render.
 */
export function flavourFor(animation: PulseAnimation, startedAt: number): string {
  const pool = FLAVOUR[animation];
  if (!pool.length) return "";
  // Rotate slowly for long activities: a new line roughly every 12 seconds.
  const tick = Math.floor(startedAt / 1000 / 12);
  return pool[Math.abs(tick) % pool.length];
}

/** mm:ss for the elapsed clock. Hours appear only once they exist. */
export function formatElapsed(ms: number): string {
  const total = Math.max(0, Math.floor(ms / 1000));
  const s = total % 60;
  const m = Math.floor(total / 60) % 60;
  const h = Math.floor(total / 3600);
  const pad = (n: number) => String(n).padStart(2, "0");
  return h > 0 ? `${h}:${pad(m)}:${pad(s)}` : `${pad(m)}:${pad(s)}`;
}

/** One line in the activity drawer — the beginning of the full timeline. */
export interface ActivityEntry {
  at: number;
  text: string;
  animation: PulseAnimation;
  /** Optional evidence: an error, a reason, a command's output. */
  detail?: string;
}

/** Cap on retained entries, so a long autonomous run can't grow without bound. */
export const ACTIVITY_LIMIT = 200;

export function pushActivity(log: ActivityEntry[], entry: ActivityEntry): ActivityEntry[] {
  log.push(entry);
  if (log.length > ACTIVITY_LIMIT) log.splice(0, log.length - ACTIVITY_LIMIT);
  return log;
}
