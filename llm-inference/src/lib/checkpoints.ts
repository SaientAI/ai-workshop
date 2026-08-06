/**
 * Building a checkpoint from live app state.
 *
 * The point of this module is what gets captured. Saving the conversation alone
 * leaves Saient able to recall the discussion and nothing about where it was —
 * which goal was active, which step was in flight, what the terminal was pointed
 * at, what it still owed the user. Restoring that gives you memories with no
 * shoes. So `buildSessionState` gathers the working state alongside the words,
 * and the pure helpers below are unit-tested because a checkpoint that silently
 * omits a field is only discovered when someone needs it back.
 */

import type { Plan } from "./types.js";
import type { SessionState, CheckpointKind, CheckpointMeta } from "./tauri.js";
import type { TurnState } from "./turnState.js";

/** When to checkpoint without being asked. */
export type AutoSavePolicy = "off" | "turn" | "task";

export const AUTO_SAVE_POLICIES: readonly AutoSavePolicy[] = ["off", "turn", "task"];

export const AUTO_SAVE_LABELS: Record<AutoSavePolicy, string> = {
  off: "Off",
  turn: "Every turn",
  task: "Every completed task",
};

/**
 * Whether a turn reaching `state` should trigger an automatic checkpoint.
 *
 * "Every turn" fires on any terminal state, including failures — a failed run is
 * often the one you most want back. "Every completed task" is narrower and skips
 * interruptions, which are usually the user changing their mind rather than a
 * result worth keeping.
 */
export function shouldAutoSave(policy: AutoSavePolicy, state: TurnState): boolean {
  switch (policy) {
    case "off":
      return false;
    case "turn":
      return state === "COMPLETED" || state === "FAILED" || state === "INTERRUPTED";
    case "task":
      return state === "COMPLETED" || state === "FAILED";
  }
}

/**
 * Steps still owed: anything not finished.
 *
 * Skipped counts as outstanding on purpose — a step skipped because its
 * prerequisite failed is work still to do, not work completed.
 */
export function outstandingFrom(plan: Plan | null): string[] {
  if (!plan?.steps) return [];
  return plan.steps
    .filter((s) => {
      const st = String(s.status ?? "").toLowerCase();
      return st !== "done";
    })
    .map((s) => s.description || s.tool || "(unnamed step)");
}

/** Which step was in flight, 1-based, or null when nothing is running. */
export function currentStep(plan: Plan | null): { index: number | null; total: number | null } {
  if (!plan?.steps?.length) return { index: null, total: null };
  const running = plan.steps.findIndex((s) =>
    ["running", "retrying"].includes(String(s.status ?? "").toLowerCase()),
  );
  if (running >= 0) return { index: running + 1, total: plan.steps.length };
  // Nothing running: report progress instead, so a paused session still records
  // how far it got.
  const done = plan.steps.filter((s) => String(s.status ?? "").toLowerCase() === "done").length;
  return { index: done, total: plan.steps.length };
}

export interface SessionSources {
  goal?: string;
  turn?: TurnState;
  terminalCwd?: string;
  plan?: Plan | null;
  conversation?: unknown;
  terminal?: string[];
}

/**
 * Assemble everything a checkpoint needs from the live state.
 *
 * Every field is defaulted rather than passed through. The Rust side types these
 * as non-optional, so a single undefined here fails the whole save at
 * deserialization — and it would fail at exactly the moment someone was trying
 * to preserve their work.
 */
export function buildSessionState(src: SessionSources): SessionState {
  const step = currentStep(src.plan ?? null);
  return {
    goal: src.goal ?? "",
    turn_state: src.turn ?? "IDLE",
    terminal_cwd: src.terminalCwd ?? "",
    step_index: step.index,
    step_total: step.total,
    outstanding: outstandingFrom(src.plan ?? null),
    conversation: src.conversation ?? [],
    plan: src.plan ?? null,
    terminal: src.terminal ?? [],
  };
}

/** A name a person will recognise in a list six checkpoints later. */
export function suggestName(goal: string, kind: CheckpointKind): string {
  const g = (goal ?? "").trim();
  if (g) return g.length > 60 ? `${g.slice(0, 60)}…` : g;
  return kind === "manual" ? "Manual save" : "Auto-save";
}

/** Group checkpoints by day for display. Newest day first, newest first within. */
export function groupByDay(list: CheckpointMeta[]): { day: string; items: CheckpointMeta[] }[] {
  const groups = new Map<string, CheckpointMeta[]>();
  for (const cp of list) {
    const day = new Date(cp.created_at * 1000).toDateString();
    const bucket = groups.get(day);
    if (bucket) bucket.push(cp);
    else groups.set(day, [cp]);
  }
  return [...groups.entries()].map(([day, items]) => ({ day, items }));
}

/** "3 files · 12 KiB" — enough to tell a full snapshot from a trivial one. */
export function describeSize(meta: CheckpointMeta): string {
  const files = `${meta.file_count} file${meta.file_count === 1 ? "" : "s"}`;
  const kib = meta.total_bytes / 1024;
  const size = kib < 1024 ? `${Math.round(kib)} KiB` : `${(kib / 1024).toFixed(1)} MiB`;
  return `${files} · ${size}`;
}
