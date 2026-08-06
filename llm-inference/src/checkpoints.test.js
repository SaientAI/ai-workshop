// checkpoints.test.js — run with: node src/checkpoints.test.js
//
// A checkpoint that quietly omits a field is only discovered when someone needs
// it back, which is the worst possible moment. These pin down what gets captured.

import assert from "node:assert/strict";
import {
  shouldAutoSave,
  shouldPrompt,
  outstandingFrom,
  currentStep,
  buildSessionState,
  suggestName,
  groupByDay,
  describeSize,
} from "./lib/checkpoints.ts";
import { TURN_STATES } from "./lib/turnState.ts";

let passed = 0;
function test(name, fn) {
  try {
    fn();
    passed++;
  } catch (err) {
    console.error(`✗ ${name}\n  ${err.message}`);
    process.exitCode = 1;
  }
}

const plan = (statuses) => ({
  steps: statuses.map((status, i) => ({
    description: `step ${i + 1}`,
    tool: "fs_write",
    status,
  })),
});

// ── What counts as outstanding ──────────────────────────────────────────────

test("outstanding covers everything not finished", () => {
  const out = outstandingFrom(plan(["done", "pending", "failed", "running"]));
  assert.deepEqual(out, ["step 2", "step 3", "step 4"]);
});

test("a skipped step is still owed", () => {
  // Skipped means a prerequisite failed — that is work to do, not work done.
  assert.deepEqual(outstandingFrom(plan(["done", "skipped"])), ["step 2"]);
});

test("no plan means nothing outstanding, not a crash", () => {
  assert.deepEqual(outstandingFrom(null), []);
  assert.deepEqual(outstandingFrom({}), []);
});

// ── Where it had got to ─────────────────────────────────────────────────────

test("records the step actually in flight", () => {
  assert.deepEqual(currentStep(plan(["done", "running", "pending"])), { index: 2, total: 3 });
  assert.deepEqual(currentStep(plan(["done", "retrying"])), { index: 2, total: 2 });
});

test("with nothing running it records progress instead", () => {
  // A paused or finished session should still say how far it got.
  assert.deepEqual(currentStep(plan(["done", "done", "pending"])), { index: 2, total: 3 });
  assert.deepEqual(currentStep(null), { index: null, total: null });
});

// ── The whole capture ───────────────────────────────────────────────────────

test("captures the shoes, not just the memories", () => {
  const s = buildSessionState({
    goal: "refactor the scheduler",
    turn: "VERIFYING",
    terminalCwd: "/workspace/src",
    plan: plan(["done", "running"]),
    conversation: [{ role: "user", content: "go" }],
    terminal: ["$ cargo test"],
  });

  assert.equal(s.goal, "refactor the scheduler");
  assert.equal(s.turn_state, "VERIFYING");
  assert.equal(s.terminal_cwd, "/workspace/src");
  assert.deepEqual({ i: s.step_index, t: s.step_total }, { i: 2, t: 2 });
  assert.deepEqual(s.outstanding, ["step 2"]);
  assert.deepEqual(s.terminal, ["$ cargo test"]);
  assert.equal(s.conversation[0].content, "go");
});

test("missing sources produce a valid record rather than undefined fields", () => {
  const s = buildSessionState({});
  for (const key of ["goal", "turn_state", "terminal_cwd", "outstanding", "conversation", "terminal"]) {
    assert.notEqual(s[key], undefined, `${key} must not be undefined`);
  }
  assert.deepEqual(s.outstanding, []);
});

// ── Auto-save policy ────────────────────────────────────────────────────────

test("off never fires", () => {
  for (const s of TURN_STATES) assert.equal(shouldAutoSave("off", s), false);
});

test("every-turn keeps failures too", () => {
  // A failed run is often the one you most want back.
  assert.equal(shouldAutoSave("turn", "COMPLETED"), true);
  assert.equal(shouldAutoSave("turn", "FAILED"), true);
  assert.equal(shouldAutoSave("turn", "INTERRUPTED"), true);
});

test("every-completed-task skips interruptions", () => {
  // An interruption is usually the user changing their mind, not a result.
  assert.equal(shouldAutoSave("task", "COMPLETED"), true);
  assert.equal(shouldAutoSave("task", "FAILED"), true);
  assert.equal(shouldAutoSave("task", "INTERRUPTED"), false);
});

test("no policy fires mid-work", () => {
  for (const policy of ["turn", "task"]) {
    for (const s of ["SAIENT_THINKING", "SAIENT_ACTING", "WAITING_FOR_TOOL", "VERIFYING", "RETRYING"]) {
      assert.equal(shouldAutoSave(policy, s), false, `${policy} fired during ${s}`);
    }
  }
});

// ── Presentation ────────────────────────────────────────────────────────────

test("names default to the goal, then to the kind", () => {
  assert.equal(suggestName("ship the release", "manual"), "ship the release");
  assert.equal(suggestName("", "manual"), "Manual save");
  assert.equal(suggestName("   ", "auto_turn"), "Auto-save");
  assert.ok(suggestName("x".repeat(90), "manual").endsWith("…"));
});

test("checkpoints group by day", () => {
  const day = 1_700_000_000;
  const groups = groupByDay([
    { created_at: day, name: "a" },
    { created_at: day + 60, name: "b" },
    { created_at: day + 86_400 * 3, name: "c" },
  ]);
  assert.equal(groups.length, 2);
  assert.equal(groups[0].items.length, 2);
});

test("size reads at a glance", () => {
  assert.equal(describeSize({ file_count: 1, total_bytes: 2048 }), "1 file · 2 KiB");
  assert.equal(describeSize({ file_count: 3, total_bytes: 12 * 1024 }), "3 files · 12 KiB");
  assert.ok(describeSize({ file_count: 9, total_bytes: 5 * 1024 * 1024 }).endsWith("5.0 MiB"));
});

// ── Ask policy ──────────────────────────────────────────────────────────────

test("ask prompts only once the turn has settled", () => {
  for (const s of ["COMPLETED", "FAILED", "INTERRUPTED"]) {
    assert.equal(shouldPrompt("ask", s), true, `${s} should prompt`);
  }
  for (const s of ["SAIENT_THINKING", "VERIFYING", "RETRYING", "IDLE"]) {
    assert.equal(shouldPrompt("ask", s), false, `${s} must not prompt`);
  }
});

test("a decided policy never prompts", () => {
  // An auto-save that still asks is just a slower prompt.
  for (const policy of ["off", "turn", "task"]) {
    for (const s of TURN_STATES) {
      assert.equal(shouldPrompt(policy, s), false, `${policy} prompted at ${s}`);
    }
  }
});

test("ask never saves silently", () => {
  for (const s of TURN_STATES) assert.equal(shouldAutoSave("ask", s), false);
});

console.log(`checkpoints.test.js — ${passed} passed`);
