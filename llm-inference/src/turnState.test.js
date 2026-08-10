// turnState.test.js — run with: node src/turnState.test.js
//
// Guards the rule that the input is only handed back when Saient has genuinely
// stopped. The regression these exist for: `plan-done` marked the run idle
// before deciding whether the autonomous loop was continuing, so the user got
// the keyboard back while a completion check and often a whole second agent run
// were still in flight.
//
// Node 24 strips types on import, so this tests the real module, not a copy.

import assert from "node:assert/strict";
import {
  TURN_STATES,
  ownsInput,
  inputLabel,
  activityText,
  restingText,
  isWorking,
  isTerminal,
  canTransition,
  transition,
  retryMessage,
} from "./lib/turnState.ts";

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

// ── Input ownership ─────────────────────────────────────────────────────────

test("only terminal states hand the keyboard back", () => {
  const userOwned = TURN_STATES.filter((s) => ownsInput(s) === "user");
  assert.deepEqual(
    [...userOwned].sort(),
    ["COMPLETED", "FAILED", "IDLE", "INTERRUPTED", "USER_TYPING"],
  );
});

test("quiet working states still belong to Saient", () => {
  // These are the states that look like nothing is happening. They were the
  // ones the old code could not distinguish from idle.
  for (const s of ["VERIFYING", "RETRYING", "WAITING_FOR_TOOL"]) {
    assert.equal(ownsInput(s), "saient", `${s} must not release the input`);
    assert.equal(isWorking(s), true);
  }
});

test("a continuing loop keeps the input even at COMPLETED", () => {
  // The exact gap the bug lived in: one inference ended, another is coming.
  assert.equal(ownsInput("COMPLETED", true), "saient");
  assert.equal(inputLabel("COMPLETED", true), "SAIENT — CONTINUING TASK");
});

test("a finished loop does release the input", () => {
  assert.equal(ownsInput("COMPLETED", false), "user");
  assert.equal(inputLabel("COMPLETED", false), "User");
});

test("every state has a label and an activity line", () => {
  for (const s of TURN_STATES) {
    assert.ok(inputLabel(s).length > 0, `${s} has no label`);
    assert.ok(activityText(s).length > 0, `${s} has no activity text`);
  }
});

test("terminal states are exactly the three specified", () => {
  const terminal = TURN_STATES.filter(isTerminal);
  assert.deepEqual([...terminal].sort(), ["COMPLETED", "FAILED", "INTERRUPTED"]);
});

// ── Transitions ─────────────────────────────────────────────────────────────

test("no working state may drop straight to IDLE", () => {
  // Reaching rest has to pass through a claim someone can check.
  for (const s of ["SAIENT_THINKING", "SAIENT_ACTING", "WAITING_FOR_TOOL", "VERIFYING", "RETRYING"]) {
    assert.equal(canTransition(s, "IDLE"), false, `${s} → IDLE must be refused`);
  }
});

test("working states may reach every terminal state", () => {
  for (const s of ["SAIENT_THINKING", "SAIENT_ACTING", "WAITING_FOR_TOOL", "VERIFYING", "RETRYING"]) {
    for (const t of ["COMPLETED", "FAILED", "INTERRUPTED"]) {
      assert.equal(canTransition(s, t), true, `${s} → ${t} should be allowed`);
    }
  }
});

test("verifying can loop back into thinking when the goal is not met", () => {
  // This is the autonomous loop continuing, and it must be expressible without
  // passing through a user-owned state.
  assert.equal(canTransition("VERIFYING", "SAIENT_THINKING"), true);
});

test("an illegal transition is refused and keeps the current state", () => {
  const r = transition("VERIFYING", "IDLE");
  assert.equal(r.ok, false);
  assert.equal(r.state, "VERIFYING");
  assert.match(r.reason, /illegal/);
});

test("a legal transition is applied", () => {
  const r = transition("WAITING_FOR_TOOL", "RETRYING");
  assert.equal(r.ok, true);
  assert.equal(r.state, "RETRYING");
});

test("staying in the same state is always allowed", () => {
  for (const s of TURN_STATES) assert.equal(canTransition(s, s), true);
});

// ── Retry reporting ─────────────────────────────────────────────────────────

test("retry text names the step and the reason", () => {
  const msg = retryMessage({ step: 2, total: 3, reason: "command returned exit code 1" });
  assert.equal(msg, "Retrying step 2 of 3\nReason: command returned exit code 1");
});

// ── Sleep, not idle ─────────────────────────────────────────────────────────

test("a project running the loop rests as sleeping, not idle", () => {
  // Its state is on disk and resumes where it left off. "Idle" invites the
  // reading that something was lost, which the help page then has to undo.
  assert.equal(restingText("IDLE", true), "Saient is sleeping");
  assert.equal(restingText("USER_TYPING", true), "Saient is sleeping");
});

test("a plain agent is still idle", () => {
  assert.equal(restingText("IDLE", false), "Idle");
});

test("sleeping never masks real work or a real outcome", () => {
  for (const s of ["SAIENT_THINKING", "WAITING_FOR_TOOL", "VERIFYING", "RETRYING", "FAILED", "COMPLETED"]) {
    assert.notEqual(restingText(s, true), "Saient is sleeping", `${s} must not read as asleep`);
    assert.equal(restingText(s, true), activityText(s));
  }
});

test("restingText does not claim sleep on a surface it cannot see", () => {
  // The Terminal tab drives no turn events, so `agent.turn` stays IDLE for the
  // whole session. Reading that as sleep is how the bar came to say "Saient is
  // sleeping" during a live turn and through four hours of pegged GPU.
  assert.equal(
    restingText("IDLE", true, false),
    "Terminal session — activity not tracked here",
  );
  assert.equal(
    restingText("SAIENT_THINKING", true, false),
    "Terminal session — activity not tracked here",
  );
});

test("restingText still reports sleep where the state is real", () => {
  assert.equal(restingText("IDLE", true), "Saient is sleeping");
  assert.equal(restingText("IDLE", true, true), "Saient is sleeping");
  assert.notEqual(restingText("IDLE", false), "Saient is sleeping");
});

console.log(`turnState.test.js — ${passed} passed`);
