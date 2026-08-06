// pulse.test.js — run with: node src/pulse.test.js
//
// The activity bar's job is to be true. These guard the two rules that make it
// worth having: the animation follows real events, and the main line is factual
// rather than flavour.

import assert from "node:assert/strict";
import {
  animationFor,
  activityLine,
  flavourFor,
  formatElapsed,
  shorten,
  pushActivity,
  ACTIVITY_LIMIT,
} from "./lib/pulse.ts";
import { TURN_STATES, activityText } from "./lib/turnState.ts";

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

// ── Animation follows real state ────────────────────────────────────────────

test("every turn state maps to an animation", () => {
  for (const s of TURN_STATES) {
    assert.ok(animationFor(s), `${s} has no animation`);
  }
});

test("resting states do not animate as if working", () => {
  for (const s of ["IDLE", "USER_TYPING", "INTERRUPTED"]) {
    assert.equal(animationFor(s), "idle", `${s} should look at rest`);
  }
});

test("the robot reflects the kind of work, not just that work exists", () => {
  assert.equal(animationFor("SAIENT_ACTING", { tool: "fs_read" }), "reading");
  assert.equal(animationFor("SAIENT_ACTING", { tool: "fs_write" }), "saving");
  assert.equal(animationFor("SAIENT_ACTING", { tool: "exec" }), "typing");
  assert.equal(animationFor("SAIENT_ACTING", { tool: "security_scan" }), "scanning");
  assert.equal(animationFor("VERIFYING"), "verifying");
  assert.equal(animationFor("FAILED"), "failed");
  assert.equal(animationFor("COMPLETED"), "completed");
});

test("tool detail cannot strand the robot mid-gesture", () => {
  // A stale tool from a finished step must not keep the reading animation
  // running once the state has moved on.
  assert.equal(animationFor("COMPLETED", { tool: "fs_read" }), "completed");
  assert.equal(animationFor("IDLE", { tool: "exec" }), "idle");
});

// ── The main line is factual ────────────────────────────────────────────────

test("names the actual file being written", () => {
  const line = activityLine("SAIENT_ACTING", { tool: "fs_write", target: "src/runtime.rs" }, "x");
  assert.equal(line, "Writing src/runtime.rs");
});

test("names the actual command being run", () => {
  const line = activityLine("WAITING_FOR_TOOL", { tool: "exec", target: "cargo test" }, "x");
  assert.equal(line, "Running cargo test");
});

test("degrades to less specific text rather than inventing detail", () => {
  // Verb but no subject.
  assert.equal(activityLine("SAIENT_ACTING", { tool: "fs_read" }, "fallback"), "Reading");
  // Neither: fall through to the state's own description.
  assert.equal(activityLine("SAIENT_ACTING", {}, "fallback"), "fallback");
  assert.equal(activityLine("SAIENT_THINKING", undefined, "Waiting for model response"),
    "Waiting for model response");
});

test("no state produces an empty main line", () => {
  for (const s of TURN_STATES) {
    const line = activityLine(s, undefined, activityText(s));
    assert.ok(line && line.trim().length > 0, `${s} produced an empty line`);
  }
});

test("long paths are trimmed keeping the informative end", () => {
  const long = "crates/engine/src/very/deeply/nested/module/scheduler.rs";
  const out = shorten(long, 24);
  assert.ok(out.length <= 24, `too long: ${out}`);
  assert.ok(out.endsWith("scheduler.rs"), `lost the filename: ${out}`);
  assert.equal(shorten("short.rs", 24), "short.rs");
});

// ── Flavour stays in its place ──────────────────────────────────────────────

test("flavour never leaks into the factual line", () => {
  const line = activityLine("SAIENT_ACTING", { tool: "fs_write", target: "a.rs" }, "x");
  const flavour = flavourFor("saving", 0);
  assert.notEqual(line, flavour);
  assert.ok(!line.includes("…") || line.startsWith("…"), "factual line reads like flavour");
});

test("flavour is stable for one activity, not per render", () => {
  const t = 1_700_000_000_000;
  assert.equal(flavourFor("thinking", t), flavourFor("thinking", t));
});

test("every animation has flavour available", () => {
  for (const a of ["idle","thinking","typing","reading","scanning","saving","verifying","failed","completed"]) {
    assert.ok(flavourFor(a, 0).length > 0, `${a} has no flavour`);
  }
});

// ── Clock and log ───────────────────────────────────────────────────────────

test("elapsed reads as mm:ss, growing to hours only when needed", () => {
  assert.equal(formatElapsed(0), "00:00");
  assert.equal(formatElapsed(18_000), "00:18");
  assert.equal(formatElapsed(95_000), "01:35");
  assert.equal(formatElapsed(3_661_000), "1:01:01");
  assert.equal(formatElapsed(-5), "00:00");
});

test("the activity log is capped so a long run cannot grow forever", () => {
  const log = [];
  for (let i = 0; i < ACTIVITY_LIMIT + 50; i++) {
    pushActivity(log, { at: i, text: `e${i}`, animation: "typing" });
  }
  assert.equal(log.length, ACTIVITY_LIMIT);
  // The oldest are dropped, the newest kept.
  assert.equal(log[log.length - 1].text, `e${ACTIVITY_LIMIT + 49}`);
});

console.log(`pulse.test.js — ${passed} passed`);
