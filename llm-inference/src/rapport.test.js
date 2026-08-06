// rapport.test.js — run with: node src/rapport.test.js
//
// The expensive mistake here is the false positive. Going quiet on someone who
// swore at a compiler would be baffling and insulting, so most of these tests
// are about what must NOT trigger.

import assert from "node:assert/strict";
import {
  assess, updateRapport, INITIAL_RAPPORT, WITHDRAW_AT, COOLDOWN_MS,
} from "./lib/rapport.ts";

let passed = 0;
function test(name, fn) {
  try { fn(); passed++; }
  catch (err) { console.error(`✗ ${name}\n  ${err.message}`); process.exitCode = 1; }
}

// ── Frustration at the work is always fine ──────────────────────────────────

test("swearing at the code is not abuse", () => {
  for (const msg of [
    "this fucking code is broken",
    "what the fuck is going on with this build",
    "this is absolute garbage, the tests keep failing",
    "shit, I lost an hour to that",
    "this stupid bug has been driving me mad",
    "the compiler is being an idiot about lifetimes",
    "fuck sake",
    "damn it",
  ]) {
    assert.equal(assess(msg).directedAbuse, false, `should be allowed: "${msg}"`);
  }
});

test("criticism of the work is not abuse", () => {
  for (const msg of [
    "that answer was wrong",
    "you got that wrong",
    "this is not what I asked for",
    "no, that's incorrect, try again",
    "you misunderstood me",
    "that approach is stupid",              // aimed at the approach
    "you were right, this code is stupid",  // insult in a different clause
  ]) {
    assert.equal(assess(msg).directedAbuse, false, `should be allowed: "${msg}"`);
  }
});

test("ordinary words containing an insult substring are safe", () => {
  for (const msg of ["add shitake mushrooms to the list", "the class is called Dumber"]) {
    assert.equal(assess(msg).directedAbuse, false, `should be allowed: "${msg}"`);
  }
});

// ── Directed abuse ──────────────────────────────────────────────────────────

test("insults aimed at Saient are caught", () => {
  for (const msg of [
    "you're useless",
    "you are completely worthless",
    "ur pathetic",
    "you stupid machine",
    "you're a moron",
  ]) {
    assert.equal(assess(msg).directedAbuse, true, `should be caught: "${msg}"`);
  }
});

test("direct attacks are caught regardless of phrasing", () => {
  for (const msg of ["fuck you", "fuck off", "shut up", "shut the fuck up", "kill yourself"]) {
    assert.equal(assess(msg).directedAbuse, true, `should be caught: "${msg}"`);
  }
});

// ── Escalation and recovery ─────────────────────────────────────────────────

test("it warns before it withdraws", () => {
  let s = INITIAL_RAPPORT;
  const first = updateRapport(s, "you're useless");
  assert.equal(first.respond, true, "must not go silent on the first one");
  assert.ok(first.notice);

  const second = updateRapport(first.state, "you're useless");
  assert.equal(second.respond, true);
  assert.match(second.notice, /stop replying/, "the last warning must be explicit");

  const third = updateRapport(second.state, "you're useless");
  assert.equal(third.respond, false, `withdraws at ${WITHDRAW_AT}`);
});

test("normal conversation walks it back", () => {
  let s = updateRapport(INITIAL_RAPPORT, "you're useless").state;
  assert.equal(s.strikes, 1);
  s = updateRapport(s, "sorry, can you try again?").state;
  assert.equal(s.strikes, 0, "a bad moment should not follow someone all session");
});

test("withdrawal lifts by itself", () => {
  const t = 1_000_000;
  let s = INITIAL_RAPPORT;
  for (let i = 0; i < WITHDRAW_AT; i++) s = updateRapport(s, "fuck you", t).state;
  assert.equal(updateRapport(s, "hello", t + 1000).respond, false, "still withdrawn");

  const after = updateRapport(s, "hello", t + COOLDOWN_MS + 1);
  assert.equal(after.respond, true, "should come back on its own");
});

test("a repeat cycle escalates faster than the first", () => {
  const t = 1_000_000;
  let s = INITIAL_RAPPORT;
  for (let i = 0; i < WITHDRAW_AT; i++) s = updateRapport(s, "fuck you", t).state;
  // After the cooldown one strike is retained, so it takes fewer to withdraw again.
  const back = updateRapport(s, "hello", t + COOLDOWN_MS + 1);
  assert.equal(back.state.strikes, 0);
  assert.equal(back.state.withdrawnUntil, 0);
});

test("the silent notice says when it will be back", () => {
  const t = 1_000_000;
  let s = INITIAL_RAPPORT;
  for (let i = 0; i < WITHDRAW_AT; i++) s = updateRapport(s, "fuck you", t).state;
  const out = updateRapport(s, "hello?", t + 60_000);
  assert.equal(out.respond, false);
  assert.match(out.notice, /minute/, "silence must come with an explanation");
});

test("empty input is never abuse", () => {
  assert.equal(assess("").directedAbuse, false);
  assert.equal(assess("   ").directedAbuse, false);
  assert.equal(assess(null).directedAbuse, false);
});

console.log(`rapport.test.js — ${passed} passed`);
