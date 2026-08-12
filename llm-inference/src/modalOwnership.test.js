import assert from "node:assert/strict";
import { readFileSync } from "node:fs";

const app = readFileSync(new URL("./App.svelte", import.meta.url), "utf8");
const pulse = readFileSync(new URL("./components/SaientPulse.svelte", import.meta.url), "utf8");
const autonomy = readFileSync(new URL("./components/AutonomyConfirm.svelte", import.meta.url), "utf8");
const shortcuts = readFileSync(new URL("./lib/shortcuts.ts", import.meta.url), "utf8");
const source = app + "\n" + pulse;
const mounts = source.match(/<AutonomyConfirm\b/g) ?? [];

assert.equal(mounts.length, 1, "the mode dialog must have exactly one DOM owner");
assert.doesNotMatch(pulse, /import\s+AutonomyConfirm|<AutonomyConfirm\b/,
  "the status bar may request the dialog but must not mount its own copy");
assert.match(pulse, /onLevelRequest/, "the status bar must route requests to the owner");
assert.match(shortcuts, /T\.saientSetEnabled\(/,
  "the keyboard master switch must update the backend loop, not only UI state");
assert.match(shortcuts, /needsLoop\(effectiveAgiLevel\(/,
  "keyboard loop synchronization must preserve per-project mode semantics");
assert.match(autonomy, /onDismiss:\s*\(\)\s*=>\s*void/,
  "the mode dialog must expose a non-mutating dismiss action");
assert.match(autonomy, /aria-label="Close autonomy settings"/,
  "the mode dialog must render an accessible close button");
assert.match(app, /onDismiss=\{\(\)\s*=>\s*\{/,
  "the dialog owner must handle dismiss separately from changing the level");

console.log("modalOwnership.test.js — 8 passed");
