import assert from "node:assert/strict";
import { readFileSync } from "node:fs";

const read = (path) => readFileSync(new URL(path, import.meta.url), "utf8");
const bridge = read("../src-tauri/resources/saient/binding_bridge.py");
const rust = read("../src-tauri/src/binding.rs");
const sidebar = read("./components/chat/Sidebar.svelte");
const input = read("./components/chat/InputBar.svelte");
const title = read("./components/TitleBar.svelte");

assert.match(bridge, /def require_binding\(/,
  "the runtime needs a manifest-only operation for user inference");
assert.match(bridge, /manifest, path = require_binding\(endpoint, directory\)/,
  "bound_chat must fail fast instead of profiling inside a user turn");
assert.doesNotMatch(
  bridge.match(/def bound_chat[\s\S]*?^\s*return \{/m)?.[0] ?? "",
  /ensure_binding\(/,
  "bound_chat must never call the profiling path");
assert.match(rust, /pub async fn require\(/,
  "Rust user-operation paths need the manifest-only binding call");
assert.match(rust, /let manifest = self\.require\(port\)\.await\?/,
  "planner context must not profile while waiting for inference");
assert.match(sidebar, /Binding Saient to this model/,
  "formal profiling must be a visible model phase");
assert.match(sidebar, /await bindSaientModel\(\)/,
  "model load must finish explicit binding before enabling chat");
assert.match(input, /model\.bindingStatus === "bound"/,
  "Saient chat must stay disabled until explicit binding succeeds");
assert.match(input, /recordBoundaryClean: reply\.record_boundary_clean/,
  "the UI must retain proof that returned text matched the completed tick");
assert.match(title, /!ui\.saientEnabled[\s\S]*?T\.stopGenerate\(\)/,
  "turning Saient off must stop an in-flight explicit bind");
assert.match(rust, /saient-binding-progress/,
  "formal binding must surface its real per-sample progress");

assert.match(bridge, /state_ownership_self_model_grounding/,
  "the reusable binding manifest must cover state ownership and self-model grounding");
assert.match(bridge, /state_ownership_nonce/,
  "formal binding must challenge exact hidden-state retrieval");
assert.match(bridge, /state_ownership_absence/,
  "formal binding must challenge calibrated UNKNOWN instead of inference");
assert.match(bridge, /state_ownership_relational_provenance/,
  "formal binding must keep historical action provenance atomic");
assert.match(bridge, /state_field_boundary_clean/,
  "chat must enforce exact queried state at the final output boundary");

console.log("bindingLifecycle.test.js — 16 passed");
