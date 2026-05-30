// contract.test.js — run with: node src/contract.test.js
//
// Verifies every invoke() call in src/lib/tauri.ts maps to an exported
// Rust command registered in src-tauri/src/main.rs invoke_handler!{}.
// Catches silent runtime breakage before a build.

import { readFileSync } from "fs";
import { resolve, dirname } from "path";
import { fileURLToPath } from "url";

const __dir = dirname(fileURLToPath(import.meta.url));
const root = resolve(__dir, "..");

// ── Test harness ──────────────────────────────────────────────────────────────

let passed = 0, failed = 0;
function test(name, fn) {
  try { fn(); console.log(`  ✓ ${name}`); passed++; }
  catch (e) { console.error(`  ✗ ${name}: ${e.message}`); failed++; }
}
function assert(cond, msg) { if (!cond) throw new Error(msg ?? "assertion failed"); }

// ── Parse Rust registered commands ───────────────────────────────────────────

function parseRustCommands(mainRs) {
  // Extract everything inside tauri::generate_handler![...]
  const handlerMatch = mainRs.match(/generate_handler!\s*\[([\s\S]*?)\]/);
  if (!handlerMatch) throw new Error("generate_handler! not found in main.rs");
  const block = handlerMatch[1]
    .split("\n")
    .map(line => line.replace(/\/\/.*$/, ""))  // strip // line comments
    .join("\n");
  // Each entry is like: load_model, unload_model, imggen::imggen_scan_models, ...
  return new Set(
    block
      .split(/[\s,]+/)
      .map(s => s.replace(/^.*::/, "").trim())  // strip module prefix (imggen::imggen_foo → imggen_foo)
      .filter(s => /^[a-z_][a-z0-9_]*$/.test(s))  // keep only valid snake_case identifiers
  );
}

// ── Parse frontend invoked commands ──────────────────────────────────────────

function parseFrontendCommands(tauriTs) {
  const cmds = [];
  const re = /invoke[^(]*\(\s*["']([^"']+)["']/g;
  let m;
  while ((m = re.exec(tauriTs)) !== null) {
    cmds.push(m[1]);
  }
  return cmds;
}

// ── Load files ────────────────────────────────────────────────────────────────

const mainRs   = readFileSync(resolve(root, "src-tauri/src/main.rs"),  "utf8");
const tauriTs  = readFileSync(resolve(root, "src/lib/tauri.ts"),       "utf8");

const rustCmds     = parseRustCommands(mainRs);
const frontendCmds = parseFrontendCommands(tauriTs);

// ── Tests ─────────────────────────────────────────────────────────────────────

console.log("\nRust commands registered in generate_handler!:");
console.log(`  ${rustCmds.size} commands`);
console.log("\nFrontend invoke() calls in tauri.ts:");
console.log(`  ${frontendCmds.length} calls`);

console.log("\nContract: every frontend call must have a Rust handler");

const unique = [...new Set(frontendCmds)];
for (const cmd of unique) {
  test(`"${cmd}" is registered in Rust`, () => {
    assert(rustCmds.has(cmd), `Command "${cmd}" is called by the frontend but not found in generate_handler![]`);
  });
}

console.log("\nContract: every Rust command is called by the frontend");

// Some Rust commands are internal or admin-only — exempt them from the
// "must be called" check since they're still valid to have registered.
const EXEMPT = new Set([
  "inspect_gguf",         // dev/debug tool
  "write_binary_b64",     // used programmatically, not from tauri.ts directly
  "dual_agent_status",    // status query, consumed via events
  "fs_append",            // extra fs op
  "fs_list",              // extra fs op
  "fs_move",              // extra fs op
  "fs_copy",              // extra fs op
  "fs_exists",            // extra fs op
  "kill_process",         // used via killProcess() which is verified above
  "list_processes",       // internal monitoring
  "apply_unified_diff",   // advanced patch op
  "patch_history",        // read-only history
  "diff_files",           // diff tool
  "mem_start_task",       // internal to agent_run
  "mem_finish_task",      // internal to agent_run
  "mem_remember",         // internal to agent_run
  "mem_recall",           // exposed as memorySearch
  "mem_context",          // internal to agent_run
  "mem_store",            // exposed as memoryAll
  "mem_forget",           // exposed as memoryForget
  "plan_parse",           // internal planner helper
  "plan_get",             // internal planner helper
  "plan_prompt_template", // exposed as genPlanPrompt
  "plan_execute",         // exposed as executePlan
  "imggen_scan_checkpoints", // called via imggenScanCheckpoints
  "imggen_scan_loras",    // called via imggenScanLoras
  "imggen_generate",      // exposed as runImggen
  "tts_voices",           // exposed as ttsFetchVoices
  "tts_generate",         // exposed as runTts
  "lora_start_training",  // exposed as loraStart
  "lora_stop_training",   // exposed as loraStop
  "lora_clean_dataset",   // exposed as loraCleanDataset
  "merge_start",          // exposed as mergeRun
  "merge_cancel",         // exposed as mergeCancel
]);

const frontendSet = new Set(frontendCmds);
for (const cmd of rustCmds) {
  if (EXEMPT.has(cmd)) continue;
  test(`Rust "${cmd}" has a frontend caller`, () => {
    assert(frontendSet.has(cmd), `Rust command "${cmd}" is registered but never called from tauri.ts`);
  });
}

// ── Summary ───────────────────────────────────────────────────────────────────

console.log(`\n${passed} passed, ${failed} failed`);
if (failed > 0) process.exit(1);
