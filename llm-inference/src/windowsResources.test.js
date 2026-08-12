import assert from "node:assert/strict";
import { readFileSync } from "node:fs";

const config = JSON.parse(readFileSync(new URL("../src-tauri/tauri.windows.conf.json", import.meta.url), "utf8"));
const resources = config.bundle.resources;
assert.equal(Array.isArray(resources), false,
  "Windows resources must use explicit source-to-package mappings");

const requiredScripts = [
  "clean_lora_dataset.py",
  "enhance_video.py",
  "generate_cogvideo.py",
  "generate_sdxl.py",
  "generate_video.py",
  "generate_wan_i2v.py",
  "merge_checkpoints.py",
  "saient_paths.py",
  "train_lora_sdxl.py",
  "tts_kokoro.py",
  "vision.py",
];

for (const script of requiredScripts) {
  assert.equal(resources[`../../scripts/${script}`], `resources/scripts/${script}`,
    `Windows package is missing ${script}`);
}
assert.equal(resources["resources/engine/tinyq4-cpu.exe"], "resources/engine/tinyq4-cpu.exe",
  "Windows CPU engine must remain bundled");
assert.equal(resources["resources/saient/"], "resources/saient/",
  "Windows autonomous helper resources must remain bundled");

console.log(`windowsResources.test.js — ${requiredScripts.length + 3} passed`);
