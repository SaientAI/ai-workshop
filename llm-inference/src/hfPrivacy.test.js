import assert from "node:assert/strict";
import { readFileSync } from "node:fs";

const paths = readFileSync(new URL("../src-tauri/src/paths.rs", import.meta.url), "utf8");
const setup = readFileSync(new URL("../src-tauri/src/setup.rs", import.meta.url), "utf8");
const settings = readFileSync(new URL("./components/SettingsModal.svelte", import.meta.url), "utf8");
const pythonPaths = readFileSync(new URL("../../scripts/saient_paths.py", import.meta.url), "utf8");
const tts = readFileSync(new URL("../../scripts/tts_kokoro.py", import.meta.url), "utf8");
const vision = readFileSync(new URL("../../scripts/vision.py", import.meta.url), "utf8");
const imggen = readFileSync(new URL("../src-tauri/src/imggen.rs", import.meta.url), "utf8");
const devLauncher = readFileSync(new URL("../../launch-dev.sh", import.meta.url), "utf8");

const childEnv = paths.slice(paths.indexOf("pub fn apply_child_env"), paths.indexOf("pub fn legacy_config_dir"));
assert.match(childEnv, /\.env\("HF_HUB_OFFLINE", "1"\)/,
  "every runtime child must force Hugging Face Hub offline");
assert.match(childEnv, /\.env\("TRANSFORMERS_OFFLINE", "1"\)/,
  "every runtime child must force Transformers offline");
assert.match(childEnv, /\.env\("HF_DATASETS_OFFLINE", "1"\)/,
  "every runtime child must force datasets offline");
assert.doesNotMatch(childEnv, /if !crate::internet::enabled/,
  "runtime library isolation must not depend on the general Internet switch");

assert.match(paths, /data_dir\(\)\.join\("runtime-assets"\)/,
  "named voice and vision assets must have a visible managed directory");
assert.match(paths, /data_dir\(\)\.join\("runtime-tmp"\)\.join\("huggingface"\)/,
  "unavoidable library state must be isolated as temporary data");
assert.match(pythonPaths, /def configure_hf_cache\(\*, offline: bool = True\)/,
  "bundled Python scripts must default to offline library behavior");
assert.match(devLauncher, /SAIENT_DATA_DIR\/runtime-tmp\/huggingface/,
  "the development launcher must isolate library state as temporary data");
assert.doesNotMatch(devLauncher, /SAIENT_DATA_DIR\/huggingface}/,
  "the development launcher must not recreate the legacy cache");

const assetSetup = setup.slice(setup.indexOf("fn prefetch_runtime_assets"), setup.indexOf("/// Full = creative"));
assert.doesNotMatch(assetSetup, /snapshot_download|hf_hub_download/,
  "Full Setup must not populate a Hugging Face cache");
assert.match(assetSetup, /f3ff3571791e39611d31c381e3a41a3af07b4987/,
  "Kokoro runtime files must be revision-pinned");
assert.match(assetSetup, /6b714b26eea5cbd9f31e4edb2541c170afa935ba/,
  "Moondream runtime files must be revision-pinned");
assert.match(assetSetup, /part\.unlink\(\)/,
  "failed managed downloads must remove partial files");

assert.match(tts, /KModel\([\s\S]*config=str\(config_path\), model=str\(model_path\)/,
  "Kokoro must load its model from named local files");
assert.match(tts, /voice=str\(voice_path\)/,
  "Kokoro voices must load from named local files");
assert.match(vision, /str\(repo\), trust_remote_code=True, torch_dtype=dtype, local_files_only=True/,
  "Moondream must load a local managed directory in offline mode");
assert.match(vision, /Tokenizer\.from_file\(str\(starmie\)\)/,
  "Moondream's transitive tokenizer lookup must be redirected locally");
assert.match(vision, /for source in repo\.glob\("\*\.py"\):[\s\S]*shutil\.copy2/,
  "all pinned Moondream modules must be staged in run-only library state");

assert.doesNotMatch(imggen, /huggingface\.co/,
  "the Image Gen backend must not send prompts to Hugging Face");
assert.match(settings, /Clear legacy Hugging Face cache/,
  "Settings must expose legacy cache removal");
assert.match(settings, /Saient no longer uses a persistent Hugging Face cache/,
  "Settings must explain the durable storage boundary");

console.log("hfPrivacy.test.js — 20 passed");
