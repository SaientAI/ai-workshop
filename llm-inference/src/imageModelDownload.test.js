import assert from "node:assert/strict";
import { readFileSync } from "node:fs";

const browser = readFileSync(new URL("./components/HfBrowser.svelte", import.meta.url), "utf8");
const imageScreen = readFileSync(new URL("./components/screens/ImageGenScreen.svelte", import.meta.url), "utf8");
const resolve = readFileSync(new URL("../src-tauri/src/resolve.rs", import.meta.url), "utf8");
const setup = readFileSync(new URL("../src-tauri/src/setup.rs", import.meta.url), "utf8");

assert.match(imageScreen, /const BASE_MODELS = \[/,
  "curated entries must be identified as base models, not checkpoints");
assert.equal((imageScreen.match(/install: "repo" as const/g) ?? []).length, 5,
  "every curated base model must request a full repository install");
assert.match(browser, /s\.install === "repo"[\s\S]*downloadRepo\(s\.repo, "model"\)/,
  "a curated repository suggestion must bypass checkpoint-file listing");
assert.match(browser, /Suggested base models — install Diffusers folder/,
  "the browser must describe the storage format it installs");
assert.match(browser, /Search results let you choose a tuned single-file checkpoint/,
  "ordinary search must remain available for checkpoint downloads");
assert.match(resolve, /pub fn image_models_download_dir\(\)[\s\S]*image_models_dir\(\)/,
  "full image repositories must resolve into models/image");
assert.match(setup, /image_models_download_dir\(\)\.join\(folder\)/,
  "the repository downloader must use the Image Gen scan directory");
assert.doesNotMatch(setup, /crate::resolve::models_download_dir\(\)\.join\(folder\)/,
  "the repository downloader must not hide image models in the general models root");

console.log("imageModelDownload.test.js — 8 passed");
