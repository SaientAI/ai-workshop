import assert from "node:assert/strict";
import { readFileSync } from "node:fs";

const settings = readFileSync(new URL("./components/SettingsModal.svelte", import.meta.url), "utf8");
const app = readFileSync(new URL("./App.svelte", import.meta.url), "utf8");
const wizard = readFileSync(new URL("./components/SetupWizard.svelte", import.meta.url), "utf8");
const setup = readFileSync(new URL("../src-tauri/src/setup.rs", import.meta.url), "utf8");
const resolve = readFileSync(new URL("../src-tauri/src/resolve.rs", import.meta.url), "utf8");
const imggen = readFileSync(new URL("../src-tauri/src/imggen.rs", import.meta.url), "utf8");

assert.match(settings, /type Tab = "internet" \| "setup" \| "security"/,
  "Settings must expose a dedicated Setup section");
assert.match(settings, /"Run setup again"/,
  "Settings must expose the setup recovery action");
assert.match(settings, /await onSetup\(\)/,
  "the recovery action must wait for the backend reset and surface failures");
assert.match(app, /async function reopenSetup\(\)[\s\S]*await T\.resetSetup\(\)[\s\S]*showSetup = true/,
  "the app must clear the marker before reopening the wizard");
assert.match(setup, /remove_setup_marker\(&setup_marker\(\)\)/,
  "reset_setup must use the checked marker-removal path");
assert.match(wizard, /confirmingSkip[\s\S]*Skip without installing/,
  "skipping setup must require a deliberate second action");
assert.doesNotMatch(wizard, /T\.skipSetup\(\)\.catch/,
  "skip marker write failures must not be hidden");
assert.match(resolve, /Open Settings → Setup and run Full setup/,
  "missing Image Gen dependencies must explain the recovery route");
assert.match(imggen, /resolve::find_image_python\(\)/,
  "Image Gen must preflight its creative Python modules");
assert.match(settings, /keeps existing models, settings, downloads, and environment files/,
  "Settings must accurately state that recovery is non-destructive");
assert.match(settings, /\.settings-card\s*\{[\s\S]*display:\s*flex;\s*flex-direction:\s*column/,
  "the Settings card must constrain its body within the viewport");
assert.match(settings, /\.settings-body\s*\{[\s\S]*min-height:\s*0;[\s\S]*overflow:\s*hidden/,
  "the Settings body must allow the panel to shrink and scroll");

console.log("setupRecovery.test.js — 12 passed");
