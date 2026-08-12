import assert from "node:assert/strict";
import { readFileSync } from "node:fs";

const wizard = readFileSync(new URL("./components/SetupWizard.svelte", import.meta.url), "utf8");
const backend = readFileSync(new URL("../src-tauri/src/internet.rs", import.meta.url), "utf8");
const setup = readFileSync(new URL("../src-tauri/src/setup.rs", import.meta.url), "utf8");

assert.match(wizard, /step === "network"/,
  "the setup modal itself must expose the network authorization state");
assert.match(wizard, /Allow for setup and continue/,
  "setup network access must require an explicit user action");
assert.match(wizard, /setSetupInternetAuthorized\(true\)/,
  "the explicit action must grant setup-scoped authority");
assert.match(wizard, /onDestroy\(\(\) => \{ void T\.setSetupInternetAuthorized\(false\); \}\)/,
  "closing setup must revoke temporary authority");
assert.match(wizard, /await releaseSetupNetwork\(\);\s*onDone\(\)/,
  "finishing setup must revoke temporary authority before leaving");
assert.match(backend, /static SETUP_INTERNET_AUTHORIZED: AtomicBool/,
  "setup authority must not reuse the durable runtime preference");
assert.match(backend, /SETUP_INTERNET_AUTHORIZED\.store\(false/,
  "startup must fail closed after a crash or restart");
assert.match(setup, /require_setup_enabled\("Full setup downloads"\)/,
  "Full Setup must enforce the scoped backend gate");
assert.match(setup, /require_setup_enabled\("Starter model download"\)/,
  "the in-wizard model download must enforce the same gate");
assert.match(setup, /"moondream\/starmie-v1"/,
  "Full Setup must download Moondream's separate runtime tokenizer");
assert.match(setup, /managed runtime asset missing after download/,
  "Full Setup must validate required files in the managed assets directory");
assert.match(setup, /This deliberately does not import huggingface_hub/,
  "Full Setup must not use the shared Hugging Face cache downloader");
assert.match(setup, /"transformers==4\.52\.4"/,
  "Full Setup must retain the Transformers version declared by the bundled Moondream revision");

console.log("setupNetwork.test.js — 13 passed");
