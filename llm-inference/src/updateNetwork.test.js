import assert from "node:assert/strict";
import { readFileSync } from "node:fs";

const box = readFileSync(new URL("./components/UpdateBox.svelte", import.meta.url), "utf8");
const app = readFileSync(new URL("./App.svelte", import.meta.url), "utf8");
const settings = readFileSync(new URL("./components/SettingsModal.svelte", import.meta.url), "utf8");
const internet = readFileSync(new URL("../src-tauri/src/internet.rs", import.meta.url), "utf8");
const update = readFileSync(new URL("../src-tauri/src/update.rs", import.meta.url), "utf8");
const main = readFileSync(new URL("../src-tauri/src/main.rs", import.meta.url), "utf8");

assert.match(internet, /static UPDATE_INTERNET_AUTHORIZED: AtomicBool/,
  "manual updates must have a capability separate from the durable Internet preference");
assert.match(internet, /UPDATE_INTERNET_AUTHORIZED\.store\(false/,
  "update authority must fail closed after restart");
assert.match(update, /require_update_enabled\("Update checks"\)/,
  "update checks must enforce their scoped backend capability");
assert.match(update, /require_update_enabled\("Installing updates"\)/,
  "update installation must enforce the same scoped backend capability");
assert.match(main, /set_update_internet_authorized/,
  "the scoped capability command must be registered with Tauri");
assert.match(box, /await grantUpdateNetwork\(\);[\s\S]*await T\.checkUpdate\(\)/,
  "Check now must explicitly grant temporary authority before the request");
assert.match(box, /finally \{[\s\S]*await releaseUpdateNetwork\(\)/,
  "temporary update authority must be revoked after requests");
assert.match(box, /onDestroy\(\(\) => \{ void releaseUpdateNetwork\(\); \}\)/,
  "closing the update dialog must revoke temporary authority");
assert.doesNotMatch(box, /setInternetEnabled/,
  "manual update checks must not change the durable agent Internet setting");
assert.doesNotMatch(box, /Couldn't reach the update server/,
  "the UI must not replace every backend error with a false server diagnosis");
assert.match(box, /Update check failed: \$\{raw/,
  "the updater must retain the real backend failure reason");
assert.match(box, /does not turn on Internet access for the agent/,
  "the dialog must explain the temporary capability boundary");
assert.doesNotMatch(app, /setUpdateInternetAuthorized/,
  "the silent startup check must never grant network authority");
assert.match(settings, /manual update check can request temporary update-only access/,
  "Settings copy must distinguish silent checks from explicit manual checks");

console.log("updateNetwork.test.js — 14 passed");
