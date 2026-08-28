import assert from "node:assert/strict";
import { readFileSync } from "node:fs";

const read = (p) => readFileSync(new URL(p, import.meta.url), "utf8");
const hflink = read("../src-tauri/src/hflink.rs");
const main = read("../src-tauri/src/main.rs");
const app = read("./App.svelte");
const importer = read("./components/HfImport.svelte");
const config = JSON.parse(read("../src-tauri/tauri.conf.json"));
const caps = JSON.parse(read("../src-tauri/capabilities/default.json"));

// ── The scheme is declared where the bundler can register it ──────────────────
assert.deepEqual(config.plugins["deep-link"].desktop.schemes, ["saient"],
  "the installed build must claim exactly the saient:// scheme");
assert.ok(caps.permissions.includes("deep-link:default"),
  "the main window needs the deep-link capability");
// A .desktop handler without %u is launched with NO argument — measured, not
// assumed — so the packaged build must use the template that adds it.
const desktopTemplate = read("../src-tauri/linux/main.desktop");
assert.equal(config.bundle.linux.deb.desktopTemplate, "linux/main.desktop",
  "the deb must use the template that passes the URL through");
assert.match(desktopTemplate, /^Exec=\{\{exec\}\} %u$/m,
  "the launcher entry must take a URL argument, or the scheme silently does nothing");
assert.match(desktopTemplate, /^StartupWMClass=\{\{exec\}\}$/m,
  "StartupWMClass must stay the bare executable name");
assert.match(desktopTemplate, /MimeType=\{\{mime_type\}\}/,
  "the template must keep the bundler's scheme registration");
assert.match(main, /tauri_plugin_single_instance::init/,
  "a second launch must hand its link to the running app, not open a rival instance");
assert.match(main, /single_instance::init[\s\S]{0,400}?tauri_plugin_deep_link::init/,
  "single-instance must be registered before the other plugins");
assert.match(main, /hflink::hf_pending_deeplink/,
  "the pending-link command must be registered");

// ── A link is untrusted input: it may not carry network authority ─────────────
assert.doesNotMatch(hflink, /set_internet_enabled|set_setup_authorized|set_update_authorized|deeplink_authorized/,
  "a deep link must never touch the Internet gate — any web page can send one");
assert.doesNotMatch(hflink, /reqwest|resolve\/main|api\/models/,
  "the link handler must not fetch anything itself; downloads stay in setup.rs behind their gate");
assert.match(hflink, /fn valid_repo/,
  "the repo id must be validated in Rust, not only in the UI");
assert.match(hflink, /is_ascii_alphanumeric\(\)/,
  "repo segments must be restricted to a known-safe character set");

// ── One link opens one prompt, and the prompt asks first ──────────────────────
assert.match(hflink, /slot\.take\(\)/, "a pending link must be consumed, not replayed");
assert.match(app, /"hf-deeplink"/, "the app must listen for the live link event");
assert.match(app, /hfPendingDeeplink\(\)/, "the app must also drain a link that arrived before it mounted");
assert.match(app, /<HfImport\b/, "the import prompt must have a DOM owner");
assert.equal((app.match(/<HfImport\b/g) ?? []).length, 1,
  "the import prompt must have exactly one DOM owner");
assert.match(importer, /T\.hfListGguf\(/,
  "the import must go through the strictly gated file listing before downloading");
assert.match(importer, /Nothing is downloaded until you pick a file/,
  "the prompt must state that it does not auto-download");
assert.doesNotMatch(importer, /onMount|\$effect/,
  "the prompt must not start work on its own — the user picks a file");

console.log("hfDeeplink.test.js — 20 passed");
