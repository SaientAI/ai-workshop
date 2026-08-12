import assert from "node:assert/strict";
import { existsSync, readFileSync } from "node:fs";

const root = new URL("../", import.meta.url);
const config = JSON.parse(readFileSync(new URL("../src-tauri/tauri.conf.json", import.meta.url), "utf8"));
const main = readFileSync(new URL("../src-tauri/src/main.rs", import.meta.url), "utf8");
const engine = readFileSync(new URL("../src-tauri/src/engine.rs", import.meta.url), "utf8");
const binding = readFileSync(new URL("../src-tauri/resources/saient/binding_bridge.py", import.meta.url), "utf8");

assert.equal(existsSync(new URL("../src-tauri/src/remote.rs", import.meta.url)), false,
  "the removed phone/desktop HTTP service must not return");
assert.doesNotMatch(main, /mod remote|remote::|start_remote_server|pairing_qr/,
  "desktop commands must not re-register the removed HTTP service");
assert.doesNotMatch(config.app.security.csp, /https?:\/\/|wss?:\/\//,
  "the production WebView CSP must not expose a localhost or external network client");
assert.match(config.app.security.devCsp, /127\.0\.0\.1:1421/,
  "only the development CSP may name the Vite listener");
assert.match(engine, /const DESKTOP_BIND_HOST: &str = "127\.0\.0\.1"/,
  "the bundled model listener must have a fixed numeric loopback parent policy");
assert.match(engine, /\.env\("TINYQ4_BIND", DESKTOP_BIND_HOST\)/,
  "the parent must override inherited model bind settings");
assert.match(binding, /if not address\.is_loopback:/,
  "formal binding must reject a selected model outside loopback");
assert.match(binding, /model endpoint must use a numeric loopback address/,
  "formal binding must not delegate endpoint resolution to DNS or a proxy");

console.log("networkSurface.test.js — 8 passed");
