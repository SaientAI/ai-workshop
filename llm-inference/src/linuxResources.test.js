import assert from "node:assert/strict";
import { readFileSync } from "node:fs";

const common = JSON.parse(readFileSync(new URL("../src-tauri/tauri.conf.json", import.meta.url), "utf8"));
const linux = JSON.parse(readFileSync(new URL("../src-tauri/tauri.linux.conf.json", import.meta.url), "utf8"));
assert.ok(common.bundle.linux.deb.depends.includes("python3 (>= 3.10)"),
  "The packaged agent and binding runtime require Python 3.10+ on fresh Linux installs");
assert.equal(common.bundle.resources["resources/saient/"], "resources/saient/");
for (const name of ["tinyq4-cpu", "tinyq4-cuda", "libcudart.so.12"]) {
  assert.equal(linux.bundle.resources[`resources/engine/${name}`], `resources/engine/${name}`);
}
console.log("linuxResources.test.js — 5 passed");
