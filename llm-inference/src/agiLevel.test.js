/**
 * Regression contract for the four Saient execution levels.
 *
 * Extract the deliberately pure level helpers from the TypeScript module so
 * this remains runnable in the dependency-light contract suite.
 */
import assert from "node:assert/strict";
import fs from "node:fs";
import vm from "node:vm";

const source = fs.readFileSync(new URL("./lib/agiLevel.ts", import.meta.url), "utf8");
const start = source.indexOf("export const AGI_LEVEL_INFO");
const end = source.indexOf("/**\n * What swapping the model", start);
assert.ok(start >= 0 && end > start, "level contract block is present");

const runnable = source
  .slice(start, end)
  .replace(/export /g, "")
  .replace(/: Record<AgiLevel, AgiLevelInfo>/, "")
  .replace(/: AgiLevel =/, " =")
  .replace(/\(level: AgiLevel\)/g, "(level)")
  .replace(/\): boolean/g, ")")
  .replace(/\(value: unknown\): value is AgiLevel/g, "(value)")
  .replace(/\(value: unknown\): AgiLevel/g, "(value)")
  .replace(/ as readonly string\[\]/g, "")
  .concat("\nresult = { AGI_LEVEL_INFO, needsLoop, actsUnprompted, parseAgiLevel };\n");

const sandbox = { result: null, AGI_LEVELS: ["off", "guided", "companion", "autonomous"] };
vm.runInNewContext(runnable, sandbox);
const { AGI_LEVEL_INFO, needsLoop, actsUnprompted, parseAgiLevel } = sandbox.result;

assert.equal(needsLoop("off"), false, "off has no persistent Saient loop");
assert.equal(needsLoop("guided"), false, "guided proposes goals without a background loop");
assert.equal(needsLoop("companion"), true, "companion retains the persistent Saient loop");
assert.equal(needsLoop("autonomous"), true, "autonomous retains the persistent Saient loop");
assert.equal(actsUnprompted("autonomous"), true, "only autonomous acts without a prompt");
assert.equal(actsUnprompted("companion"), false, "companion does not gain autonomous semantics");
assert.equal(parseAgiLevel("corrupt-value"), "off", "unknown persisted levels fail closed");
for (const info of Object.values(AGI_LEVEL_INFO)) {
  assert.equal(info.toolGuardsApply, true, `${info.id} preserves tool guards`);
}

console.log("agiLevel tests: 11 passed");
