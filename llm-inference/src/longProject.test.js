import assert from "node:assert/strict";
import test from "node:test";
import { buildLongProjectCommand } from "./lib/longProject.ts";

const options = { goal: "Build the project", hours: 24, steps: 10000, toolTimeout: 3600,
  allowWrite: false, allowShell: false, verifyCommand: "" };

test("project input is one JSON-bearing line with explicit permission defaults", () => {
  const goal = 'Build C:\\Users\\tiny\\Desktop\\工程\nwith "quotes", `backticks` and $(commands)';
  const input = buildLongProjectCommand({ ...options, goal });
  assert.equal(input.split("\n").length, 1);
  const payload = JSON.parse(input.slice("/project start ".length));
  assert.equal(payload.goal, goal);
  assert.equal(payload.allow_shell, false);
  assert.equal(payload.allow_write, false);
});

test("days-long limits and explicit acceptance are retained", () => {
  const payload = JSON.parse(buildLongProjectCommand({ ...options, hours: 72, verifyCommand: "npm test" }).slice(15));
  assert.equal(payload.hours, 72);
  assert.equal(payload.verify_command, "npm test");
});

test("invalid limits and empty goals cannot generate start commands", () => {
  for (const invalid of [{ goal: "  " }, { hours: Infinity }, { hours: 0 }, { hours: 8785 },
    { steps: 1.2 }, { steps: 0 }, { toolTimeout: NaN }, { toolTimeout: 604801 }, { allowShell: "yes" }]) {
    assert.throws(() => buildLongProjectCommand({ ...options, ...invalid }));
  }
});
