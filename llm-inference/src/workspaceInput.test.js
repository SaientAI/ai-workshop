import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { createContext, runInContext } from "node:vm";
import test from "node:test";
import ts from "typescript";

// Exercise the component's actual handlers and workspace effects with the Tauri
// boundary replaced by controllable promises. No shell or model is started.
const source = readFileSync(new URL("./components/screens/AgentScreen.svelte", import.meta.url), "utf8");
const script = source.match(/<script lang="ts">([\s\S]*?)<\/script>/)[1];
const parsed = ts.createSourceFile("AgentScreen.ts", script, ts.ScriptTarget.Latest, true);
const withoutImports = ts.factory.updateSourceFile(parsed, parsed.statements.filter((s) => !ts.isImportDeclaration(s)));
const compiled = ts.transpileModule(ts.createPrinter().printFile(withoutImports), {
  compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.None },
}).outputText;
const workspaceInput = source.match(/<input\b[^>]*aria-label="Workspace folder\.[\s\S]*?\/>/)[0];
const binding = workspaceInput.match(/bind:value=\{([^}]+)\}/)[1];
const changed = workspaceInput.match(/onchange=\{([^}]+)\}/)[1];
const deferred = () => {
  let resolve, reject;
  const promise = new Promise((yes, no) => { resolve = yes; reject = no; });
  return { promise, resolve, reject };
};
const folder = (path) => ({ path, name: path.split("/").at(-1), is_dir: true, depth: 0, children: [] });

async function harness(overrides = {}) {
  const calls = { roots: [], terminals: [], trees: [], toasts: [] };
  let backendRoot = "/workspace/original";
  let activeEffect = null;
  const subscriptions = new Map();
  const pending = new Set();
  const effects = [];
  const agent = new Proxy({
    sandboxRoot: backendRoot, workspaceEpoch: 1, tree: [], selPath: null,
    content: "", dirty: false, tab: "files", turn: "IDLE", continuing: false,
  }, {
    get(target, key) {
      if (activeEffect) {
        if (!subscriptions.has(key)) subscriptions.set(key, new Set());
        subscriptions.get(key).add(activeEffect);
      }
      return target[key];
    },
    set(target, key, value) {
      if (target[key] !== value) {
        target[key] = value;
        for (const effect of subscriptions.get(key) ?? []) pending.add(effect);
      }
      return true;
    },
  });
  const T = {
    getSandboxRoot: async () => backendRoot,
    setSandboxRoot: async (path) => { calls.roots.push(path); backendRoot = path; },
    saientSetEnabled: async () => {},
    fsTree: async (path, depth) => { calls.trees.push([path, depth]); return [folder("src")]; },
    ptySpawn: async (...args) => { calls.terminals.push(args); },
    ...overrides,
  };
  const context = createContext({
    agent, T, console,
    model: { activeServerPort: 8899 }, ui: { screen: "agent" },
    projects: { active: null }, checkpoints: { list: [] }, chat: {},
    ownsInput: () => "user", inputLabel: () => "User",
    toast: (...args) => calls.toasts.push(args), onMount: () => {},
    $state: (value) => value, $derived: (value) => value,
    $effect: (effect) => effects.push(effect),
    untrack: (fn) => {
      const previous = activeEffect;
      activeEffect = null;
      try { return fn(); } finally { activeEffect = previous; }
    },
  });
  runInContext(compiled, context);
  runInContext('term = { cols: 80, rows: 24, write() {} }; ptyWorkspace = agent.sandboxRoot;', context);
  const flush = async () => {
    for (let round = 0; round < 20; round++) {
      const next = [...pending];
      pending.clear();
      for (const effect of next) {
        activeEffect = effect;
        try { effect(); } finally { activeEffect = null; }
      }
      await new Promise(setImmediate);
      if (!pending.size) return;
    }
    assert.fail("workspace effects did not settle");
  };
  for (const effect of effects) pending.add(effect);
  await flush();
  calls.trees.length = 0;
  calls.terminals.length = 0;
  const run = (code) => runInContext(code, context);
  return {
    agent, T, calls, flush, run,
    type(value) { context.typedValue = value; run(`${binding} = typedValue`); },
    change: () => run(`(${changed})()`),
    draft: () => run(binding),
  };
}

test("typing a workspace path cannot rebind the running terminal", async () => {
  const h = await harness();
  for (const partial of ["", "/", "/home", "/home/tiny/Desktop"]) {
    h.type(partial);
    await h.flush();
    assert.equal(h.agent.sandboxRoot, "/workspace/original");
    assert.equal(h.calls.roots.length, 0);
    assert.equal(h.calls.terminals.length, 0);
  }
});

test("one accepted change publishes the backend's canonical path and starts one terminal", async () => {
  const accepted = deferred();
  const h = await harness();
  h.T.setSandboxRoot = async (path) => { h.calls.roots.push(path); await accepted.promise; };
  h.T.getSandboxRoot = async () => "/canonical/workspace";
  h.type("/workspace/link");
  const commit = h.change();
  await h.flush();
  assert.equal(h.agent.sandboxRoot, "/workspace/original");
  assert.equal(h.calls.terminals.length, 0);
  const duplicate = h.change();
  accepted.resolve();
  await Promise.all([commit, duplicate]);
  await h.flush();
  assert.deepEqual(h.calls.roots, ["/workspace/link"]);
  assert.equal(h.agent.sandboxRoot, "/canonical/workspace");
  assert.equal(h.draft(), "/canonical/workspace");
  assert.equal(h.agent.workspaceEpoch, 2);
  assert.deepEqual(h.calls.terminals.map((args) => args[0]), ["/canonical/workspace"]);
  await h.change();
  await h.flush();
  assert.equal(h.calls.roots.length, 1, "blur after Enter must not commit twice");
});

test("a rejected workspace leaves the live workspace and terminal untouched", async () => {
  const h = await harness({ setSandboxRoot: async () => { throw new Error("permission denied"); } });
  h.type("/unavailable");
  await h.change();
  await h.flush();
  assert.equal(h.agent.sandboxRoot, "/workspace/original");
  assert.equal(h.draft(), "/workspace/original");
  assert.equal(h.agent.workspaceEpoch, 1);
  assert.equal(h.calls.terminals.length, 0);
  assert.match(h.calls.toasts.at(-1)[0], /permission denied/);
});

test("project selection updates the draft and reloads its file tree", async () => {
  const h = await harness();
  h.type("/unfinished-edit");
  h.agent.sandboxRoot = "/workspace/project";
  h.agent.workspaceEpoch++;
  await h.flush();
  assert.equal(h.draft(), "/workspace/project");
  assert.deepEqual(h.calls.trees, [[".", 1]]);
  assert.deepEqual(h.calls.terminals.map((args) => args[0]), ["/workspace/project"]);
});

test("a path assignment without an accepted epoch cannot restart the terminal", async () => {
  const h = await harness();
  h.agent.sandboxRoot = "/workspace/pending";
  await h.flush();
  assert.equal(h.calls.terminals.length, 0);
  assert.equal(h.calls.trees.length, 0);
});

test("the initial tree loads one level and opens each folder on demand", async () => {
  const h = await harness();
  await h.run("loadFileTree()");
  assert.deepEqual(h.calls.trees, [[".", 1]], "opening / must not walk four levels of the filesystem");
  await h.run("toggleFolder(agent.tree[0])");
  assert.deepEqual(h.calls.trees, [[".", 1], ["src", 1]]);
  await h.run("toggleFolder(agent.tree[0])");
  await h.run("toggleFolder(agent.tree[0])");
  assert.equal(h.calls.trees.length, 2, "reopening a loaded folder uses its current tree");
});

test("late tree results cannot overwrite a newer refresh", async () => {
  const h = await harness();
  const old = deferred();
  h.T.fsTree = () => old.promise;
  const first = h.run("loadFileTree()");
  h.T.fsTree = async () => [folder("new-workspace")];
  await h.run("loadFileTree()");
  old.resolve([folder("old-workspace")]);
  await first;
  assert.equal(h.agent.tree[0].path, "new-workspace");
});

test("a failed folder expansion is visible and can be retried", async () => {
  const h = await harness();
  await h.run("loadFileTree()");
  h.T.fsTree = async () => { throw new Error("folder inaccessible"); };
  await h.run("toggleFolder(agent.tree[0])");
  assert.match(h.calls.toasts.at(-1)[0], /folder inaccessible/);
  h.T.fsTree = async () => [folder("src/nested")];
  await h.run("toggleFolder(agent.tree[0])");
  assert.equal(h.agent.tree[0].children[0].path, "src/nested");
});

test("stale folder results remain isolated across 30 refresh timing permutations", async () => {
  for (let round = 0; round < 30; round++) {
    const h = await harness();
    await h.run("loadFileTree()");
    const old = deferred();
    h.T.fsTree = () => old.promise;
    const expansion = h.run("toggleFolder(agent.tree[0])");
    if (round % 2) await h.flush();
    h.agent.workspaceEpoch++;
    h.T.fsTree = async () => [folder("src")];
    await h.flush();
    if (round % 3 === 0) old.resolve([folder("src/old")]);
    h.T.fsTree = async () => [folder("src/current")];
    await h.run("toggleFolder(agent.tree[0])");
    old.resolve([folder("src/old")]);
    await expansion;
    await h.flush();
    assert.equal(h.agent.tree[0].children[0].path, "src/current", `round ${round}`);
  }
});

test("failure to read a canonical path cannot roll the UI back after the backend changed", async () => {
  const h = await harness({ getSandboxRoot: async () => { throw new Error("connection interrupted"); } });
  h.type("/workspace/new");
  await h.change();
  await h.flush();
  assert.equal(h.agent.sandboxRoot, "/workspace/new");
  assert.equal(h.agent.workspaceEpoch, 2);
  assert.deepEqual(h.calls.terminals.map((args) => args[0]), ["/workspace/new"]);
  assert.ok(h.calls.toasts.some(([text]) => text.includes("connection interrupted")));
});
