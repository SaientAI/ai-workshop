<script lang="ts">
  import { onMount } from "svelte";
  import { open } from "@tauri-apps/plugin-dialog";
  import { listen } from "@tauri-apps/api/event";
  import { agent, model, ui } from "../../lib/state.svelte.js";
  import * as T from "../../lib/tauri.js";
  import type { TreeEntry } from "../../lib/types.js";

  // ── File tree ──────────────────────────────────────────────────────────────

  async function loadFileTree() {
    try {
      agent.tree = await T.fsTree(".", 4) as TreeEntry[];
    } catch (e) { console.error(e); }
  }

  async function selectFile(path: string) {
    agent.selPath = path;
    try {
      const r = await T.fsRead(path);
      agent.content = r.content;
      agent.dirty = false;
    } catch (e) { console.error(e); }
  }

  async function saveFile() {
    if (!agent.selPath) return;
    try {
      await T.fsWrite(agent.selPath, agent.content);
      agent.dirty = false;
    } catch (e) { alert(String(e)); }
  }

  async function newFile() {
    const name = prompt("File name:");
    if (!name) return;
    await T.fsWrite(name, "").catch(() => {});
    await loadFileTree();
    await selectFile(name);
  }

  async function browseSandboxRoot() {
    const p = await open({ directory: true }).catch(() => null);
    if (p) {
      agent.sandboxRoot = p as string;
      await T.setSandboxRoot(p as string);
      await loadFileTree();
    }
  }

  // ── xterm.js PTY terminal ─────────────────────────────────────────────────
  // The xterm container lives outside the tab conditional — it's always in the DOM
  // so scrollback history survives tab switches. CSS visibility controls show/hide.

  import type { Terminal } from "@xterm/xterm";
  import type { FitAddon } from "@xterm/addon-fit";

  let xtermEl: HTMLDivElement | undefined = $state();
  let term: Terminal | null = null;
  let fit: FitAddon | null = null;

  // Re-fit whenever the terminal tab becomes visible (display:block re-enables measurement).
  $effect(() => {
    if (agent.tab === "terminal" && fit && term) {
      requestAnimationFrame(() => {
        fit?.fit();
        if (term) T.ptyResize(term.cols, term.rows).catch(() => {});
      });
    }
  });

  // ── Planner ────────────────────────────────────────────────────────────────

  async function agentRun() {
    if (!agent.planGoal.trim() || !model.loaded) return;
    await T.agentRun(agent.planGoal).catch(e => {
      term?.write(`\x1b[31mAgent error: ${String(e)}\x1b[0m\r\n`);
    });
  }

  function startAuto() {
    if (!agent.planGoal.trim() || !model.loaded || !ui.saientEnabled) return;
    agent.autoMode = true;
    agent.autoIteration = 0;
    agent.autoStatus = "";
    agent.autoGoalDone = false;
    if (!agent.planRunning) agentRun();
  }

  function stopAuto() {
    agent.autoMode = false;
    agent.autoStatus = "Stopped by user";
    T.stopGenerate().catch(() => {});
  }

  async function executePlan() {
    if (!agent.planJson.trim()) return;
    await T.executePlan(agent.planJson, agent.planGoal).catch(e => {
      term?.write(`\x1b[31mPlan error: ${String(e)}\x1b[0m\r\n`);
    });
  }

  // ── Write mode ─────────────────────────────────────────────────────────────

  async function toggleWriteMode() {
    const enabling = !ui.agentWriteMode;
    if (enabling && !confirm(
      "Enable Agent Write Mode?\n\nThe agent will be able to write files, delete files, and run arbitrary commands inside the workspace."
    )) return;
    ui.agentWriteMode = enabling;
    localStorage.setItem("agent_write_mode", String(enabling));
    await T.setAgentWriteMode(enabling).catch(() => {});
  }

  // ── Memory ─────────────────────────────────────────────────────────────────

  async function searchMemory() {
    if (!agent.memQuery.trim()) return;
    agent.memFacts = await T.memorySearch(agent.memQuery).catch(() => []) as typeof agent.memFacts;
  }

  async function loadAllMemory() {
    const store = await T.memoryAll().catch(() => null) as { facts?: typeof agent.memFacts } | null;
    agent.memFacts = store?.facts ?? [];
  }

  async function forgetFact(id: string) {
    await T.memoryForget(id).catch(() => {});
    agent.memFacts = agent.memFacts.filter(f => (f as { id: string }).id !== id);
  }

  // ── Boot ───────────────────────────────────────────────────────────────────

  onMount(() => {
    // onMount must return its cleanup SYNCHRONOUSLY — Svelte ignores the resolved
    // value of an async callback, so an `async () => { ... return cleanup }` would
    // silently never tear down the PTY or listeners. Run setup in an inner async
    // IIFE and hand the teardown back through a sync closure guarded by `disposed`.
    let disposed = false;
    let teardown: (() => void) | null = null;

    void (async () => {
      const cleanup = await boot();
      if (disposed) cleanup?.();          // unmounted before async setup finished
      else teardown = cleanup ?? null;
    })();

    return () => {
      disposed = true;
      teardown?.();
    };

    async function boot(): Promise<(() => void) | void> {
      loadFileTree();

      // Dynamically import xterm so it doesn't bloat the initial bundle parse.
      const [{ Terminal }, { FitAddon }] = await Promise.all([
        import("@xterm/xterm"),
        import("@xterm/addon-fit"),
      ]);
      // xterm CSS must be in global scope; import as side-effect.
      await import("@xterm/xterm/css/xterm.css");

      if (!xtermEl) return;

      term = new Terminal({
      theme: {
        background:          "#0a0a0c",
        foreground:          "#c9d1d9",
        cursor:              "#c9d1d9",
        cursorAccent:        "#0a0a0c",
        selectionBackground: "#3d444d",
        black:   "#0d1117", brightBlack:   "#484f58",
        red:     "#f87171", brightRed:     "#ffa198",
        green:   "#56d364", brightGreen:   "#56d364",
        yellow:  "#e3b341", brightYellow:  "#f2cc60",
        blue:    "#6c8ef5", brightBlue:    "#79c0ff",
        magenta: "#d2a8ff", brightMagenta: "#d2a8ff",
        cyan:    "#79c0ff", brightCyan:    "#79c0ff",
        white:   "#c9d1d9", brightWhite:   "#ffffff",
      },
      fontFamily: "'JetBrains Mono', 'Fira Code', Consolas, monospace",
      fontSize: 13,
      lineHeight: 1.4,
      cursorBlink: true,
      cursorStyle: "block",
      scrollback: 5000,
      allowProposedApi: true,
    });

    fit = new FitAddon();
    term.loadAddon(fit);
    term.open(xtermEl);
    fit.fit();

    // Resize observer: keep PTY kernel size in sync with terminal element size.
    const ro = new ResizeObserver(() => {
      if (agent.tab !== "terminal") return;
      fit?.fit();
      if (term) T.ptyResize(term.cols, term.rows).catch(() => {});
    });
    ro.observe(xtermEl);

    // Keystrokes from xterm → PTY master.
    term.onData(data => T.ptyWrite(data).catch(() => {}));

    // PTY output events → xterm.
    const unlistenPty = await listen<string>("pty-data", e => {
      term?.write(e.payload);
    });

    // Plan executor (exec_command) output → xterm with amber colour so it's
    // visually distinct from interactive shell output.
    const unlistenOut = await listen<{ line: string } | string>("exec-stdout", e => {
      const text = typeof e.payload === "string" ? e.payload : e.payload.line;
      term?.write(text + "\r\n");
    });
    const unlistenErr = await listen<{ line: string } | string>("exec-stderr", e => {
      const text = typeof e.payload === "string" ? e.payload : e.payload.line;
      term?.write(`\x1b[33m${text}\x1b[0m\r\n`);
    });

      // Spawn the shell at the workspace root; hand the agent CLI the model port.
      await T.ptySpawn(agent.sandboxRoot || ".", term.cols, term.rows, model.activeServerPort).catch(err => {
        term?.write(`\x1b[31mFailed to start shell: ${String(err)}\x1b[0m\r\n`);
      });

      return () => {
        ro.disconnect();
        unlistenPty();
        unlistenOut();
        unlistenErr();
        term?.dispose();
        term = null;
        fit = null;
        T.ptyKill().catch(() => {});
      };
    }
  });
</script>

<div class="agent-sidebar sidebar">
  <div class="sidebar-section" style="flex-shrink:0;">
    <div class="section-label">Workspace</div>
    <div style="display:flex;gap:6px;align-items:center;">
      <input type="text" bind:value={agent.sandboxRoot} placeholder="~/agent-workspace"
        style="flex:1;font-size:11px;"
        onchange={() => T.setSandboxRoot(agent.sandboxRoot)} />
      <button class="tab-action" onclick={browseSandboxRoot}>…</button>
    </div>
  </div>

  <div class="sidebar-section" style="flex:1;overflow-y:auto;padding-top:8px;">
    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">
      <div class="section-label" style="margin-bottom:0;">Files</div>
      <div style="display:flex;gap:4px;">
        <button class="tab-action" onclick={loadFileTree} title="Refresh">↻</button>
        <button class="tab-action" onclick={newFile} title="New file">+</button>
      </div>
    </div>
    {@render FileTree({ entries: agent.tree, depth: 0, selectFile })}
  </div>
</div>

<div class="agent-main">
  <!-- Tab bar -->
  <div class="tabbar">
    {#each ["files","terminal","planner","memory"] as t}
      <button class="tab" class:active={agent.tab === t}
        onclick={() => (agent.tab = t as typeof agent.tab)}>
        {t.charAt(0).toUpperCase() + t.slice(1)}
      </button>
    {/each}
    <div style="flex:1;"></div>
    <button class="write-btn" class:on={ui.agentWriteMode} onclick={toggleWriteMode}>
      {ui.agentWriteMode ? "⚠ Write ON" : "Write OFF"}
    </button>
  </div>

  <!--
    xterm container — ALWAYS rendered so scrollback survives tab switches.
    Visibility is controlled by CSS: visible only when agent.tab === "terminal".
  -->
  <div class="xterm-wrap" class:xterm-hidden={agent.tab !== "terminal"} bind:this={xtermEl}></div>

  <!-- Files tab -->
  {#if agent.tab === "files"}
    <div class="file-editor">
      {#if agent.selPath}
        <div class="file-header">
          <span class="file-path">{agent.selPath}</span>
          <button class="tab-action" onclick={saveFile} disabled={!agent.dirty}>Save</button>
        </div>
        <textarea
          class="file-content"
          bind:value={agent.content}
          oninput={() => (agent.dirty = true)}
          spellcheck="false"
        ></textarea>
      {:else}
        <div class="file-placeholder">Select a file from the tree</div>
      {/if}
    </div>

  <!-- Planner tab -->
  {:else if agent.tab === "planner"}
    <div class="planner-panel">
      <div class="plan-goal-row">
        <input type="text" bind:value={agent.planGoal} placeholder="Goal for the agent…"
          oninput={() => localStorage.setItem("saient_goal", agent.planGoal)}
          class="plan-goal" />
        <button class="tab-action run-btn" onclick={agentRun}
          disabled={agent.planRunning || !model.loaded || !ui.saientEnabled || agent.autoMode}>
          {agent.planRunning && !agent.autoMode ? "Running…" : "▶ Run"}
        </button>
        <button
          class="tab-action auto-btn"
          class:auto-on={agent.autoMode}
          onclick={agent.autoMode ? stopAuto : startAuto}
          disabled={!model.loaded || !ui.saientEnabled || (!agent.autoMode && agent.planRunning)}
          title="Run autonomously — Saient re-evaluates and re-plans until the goal is achieved"
        >
          {agent.autoMode ? "■ Stop auto" : "⟳ Auto"}
        </button>
      </div>

      {#if agent.autoMode || agent.autoStatus}
        <div class="auto-bar" class:auto-done={agent.autoGoalDone} class:auto-active={agent.autoMode}>
          <div class="auto-bar-left">
            {#if agent.autoMode}
              <span class="auto-spin">⟳</span>
              <span class="auto-iter">Autonomous — iteration {agent.autoIteration + 1}/{agent.autoMaxIter}</span>
            {:else if agent.autoGoalDone}
              <span class="auto-check">✓</span>
              <span class="auto-iter">Goal achieved</span>
            {:else}
              <span class="auto-iter">Saient stopped</span>
            {/if}
            {#if agent.autoStatus}
              <span class="auto-reason">— {agent.autoStatus}</span>
            {/if}
          </div>
          {#if agent.autoMode}
            <div class="auto-bar-right">
              <span class="auto-iter-config">
                max <input type="number" class="auto-max-input" bind:value={agent.autoMaxIter} min="1" max="20" />
              </span>
            </div>
          {/if}
        </div>
      {/if}

      {#if !ui.saientEnabled}
        <div class="saient-off">Saient is disabled — toggle it in the title bar to enable autonomous runs.</div>
      {/if}

      <div class="plan-json-label">Plan JSON</div>
      <textarea
        class="plan-json"
        bind:value={agent.planJson}
        placeholder="Paste or generate a plan JSON…"
        oninput={() => localStorage.setItem("saient_plan_json", agent.planJson)}
        spellcheck="false"
      ></textarea>

      <div style="display:flex;gap:8px;">
        <button class="tab-action" onclick={executePlan} disabled={!agent.planJson.trim()}>Execute plan</button>
        <button class="tab-action" onclick={() => { agent.planJson = ""; agent.plan = null; }}>Clear</button>
      </div>

      {#if agent.plan}
        <div class="plan-steps">
          {#each agent.plan.steps as step}
            <div class="plan-step"
              class:done={step.status === "done"}
              class:failed={step.status === "failed"}
              class:running={step.status === "running"}>
              <span class="step-status">
                {step.status === "done" ? "✓" : step.status === "failed" ? "✗" : step.status === "running" ? "⟳" : "○"}
              </span>
              <span class="step-desc">{step.description}</span>
            </div>
          {/each}
        </div>
      {/if}
    </div>

  <!-- Memory tab -->
  {:else if agent.tab === "memory"}
    <div class="memory-panel">
      <div class="mem-search-row">
        <input type="text" bind:value={agent.memQuery} placeholder="Search memory…"
          onkeydown={(e) => e.key === "Enter" && searchMemory()} />
        <button class="tab-action" onclick={searchMemory}>Search</button>
        <button class="tab-action" onclick={loadAllMemory}>All</button>
      </div>
      <div class="mem-list">
        {#each agent.memFacts as fact}
          {@const f = fact as import("../../lib/types.js").MemoryFact}
          <div class="mem-fact">
            <div class="mem-fact-head">
              <span class="mem-key">{f.key}</span>
              <span class="mem-cat">{f.category}</span>
              <button class="mem-forget-btn" onclick={() => forgetFact(f.id)} title="Forget">✕</button>
            </div>
            <div class="mem-value">{f.value}</div>
            <div class="mem-meta">
              <span class="mem-conf">{Math.round(f.confidence * 100)}% confidence</span>
              <span class="mem-source">· {f.source}</span>
              <span class="mem-acc">· used {f.access_count}×</span>
            </div>
          </div>
        {/each}
        {#if agent.memFacts.length === 0}
          <div class="mem-empty">No facts yet. Run the agent to build memory.</div>
        {/if}
      </div>
    </div>
  {/if}
</div>

<!-- FileTree snippet -->
{#snippet FileTree({ entries, depth, selectFile }: { entries: TreeEntry[]; depth: number; selectFile: (p: string) => void })}
  {#each entries as e}
    <div
      class="tree-entry"
      style="padding-left:{depth * 14 + 4}px"
      role="button"
      tabindex="0"
      class:dir={e.is_dir}
      class:sel={agent.selPath === e.path}
      onclick={() => !e.is_dir && selectFile(e.path)}
      onkeydown={(ev) => ev.key === "Enter" && !e.is_dir && selectFile(e.path)}
    >
      {e.is_dir ? "📁" : "📄"} {e.name}
    </div>
    {#if e.is_dir && e.children.length > 0}
      {@render FileTree({ entries: e.children, depth: depth + 1, selectFile })}
    {/if}
  {/each}
{/snippet}

<style>
  /* Explicit backgrounds — don't rely on inherited :global rules for these. */
  .agent-sidebar { width: 220px; background: var(--bg2); border-right: 1px solid var(--border); display: flex; flex-direction: column; overflow-y: auto; flex-shrink: 0; }
  .agent-main { flex: 1; display: flex; flex-direction: column; overflow: hidden; position: relative; background: var(--bg); }
  .tabbar { display: flex; align-items: center; border-bottom: 1px solid var(--border); background: var(--bg2); flex-shrink: 0; position: relative; z-index: 1; }
  .tab { padding: 8px 14px; font-size: 12px; color: var(--text3); border: none; background: transparent; border-bottom: 2px solid transparent; border-radius: 0; }
  .tab.active { color: var(--accent); border-bottom-color: var(--accent); }
  .tab:hover { color: var(--text2); }
  .write-btn { margin: 4px 8px; font-size: 10px; padding: 3px 8px; border-radius: 4px; color: var(--text3); border-color: var(--border); }
  .write-btn.on { color: var(--amber); border-color: rgba(245,166,35,0.4); background: rgba(245,166,35,0.08); }

  /* xterm wrapper — always in DOM, show/hide via class */
  .xterm-wrap {
    position: absolute;
    inset: 0;
    /* Push below the tab bar (38px). z-index below the other panels. */
    top: 38px;
    z-index: 0;
    background: #0a0a0c;
    padding: 4px;
  }
  /* xterm needs display:flex or block to measure correctly */
  .xterm-wrap :global(.xterm) { height: 100%; }
  .xterm-hidden { display: none; }

  /* File editor */
  .file-editor { flex: 1; display: flex; flex-direction: column; overflow: hidden; position: relative; z-index: 1; background: var(--bg); }
  .file-header { display: flex; align-items: center; justify-content: space-between; padding: 8px 12px; border-bottom: 1px solid var(--border); background: var(--bg2); flex-shrink: 0; }
  .file-path { font-size: 11px; font-family: var(--mono); color: var(--text2); }
  .file-content { flex: 1; width: 100%; resize: none; font-family: var(--mono); font-size: 12px; line-height: 1.6; padding: 12px 14px; border: none; border-radius: 0; background: var(--bg); color: var(--text); }
  .file-placeholder { flex: 1; display: flex; align-items: center; justify-content: center; color: var(--text3); font-size: 13px; }

  /* Planner */
  .planner-panel { flex: 1; display: flex; flex-direction: column; padding: 12px; gap: 10px; overflow: auto; position: relative; z-index: 1; background: var(--bg); }
  .plan-goal-row { display: flex; gap: 8px; }
  .plan-goal { flex: 1; font-size: 13px; }
  .run-btn { flex-shrink: 0; }
  .auto-btn { flex-shrink: 0; color: var(--text2); }
  .auto-btn.auto-on { color: var(--red); border-color: rgba(248,113,113,0.4); background: rgba(248,113,113,0.08); }
  .auto-btn:not(.auto-on):not(:disabled):hover { color: var(--accent); border-color: rgba(108,142,245,0.4); }
  .auto-bar {
    display: flex; align-items: center; justify-content: space-between; gap: 8px;
    padding: 8px 12px; border-radius: var(--radius-sm); font-size: 11px;
    background: rgba(108,142,245,0.07); border: 1px solid rgba(108,142,245,0.25);
  }
  .auto-bar.auto-done { background: rgba(0,214,143,0.07); border-color: rgba(0,214,143,0.3); }
  .auto-bar:not(.auto-active):not(.auto-done) { background: var(--bg3); border-color: var(--border); }
  .auto-bar-left { display: flex; align-items: center; gap: 6px; flex: 1; min-width: 0; }
  .auto-bar-right { display: flex; align-items: center; gap: 6px; flex-shrink: 0; }
  .auto-spin { color: var(--accent); animation: spin 1.2s linear infinite; display: inline-block; }
  @keyframes spin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }
  .auto-check { color: var(--green); font-weight: 700; }
  .auto-iter { color: var(--text); font-weight: 600; white-space: nowrap; }
  .auto-reason { color: var(--text2); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; font-style: italic; }
  .auto-iter-config { display: flex; align-items: center; gap: 4px; color: var(--text3); white-space: nowrap; }
  .auto-max-input { width: 38px; font-size: 11px; padding: 2px 4px; text-align: center; font-family: var(--mono); }
  .plan-json-label { font-size: 10px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; color: var(--text3); }
  .plan-json { flex: 1; min-height: 120px; font-family: var(--mono); font-size: 11px; line-height: 1.5; resize: vertical; }
  .saient-off { font-size: 11px; color: var(--amber); padding: 6px 10px; background: rgba(245,166,35,0.07); border: 1px solid rgba(245,166,35,0.3); border-radius: var(--radius-sm); }
  .plan-steps { display: flex; flex-direction: column; gap: 4px; }
  .plan-step { display: flex; align-items: center; gap: 8px; padding: 6px 10px; background: var(--bg3); border: 1px solid var(--border); border-radius: var(--radius-sm); font-size: 12px; }
  .plan-step.done    { border-color: rgba(0,214,143,0.3); }
  .plan-step.failed  { border-color: rgba(248,113,113,0.3); }
  .plan-step.running { border-color: rgba(245,166,35,0.3); }
  .step-status { font-family: var(--mono); width: 14px; flex-shrink: 0; }
  .step-desc { flex: 1; color: var(--text2); }

  /* Memory */
  .memory-panel { flex: 1; display: flex; flex-direction: column; padding: 12px; gap: 10px; overflow: hidden; position: relative; z-index: 1; background: var(--bg); }
  .mem-search-row { display: flex; gap: 8px; }
  .mem-search-row input { flex: 1; }
  .mem-list { flex: 1; overflow-y: auto; display: flex; flex-direction: column; gap: 6px; }
  .mem-fact { padding: 9px 11px; background: var(--bg3); border: 1px solid var(--border); border-radius: var(--radius-sm); }
  .mem-fact-head { display: flex; align-items: center; gap: 6px; margin-bottom: 4px; }
  .mem-key { font-size: 12px; font-weight: 700; color: var(--text); flex: 1; font-family: var(--mono); }
  .mem-cat { font-size: 10px; padding: 1px 6px; background: rgba(108,142,245,0.1); color: var(--accent); border-radius: 10px; flex-shrink: 0; }
  .mem-forget-btn { font-size: 10px; padding: 1px 5px; color: var(--text3); border-color: transparent; background: transparent; flex-shrink: 0; line-height: 1; }
  .mem-forget-btn:hover { color: var(--red); border-color: rgba(248,113,113,0.3); }
  .mem-value { font-size: 12px; color: var(--text2); line-height: 1.5; margin-bottom: 5px; white-space: pre-wrap; word-break: break-word; }
  .mem-meta { display: flex; align-items: center; gap: 4px; font-size: 10px; color: var(--text3); font-family: var(--mono); flex-wrap: wrap; }
  .mem-conf { color: var(--green); }
  .mem-source { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; max-width: 160px; }
  .mem-empty { color: var(--text3); font-size: 12px; }

  /* File tree */
  .tree-entry { font-size: 12px; color: var(--text2); padding: 3px 0; cursor: pointer; border-radius: 3px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .tree-entry:hover { color: var(--text); background: var(--bg3); }
  .tree-entry.sel { color: var(--accent); background: rgba(108,142,245,0.08); }
  .tree-entry.dir { color: var(--text3); font-weight: 600; cursor: default; }
</style>
