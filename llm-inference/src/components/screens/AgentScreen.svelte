<script lang="ts">
  import { onMount } from "svelte";
  import { open } from "@tauri-apps/plugin-dialog";
  import { listen } from "@tauri-apps/api/event";
  import { agent, model, ui, chat, checkpoints, projects, setCheckpointPolicy, toast } from "../../lib/state.svelte.js";
  import * as T from "../../lib/tauri.js";
  import type { TreeEntry } from "../../lib/types.js";
  import { ownsInput, inputLabel, activityText } from "../../lib/turnState.js";
  import {
    buildSessionState, suggestName, groupByDay, describeSize,
    AUTO_SAVE_POLICIES, AUTO_SAVE_LABELS,
  } from "../../lib/checkpoints.js";

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
      await changeSandboxRoot(p as string);
    }
  }

  async function changeSandboxRoot(path: string) {
    const previous = await T.getSandboxRoot().catch(() => agent.sandboxRoot);
    try {
      await T.setSandboxRoot(path);
      agent.sandboxRoot = path;
      projects.active = null;
      await T.saientSetEnabled(false).catch(() => {});
      agent.workspaceEpoch += 1;
      agent.tree = [];
      agent.selPath = null;
      agent.content = "";
      await loadFileTree();
      toast(`Workspace changed to ${path}`, "success");
    } catch (e) {
      agent.sandboxRoot = previous;
      toast(`Could not open workspace: ${String(e)}`, "error");
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
  let ptyWorkspace = "";

  async function spawnWorkspaceTerminal(path: string, announce: boolean) {
    if (!term) return;
    const cwd = path || ".";
    if (announce) {
      term.write(`\r\n\x1b[38;2;108;142;245mworkspace\x1b[0m → ${cwd}\r\n`);
    }
    await T.ptySpawn(cwd, term.cols, term.rows, model.activeServerPort);
    ptyWorkspace = cwd;
  }

  // ProjectPicker and the folder field both update the backend's file tools.
  // Rebind the terminal as part of the same accepted change so the `saient`
  // CLI, its safe-path boundary and the file tree can never show different
  // workspaces again.
  $effect(() => {
    const epoch = agent.workspaceEpoch;
    const cwd = agent.sandboxRoot || ".";
    if (epoch === 0 || !term || cwd === ptyWorkspace) return;
    void spawnWorkspaceTerminal(cwd, true).catch((e) => {
      term?.write(`\x1b[31mFailed to switch workspace: ${String(e)}\x1b[0m\r\n`);
    });
  });

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
    agent.continuing = false;
    agent.turn = "INTERRUPTED";
    agent.retry = null;
    agent.autoStatus = "Stopped by user";
    T.stopGenerate().catch(() => {});
  }

  // ── Turn ownership ─────────────────────────────────────────────────────────
  // Derived, never stored: a second copy of "is Saient busy" is how the old code
  // came to disagree with itself.
  const turnOwner = $derived(ownsInput(agent.turn, agent.continuing));
  const turnLabel = $derived(inputLabel(agent.turn, agent.continuing));

  /** Queue a mid-task instruction for the next iteration. */
  function addInstruction() {
    const text = agent.planGoal.trim();
    if (!text) return;
    agent.pendingInstructions.push(text);
    agent.planGoal = "";
  }

  /** Stop now: kill the in-flight inference and end the loop. */
  function interrupt() {
    agent.autoMode = false;
    agent.continuing = false;
    agent.paused = false;
    agent.turn = "INTERRUPTED";
    agent.retry = null;
    agent.autoStatus = "Interrupted by user";
    T.stopGenerate().catch(() => {});
  }

  /** Stop cleanly at the next boundary rather than mid-step. */
  function togglePause() {
    agent.paused = !agent.paused;
    agent.autoStatus = agent.paused
      ? "Pause requested — will stop after the current step"
      : "Pause cancelled";
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

  // ── Checkpoints ────────────────────────────────────────────────────────────

  /** The live state a checkpoint captures. Assembled fresh at save time. */
  function session() {
    return buildSessionState({
      goal: agent.planGoal,
      turn: agent.turn,
      terminalCwd: agent.termCwd || agent.sandboxRoot,
      plan: agent.plan,
      conversation: chat.messages,
      terminal: agent.termLines.map((l) => l.text),
    });
  }

  async function refreshCheckpoints() {
    checkpoints.list = await T.checkpointList().catch(() => []);
  }

  async function saveCheckpoint() {
    checkpoints.busy = true;
    checkpoints.error = "";
    try {
      const meta = await T.checkpointSave(
        checkpoints.draftName || suggestName(agent.planGoal, "manual"),
        "manual",
        checkpoints.lastSaved?.id ?? null,
        session(),
      );
      checkpoints.lastSaved = meta;
      checkpoints.draftName = "";
      await refreshCheckpoints();
      toast(`Saved “${meta.name}”`, "success");
    } catch (e) {
      checkpoints.error = String(e);
    } finally {
      checkpoints.busy = false;
    }
  }

  async function restoreCheckpoint(id: string) {
    if (!confirm(
      "Restore this checkpoint?\n\nWorkspace files will be overwritten with the saved versions. " +
      "A safety checkpoint of the current state is taken first, so this can be undone."
    )) return;
    checkpoints.busy = true;
    try {
      checkpoints.lastRestore = await T.checkpointRestore(id, session());
      await refreshCheckpoints();
      await loadFileTree();
      toast(`Restored ${checkpoints.lastRestore.restored.length} file(s)`, "success");
    } catch (e) {
      checkpoints.error = String(e);
    } finally {
      checkpoints.busy = false;
    }
  }

  async function exportCheckpoint(id: string, format: "markdown" | "json") {
    try {
      const text = await T.checkpointExport(id, format);
      await navigator.clipboard.writeText(text);
      toast(`${format === "json" ? "JSON" : "Markdown"} copied to clipboard`, "success");
    } catch (e) {
      checkpoints.error = String(e);
    }
  }

  async function deleteCheckpoint(id: string) {
    if (!confirm("Delete this checkpoint? The saved file contents stay on disk if other checkpoints share them.")) return;
    await T.checkpointDelete(id).catch((e) => (checkpoints.error = String(e)));
    if (checkpoints.lastSaved?.id === id) checkpoints.lastSaved = null;
    await refreshCheckpoints();
  }

  // Load the list when the tab is first opened, not at boot — no point paying
  // for a directory walk the user may never look at.
  $effect(() => {
    if (agent.tab === "checkpoints" && checkpoints.list.length === 0) {
      void refreshCheckpoints();
    }
  });

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

      // Prefill the constant half of the planning prompt while the user is still
      // reading the screen. Everything above the goal is identical on every run,
      // so warming it here turns the first plan from ~14s into ~2s. Fire and
      // forget: it is an optimisation, and a failure must not block boot.
      void T.warmAgentCache().catch(() => {});

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
      await spawnWorkspaceTerminal(agent.sandboxRoot || ".", false).catch(err => {
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
      <input type="text" bind:value={agent.sandboxRoot} placeholder="data/agent-workspace"
        style="flex:1;font-size:11px;"
        aria-label="Workspace folder. Agent access is limited to this folder."
        onchange={() => changeSandboxRoot(agent.sandboxRoot)} />
      <button class="tab-action" onclick={browseSandboxRoot} title="Choose the folder Saient may access">…</button>
    </div>
    <div style="margin-top:5px;font-size:10px;line-height:1.35;color:var(--text3);">
      Saient can read and act only inside this folder. Changing it restarts the terminal in that folder.
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
    {#each ["files","terminal","planner","memory","checkpoints"] as t}
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
      <!-- Whose turn it is. Never says "User" while Saient is still working,
           including the gap between autonomous iterations. -->
      <div class="turn-row" class:turn-saient={turnOwner === "saient"}>
        <span class="turn-label">{turnLabel}</span>
        <span class="turn-activity">{activityText(agent.turn)}</span>
      </div>

      {#if agent.retry}
        <div class="retry-note">
          <strong>Retrying step {agent.retry.step} of {agent.retry.total}</strong>
          <span>Reason: {agent.retry.reason}</span>
        </div>
      {/if}

      <div class="plan-goal-row">
        <input
          type="text"
          bind:value={agent.planGoal}
          placeholder={turnOwner === "saient" ? "Add an instruction for the next step…" : "Goal for the agent…"}
          class="plan-goal"
        />
        {#if turnOwner === "saient"}
          <!-- The keyboard has not come back. Offer what is actually possible
               mid-task rather than a Run button that cannot fire. -->
          <button class="tab-action" onclick={addInstruction}
            disabled={!agent.planGoal.trim()}
            title="Queue this for the next iteration">+ Add instruction</button>
          <button class="tab-action" onclick={interrupt}>■ Interrupt</button>
          <button class="tab-action" onclick={togglePause} class:auto-on={agent.paused}
            title="Finish the current step, then stop before the next one">
            {agent.paused ? "Pausing…" : "❚❚ Pause"}
          </button>
        {:else}
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
        {/if}
      </div>

      {#if agent.pendingInstructions.length}
        <div class="queued-note">
          Queued for the next iteration:
          <ul>
            {#each agent.pendingInstructions as instruction}<li>{instruction}</li>{/each}
          </ul>
        </div>
      {/if}

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

      {#if agent.planPrefill}
        <div class="plan-prefill">
          <span class="plan-prefill-label">Reading the prompt…</span>
          <progress
            class="plan-prefill-bar"
            value={agent.planPrefill.done + 1}
            max={agent.planPrefill.total}
          ></progress>
          <span class="plan-prefill-count">
            {agent.planPrefill.done + 1} / {agent.planPrefill.total} tokens
          </span>
        </div>
      {/if}

      {#if agent.planReasoning}
        <details class="plan-thoughts" open>
          <summary>Thinking</summary>
          <pre>{agent.planReasoning}</pre>
        </details>
      {/if}

      <div class="plan-json-label">Plan JSON</div>
      <textarea
        class="plan-json"
        bind:value={agent.planJson}
        placeholder="Paste or generate a plan JSON…"
        spellcheck="false"
      ></textarea>

      {#if agent.planAbandoned.length}
        <div class="plan-abandoned">
          {agent.planAbandoned.length} step(s) never ran because a prerequisite failed:
          <ul>
            {#each agent.planAbandoned as description}
              <li>{description}</li>
            {/each}
          </ul>
        </div>
      {/if}

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

  <!-- Checkpoints tab -->
  {:else if agent.tab === "checkpoints"}
    <div class="cp-panel">
      <div class="cp-toolbar">
        <input
          class="cp-name-input"
          bind:value={checkpoints.draftName}
          placeholder="Name this checkpoint…"
          onkeydown={(e) => e.key === "Enter" && saveCheckpoint()}
        />
        <button class="tab-action" onclick={saveCheckpoint} disabled={checkpoints.busy}>
          {checkpoints.busy ? "Saving…" : "Save checkpoint"}
        </button>
        <label class="cp-auto">
          Auto-save
          <select
            value={checkpoints.policy}
            onchange={(e) => setCheckpointPolicy(e.currentTarget.value as typeof checkpoints.policy)}
          >
            {#each AUTO_SAVE_POLICIES as p}<option value={p}>{AUTO_SAVE_LABELS[p]}</option>{/each}
          </select>
        </label>
      </div>

      {#if checkpoints.error}<div class="cp-error">{checkpoints.error}</div>{/if}

      {#if checkpoints.lastRestore}
        {@const r = checkpoints.lastRestore}
        <div class="cp-restore-report">
          Restored {r.restored.length} file(s), {r.unchanged.length} already matched.
          {#if r.left_in_place.length}
            {r.left_in_place.length} newer file(s) were left untouched.
          {/if}
          <button class="cp-inline-btn" onclick={() => restoreCheckpoint(r.undo_checkpoint)}>
            Undo this restore
          </button>
        </div>
      {/if}

      <div class="cp-list">
        {#each groupByDay(checkpoints.list) as group}
          <div class="cp-day">{group.day}</div>
          {#each group.items as cp}
            <div class="cp-item" class:cp-current={cp.id === checkpoints.lastSaved?.id}>
              <div class="cp-item-head">
                <span class="cp-item-name">{cp.name}</span>
                <span class="cp-item-kind">{cp.kind.replace("_", " ")}</span>
              </div>
              <div class="cp-item-meta">
                <span>{new Date(cp.created_at * 1000).toLocaleTimeString()}</span>
                <span>· {describeSize(cp)}</span>
                {#if cp.step_index !== null && cp.step_total !== null}
                  <span>· step {cp.step_index}/{cp.step_total}</span>
                {/if}
                {#if cp.outstanding.length}
                  <span class="cp-outstanding">· {cp.outstanding.length} outstanding</span>
                {/if}
              </div>
              {#if cp.goal}<div class="cp-item-goal">{cp.goal}</div>{/if}
              <div class="cp-item-actions">
                <button class="cp-inline-btn" onclick={() => restoreCheckpoint(cp.id)}>Restore</button>
                <button class="cp-inline-btn" onclick={() => exportCheckpoint(cp.id, "markdown")}>Export MD</button>
                <button class="cp-inline-btn" onclick={() => exportCheckpoint(cp.id, "json")}>Export JSON</button>
                <button class="cp-inline-btn danger" onclick={() => deleteCheckpoint(cp.id)}>Delete</button>
              </div>
            </div>
          {/each}
        {/each}
        {#if checkpoints.list.length === 0}
          <div class="mem-empty">
            No checkpoints yet. Ctrl+S saves the conversation and project state together.
          </div>
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

  /* Checkpoints */
  .cp-panel { display: flex; flex-direction: column; gap: 8px; height: 100%; overflow: hidden; padding: 10px; }
  .cp-toolbar { display: flex; gap: 8px; align-items: center; flex-shrink: 0; }
  .cp-name-input { flex: 1; background: var(--bg); border: 1px solid var(--border);
                   border-radius: var(--radius-sm); color: var(--text); padding: 5px 8px; font-size: 12px; }
  .cp-auto { font-size: 11px; color: var(--text3); display: flex; align-items: center; gap: 5px; white-space: nowrap; }
  .cp-auto select { background: var(--bg); border: 1px solid var(--border); color: var(--text2);
                    border-radius: var(--radius-sm); font-size: 11px; padding: 3px 6px; }
  .cp-error { font-size: 11px; color: #d0553a; border-left: 2px solid #d0553a; padding-left: 8px; }
  .cp-restore-report { font-size: 11px; color: var(--text2); background: var(--bg3);
                       border-radius: var(--radius-sm); padding: 6px 8px; display: flex;
                       flex-wrap: wrap; gap: 8px; align-items: center; }
  .cp-list { flex: 1; overflow-y: auto; display: flex; flex-direction: column; gap: 6px; }
  .cp-day { font-size: 10px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.07em;
            color: var(--text3); margin-top: 6px; }
  .cp-item { border: 1px solid var(--border); border-radius: var(--radius-sm);
             background: var(--bg2); padding: 7px 9px; display: flex; flex-direction: column; gap: 3px; }
  .cp-item.cp-current { border-color: var(--accent); }
  .cp-item-head { display: flex; justify-content: space-between; gap: 8px; align-items: baseline; }
  .cp-item-name { font-size: 12px; color: var(--text); font-weight: 600; }
  .cp-item-kind { font-size: 10px; color: var(--text3); text-transform: uppercase; letter-spacing: 0.05em; }
  .cp-item-meta { font-size: 10px; color: var(--text3); display: flex; gap: 5px; flex-wrap: wrap; }
  .cp-outstanding { color: #d08a3a; }
  .cp-item-goal { font-size: 11px; color: var(--text2); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .cp-item-actions { display: flex; gap: 6px; margin-top: 3px; flex-wrap: wrap; }
  .cp-inline-btn { background: transparent; border: 1px solid var(--border); color: var(--text2);
                   border-radius: var(--radius-sm); font-size: 10px; padding: 2px 7px; cursor: pointer; }
  .cp-inline-btn:hover { color: var(--text); border-color: var(--text3); }
  .cp-inline-btn.danger:hover { color: #d0553a; border-color: #d0553a; }


  /* Planning feedback. Internal children of the screen, so a component-scoped
     block is safe here — see the note at the top of global.css. */
  .plan-prefill { display: flex; align-items: center; gap: 8px; font-size: 11px; color: var(--text2); }
  .plan-prefill-label { white-space: nowrap; }
  .plan-prefill-bar { flex: 1; height: 4px; accent-color: var(--accent); }
  .plan-prefill-count { color: var(--text3); font-family: var(--mono); white-space: nowrap; }

  .plan-thoughts { border: 1px solid var(--border); border-radius: 4px; background: var(--bg2); padding: 6px 8px; }
  .plan-thoughts summary { font-size: 10px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; color: var(--text3); cursor: pointer; }
  .plan-thoughts pre { margin: 6px 0 0; font-family: var(--mono); font-size: 11px; line-height: 1.5; color: var(--text2); white-space: pre-wrap; max-height: 180px; overflow-y: auto; }

  .plan-abandoned { font-size: 11px; color: var(--text2); border-left: 2px solid #d08a3a; padding-left: 8px; }
  .plan-abandoned ul { margin: 4px 0 0; padding-left: 16px; }

  /* Turn ownership */
  .turn-row { display: flex; align-items: baseline; gap: 10px; padding: 4px 0; }
  .turn-label { font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.06em; color: var(--text3); }
  .turn-row.turn-saient .turn-label { color: var(--accent); }
  .turn-activity { font-size: 11px; color: var(--text2); }

  .retry-note { display: flex; flex-direction: column; gap: 2px; font-size: 11px; color: var(--text2);
                border-left: 2px solid #d08a3a; padding: 4px 8px; background: var(--bg2); border-radius: 3px; }
  .retry-note strong { color: var(--text); font-weight: 600; }

  .queued-note { font-size: 11px; color: var(--text2); border-left: 2px solid var(--accent); padding-left: 8px; }
  .queued-note ul { margin: 2px 0 0; padding-left: 16px; }
</style>
