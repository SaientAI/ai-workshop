<script lang="ts">
  // Save controls: the end-of-turn prompt, and Ctrl+S.
  //
  // Mounted at app level rather than inside the agent screen so Ctrl+S works
  // wherever you are, and so a prompt cannot be dismissed by navigating away.

  import { agent, chat, checkpoints, setCheckpointPolicy, toast } from "../lib/state.svelte.js";
  import * as T from "../lib/tauri.js";
  import {
    buildSessionState, suggestName, shouldAutoSave, shouldPrompt,
    AUTO_SAVE_POLICIES, AUTO_SAVE_LABELS,
  } from "../lib/checkpoints.js";
  import { isTerminal } from "../lib/turnState.js";

  /** Everything a checkpoint needs, gathered from live state at save time. */
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

  async function save(name: string, kind: T.CheckpointKind = "manual") {
    checkpoints.busy = true;
    checkpoints.error = "";
    try {
      const meta = await T.checkpointSave(
        name || suggestName(agent.planGoal, kind),
        kind,
        checkpoints.lastSaved?.id ?? null,   // chain so history can be walked
        session(),
      );
      checkpoints.lastSaved = meta;
      checkpoints.list = await T.checkpointList();
      toast(`Saved “${meta.name}”`, "success");
      return meta;
    } catch (err) {
      checkpoints.error = String(err);
      toast(`Checkpoint failed: ${String(err)}`, "error");
      return null;
    } finally {
      checkpoints.busy = false;
    }
  }

  // Ctrl+S saves the conversation and project state together — the whole point
  // of a checkpoint is that those two travel as one.
  function onKey(e: KeyboardEvent) {
    if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "s") {
      e.preventDefault();
      save(checkpoints.draftName);
      checkpoints.draftName = "";
    }
  }

  // React to the turn settling. Reads the state machine rather than guessing
  // from planRunning, so a continuing autonomous loop never triggers a save
  // mid-task.
  let lastSettled = $state("");
  $effect(() => {
    const state = agent.turn;
    if (!isTerminal(state) || agent.continuing) {
      if (!isTerminal(state)) lastSettled = "";
      return;
    }
    // One decision per settle, not one per re-render.
    const key = `${state}:${agent.planGoal}:${checkpoints.lastSaved?.id ?? ""}`;
    if (lastSettled === key) return;
    lastSettled = key;

    if (shouldAutoSave(checkpoints.policy, state)) {
      save("", state === "COMPLETED" ? "auto_task" : "auto_turn");
    } else if (shouldPrompt(checkpoints.policy, state)) {
      checkpoints.draftName = suggestName(agent.planGoal, "manual");
      checkpoints.prompting = true;
    }
  });

  async function saveFromPrompt() {
    await save(checkpoints.draftName);
    checkpoints.prompting = false;
  }

  function discard() {
    checkpoints.prompting = false;
    checkpoints.draftName = "";
  }

  function alwaysSave() {
    setCheckpointPolicy("task");
    saveFromPrompt();
  }
</script>

<svelte:window onkeydown={onKey} />

{#if checkpoints.prompting}
  <div class="cp-backdrop" role="presentation" onclick={discard}></div>
  <div class="cp-prompt" role="dialog" aria-label="Save this response and project state?">
    <h3>Save this response and project state?</h3>
    <p class="cp-sub">
      Captures the conversation, the active goal, the current step, changed files
      and the terminal's working directory.
    </p>
    <input
      class="cp-name"
      bind:value={checkpoints.draftName}
      placeholder="Name this checkpoint…"
      onkeydown={(e) => e.key === "Enter" && saveFromPrompt()}
    />
    <div class="cp-actions">
      <button class="cp-btn primary" onclick={saveFromPrompt} disabled={checkpoints.busy}>
        {checkpoints.busy ? "Saving…" : "Save"}
      </button>
      <button class="cp-btn" onclick={discard}>Discard</button>
      <button class="cp-btn" onclick={alwaysSave} title="Switch to saving every completed task">
        Always save this project
      </button>
    </div>
    <label class="cp-policy">
      Auto-save
      <select
        value={checkpoints.policy}
        onchange={(e) => setCheckpointPolicy(e.currentTarget.value as typeof checkpoints.policy)}
      >
        {#each AUTO_SAVE_POLICIES as p}<option value={p}>{AUTO_SAVE_LABELS[p]}</option>{/each}
      </select>
    </label>
  </div>
{/if}

<style>
  .cp-backdrop { position: fixed; inset: 0; background: rgba(0,0,0,0.45); z-index: 40; }
  .cp-prompt {
    position: fixed; z-index: 41; left: 50%; top: 50%; transform: translate(-50%, -50%);
    width: 420px; max-width: calc(100vw - 40px);
    background: var(--bg2); border: 1px solid var(--border); border-radius: var(--radius);
    padding: 16px; display: flex; flex-direction: column; gap: 10px;
    box-shadow: 0 12px 40px rgba(0,0,0,0.5);
  }
  .cp-prompt h3 { margin: 0; font-size: 14px; color: var(--text); }
  .cp-sub { margin: 0; font-size: 11px; color: var(--text3); line-height: 1.5; }
  .cp-name {
    background: var(--bg); border: 1px solid var(--border); border-radius: var(--radius-sm);
    color: var(--text); padding: 6px 8px; font-size: 12px;
  }
  .cp-actions { display: flex; gap: 8px; flex-wrap: wrap; }
  .cp-btn {
    background: transparent; border: 1px solid var(--border); color: var(--text2);
    border-radius: var(--radius-sm); font-size: 11px; padding: 5px 10px; cursor: pointer;
  }
  .cp-btn:hover { color: var(--text); border-color: var(--text3); }
  .cp-btn.primary { background: var(--accent); border-color: var(--accent); color: #fff; }
  .cp-policy { font-size: 11px; color: var(--text3); display: flex; align-items: center; gap: 6px; }
  .cp-policy select {
    background: var(--bg); border: 1px solid var(--border); color: var(--text2);
    border-radius: var(--radius-sm); font-size: 11px; padding: 3px 6px;
  }
</style>
