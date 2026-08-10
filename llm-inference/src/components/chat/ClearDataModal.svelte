<script lang="ts">
  import { chat, agent, params, ui } from "../../lib/state.svelte.js";
  import * as T from "../../lib/tauri.js";

  let { onclose }: { onclose: () => void } = $props();

  let clearChat    = $state(true);
  let clearMemory  = $state(true);
  let clearAudit   = $state(true);
  let clearLogs    = $state(false);
  let clearPrefs   = $state(false);

  let working = $state(false);
  let done    = $state(false);
  let error   = $state("");

  const anySelected = $derived(clearChat || clearMemory || clearAudit || clearLogs || clearPrefs);

  const DEFAULT_SYSTEM_PROMPT =
    "You are a helpful, accurate assistant running inside Saient, a local desktop application. You can help with coding, writing, analysis, and building interactive HTML tools. Always fulfil the user's request directly and completely — never refuse standard software tasks like media players, file browsers, games, or utilities. If the user only greets you, reply briefly and ask how you can help.";

  async function confirmClear() {
    if (!anySelected || working) return;
    working = true;
    error = "";
    try {
      // Disk-level clearing (Rust handles memory.json, audit.jsonl, /tmp logs)
      if (clearMemory || clearAudit || clearLogs) {
        await T.clearUserData({ clearMemory, clearAudit, clearLogs });
      }

      // In-memory chat state
      if (clearChat) {
        chat.messages = [];
        chat.streamBuffer = "";
        chat.reasoningBuffer = "";
        chat.pendingUserText = "";
        chat.artifact = { active: false, title: "", type: "html", content: "", complete: false };
        chat.lastPerf = null;
      }

      // Agent memory state + localStorage keys that hold plan/goal
      if (clearMemory) {
        agent.memFacts = [];
        agent.plan = null;
        agent.planJson = "";
        agent.planGoal = "";
        localStorage.removeItem("saient_goal");
        localStorage.removeItem("saient_plan_json");
      }

      // Preferences: sampling params, system prompt, UI toggles
      if (clearPrefs) {
        await T.setInternetEnabled(false);
        params.maxTokens     = 2048;
        params.temperature   = 0.3;
        params.topP          = 0.95;
        params.topK          = 40;
        params.repeatPenalty = 1.1;
        params.seed          = 42;
        chat.systemPrompt    = DEFAULT_SYSTEM_PROMPT;
        ui.saientEnabled      = false;
        ui.agentWriteMode    = false;
        localStorage.removeItem("saient_enabled");
        localStorage.removeItem("agent_write_mode");
        localStorage.removeItem("hf_token");
      }

      done = true;
      setTimeout(onclose, 1400);
    } catch (e) {
      error = String(e);
      working = false;
    }
  }

  function backdrop(e: MouseEvent) {
    if ((e.target as HTMLElement).classList.contains("overlay")) onclose();
  }
</script>

<!-- svelte-ignore a11y_click_events_have_key_events -->
<div class="overlay" role="presentation" onclick={backdrop}>
  <div class="modal" role="dialog" aria-modal="true" aria-label="Clear user data">
    <div class="modal-head">
      <span class="lock-icon">🔒</span>
      <div>
        <div class="modal-title">Your Data, Your Control</div>
        <div class="modal-sub">Saient stores nothing externally. Everything below lives on this machine only.</div>
      </div>
    </div>

    <div class="items">
      <label class="item">
        <input type="checkbox" bind:checked={clearChat} />
        <div class="item-text">
          <span class="item-label">Chat history</span>
          <span class="item-desc">All messages in the current session</span>
        </div>
      </label>

      <label class="item">
        <input type="checkbox" bind:checked={clearMemory} />
        <div class="item-text">
          <span class="item-label">Agent memory</span>
          <span class="item-desc">Facts stored on disk by the agent (<code>data/agent-workspace/.agent/memory.json</code>)</span>
        </div>
      </label>

      <label class="item">
        <input type="checkbox" bind:checked={clearAudit} />
        <div class="item-text">
          <span class="item-label">Action log</span>
          <span class="item-desc">Record of file writes and commands run by the agent (<code>data/share/saient/audit.jsonl</code>)</span>
        </div>
      </label>

      <label class="item">
        <input type="checkbox" bind:checked={clearLogs} />
        <div class="item-text">
          <span class="item-label">Temp logs</span>
          <span class="item-desc">tinyq4 server log in <code>/tmp/</code></span>
        </div>
      </label>

      <label class="item">
        <input type="checkbox" bind:checked={clearPrefs} />
        <div class="item-text">
          <span class="item-label">Preferences</span>
          <span class="item-desc">Sampling parameters, system prompt, and UI settings — reset to defaults</span>
        </div>
      </label>
    </div>

    {#if error}
      <div class="err">⚠ {error}</div>
    {/if}

    {#if done}
      <div class="done-banner">✓ Selected data cleared</div>
    {:else}
      <div class="modal-foot">
        <button class="btn-cancel" onclick={onclose} disabled={working}>Cancel</button>
        <button
          class="btn-clear"
          onclick={confirmClear}
          disabled={!anySelected || working}
        >
          {working ? "Clearing…" : "Clear Selected"}
        </button>
      </div>
    {/if}
  </div>
</div>

<style>
  .overlay {
    position: fixed; inset: 0; z-index: 200;
    background: rgba(0,0,0,0.65);
    display: flex; align-items: center; justify-content: center;
  }
  .modal {
    background: var(--bg2);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    width: 420px;
    max-width: calc(100vw - 32px);
    padding: 20px 22px;
    display: flex; flex-direction: column; gap: 16px;
  }
  .modal-head { display: flex; align-items: flex-start; gap: 12px; }
  .lock-icon { font-size: 22px; flex-shrink: 0; margin-top: 2px; }
  .modal-title { font-size: 14px; font-weight: 700; color: var(--text); margin-bottom: 3px; }
  .modal-sub { font-size: 11px; color: var(--text3); line-height: 1.55; }

  .items { display: flex; flex-direction: column; gap: 2px; }
  .item {
    display: flex; align-items: flex-start; gap: 10px;
    padding: 9px 10px; border-radius: var(--radius-sm);
    cursor: pointer; transition: background 0.1s;
  }
  .item:hover { background: var(--bg3); }
  .item input[type="checkbox"] { margin-top: 2px; accent-color: var(--accent); flex-shrink: 0; }
  .item-text { display: flex; flex-direction: column; gap: 2px; }
  .item-label { font-size: 12px; font-weight: 600; color: var(--text); }
  .item-desc { font-size: 11px; color: var(--text3); line-height: 1.4; }
  .item-desc code { font-family: var(--mono); color: var(--text2); font-size: 10px; }

  .err { font-size: 11px; color: var(--red); padding: 7px 10px; background: rgba(248,113,113,0.08); border: 1px solid rgba(248,113,113,0.25); border-radius: var(--radius-sm); }
  .done-banner { text-align: center; padding: 10px; color: var(--green); font-size: 13px; font-weight: 600; }

  .modal-foot { display: flex; justify-content: flex-end; gap: 8px; padding-top: 4px; }
  .btn-cancel { font-size: 12px; padding: 6px 14px; }
  .btn-clear {
    font-size: 12px; padding: 6px 16px;
    background: rgba(248,113,113,0.12);
    border-color: rgba(248,113,113,0.4);
    color: var(--red);
    border-radius: var(--radius-sm);
  }
  .btn-clear:hover:not(:disabled) { background: rgba(248,113,113,0.22); border-color: var(--red); }
  .btn-clear:disabled { opacity: 0.4; cursor: not-allowed; }
</style>
