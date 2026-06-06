<script lang="ts">
  import { model, ui } from "../lib/state.svelte.js";
  import * as T from "../lib/tauri.js";

  let { aw }: { aw: (a: "min" | "max" | "close") => void } = $props();

  async function toggleSaient() {
    ui.saientEnabled = !ui.saientEnabled;
    localStorage.setItem("saient_enabled", String(ui.saientEnabled));
  }

  async function stopModel() {
    model.loading = true;
    model.loadError = "";
    model.loadStatus = model.activeServerPort
      ? `Stopping server on port ${model.activeServerPort}…`
      : "Stopping model server…";
    try {
      await T.stopModelServer(model.activeServerPort);
      model.loaded = false;
      model.summary = null;
      model.activeServerPort = null;
      model.loadStatus = "";
    } catch (e) {
      model.loadError = String(e);
      model.loadStatus = "";
    } finally {
      model.loading = false;
    }
  }
</script>

<div class="titlebar" data-tauri-drag-region>
  <div class="left">
    <div class="dot" class:green={model.loaded} class:amber={model.loading}></div>
    <span class="logo">Saient</span>
  </div>
  <div class="right">
    <div class="info">
      {#if model.summary}
        {model.summary.architecture} · {model.summary.quant}{model.activeServerPort ? ` · :${model.activeServerPort}` : ""}
      {:else}
        no model
      {/if}
    </div>

    <button class="saient-btn" class:on={ui.saientEnabled} onclick={toggleSaient}
      title={ui.saientEnabled ? "Saient is ON — click to disable" : "Saient is OFF — click to enable"}>
      <span class="saient-dot"></span>Saient
    </button>

    {#if model.loaded}
      <button class="wbtn stop" onclick={stopModel}>■ Stop</button>
    {/if}

    <button class="wbtn" onclick={() => (ui.showShortcuts = true)} title="Keyboard shortcuts (?)">⌨</button>
    <button class="wbtn" onclick={() => aw("min")}>─</button>
    <button class="wbtn" onclick={() => aw("max")}>□</button>
    <button class="wbtn x" onclick={() => aw("close")}>✕</button>
  </div>
</div>

<style>
  .titlebar {
    height: 36px;
    background: var(--bg2);
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 10px 0 14px;
    flex-shrink: 0;
  }
  .left { display: flex; align-items: center; gap: 8px; }
  .right { display: flex; align-items: center; gap: 6px; }
  .dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    background: var(--text3);
    flex-shrink: 0;
  }
  .dot.green { background: var(--green); box-shadow: 0 0 6px var(--green); }
  .dot.amber { background: var(--amber); box-shadow: 0 0 6px var(--amber); }
  .logo { font-size: 13px; font-weight: 700; color: var(--text); letter-spacing: 0.02em; }
  .sep { color: var(--accent); margin: 0 1px; }
  .info { font-size: 11px; color: var(--text3); font-family: var(--mono); }
  .saient-btn {
    display: flex; align-items: center; gap: 5px;
    font-size: 11px; padding: 3px 8px;
    border-radius: 12px;
    border-color: rgba(108,142,245,0.3);
    color: var(--text2);
  }
  .saient-btn.on { border-color: var(--accent); color: var(--accent); }
  .saient-dot {
    width: 6px; height: 6px; border-radius: 50%;
    background: var(--text3);
  }
  .saient-btn.on .saient-dot { background: var(--accent); box-shadow: 0 0 4px var(--accent); }
  .wbtn {
    font-size: 12px; padding: 3px 8px;
    border-radius: 4px; color: var(--text3);
    border-color: transparent;
    background: transparent;
  }
  .wbtn:hover { color: var(--text); background: var(--bg3); }
  .wbtn.x:hover { color: var(--red); }
  .wbtn.stop { color: var(--amber); border-color: rgba(245,166,35,0.3); }
  .wbtn.stop:hover { border-color: var(--amber); }
</style>
