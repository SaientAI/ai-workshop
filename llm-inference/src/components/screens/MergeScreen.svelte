<script lang="ts">
  import { merge, model } from "../../lib/state.svelte.js";
  import * as T from "../../lib/tauri.js";
  import HfBrowser from "../HfBrowser.svelte";

  // No dedicated merge scan command — uses the main model list from the sidebar
  let showBrowser = $state(false);
  async function rescanModels() {
    model.models = await T.scanModelsDir().catch(() => model.models);
  }

  async function run() {
    if (!merge.modelA || !merge.modelB) { merge.error = "Select at least model A and B."; return; }
    merge.running = true; merge.done = false; merge.error = ""; merge.log = [];
    const dir = merge.outputDir || ".";
    const outputPath = `${dir}/${merge.outputName}.safetensors`;
    try {
      await T.mergeRun({
        model_a: merge.modelA, model_b: merge.modelB,
        model_c: merge.modelC || undefined,
        method: merge.method, weight: merge.weight,
        output_path: outputPath,
      });
    } catch (e) { merge.error = String(e); merge.running = false; }
  }

  const pct = $derived(merge.total > 0 ? Math.round((merge.progress / merge.total) * 100) : 0);
</script>

<div class="ig-layout">
  <div class="ig-sidebar">
    <div class="ig-section-label">Models</div>
    <button class="ig-refresh-btn" onclick={() => (showBrowser = true)} style="margin-bottom:8px;width:100%">⬇ Find a model</button>

    <div class="ig-section-label">Model A</div>
    <select class="ig-select" bind:value={merge.modelA}>
      <option value="">— select —</option>
      {#each model.models as m}<option value={m.gguf_path}>{m.name}</option>{/each}
    </select>

    <div class="ig-section-label" style="margin-top:10px">Model B</div>
    <select class="ig-select" bind:value={merge.modelB}>
      <option value="">— select —</option>
      {#each model.models as m}<option value={m.gguf_path}>{m.name}</option>{/each}
    </select>

    <div class="ig-section-label" style="margin-top:10px">Model C (optional)</div>
    <select class="ig-select" bind:value={merge.modelC}>
      <option value="">— none —</option>
      {#each model.models as m}<option value={m.gguf_path}>{m.name}</option>{/each}
    </select>

    <div class="ig-section-label" style="margin-top:10px">Method</div>
    <select class="ig-select" bind:value={merge.method}>
      <option value="weighted_sum">Weighted sum</option>
      <option value="add_difference">Add difference (A + (B − C))</option>
    </select>

    <div class="sl-field" style="margin-top:10px">
      <div class="sl-row">Weight (α)<span>{merge.weight.toFixed(2)}</span></div>
      <input type="range" min="0" max="1" step="0.01" bind:value={merge.weight} />
    </div>

    <div class="ig-section-label" style="margin-top:10px">Output</div>
    <input type="text" bind:value={merge.outputName} placeholder="merged_model" />
    <input type="text" bind:value={merge.outputDir} placeholder="~/models/" style="margin-top:4px;" />
  </div>

  <div class="ig-main">
    {#if merge.running}
      <div class="train-status">
        <div class="train-header">
          <span class="train-dot"></span>
          <span>Merging… {pct}%</span>
        </div>
        <div class="ig-progress-bar"><div class="ig-progress-fill" style="width:{pct}%"></div></div>
        <button class="tab-action btn-danger" onclick={() => T.mergeCancel()} style="margin-top:8px;">■ Cancel</button>
      </div>
    {:else}
      <button class="ig-generate-btn" onclick={run} disabled={!merge.modelA || !merge.modelB}>
        🔀 Merge
      </button>
    {/if}

    {#if merge.error}
      <div class="ig-error">{merge.error}</div>
    {/if}
    {#if merge.done}
      <div class="done-box">✓ Merge complete: <code>{merge.outputPath}</code></div>
    {/if}

    <div class="log-box">
      {#each merge.log as line}
        <div class="log-line" class:err={line.type === "err"}>{line.text}</div>
      {/each}
    </div>
  </div>
</div>

{#if showBrowser}
  <HfBrowser
    target="gguf"
    filter="text-generation"
    exts={[".gguf"]}
    title="Find a model"
    onClose={() => (showBrowser = false)}
    onDone={rescanModels}
  />
{/if}

<style>
  .ig-layout { display: flex; flex: 1; overflow: hidden; }
  .ig-sidebar { width: 240px; flex-shrink: 0; border-right: 1px solid var(--border); padding: 16px; background: var(--bg2); overflow-y: auto; }
  .ig-section-label { font-size: 10px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; color: var(--text3); margin-bottom: 8px; }
  .ig-select { width: 100%; margin-bottom: 4px; }
  .ig-main { flex: 1; display: flex; flex-direction: column; padding: 16px; gap: 12px; overflow: hidden; }
  .ig-generate-btn { padding: 10px; font-size: 13px; font-weight: 600; background: rgba(108,142,245,0.12); border-color: rgba(108,142,245,0.4); color: var(--accent); border-radius: var(--radius); }
  .ig-progress-bar { height: 4px; background: var(--border); border-radius: 2px; overflow: hidden; margin-top: 8px; }
  .ig-progress-fill { height: 100%; background: var(--accent); transition: width 0.3s; }
  .ig-error { color: var(--red); font-size: 12px; padding: 8px 10px; background: rgba(248,113,113,0.07); border: 1px solid rgba(248,113,113,0.25); border-radius: var(--radius-sm); }
  .train-status { padding: 12px; background: var(--bg2); border: 1px solid var(--border); border-radius: var(--radius-sm); }
  .train-header { display: flex; align-items: center; gap: 8px; font-size: 12px; margin-bottom: 6px; }
  .train-dot { width: 6px; height: 6px; border-radius: 50%; background: var(--amber); animation: pulse 1.2s infinite; }
  @keyframes pulse { 0%,100%{opacity:1}50%{opacity:.3} }
  .done-box { padding: 10px 12px; background: rgba(0,214,143,0.07); border: 1px solid rgba(0,214,143,0.3); border-radius: var(--radius-sm); font-size: 12px; color: var(--green); }
  .done-box code { font-family: var(--mono); }
  .log-box { flex: 1; overflow-y: auto; background: var(--bg3); border: 1px solid var(--border); border-radius: var(--radius-sm); padding: 8px 10px; font-family: var(--mono); font-size: 11px; }
  .log-line { color: var(--text2); margin-bottom: 2px; white-space: pre-wrap; word-break: break-all; }
  .log-line.err { color: var(--red); }
</style>
