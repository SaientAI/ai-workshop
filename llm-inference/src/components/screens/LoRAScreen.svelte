<script lang="ts">
  import { onMount } from "svelte";
  import { open } from "@tauri-apps/plugin-dialog";
  import { lora, model } from "../../lib/state.svelte.js";
  import * as T from "../../lib/tauri.js";

  onMount(() => scanModels());

  async function scanModels() {
    // No dedicated scan command — LoRA uses the main model list from the sidebar
  }

  async function browseDataset() {
    const p = await open({ directory: true }).catch(() => null);
    if (p) lora.datasetDir = p as string;
  }

  async function cleanDataset() {
    if (!lora.datasetDir) return;
    lora.cleaning = true;
    await T.loraCleanDataset(lora.datasetDir).catch(e => { lora.error = String(e); });
    lora.cleaning = false;
  }

  async function startTraining() {
    if (!lora.modelPath || !lora.datasetDir) { lora.error = "Select a base model and dataset folder first."; return; }
    lora.training = true; lora.done = false; lora.error = ""; lora.log = [];
    try {
      await T.loraStart({
        model_path: lora.modelPath, dataset_dir: lora.datasetDir,
        output_name: lora.outputName, output_dir: lora.outputDir,
        rank: lora.rank, alpha: lora.alpha, lr: lora.lr,
        epochs: lora.epochs, batch_size: lora.batchSize, resolution: lora.resolution,
      });
    } catch (e) { lora.error = String(e); lora.training = false; }
  }

  async function stopTraining() {
    await T.loraStop().catch(() => {});
    lora.training = false;
  }

  const progress = $derived(lora.totalSteps > 0 ? Math.round((lora.step / lora.totalSteps) * 100) : 0);
</script>

<div class="ig-layout">
  <div class="ig-sidebar">
    <div class="ig-section-label">Base model</div>
    <select class="ig-select" bind:value={lora.modelPath}>
      <option value="">— select model —</option>
      {#each model.models as m}
        <option value={m.gguf_path}>{m.name}</option>
      {/each}
    </select>

    <div class="ig-section-label" style="margin-top:12px">Dataset folder</div>
    <div style="display:flex;gap:6px;">
      <input type="text" bind:value={lora.datasetDir} placeholder="~/dataset/" style="flex:1;" />
      <button class="tab-action" onclick={browseDataset}>…</button>
    </div>
    <button class="tab-action" onclick={cleanDataset} disabled={lora.cleaning} style="width:100%;margin-top:4px;">
      {lora.cleaning ? "Cleaning…" : "🧹 Clean dataset"}
    </button>

    <div class="ig-section-label" style="margin-top:12px">Output</div>
    <input type="text" bind:value={lora.outputName} placeholder="my_lora" />
    <input type="text" bind:value={lora.outputDir} placeholder="~/loras/" style="margin-top:4px;" />

    <div class="ig-section-label" style="margin-top:12px">Hyperparams</div>
    <div class="sl-field"><div class="sl-row">Rank<span>{lora.rank}</span></div><input type="range" min="4" max="128" step="4" bind:value={lora.rank} /></div>
    <div class="sl-field"><div class="sl-row">Alpha<span>{lora.alpha}</span></div><input type="range" min="4" max="128" step="4" bind:value={lora.alpha} /></div>
    <div class="sl-field"><div class="sl-row">Learning rate<span>{lora.lr.toExponential(1)}</span></div><input type="range" min="0.000001" max="0.001" step="0.000001" bind:value={lora.lr} /></div>
    <div class="sl-field"><div class="sl-row">Epochs<span>{lora.epochs}</span></div><input type="range" min="1" max="100" step="1" bind:value={lora.epochs} /></div>
    <div class="sl-field"><div class="sl-row">Batch size<span>{lora.batchSize}</span></div><input type="range" min="1" max="8" step="1" bind:value={lora.batchSize} /></div>
    <div class="sl-field"><div class="sl-row">Resolution<span>{lora.resolution}</span></div>
      <select bind:value={lora.resolution}><option>512</option><option>768</option><option>1024</option></select>
    </div>
  </div>

  <div class="ig-main">
    {#if lora.training}
      <div class="train-status">
        <div class="train-header">
          <span class="train-dot"></span>
          <span>Epoch {lora.epoch}/{lora.totalEpochs} · Step {lora.step}/{lora.totalSteps}</span>
          {#if lora.loss !== null}<span class="loss">loss: {lora.loss.toFixed(4)}</span>{/if}
        </div>
        <div class="ig-progress-bar"><div class="ig-progress-fill" style="width:{progress}%"></div></div>
        <button class="tab-action btn-danger" onclick={stopTraining} style="margin-top:8px;">■ Stop</button>
      </div>
    {:else}
      <button class="ig-generate-btn" onclick={startTraining} disabled={!lora.modelPath || !lora.datasetDir}>
        🎛 Start Training
      </button>
    {/if}

    {#if lora.error}
      <div class="ig-error">{lora.error}</div>
    {/if}

    {#if lora.done}
      <div class="done-box">✓ Training complete: <code>{lora.outputPath}</code></div>
    {/if}

    <div class="log-box">
      {#each lora.log as line}
        <div class="log-line" class:err={line.type === "err"}>{line.text}</div>
      {/each}
    </div>
  </div>
</div>

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
  .loss { margin-left: auto; font-family: var(--mono); font-size: 11px; color: var(--text3); }
  .done-box { padding: 10px 12px; background: rgba(0,214,143,0.07); border: 1px solid rgba(0,214,143,0.3); border-radius: var(--radius-sm); font-size: 12px; color: var(--green); }
  .done-box code { font-family: var(--mono); }
  .log-box { flex: 1; overflow-y: auto; background: var(--bg3); border: 1px solid var(--border); border-radius: var(--radius-sm); padding: 8px 10px; font-family: var(--mono); font-size: 11px; }
  .log-line { color: var(--text2); margin-bottom: 2px; white-space: pre-wrap; word-break: break-all; }
  .log-line.err { color: var(--red); }
</style>
