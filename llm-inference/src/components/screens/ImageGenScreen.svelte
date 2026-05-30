<script lang="ts">
  import { onMount, onDestroy } from "svelte";
  import { save } from "@tauri-apps/plugin-dialog";
  import { listen } from "@tauri-apps/api/event";
  import { ig } from "../../lib/state.svelte.js";
  import * as T from "../../lib/tauri.js";

  const SCHEDULERS = ["dpm++2m_karras","euler_a","euler","ddim","pndm","lms"];
  const SIZES = [512, 768, 1024, 1280];

  async function igScanModels() {
    try {
      const [models, checkpoints, loras] = await Promise.all([
        T.imggenScanModels(),
        T.imggenScanCheckpoints(),
        T.imggenScanLoras(),
      ]);
      ig.models = models as typeof ig.models;
      ig.checkpoints = checkpoints as typeof ig.checkpoints;
      ig.loras = loras as typeof ig.loras;
    } catch (e) { console.error(e); }
  }

  async function generate() {
    if (!ig.modelPath || ig.generating) return;
    ig.generating = true; ig.error = ""; ig.resultB64 = ""; ig.progress = 0;
    try {
      const r = await T.runImggen({
        model_path: ig.modelPath, lora_path: ig.loraPath || undefined,
        prompt: ig.prompt, neg_prompt: ig.negPrompt,
        steps: ig.steps, cfg_scale: ig.cfg, seed: ig.seed,
        width: ig.width, height: ig.height, device: ig.device, scheduler: ig.scheduler,
      });
      ig.resultB64 = r.base64_png;
      ig.elapsed = r.elapsed;
    } catch (e) { ig.error = String(e); }
    finally { ig.generating = false; }
  }

  async function saveImage() {
    if (!ig.resultB64) return;
    const path = await save({ filters: [{ name: "PNG", extensions: ["png"] }], defaultPath: "image.png" }).catch(() => null);
    if (!path) return;
    await T.writeBinaryB64(path, ig.resultB64).catch((e: unknown) => { ig.error = String(e); });
  }

  const allModels = $derived([
    ...ig.models.map(m => ({ path: (m as unknown as {path:string}).path, label: `[HF] ${(m as unknown as {label:string}).label}` })),
    ...ig.checkpoints.map(m => ({ path: (m as unknown as {path:string}).path, label: `[CKPT] ${(m as unknown as {label:string}).label}` })),
  ]);
  const pct = $derived(ig.progressTotal > 0 ? Math.round((ig.progress / ig.progressTotal) * 100) : 0);

  let lightbox = $state(false);
  let hotModel = $state<string | null>(null);
  let loading = $state(false);
  let loadStatus = $state("");
  let loadError = $state("");

  let unlistenProgress: (() => void) | null = null;

  onMount(async () => {
    igScanModels();
    hotModel = await T.imggenLoadedModel().catch(() => null);
    unlistenProgress = await listen<string>("igload-progress", (e) => {
      loadStatus = e.payload;
    });
  });

  onDestroy(() => {
    unlistenProgress?.();
  });

  const hotLabel = $derived.by(() => {
    if (!hotModel) return null;
    const m = allModels.find(m => m.path === hotModel);
    return m ? m.label : hotModel.split("/").pop() ?? hotModel;
  });

  const modelChanged = $derived(!!hotModel && !!ig.modelPath && ig.modelPath !== hotModel);

  async function loadModel() {
    if (!ig.modelPath || loading) return;
    loading = true;
    loadError = "";
    loadStatus = "Connecting…";
    try {
      const dev = await T.imggenLoad(ig.modelPath, ig.loraPath || "", ig.device);
      hotModel = ig.modelPath;
      ig.vramFreed = false;
      loadStatus = "";
      console.log("Model loaded on", dev);
    } catch (e) {
      loadError = String(e);
      loadStatus = "";
    } finally {
      loading = false;
    }
  }

  async function unloadModel() {
    await T.imggenUnload().catch(() => null);
    hotModel = null;
    ig.vramFreed = true;
  }

  function openLightbox() { if (ig.resultB64) lightbox = true; }
  function closeLightbox() { lightbox = false; }

  function onKeydown(e: KeyboardEvent) {
    if (lightbox && e.key === "Escape") closeLightbox();
  }
</script>

<div class="ig-layout">
  <div class="ig-sidebar">
    <div class="ig-section-label">Model</div>
    <select class="ig-select" bind:value={ig.modelPath}>
      <option value="">— select model —</option>
      {#each allModels as m}
        <option value={m.path}>{m.label}</option>
      {/each}
    </select>
    <button class="ig-refresh-btn" onclick={igScanModels}>⟳ Refresh</button>

    {#if hotModel && !modelChanged}
      <div class="hot-badge">🔥 Hot — {hotLabel}</div>
      <button class="ig-unload-btn" onclick={unloadModel}>Unload / Free VRAM</button>
    {:else}
      <button class="ig-load-btn" onclick={loadModel} disabled={!ig.modelPath || loading}>
        {loading ? "⏳ Loading…" : modelChanged ? "↺ Reload Model" : "⚡ Load Model"}
      </button>
      {#if loading && loadStatus}
        <div class="load-status">{loadStatus}</div>
      {/if}
      {#if loadError}<div class="load-error">{loadError}</div>{/if}
    {/if}

    {#if ig.loras.length > 0}
      <div class="ig-section-label" style="margin-top:12px">LoRA</div>
      <select class="ig-select" bind:value={ig.loraPath}>
        <option value="">— none —</option>
        {#each ig.loras as l}
          <option value={(l as unknown as {path:string}).path}>{(l as unknown as {label:string}).label}</option>
        {/each}
      </select>
    {/if}

    <div class="ig-section-label" style="margin-top:12px">Device</div>
    <select class="ig-select" bind:value={ig.device}>
      <option value="auto">Auto</option>
      <option value="cuda">CUDA</option>
      <option value="cpu">CPU</option>
    </select>

    <div class="ig-section-label" style="margin-top:12px">Scheduler</div>
    <select class="ig-select" bind:value={ig.scheduler}>
      {#each SCHEDULERS as s}
        <option value={s}>{s}</option>
      {/each}
    </select>

    <div class="ig-section-label" style="margin-top:12px">Steps / CFG / Seed</div>
    <div class="num-row">
      <div class="sl-field">
        <div class="sl-row">Steps<span>{ig.steps}</span></div>
        <input type="range" min="10" max="150" step="1" bind:value={ig.steps} />
      </div>
      <div class="sl-field">
        <div class="sl-row">CFG<span>{ig.cfg.toFixed(1)}</span></div>
        <input type="range" min="1" max="20" step="0.5" bind:value={ig.cfg} />
      </div>
    </div>
    <input type="number" placeholder="Seed (-1 = random)" bind:value={ig.seed} style="width:100%;margin-top:6px;" />

    <div class="ig-section-label" style="margin-top:12px">Size</div>
    <div class="size-row">
      <select bind:value={ig.width}>{#each SIZES as s}<option>{s}</option>{/each}</select>
      <span>×</span>
      <select bind:value={ig.height}>{#each SIZES as s}<option>{s}</option>{/each}</select>
    </div>
  </div>

  <div class="ig-main">
    <div class="ig-prompt-area">
      <textarea class="ig-prompt" placeholder="Positive prompt…" bind:value={ig.prompt} rows="3"></textarea>
      <textarea class="ig-prompt" placeholder="Negative prompt…" bind:value={ig.negPrompt} rows="2" style="font-size:11px;color:var(--text3);"></textarea>
      <button class="ig-generate-btn" onclick={generate} disabled={!hotModel || ig.generating || !!modelChanged}>
        {ig.generating ? `Generating… ${pct}%` : !hotModel ? "Load a model first" : modelChanged ? "Reload model to generate" : "🖼 Generate"}
      </button>
    </div>

    {#if ig.generating}
      <div class="ig-progress-bar">
        <div class="ig-progress-fill" style="width:{pct}%"></div>
      </div>
    {/if}

    <div class="ig-canvas">
      {#if ig.error}
        <div class="ig-error">{ig.error}</div>
      {:else if ig.resultB64}
        <div class="img-wrap" role="button" tabindex="0" onclick={openLightbox} onkeydown={(e) => e.key === "Enter" && openLightbox()} title="Click to expand">
          <img src="data:image/png;base64,{ig.resultB64}" alt="Generated" class="gen-img" />
          <div class="expand-hint">⛶ Expand</div>
        </div>
        <div class="img-footer">
          <span class="img-meta">{ig.elapsed.toFixed(1)}s</span>
          <button class="tab-action" onclick={saveImage}>💾 Save PNG</button>
        </div>
      {:else}
        <div class="ig-placeholder">
          <div style="font-size:48px;opacity:0.15">🖼</div>
          <div style="color:var(--text3);font-size:13px;margin-top:8px">Image will appear here</div>
        </div>
      {/if}
    </div>
  </div>
</div>

<svelte:window onkeydown={onKeydown} />

{#if lightbox}
  <!-- svelte-ignore a11y_click_events_have_key_events -->
  <div class="lightbox-backdrop" onclick={closeLightbox} role="presentation">
    <button class="lightbox-close" onclick={closeLightbox}>✕</button>
    <img
      src="data:image/png;base64,{ig.resultB64}"
      alt="Generated fullscreen"
      class="lightbox-img"
      onclick={(e) => e.stopPropagation()}
    />
  </div>
{/if}

<style>
  .ig-layout { display: flex; flex: 1; overflow: hidden; }
  .ig-sidebar { width: 220px; flex-shrink: 0; border-right: 1px solid var(--border); padding: 16px; background: var(--bg2); overflow-y: auto; }
  .ig-section-label { font-size: 10px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; color: var(--text3); margin-bottom: 8px; }
  .ig-select { width: 100%; margin-bottom: 4px; }
  .ig-refresh-btn { font-size: 10px; padding: 3px 8px; width: 100%; margin-top: 4px; }
  .hot-badge { font-size: 10px; color: var(--amber, #f59e0b); margin: 6px 0 2px; font-weight: 600; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .ig-load-btn { width: 100%; margin-top: 6px; padding: 6px; font-size: 11px; font-weight: 600; background: rgba(108,142,245,0.15); border-color: rgba(108,142,245,0.4); color: var(--accent); border-radius: var(--radius); }
  .ig-unload-btn { width: 100%; margin-top: 4px; padding: 4px; font-size: 10px; background: rgba(248,113,113,0.08); border-color: rgba(248,113,113,0.3); color: var(--red, #f87171); border-radius: var(--radius); }
  .load-status { font-size: 10px; color: var(--text3); margin-top: 4px; word-break: break-word; font-style: italic; }
  .load-error { font-size: 10px; color: var(--red, #f87171); margin-top: 4px; word-break: break-word; }
  .num-row { display: flex; flex-direction: column; gap: 6px; }
  .size-row { display: flex; align-items: center; gap: 8px; }
  .size-row select { flex: 1; }
  .ig-main { flex: 1; display: flex; flex-direction: column; padding: 16px; gap: 12px; overflow: hidden; }
  .ig-prompt-area { display: flex; flex-direction: column; gap: 8px; flex-shrink: 0; }
  .ig-prompt { width: 100%; resize: none; font-family: var(--sans); font-size: 13px; line-height: 1.6; }
  .ig-generate-btn { padding: 10px; font-size: 13px; font-weight: 600; background: rgba(108,142,245,0.12); border-color: rgba(108,142,245,0.4); color: var(--accent); border-radius: var(--radius); }
  .ig-progress-bar { height: 4px; background: var(--border); border-radius: 2px; overflow: hidden; flex-shrink: 0; }
  .ig-progress-fill { height: 100%; background: var(--accent); transition: width 0.3s; }
  .ig-error { color: var(--red); font-size: 12px; padding: 8px 10px; background: rgba(248,113,113,0.07); border: 1px solid rgba(248,113,113,0.25); border-radius: var(--radius-sm); }
  .ig-canvas { flex: 1; min-height: 0; display: flex; flex-direction: column; gap: 8px; overflow-y: auto; }
  .ig-placeholder { display: flex; flex-direction: column; align-items: center; justify-content: center; flex: 1; }
  /* thumbnail in main panel */
  .img-wrap { position: relative; cursor: zoom-in; display: inline-block; width: 100%; }
  .img-wrap:hover .expand-hint { opacity: 1; }
  .expand-hint { position: absolute; top: 8px; right: 8px; background: rgba(0,0,0,0.6); color: #fff; font-size: 11px; padding: 3px 8px; border-radius: 4px; opacity: 0; transition: opacity 0.15s; pointer-events: none; }
  .gen-img { width: 100%; height: auto; display: block; border-radius: var(--radius); border: 1px solid var(--border); }
  .img-footer { display: flex; align-items: center; gap: 12px; flex-shrink: 0; padding-bottom: 4px; }
  .img-meta { font-size: 11px; color: var(--text3); font-family: var(--mono); }
  /* lightbox */
  .lightbox-backdrop { position: fixed; inset: 0; z-index: 9999; background: rgba(0,0,0,0.92); display: flex; align-items: center; justify-content: center; cursor: zoom-out; }
  .lightbox-img { max-width: 95vw; max-height: 95vh; object-fit: contain; border-radius: var(--radius); box-shadow: 0 8px 48px rgba(0,0,0,0.8); cursor: default; }
  .lightbox-close { position: absolute; top: 16px; right: 20px; background: rgba(255,255,255,0.1); border: 1px solid rgba(255,255,255,0.2); color: #fff; font-size: 18px; width: 36px; height: 36px; border-radius: 50%; cursor: pointer; display: flex; align-items: center; justify-content: center; }
  .lightbox-close:hover { background: rgba(255,255,255,0.2); }
</style>
