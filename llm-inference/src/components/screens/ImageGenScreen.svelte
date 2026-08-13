<script lang="ts">
  import { onMount, onDestroy } from "svelte";
  import { save } from "@tauri-apps/plugin-dialog";
  import { listen } from "@tauri-apps/api/event";
  import { ig, toast } from "../../lib/state.svelte.js";
  import * as T from "../../lib/tauri.js";
  import HfBrowser from "../HfBrowser.svelte";

  // Curated model repos for the downloader. Prefer tuned checkpoints for game assets;
  // raw base/HF folders are still supported but are not the best default.
  const BASE_MODELS = [
    { label: "SDXL Base 1.0", repo: "stabilityai/stable-diffusion-xl-base-1.0", install: "repo" as const },
    { label: "SDXL Turbo (fast)", repo: "stabilityai/sdxl-turbo", install: "repo" as const },
    { label: "DreamShaper XL", repo: "Lykon/dreamshaper-xl-1-0", install: "repo" as const },
    { label: "SD 1.5", repo: "stable-diffusion-v1-5/stable-diffusion-v1-5", install: "repo" as const },
    { label: "SD 2.1", repo: "stabilityai/stable-diffusion-2-1", install: "repo" as const },
  ];

  let browser = $state<{ target: string; filter: string; exts: string[]; title: string; suggestions: { label: string; repo: string; install?: "repo" | "files" }[] } | null>(null);
  function findModel(target: "checkpoint" | "lora") {
    browser = target === "lora"
      ? { target: "lora", filter: "text-to-image", exts: [".safetensors"], title: "Find a LoRA", suggestions: [] }
      : { target: "checkpoint", filter: "text-to-image", exts: [".safetensors", ".ckpt"], title: "Find an image model", suggestions: BASE_MODELS };
  }

  const SCHEDULERS = [
    { id: "auto", label: "Auto" },
    { id: "euler_a", label: "Euler a" },
    { id: "dpm++2m_karras", label: "DPM++ 2M Karras" },
    { id: "dpm++2m", label: "DPM++ 2M" },
    { id: "euler", label: "Euler" },
    { id: "ddim", label: "DDIM" },
    { id: "pndm", label: "PNDM" },
    { id: "lms", label: "LMS" },
  ];
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

  async function selectInstalledAsset(installedPath: string, installedTarget: string) {
    await igScanModels();
    if (installedTarget === "lora") ig.loraPath = installedPath;
    else ig.modelPath = installedPath;
    loadError = "";
  }

  async function generate() {
    if (!ig.modelPath || ig.generating) return;
    ig.generating = true; ig.error = ""; ig.resultB64 = ""; ig.progress = 0; ig.progressTotal = ig.steps;
    try {
      const r = await T.runImggen({
        model_path: ig.modelPath, lora_path: ig.loraPath || undefined,
        prompt: ig.prompt, neg_prompt: ig.negPrompt,
        steps: ig.steps, cfg_scale: ig.cfg, seed: ig.seed,
        width: ig.width, height: ig.height, device: ig.device, scheduler: ig.scheduler,
        face_detail: ig.faceDetail,
        asset_guard: ig.assetGuard,
        asset_kind: ig.assetKind,
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
    ...ig.checkpoints.map(m => ({ path: (m as unknown as {path:string}).path, label: `[CKPT] ${(m as unknown as {label:string}).label}` })),
    ...ig.models.map(m => ({ path: (m as unknown as {path:string}).path, label: `[BASE] ${(m as unknown as {label:string}).label}` })),
  ]);
  const pct = $derived(ig.progressTotal > 0 ? Math.round((ig.progress / ig.progressTotal) * 100) : 0);

  let lightbox = $state(false);
  let hotModel = $state<string | null>(null);
  let loading = $state(false);
  let loadStatus = $state("");
  let loadError = $state("");

  let unlistenProgress: (() => void) | null = null;
  let unlistenArch: (() => void) | null = null;
  let unlistenGenProgress: (() => void) | null = null;
  let loadedFamily = $state("");

  onMount(async () => {
    igScanModels();
    hotModel = await T.imggenLoadedModel().catch(() => null);
    unlistenProgress = await listen<string>("igload-progress", (e) => {
      loadStatus = e.payload;
    });
    // Reset CFG/steps to what the just-loaded model actually wants (SD3.5 wants
    // cfg≈4.5, not the SDXL-era 7.0 default) rather than carrying over whatever was
    // set for the previously loaded model. Still fully user-editable afterward.
    unlistenArch = await listen<{
      family: string; default_cfg: number; default_steps: number;
    }>("igload-arch", (e) => {
      ig.cfg = e.payload.default_cfg;
      ig.steps = e.payload.default_steps;
      loadedFamily = e.payload.family;
    });
    // Rust/Python already emit real per-step progress during generation — nothing was
    // listening for it, so the progress bar sat at 0% for the whole run regardless of
    // model or step count.
    unlistenGenProgress = await listen<{ step: number; total: number }>("imggen_progress", (e) => {
      ig.progress = e.payload.step;
      ig.progressTotal = e.payload.total;
    });
  });

  onDestroy(() => {
    unlistenProgress?.();
    unlistenArch?.();
    unlistenGenProgress?.();
  });

  const hotLabel = $derived.by(() => {
    if (!hotModel) return null;
    const m = allModels.find(m => m.path === hotModel);
    return m ? m.label : hotModel.split("/").pop() ?? hotModel;
  });
  const selectedModelLabel = $derived.by(() => {
    if (!ig.modelPath) return "";
    const m = allModels.find(m => m.path === ig.modelPath);
    return m ? m.label : ig.modelPath.split("/").pop() ?? ig.modelPath;
  });
  const legacySdxlBaseCheckpointSelected = $derived(
    /(?:^|[\\/])sd[_-]?xl[_-]?base[_-]?1(?:[._-]?0)?\.safetensors$/i.test(ig.modelPath),
  );
  const installedSdxlBase = $derived.by(() => ig.models.find((model) => {
    const entry = model as unknown as { path: string; label: string };
    const normalized = entry.path.replaceAll("\\", "/").toLowerCase();
    return normalized.endsWith("/stable-diffusion-xl-base-1.0")
      || entry.label.toLowerCase() === "stable-diffusion-xl-base-1.0";
  }));
  const baseModelSelected = $derived.by(() => selectedModelLabel.startsWith("[BASE]"));
  const sketchBiasedModel = $derived.by(() => {
    const text = `${ig.modelPath} ${selectedModelLabel}`.toLowerCase();
    return /pencil|sketch|line.?art|monochrome|grayscale|greyscale|black.?white|\bbw\b/.test(text);
  });
  const modelChanged = $derived(!!hotModel && !!ig.modelPath && ig.modelPath !== hotModel);

  async function loadModel() {
    if (!ig.modelPath || loading) return;
    let modelPath = ig.modelPath;
    if (legacySdxlBaseCheckpointSelected) {
      if (!installedSdxlBase) {
        toast("That old SDXL Base file is incomplete. Choose SDXL Base 1.0 below to repair it.", "error", 7000);
        findModel("checkpoint");
        return;
      }
      modelPath = (installedSdxlBase as unknown as { path: string }).path;
      ig.modelPath = modelPath;
      toast("Switched from the old checkpoint to the complete SDXL Base folder.", "success");
    }
    loading = true;
    loadError = "";
    loadStatus = "Connecting…";
    try {
      const dev = await T.imggenLoad(modelPath, ig.loraPath || "", ig.device);
      hotModel = modelPath;
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
    <div style="display:flex;gap:4px;">
      <button class="ig-refresh-btn" onclick={igScanModels} style="flex:1">⟳ Refresh</button>
      <button class="ig-refresh-btn" onclick={() => findModel("checkpoint")} style="flex:1">⬇ Find a model</button>
    </div>

    {#if hotModel && !modelChanged}
      <div class="hot-badge">🔥 Hot — {hotLabel}{loadedFamily ? ` (${loadedFamily})` : ""}</div>
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
    {#if baseModelSelected}
      <div class="model-warning">Base/HF model folder. It may load, but tuned CKPT checkpoints usually give better game-asset output.</div>
    {/if}
    {#if legacySdxlBaseCheckpointSelected}
      <div class="model-warning">Old single-file SDXL Base download. {installedSdxlBase ? "Load will switch to the complete base folder." : "Press Load to open the repair download."}</div>
    {/if}
    {#if sketchBiasedModel}
      <div class="model-warning">Sketch-biased model. Good for drawn styles; use a color/game checkpoint for stronger color.</div>
    {/if}

    <div class="ig-section-label" style="margin-top:12px">LoRA</div>
    {#if ig.loras.length > 0}
      <select class="ig-select" bind:value={ig.loraPath}>
        <option value="">— none —</option>
        {#each ig.loras as l}
          <option value={(l as unknown as {path:string}).path}>{(l as unknown as {label:string}).label}</option>
        {/each}
      </select>
    {/if}
    <button class="ig-refresh-btn" onclick={() => findModel("lora")}>⬇ Find a LoRA</button>

    <div class="ig-section-label" style="margin-top:12px">Device</div>
    <select class="ig-select" bind:value={ig.device}>
      <option value="auto">Auto</option>
      <option value="cuda">CUDA</option>
      <option value="cpu">CPU</option>
    </select>

    <div class="ig-section-label" style="margin-top:12px">Scheduler</div>
    <select class="ig-select" bind:value={ig.scheduler}>
      {#each SCHEDULERS as s}
        <option value={s.id}>{s.label}</option>
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

    <label class="fd-row" title="Re-detail small/blurry faces at hi-res after generating (auto-skips close-ups). Adds a few seconds.">
      <input type="checkbox" bind:checked={ig.faceDetail} />
      <span>Face detailer</span>
    </label>
    <label class="fd-row" title="Bias generations toward one centered game-ready subject and away from character sheets, duplicates, palettes, text, and crops.">
      <input type="checkbox" bind:checked={ig.assetGuard} />
      <span>Game asset guard</span>
    </label>
    {#if ig.assetGuard}
      <div class="ig-section-label" style="margin-top:10px">Asset type</div>
      <select class="ig-select" bind:value={ig.assetKind}>
        <option value="humanoid">Humanoid</option>
        <option value="creature">Creature</option>
        <option value="building">Building</option>
        <option value="prop">Prop</option>
        <option value="free">Free prompt</option>
      </select>
    {/if}
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
  <div
    class="lightbox-backdrop"
    onclick={(e) => { if (e.target === e.currentTarget) closeLightbox(); }}
    role="presentation"
  >
    <button class="lightbox-close" onclick={closeLightbox}>✕</button>
    <img
      src="data:image/png;base64,{ig.resultB64}"
      alt="Generated fullscreen"
      class="lightbox-img"
    />
  </div>
{/if}

{#if browser}
  <HfBrowser
    target={browser.target}
    filter={browser.filter}
    exts={browser.exts}
    title={browser.title}
    suggestions={browser.suggestions}
    onClose={() => (browser = null)}
    onDone={selectInstalledAsset}
  />
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
  .model-warning { font-size: 10px; line-height: 1.35; color: var(--amber, #f59e0b); margin-top: 6px; padding: 6px 7px; border: 1px solid rgba(245,158,11,0.28); border-radius: var(--radius-sm); background: rgba(245,158,11,0.08); }
  .num-row { display: flex; flex-direction: column; gap: 6px; }
  .size-row { display: flex; align-items: center; gap: 8px; }
  .size-row select { flex: 1; }
  .fd-row { display: flex; align-items: center; gap: 6px; margin-top: 10px; font-size: 12px; color: var(--text2); cursor: pointer; user-select: none; }
  .fd-row input { cursor: pointer; }
  .ig-main { flex: 1; display: flex; flex-direction: column; padding: 16px; gap: 12px; overflow: hidden; }
  .ig-prompt-area { display: flex; flex-direction: column; gap: 8px; flex-shrink: 0; }
  .ig-prompt { width: 100%; resize: none; font-family: var(--sans); font-size: 13px; line-height: 1.6; }
  .ig-generate-btn { padding: 10px; font-size: 13px; font-weight: 600; background: rgba(108,142,245,0.12); border-color: rgba(108,142,245,0.4); color: var(--accent); border-radius: var(--radius); }
  .ig-progress-bar { height: 4px; background: var(--border); border-radius: 2px; overflow: hidden; flex-shrink: 0; }
  .ig-progress-fill { height: 100%; background: var(--accent); transition: width 0.3s; }
  .ig-error { color: var(--red); font-size: 12px; padding: 8px 10px; background: rgba(248,113,113,0.07); border: 1px solid rgba(248,113,113,0.25); border-radius: var(--radius-sm); }
  .ig-canvas { flex: 1; min-height: 0; display: flex; flex-direction: column; gap: 8px; overflow: hidden; }
  .ig-placeholder { display: flex; flex-direction: column; align-items: center; justify-content: center; flex: 1; }
  /* thumbnail in main panel */
  .img-wrap { position: relative; cursor: zoom-in; display: flex; align-items: center; justify-content: center; flex: 1; min-height: 0; width: 100%; overflow: hidden; }
  .img-wrap:hover .expand-hint { opacity: 1; }
  .expand-hint { position: absolute; top: 8px; right: 8px; background: rgba(0,0,0,0.6); color: #fff; font-size: 11px; padding: 3px 8px; border-radius: 4px; opacity: 0; transition: opacity 0.15s; pointer-events: none; }
  .gen-img { max-width: 100%; max-height: 100%; width: auto; height: auto; display: block; object-fit: contain; border-radius: var(--radius); border: 1px solid var(--border); }
  .img-footer { display: flex; align-items: center; gap: 12px; flex-shrink: 0; padding-bottom: 4px; }
  .img-meta { font-size: 11px; color: var(--text3); font-family: var(--mono); }
  /* lightbox */
  .lightbox-backdrop { position: fixed; inset: 0; z-index: 9999; background: rgba(0,0,0,0.92); display: flex; align-items: center; justify-content: center; cursor: zoom-out; }
  .lightbox-img { max-width: 95vw; max-height: 95vh; object-fit: contain; border-radius: var(--radius); box-shadow: 0 8px 48px rgba(0,0,0,0.8); cursor: default; }
  .lightbox-close { position: absolute; top: 16px; right: 20px; background: rgba(255,255,255,0.1); border: 1px solid rgba(255,255,255,0.2); color: #fff; font-size: 18px; width: 36px; height: 36px; border-radius: 50%; cursor: pointer; display: flex; align-items: center; justify-content: center; }
  .lightbox-close:hover { background: rgba(255,255,255,0.2); }
</style>
