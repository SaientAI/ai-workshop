<script lang="ts">
  import { listen } from "@tauri-apps/api/event";
  import { model, params, dual, chat, toast } from "../../lib/state.svelte.js";
  import * as T from "../../lib/tauri.js";
  import { friendlyLoadError, prettyPath } from "../../lib/format.js";
  import { STARTER_MODELS } from "../../lib/models.js";
  import ClearDataModal from "./ClearDataModal.svelte";

  let showClearModal = $state(false);

  // ── Model download ─────────────────────────────────────────────────────────
  let showDownload = $state(false);
  let downloading = $state("");
  let dlProgress = $state<{ downloaded: number; total: number; done?: boolean }>({ downloaded: 0, total: 0 });
  let dlError = $state("");
  const dlPct = $derived(dlProgress.total > 0 ? Math.round((dlProgress.downloaded / dlProgress.total) * 100) : 0);
  const fmtGB = (b: number) => (b / 1e9).toFixed(2) + " GB";

  // ── Hugging Face: paste any repo, pick a quant ───────────────────────────────
  let hfInput = $state("");
  let hfRepo = $state("");
  let hfFiles = $state<T.HfFile[]>([]);
  let hfListing = $state(false);
  let hfError = $state("");
  let hfToken = $state(localStorage.getItem("hf_token") ?? "");
  let hfShowToken = $state(false);
  $effect(() => { localStorage.setItem("hf_token", hfToken); });

  // Accept "owner/name", a repo URL, or a direct file URL (resolve/blob).
  function parseHf(raw: string): { repo: string; file?: string } | null {
    const s = raw.trim();
    if (!s) return null;
    let parts: string[];
    if (/^https?:\/\//i.test(s)) {
      try { parts = new URL(s).pathname.replace(/^\/+|\/+$/g, "").split("/"); }
      catch { return null; }
      if (parts.length < 2) return null;
      const repo = `${parts[0]}/${parts[1]}`;
      const marker = Math.max(parts.indexOf("resolve"), parts.indexOf("blob"));
      if (marker >= 0 && parts.length > marker + 2) {
        return { repo, file: decodeURIComponent(parts.slice(marker + 2).join("/")) };
      }
      return { repo };
    }
    parts = s.replace(/^\/+|\/+$/g, "").split("/");
    if (parts.length < 2) return null;
    if (parts.length > 2 && parts[parts.length - 1].toLowerCase().endsWith(".gguf")) {
      return { repo: `${parts[0]}/${parts[1]}`, file: parts.slice(2).join("/") };
    }
    return { repo: `${parts[0]}/${parts[1]}` };
  }

  async function listHf() {
    const parsed = parseHf(hfInput);
    if (!parsed) { hfError = 'Enter a repo like "owner/name" or a Hugging Face link.'; return; }
    hfRepo = parsed.repo;
    if (parsed.file) { hfFiles = []; await downloadFile(parsed.repo, parsed.file); return; }
    hfListing = true; hfError = ""; hfFiles = [];
    try {
      hfFiles = await T.hfListGguf(parsed.repo, hfToken);
    } catch (e) {
      hfError = String(e);
    } finally {
      hfListing = false;
    }
  }

  // Generic download: any repo + file → managed models dir.
  async function downloadFile(repo: string, file: string) {
    if (downloading) return;
    downloading = file.split("/").pop() ?? file;
    dlError = ""; dlProgress = { downloaded: 0, total: 0 };
    const dir = await T.getModelsDir().catch(() => model.modelsDir);
    const unlisten = await listen<{ downloaded: number; total: number; done?: boolean }>(
      "model-progress", (e) => { dlProgress = e.payload; });
    try {
      await T.downloadStarterModel(repo, file, dir, hfToken);
      await scanModels();          // surface the new model in the picker
      hfFiles = [];                // collapse the picker; the model is now in the list
      toast(`Downloaded ${file.split("/").pop()} — select it and Start server`, "success");
    } catch (e) {
      dlError = String(e);
      toast("Download failed — see the panel for details", "error");
    } finally {
      unlisten(); downloading = "";
    }
  }

  const downloadModel = (m: typeof STARTER_MODELS[number]) => downloadFile(m.repo, m.file);

  async function loadModel() {
    if (!model.path) { model.loadError = "No model selected — click a card in the list above."; return; }
    model.loading = true;
    model.loaded = false;
    model.loadError = "";
    model.loadStatus = "Starting model server…";
    try {
      const summary = await T.loadModel(model.path, model.gpuLayers, model.ctxSize);
      model.activeServerPort = await T.currentModelPort().catch(() => null);
      model.summary = summary;
      model.loaded = true;
      model.loadStatus = "";
    } catch (e) {
      model.loadError = friendlyLoadError(String(e));
      model.loadStatus = "";
    } finally {
      model.loading = false;
    }
  }

  async function scanModels() {
    try {
      model.modelsDir = await T.getModelsDir().catch(() => model.modelsDir);
      model.models = await T.scanModelsDir();
      if (model.models.length === 1) {
        model.selectedModel = model.models[0];
        model.path = model.models[0].gguf_path;
      }
    } catch (e) { console.error("scanModels", e); }
  }

  async function openModelsFolder() {
    model.modelsDir = await T.getModelsDir().catch(() => model.modelsDir);
    await T.openModelsDir().catch(console.error);
  }

  async function scanRunningServers() {
    try {
      model.runningServers = await T.scanRunningServers();
      model.runningServersScanned = true;
    } catch (e) { console.error(e); }
  }

  async function attachServer(port: number) {
    const srv = model.runningServers.find(s => s.port === port);
    if (!model.path && srv?.model_path) {
      model.path = srv.model_path;
      model.selectedModel = model.models.find(m => m.gguf_path === srv.model_path) ?? null;
    }
    if (!model.path) {
      model.loadError = "Select a model card first so the summary can be read from the GGUF file.";
      return;
    }
    model.loading = true; model.loaded = false; model.loadError = "";
    model.loadStatus = `Attaching to port ${port}…`;
    try {
      const summary = await T.attachModel(model.path, port);
      model.activeServerPort = port;
      model.summary = summary;
      model.loaded = true;
      model.loadStatus = "";
    } catch (e) {
      model.loadError = friendlyLoadError(String(e));
      model.loadStatus = "";
    } finally { model.loading = false; }
  }

  async function loadDrafter() {
    if (!dual.drafterPath) return;
    dual.drafterLoading = true; dual.drafterError = "";
    try { dual.drafterSummary = await T.loadDrafter(dual.drafterPath, model.gpuLayers, model.ctxSize); }
    catch (e) { dual.drafterError = String(e); }
    finally { dual.drafterLoading = false; }
  }

  async function loadCritic() {
    if (!dual.criticPath) return;
    dual.criticLoading = true; dual.criticError = "";
    try { dual.criticSummary = await T.loadCritic(dual.criticPath, model.gpuLayers, model.ctxSize); }
    catch (e) { dual.criticError = String(e); }
    finally { dual.criticLoading = false; }
  }

  function selectModel(idx: number) {
    const m = model.models[idx];
    if (!m) return;
    model.selectedModel = m;
    model.path = m.gguf_path;
  }

  // GPU health: true if we think we're running on CPU when the user didn't ask for it
  const gpuWarn = $derived.by(() => {
    if (!model.loaded || model.gpuLayers === 0) return null;
    const gpu = model.gpu as { available?: boolean } | null;
    if (gpu && gpu.available === false) {
      return "No NVIDIA GPU detected — inference will be very slow. Set GPU layers = 0 to silence this.";
    }
    if (chat.lastPerf && chat.lastPerf.tokens_per_sec < 3 && chat.lastPerf.tokens_generated > 5) {
      return `Generating at ${chat.lastPerf.tokens_per_sec.toFixed(1)} tok/s — likely running on CPU. Rebuild tinyq4 with --features cuda for GPU speed.`;
    }
    return null;
  });

  function sl(label: string, key: keyof typeof params, min: number, max: number, step: number) {
    return { label, key, min, max, step };
  }

  const sliders = [
    sl("Max tokens", "maxTokens", 64, 16384, 64),
    sl("Temperature", "temperature", 0, 2, 0.01),
    sl("Top-P", "topP", 0, 1, 0.01),
    sl("Top-K", "topK", 1, 200, 1),
    sl("Repeat penalty", "repeatPenalty", 1, 2, 0.01),
  ];
</script>

<div class="sidebar">
  <!-- Models -->
  <div class="sidebar-section">
    <div class="section-label" style="display:flex;justify-content:space-between;align-items:center;">
      Models
      <span class="dir-label" title={model.modelsDir}>{prettyPath(model.modelsDir)}</span>
    </div>

    {#if model.models.length === 0}
      <div class="no-models">
        <div class="no-icon">📂</div>
        <div class="no-title">No models yet</div>
        <div class="no-hint">Download one to get started, or drop <code>.gguf</code> files into the models folder.</div>
        <button class="load-btn" style="margin-top:8px" onclick={() => (showDownload = true)}>⬇ Download a model</button>
      </div>
    {:else}
      <div class="model-list">
        {#each model.models as m, i}
          <div
            class="model-card"
            class:model-card-sel={model.selectedModel?.gguf_path === m.gguf_path}
            role="button"
            tabindex="0"
            onclick={() => selectModel(i)}
            onkeydown={(e) => e.key === "Enter" && selectModel(i)}
          >
            <span class="model-name">{m.name}</span>
            <span class="model-size">{m.size_gb.toFixed(1)} GB</span>
          </div>
        {/each}
      </div>
    {/if}

    <div class="btn-row" style="margin-top:6px;">
      <button class="tab-action" onclick={openModelsFolder}>Open folder</button>
      <button class="tab-action" onclick={scanModels}>↻ Refresh</button>
      <button class="tab-action" onclick={() => (showDownload = !showDownload)}>⬇ Download</button>
    </div>

    {#if showDownload}
      <div class="dl-panel">
        {#if downloading}
          <div class="dl-name">{downloading}</div>
          <div class="dl-bar"><div class="dl-fill" style="width:{dlPct}%"></div></div>
          <div class="dl-meta">{dlPct}% · {fmtGB(dlProgress.downloaded)}{dlProgress.total ? ` / ${fmtGB(dlProgress.total)}` : ""}</div>
        {:else}
          {#each STARTER_MODELS as m}
            <button class="dl-model" onclick={() => downloadModel(m)}>
              <span class="dl-model-name">{m.name}</span>
              <span class="dl-model-size">⬇ {m.size}</span>
              <span class="dl-model-desc">{m.desc}</span>
            </button>
          {/each}

          <!-- Any Hugging Face GGUF repo -->
          <div class="hf-divider"><span>or any Hugging Face repo</span></div>
          <div class="hf-row">
            <input
              class="hf-input"
              placeholder="owner/name or HF link"
              bind:value={hfInput}
              onkeydown={(e) => e.key === "Enter" && listHf()}
              spellcheck="false"
              autocomplete="off"
            />
            <button class="tab-action" onclick={listHf} disabled={hfListing || !hfInput.trim()}>
              {hfListing ? "…" : "List"}
            </button>
          </div>
          <div class="hf-hint">Paste a GGUF repo (e.g. <code>bartowski/Llama-3.2-3B-Instruct-GGUF</code>) — downloads straight into your models folder.</div>
          <button class="hf-token-toggle" onclick={() => (hfShowToken = !hfShowToken)}>
            {hfShowToken ? "▾" : "▸"} HF access token (for gated models){hfToken ? " ✓" : ""}
          </button>
          {#if hfShowToken}
            <input
              class="hf-input"
              type="password"
              placeholder="hf_…  (stored locally, optional)"
              bind:value={hfToken}
              spellcheck="false"
              autocomplete="off"
            />
          {/if}

          {#if hfError}<div class="dl-err">{hfError}</div>{/if}

          {#if hfFiles.length}
            <div class="hf-files">
              <div class="hf-files-head">{hfRepo} · pick a quant</div>
              {#each hfFiles as f}
                <button class="hf-file" onclick={() => downloadFile(hfRepo, f.file)}>
                  <span class="hf-file-name">{f.file}</span>
                  <span class="hf-file-size">⬇ {fmtGB(f.size)}</span>
                </button>
              {/each}
            </div>
          {/if}

          {#if dlError}<div class="dl-err">{dlError}</div>{/if}
        {/if}
      </div>
    {/if}

    <div class="field">
      <label for="gpu-layers">GPU layers <span class="hint-label">(-1 = auto · 0 = CPU)</span></label>
      <input id="gpu-layers" type="number" bind:value={model.gpuLayers} min="-1" max="999" />
    </div>
    <div class="field">
      <label for="ctx-size">Context length</label>
      <input id="ctx-size" type="number" bind:value={model.ctxSize} min="512" max="131072" step="512" />
    </div>

    <div class="btn-row">
      <button class="load-btn" class:loading={model.loading} onclick={loadModel} disabled={model.loading}>
        {model.loading ? "⟳ Working…" : model.loaded ? "⟳ Restart server" : "▶ Start server"}
      </button>
      {#if model.loaded}
        <button class="tab-action btn-danger" onclick={() => T.stopModelServer(model.activeServerPort).then(() => { model.loaded = false; model.summary = null; model.activeServerPort = null; })}>Stop</button>
      {/if}
    </div>

    {#if model.loading && model.loadPhases.length}
      <div class="load-timeline">
        {#each model.loadPhases as phase}
          <div class="phase-row">
            <span class:done={phase.done}>{phase.done ? "✓" : "⟳"}</span>
            <span class:done={phase.done}>{phase.label}</span>
          </div>
        {/each}
      </div>
    {:else if model.loadStatus}
      <div class="load-status">⟳ {model.loadStatus}</div>
    {/if}

    {#if model.loadError}
      <div class="load-error">⚠ {model.loadError}</div>
    {/if}

    {#if model.summary}
      <div class="model-info">
        <div class="perf-row">Arch<span>{model.summary.architecture}</span></div>
        <div class="perf-row">Quant<span>{model.summary.quant}</span></div>
        <div class="perf-row">Layers<span>{model.summary.block_count}</span></div>
        <div class="perf-row">Context<span>{model.summary.context_length}</span></div>
      </div>
    {/if}
  </div>

  <!-- Running servers -->
  <div class="sidebar-section">
    <div class="section-label" style="display:flex;justify-content:space-between;align-items:center;">
      Running servers
      <button class="tab-action" onclick={scanRunningServers}>Scan</button>
    </div>
    {#if model.runningServers.length === 0}
      <div style="font-size:11px;color:var(--text3);">{model.runningServersScanned ? "None detected" : "Click Scan to detect"}</div>
    {:else}
      {#each model.runningServers as srv}
        <div class="server-row">
          <span class="server-id">{srv.model_id}</span>
          <span class="server-port">:{srv.port}</span>
          <button class="tab-action" onclick={() => attachServer(srv.port)}>Attach</button>
        </div>
      {/each}
    {/if}
  </div>

  <!-- Dual agent -->
  <div class="sidebar-section">
    <div class="section-label" style="display:flex;justify-content:space-between;align-items:center;">
      Dual Agent
      <label class="toggle-wrap">
        <input type="checkbox" bind:checked={dual.enabled} />
        <span class="toggle-slider"></span>
      </label>
    </div>
    {#if dual.enabled}
      <div class="field">
        <label for="drafter-select">Drafter model</label>
        <select id="drafter-select" onchange={(e) => { dual.drafterPath = e.currentTarget.value; dual.selectedDrafter = model.models.find(m => m.gguf_path === e.currentTarget.value) ?? null; }}>
          <option value="">— select —</option>
          {#each model.models as m}
            <option value={m.gguf_path} selected={dual.drafterPath === m.gguf_path}>{m.name}</option>
          {/each}
        </select>
        <button class="tab-action" onclick={loadDrafter} disabled={!dual.drafterPath || dual.drafterLoading}>
          {dual.drafterLoading ? "Loading…" : dual.drafterSummary ? "✓ Loaded" : "Load"}
        </button>
        {#if dual.drafterError}<div class="small-error">{dual.drafterError}</div>{/if}
      </div>
      <div class="field">
        <label for="critic-select">Critic model</label>
        <select id="critic-select" onchange={(e) => { dual.criticPath = e.currentTarget.value; dual.selectedCritic = model.models.find(m => m.gguf_path === e.currentTarget.value) ?? null; }}>
          <option value="">— select —</option>
          {#each model.models as m}
            <option value={m.gguf_path} selected={dual.criticPath === m.gguf_path}>{m.name}</option>
          {/each}
        </select>
        <button class="tab-action" onclick={loadCritic} disabled={!dual.criticPath || dual.criticLoading}>
          {dual.criticLoading ? "Loading…" : dual.criticSummary ? "✓ Loaded" : "Load"}
        </button>
        {#if dual.criticError}<div class="small-error">{dual.criticError}</div>{/if}
      </div>
    {/if}
  </div>

  <!-- Parameters -->
  <div class="sidebar-section">
    <div class="section-label">Parameters</div>
    {#each sliders as s}
      <div class="sl-field">
        <div class="sl-row">{s.label}<span>{params[s.key]}</span></div>
        <input type="range" min={s.min} max={s.max} step={s.step}
          value={params[s.key]}
          oninput={(e) => { (params as unknown as Record<string, number>)[s.key] = +e.currentTarget.value; }} />
      </div>
    {/each}
    <div class="sl-field">
      <div class="sl-row">Seed<span>{params.seed}</span></div>
      <input type="number" bind:value={params.seed} />
    </div>
  </div>

  <!-- GPU/speed warning -->
  {#if gpuWarn}
    <div class="gpu-warn">
      <div class="gpu-warn-head">⚠ Slow inference detected</div>
      <div class="gpu-warn-body">{gpuWarn}</div>
    </div>
  {/if}

  <!-- Last run perf -->
  {#if chat.lastPerf}
    <div class="sidebar-section">
      <div class="section-label">Last run</div>
      <div class="perf-row">Tokens<span>{chat.lastPerf.tokens_generated}</span></div>
      <div class="perf-row">Speed<span>{chat.lastPerf.tokens_per_sec.toFixed(1)} tok/s</span></div>
      <div class="perf-row">Time<span>{(chat.lastPerf.elapsed_ms / 1000).toFixed(1)}s</span></div>
    </div>
  {/if}

  <!-- Dep report -->
  {#if model.depReport && !model.depReport.all_ok}
    <div class="sidebar-section dep-report">
      <div class="section-label">Startup check</div>
      {#each model.depReport.deps as dep}
        <div class="dep-row" class:ok={dep.ok}>
          <span>{dep.ok ? "✓" : "✗"}</span>
          <span>{dep.name}</span>
          {#if !dep.ok}<span class="dep-detail">{dep.detail}</span>{/if}
        </div>
      {/each}
    </div>
  {/if}

  <!-- GPU -->
  <div class="sidebar-section" style="margin-top:auto;">
    <div class="section-label">GPU</div>
    {#if model.gpu}
      {#each Object.entries(model.gpu) as [k, v]}
        <div class="perf-row">{k}<span>{String(v)}</span></div>
      {/each}
    {:else}
      <div style="font-size:11px;color:var(--text3);">Detecting…</div>
    {/if}
  </div>

  <!-- Privacy footer -->
  <div class="privacy-footer">
    <button class="privacy-btn" onclick={() => (showClearModal = true)}>
      🔒 Privacy · Clear data
    </button>
    <div class="privacy-note">No data is ever sent externally</div>
  </div>
</div>

{#if showClearModal}
  <ClearDataModal onclose={() => (showClearModal = false)} />
{/if}

<style>
  .sidebar { width: 240px; min-width: 200px; background: var(--bg2); border-right: 1px solid var(--border); display: flex; flex-direction: column; overflow-y: auto; flex-shrink: 0; }
  .dir-label { font-size: 9px; color: var(--text3); font-family: var(--mono); text-transform: none; font-weight: 400; max-width: 120px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .no-models { text-align: center; padding: 12px 0; }
  .no-icon { font-size: 26px; opacity: 0.3; margin-bottom: 6px; }
  .no-title { font-size: 11px; font-weight: 700; color: var(--text2); margin-bottom: 4px; }
  .no-hint { font-size: 10px; color: var(--text3); line-height: 1.6; }
  .no-hint code { color: var(--accent); }
  .model-list { display: flex; flex-direction: column; gap: 3px; margin-bottom: 6px; }
  .model-name { font-size: 11px; font-weight: 700; flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .model-size { font-size: 10px; color: var(--text3); flex-shrink: 0; }
  .model-card { display: flex; align-items: center; gap: 4px; }
  .btn-row { display: flex; gap: 6px; flex-wrap: wrap; }

  /* Download panel */
  .dl-panel { margin-top: 8px; display: flex; flex-direction: column; gap: 6px; }
  .dl-model {
    display: grid; grid-template-columns: 1fr auto; gap: 2px 8px;
    text-align: left; padding: 8px 10px; border-radius: var(--radius-sm);
    background: var(--bg3); border: 1px solid var(--border); cursor: pointer;
  }
  .dl-model:hover { border-color: var(--accent); }
  .dl-model-name { font-size: 12px; font-weight: 600; color: var(--text); }
  .dl-model-size { font-size: 10px; color: var(--accent); font-family: var(--mono); align-self: center; }
  .dl-model-desc { grid-column: 1 / -1; font-size: 10px; color: var(--text3); line-height: 1.4; }
  /* Hugging Face arbitrary download */
  .hf-divider { display: flex; align-items: center; gap: 8px; margin: 4px 0 2px; }
  .hf-divider span { font-size: 9px; text-transform: uppercase; letter-spacing: 0.06em; color: var(--text3); white-space: nowrap; }
  .hf-divider::before, .hf-divider::after { content: ""; flex: 1; height: 1px; background: var(--border); }
  .hf-row { display: flex; gap: 6px; }
  .hf-input { flex: 1; font-size: 11px; font-family: var(--mono); padding: 5px 8px; }
  .hf-hint { font-size: 9px; color: var(--text3); line-height: 1.5; }
  .hf-hint code { color: var(--accent); font-size: 9px; word-break: break-all; }
  .hf-token-toggle { background: none; border: 0; padding: 2px 0; color: var(--text3); font-size: 10px; cursor: pointer; text-align: left; }
  .hf-token-toggle:hover { color: var(--text2); }
  .hf-files { display: flex; flex-direction: column; gap: 3px; margin-top: 2px; }
  .hf-files-head { font-size: 9px; color: var(--text3); font-family: var(--mono); margin-bottom: 2px; word-break: break-all; }
  .hf-file {
    display: flex; align-items: center; justify-content: space-between; gap: 8px;
    text-align: left; padding: 6px 8px; border-radius: var(--radius-sm);
    background: var(--bg3); border: 1px solid var(--border); cursor: pointer;
  }
  .hf-file:hover { border-color: var(--accent); }
  .hf-file-name { font-size: 10px; font-family: var(--mono); color: var(--text); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .hf-file-size { font-size: 10px; color: var(--accent); font-family: var(--mono); flex-shrink: 0; }
  .dl-name { font-size: 11px; font-family: var(--mono); color: var(--text2); word-break: break-all; }
  .dl-bar { height: 6px; background: var(--bg3); border-radius: 3px; overflow: hidden; }
  .dl-fill { height: 100%; background: var(--accent); border-radius: 3px; transition: width 0.3s ease; }
  .dl-meta { font-size: 10px; color: var(--text3); font-family: var(--mono); }
  .dl-err { font-size: 10px; color: var(--red); line-height: 1.4; }
  .load-timeline { margin-top: 8px; padding: 8px 10px; background: var(--bg3); border: 1px solid var(--border); border-radius: var(--radius-sm); }
  .phase-row { display: flex; align-items: center; gap: 6px; margin-bottom: 3px; font-size: 11px; font-family: var(--mono); color: var(--amber); }
  .phase-row span.done { color: var(--green); }
  .load-status { font-size: 10px; color: var(--amber); font-family: var(--mono); margin-top: 6px; }
  .load-error { font-size: 11px; color: var(--red); margin-top: 8px; padding: 7px 9px; background: rgba(248,113,113,0.07); border: 1px solid rgba(248,113,113,0.25); border-radius: var(--radius-sm); line-height: 1.5; }
  .model-info { margin-top: 10px; background: var(--bg3); border: 1px solid var(--border); border-radius: var(--radius-sm); padding: 8px 10px; }
  .server-row { display: flex; align-items: center; gap: 6px; margin-bottom: 6px; font-size: 11px; }
  .server-id { flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; color: var(--text2); }
  .server-port { font-family: var(--mono); color: var(--text3); }
  .hint-label { font-size: 9px; color: var(--text3); font-weight: 400; text-transform: none; letter-spacing: 0; }
  .toggle-wrap { display: flex; align-items: center; gap: 0; cursor: pointer; }
  .toggle-wrap input { display: none; }
  .toggle-slider { width: 28px; height: 15px; background: var(--border); border-radius: 8px; position: relative; transition: background 0.2s; }
  .toggle-wrap input:checked + .toggle-slider { background: var(--accent); }
  .toggle-slider::after { content: ""; position: absolute; left: 2px; top: 2px; width: 11px; height: 11px; border-radius: 50%; background: #fff; transition: transform 0.2s; }
  .toggle-wrap input:checked + .toggle-slider::after { transform: translateX(13px); }
  .small-error { font-size: 10px; color: var(--red); margin-top: 4px; }
  .dep-report .dep-row { display: flex; gap: 6px; font-size: 11px; color: var(--red); margin-bottom: 4px; }
  .dep-row.ok { color: var(--green); }
  .dep-detail { color: var(--text3); font-size: 10px; font-family: var(--mono); }
  .gpu-warn { margin: 4px 12px 0; padding: 8px 10px; background: rgba(251,191,36,0.08); border: 1px solid rgba(251,191,36,0.35); border-radius: var(--radius-sm); }
  .gpu-warn-head { font-size: 10px; font-weight: 700; color: var(--amber); margin-bottom: 4px; }
  .gpu-warn-body { font-size: 10px; color: var(--text2); line-height: 1.55; }
  .privacy-footer {
    border-top: 1px solid var(--border);
    padding: 10px 12px;
    display: flex; flex-direction: column; gap: 4px;
  }
  .privacy-btn {
    font-size: 11px; padding: 5px 10px;
    color: var(--text2); border-color: var(--border);
    background: transparent; border-radius: var(--radius-sm);
    text-align: left; cursor: pointer; width: 100%;
    transition: background 0.15s;
  }
  .privacy-btn:hover { background: var(--bg3); color: var(--text); }
  .privacy-note { font-size: 9px; color: var(--text3); padding-left: 2px; }
</style>
