<script lang="ts">
  import { onMount } from "svelte";
  import { save } from "@tauri-apps/plugin-dialog";
  import { listen } from "@tauri-apps/api/event";
  import { video } from "../../lib/state.svelte.js";
  import * as T from "../../lib/tauri.js";

  const loaded = $derived(!!video.loadedPath && video.loadedPath === video.modelPath);
  const pct = $derived(video.progressTotal > 0 ? Math.round((video.progress / video.progressTotal) * 100) : 0);

  onMount(async () => {
    video.models = await T.videoScanModels().catch(() => []) as typeof video.models;
    const cur = await T.videoLoadedModel().catch(() => null);
    if (cur) { video.loadedPath = cur; video.modelPath = cur; }
    else if (video.models.length && !video.modelPath) video.modelPath = video.models[0].path;
  });

  async function refresh() {
    video.models = await T.videoScanModels().catch(() => video.models) as typeof video.models;
  }

  async function load() {
    if (!video.modelPath || video.loading) return;
    video.loading = true; video.error = ""; video.loadStatus = "Starting…";
    const un = await listen<string>("vidload-progress", (e) => { video.loadStatus = e.payload; });
    try {
      await T.videoLoad(video.modelPath);
      video.loadedPath = video.modelPath; video.loadStatus = "";
    } catch (e) { video.error = String(e); }
    finally { un(); video.loading = false; }
  }

  async function generate() {
    if (!loaded || video.generating || !video.prompt.trim()) return;
    video.generating = true; video.error = ""; video.resultB64 = "";
    video.progress = 0; video.progressTotal = video.steps;
    video.loadStatus = "Preparing…";
    // Phase messages (text-encoder load, then denoise) come over vidload-progress;
    // step progress comes over video_progress. The first real step clears the phase text.
    const unStatus = await listen<string>("vidload-progress", (e) => { video.loadStatus = e.payload; });
    const un = await listen<{ step: number; total: number }>("video_progress", (e) => {
      video.progress = e.payload.step; video.progressTotal = e.payload.total;
      video.loadStatus = "";
    });
    try {
      const r = await T.videoGenerate({
        prompt: video.prompt, neg_prompt: video.negPrompt, model_path: video.modelPath,
        num_frames: video.numFrames, steps: video.steps, cfg_scale: video.cfg,
        width: video.width, height: video.height, fps: video.fps, seed: video.seed,
      });
      video.resultB64 = r.base64_mp4; video.frames = r.frames; video.elapsed = r.elapsed;
    } catch (e) { video.error = String(e); }
    finally { un(); unStatus(); video.generating = false; video.loadStatus = ""; }
  }

  async function saveMp4() {
    if (!video.resultB64) return;
    const path = await save({ filters: [{ name: "MP4", extensions: ["mp4"] }], defaultPath: "clip.mp4" }).catch(() => null);
    if (path) await T.writeBinaryB64(path, video.resultB64).catch(() => {});
  }
</script>

<div class="ig-layout">
  <div class="ig-sidebar">
    <div class="vlabel">Model</div>
    <select class="vsel" bind:value={video.modelPath}>
      <option value="">— select a video model —</option>
      {#each video.models as m}<option value={m.path}>{m.label}</option>{/each}
    </select>
    <button class="vbtn-ghost" onclick={refresh}>⟳ Refresh</button>
    <button class="load-btn" onclick={load} disabled={!video.modelPath || video.loading}>
      {video.loading ? `⟳ ${video.loadStatus || "Loading…"}` : loaded ? "✓ Loaded · Reload" : "▶ Load Model"}
    </button>
    {#if video.models.length === 0}
      <div class="vhint">No video models found. Drop a diffusers Wan/LTX folder into your models dir, then Refresh.</div>
    {/if}

    <div class="vlabel" style="margin-top:14px">Params</div>
    <div class="vrow"><span>Frames</span><input type="number" bind:value={video.numFrames} min="9" max="161" step="4" /></div>
    <div class="vrow"><span>Steps</span><input type="number" bind:value={video.steps} min="5" max="60" /></div>
    <div class="vrow"><span>CFG</span><input type="number" bind:value={video.cfg} min="1" max="15" step="0.5" /></div>
    <div class="vrow"><span>Width</span><input type="number" bind:value={video.width} min="256" max="1280" step="16" /></div>
    <div class="vrow"><span>Height</span><input type="number" bind:value={video.height} min="256" max="720" step="16" /></div>
    <div class="vrow"><span>FPS</span><input type="number" bind:value={video.fps} min="8" max="30" /></div>
    <div class="vrow"><span>Seed</span><input type="number" bind:value={video.seed} /></div>
    <div class="vhint" style="margin-top:4px">−1 seed = random. ~49 frames @ 16 fps ≈ 3 s clip.</div>
  </div>

  <div class="ig-main">
    <textarea class="vprompt" placeholder="Describe the video…" bind:value={video.prompt}></textarea>
    <textarea class="vprompt vneg" placeholder="Negative prompt (what to avoid)" bind:value={video.negPrompt}></textarea>
    <button class="vgen" onclick={generate} disabled={!loaded || video.generating || !video.prompt.trim()}>
      {video.generating ? (video.progress > 0 ? `Generating… ${pct}%` : video.loadStatus || "Preparing…") : loaded ? "🎬 Generate" : "Load a model first"}
    </button>

    {#if video.generating}
      {#if video.progress > 0}
        <div class="vbar"><div class="vbar-fill" style="width:{pct}%"></div></div>
        <div class="vbar-label">Step {video.progress} / {video.progressTotal}</div>
      {:else}
        <div class="vbar vbar-indet"><div class="vbar-fill"></div></div>
        <div class="vbar-label">{video.loadStatus || "Preparing…"}</div>
      {/if}
    {/if}

    {#if video.error}
      <div class="verr">{video.error}</div>
    {:else if video.resultB64}
      <div class="vresult">
        <!-- svelte-ignore a11y_media_has_caption -->
        <video class="vplayer" controls autoplay loop src="data:video/mp4;base64,{video.resultB64}"></video>
        <div class="vmeta">{video.frames ?? ""} frames · generated in {video.elapsed}s</div>
        <button class="vbtn-ghost" onclick={saveMp4}>💾 Save MP4</button>
      </div>
    {:else if !video.generating}
      <div class="vplaceholder">
        <div style="font-size:48px;opacity:0.15">🎬</div>
        <div style="color:var(--text3);font-size:13px;margin-top:8px">Your clip will appear here</div>
      </div>
    {/if}
  </div>
</div>

<style>
  .vlabel { font-size: 10px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; color: var(--text3); margin-bottom: 6px; }
  .vsel { width: 100%; margin-bottom: 4px; }
  .vbtn-ghost { background: transparent; border: 1px solid var(--border); color: var(--text2); border-radius: var(--radius-sm); padding: 4px 8px; font-size: 11px; cursor: pointer; width: 100%; }
  .vbtn-ghost:hover { background: var(--bg3); color: var(--text); }
  .vhint { font-size: 9px; color: var(--text3); margin-top: 4px; line-height: 1.5; }
  .vrow { display: flex; align-items: center; gap: 6px; margin-bottom: 6px; }
  .vrow span { font-size: 11px; color: var(--text2); min-width: 48px; }
  .vrow input { flex: 1; font-family: var(--mono); }

  .ig-main { gap: 10px; }
  .vprompt { width: 100%; min-height: 70px; resize: vertical; font-family: var(--sans); font-size: 13px; line-height: 1.6; padding: 10px 12px; }
  .vneg { min-height: 42px; color: var(--text2); }
  .vgen { align-self: flex-start; padding: 9px 22px; font-size: 14px; font-weight: 600; background: var(--accent); border: none; color: #fff; border-radius: var(--radius); cursor: pointer; }
  .vgen:hover:not(:disabled) { background: #7a9cf7; }
  .vgen:disabled { opacity: 0.5; cursor: not-allowed; }

  .vbar { height: 5px; background: var(--bg3); border-radius: 3px; overflow: hidden; }
  .vbar-fill { height: 100%; background: var(--accent); border-radius: 3px; transition: width 0.3s; }
  /* Indeterminate (no step count yet — e.g. text-encoder loading): sliding sweep. */
  .vbar-indet .vbar-fill { width: 35%; transition: none; animation: vbar-sweep 1.1s ease-in-out infinite; }
  @keyframes vbar-sweep { 0% { margin-left: -35%; } 100% { margin-left: 100%; } }
  .vbar-label { font-size: 11px; color: var(--text3); font-family: var(--mono); }
  .verr { background: rgba(248,113,113,0.08); border: 1px solid rgba(248,113,113,0.3); color: var(--red); font-size: 12px; padding: 10px 12px; border-radius: var(--radius-sm); line-height: 1.5; white-space: pre-wrap; }

  .vresult { display: flex; flex-direction: column; gap: 8px; align-items: flex-start; }
  .vplayer { max-width: 100%; max-height: 60vh; border-radius: var(--radius); border: 1px solid var(--border); background: #000; }
  .vmeta { font-size: 11px; color: var(--text3); font-family: var(--mono); }
  .vplaceholder { flex: 1; display: flex; flex-direction: column; align-items: center; justify-content: center; min-height: 280px; }
</style>
