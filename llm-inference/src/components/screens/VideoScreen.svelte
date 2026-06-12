<script lang="ts">
  import { onMount } from "svelte";
  import { save } from "@tauri-apps/plugin-dialog";
  import { listen } from "@tauri-apps/api/event";
  import { video } from "../../lib/state.svelte.js";
  import * as T from "../../lib/tauri.js";

  const loaded = $derived(!!video.loadedPath && video.loadedPath === video.modelPath);
  const pct = $derived(video.progressTotal > 0 ? Math.round((video.progress / video.progressTotal) * 100) : 0);

  // ── Activity log (non-typable mini terminal under Params) ──────────────────
  let termEl: HTMLDivElement | undefined = $state();
  function logln(s: string) {
    const t = new Date().toLocaleTimeString([], { hour12: false });
    const last = video.log[video.log.length - 1];
    if (last && last.endsWith(s)) return;       // skip repeated identical lines
    video.log = [...video.log, `${t}  ${s}`].slice(-300);
  }
  // Auto-scroll to the newest line whenever the log grows.
  $effect(() => {
    void video.log.length;
    if (termEl) termEl.scrollTop = termEl.scrollHeight;
  });

  onMount(async () => {
    video.models = await T.videoScanModels().catch(() => []) as typeof video.models;
    video.loras = await T.videoScanLoras().catch(() => []) as typeof video.loras;
    const cur = await T.videoLoadedModel().catch(() => null);
    if (cur) { video.loadedPath = cur; video.modelPath = cur; }
    else if (video.models.length && !video.modelPath) video.modelPath = video.models[0].path;
    autoSet();   // snap params to the initially-selected model
  });

  async function refresh() {
    video.models = await T.videoScanModels().catch(() => video.models) as typeof video.models;
    video.loras = await T.videoScanLoras().catch(() => video.loras) as typeof video.loras;
  }

  // Per-model best/required settings. Different families want very different params
  // (FastWan: 3 steps/CFG1; CogVideoX: hard-locked 720×480/50/CFG6/8fps). "Auto"
  // snaps everything to the loaded model's sweet spot so you don't fight it.
  function presetFor(label: string, pipeline: string) {
    const l = label.toLowerCase(), p = pipeline.toLowerCase();
    if (p.includes("cogvideo"))
      // 49 frames is NATIVE for CogVideoX-5b-I2V — it MUST be 49 (the model is trained at
      // 49 and its temporal position embeddings assume it). A previous "25 frames to reduce
      // drift" tweak actually CORRUPTED the temporal embeddings → the whole clip diverged
      // into a rainbow/waffle field by mid-clip. 49 is both coherent AND fits 16 GB (~14.5 GB
      // peak with VAE tiling). Do not lower this below 49.
      return { width: 720, height: 480, numFrames: 49, steps: 50, cfg: 6, fps: 8, resLocked: true };
    if (l.includes("fastwan"))
      return { width: 832, height: 480, numFrames: 49, steps: 3,  cfg: 1, fps: 16, resLocked: false };
    if (l.includes("ti2v-5b") || l.includes("wan2.2"))
      return { width: 832, height: 480, numFrames: 49, steps: 30, cfg: 5, fps: 16, resLocked: false };
    return { width: 832, height: 480, numFrames: 49, steps: 30, cfg: 5, fps: 16, resLocked: false }; // Wan2.1 default
  }
  function autoSet() {
    const m = video.models.find((x) => x.path === video.modelPath);
    if (!m) return;
    const pr = presetFor(m.label, (m as any).pipeline ?? "");
    video.width = pr.width; video.height = pr.height; video.numFrames = pr.numFrames;
    video.steps = pr.steps; video.cfg = pr.cfg; video.fps = pr.fps; video.resLocked = pr.resLocked;
  }
  // One-click SD/HD. 720p rounds to 704 because Wan requires multiples of 32 (1280×720
  // would be silently adjusted to 1280×704 anyway) — set the real value so there's no
  // surprise. HD fits a CLEAN 16 GB card (~14 GB peak); the loader now guarantees that.
  function setRes(w: number, h: number) {
    if (video.resLocked) return;
    video.width = w; video.height = h;
  }

  // i2v: pick a still image, read it to base64 (strip the data: prefix for the daemon)
  function pickImage(e: Event) {
    const file = (e.target as HTMLInputElement).files?.[0];
    if (!file) return;
    const r = new FileReader();
    r.onload = () => {
      video.imageB64 = String(r.result).split(",")[1] || "";
      video.imageName = file.name;
    };
    r.readAsDataURL(file);
  }
  function clearImage() { video.imageB64 = ""; video.imageName = ""; }

  async function load() {
    if (!video.modelPath || video.loading) return;
    video.loading = true; video.error = ""; video.loadStatus = "Starting…";
    logln(`▶ loading ${video.models.find(m => m.path === video.modelPath)?.label ?? "model"}…`);
    const un = await listen<string>("vidload-progress", (e) => { video.loadStatus = e.payload; logln(e.payload); });
    try {
      // One GPU, one model. Drop the chat LLM first so its VRAM is gone before the video
      // model loads — otherwise HD generation OOMs on top of a resident chat server. (The
      // backend also kills stray servers and waits for the memory, but unloading here keeps
      // the chat engine's own state clean so it reloads correctly next time.)
      await T.unloadModel().catch(() => null);
      await T.videoLoad(video.modelPath, video.loraPath, video.loraStrength, video.numFrames, video.qualityMode ? "quality" : "fast");
      video.loadedPath = video.modelPath; video.loadStatus = "";
      logln(video.loraPath ? `✓ model ready (+ LoRA @ ${video.loraStrength})` : "✓ model ready");
    } catch (e) { video.error = String(e); logln(`✗ load failed: ${e}`); }
    finally { un(); video.loading = false; }
  }

  async function unload() {
    await T.videoUnload().catch(() => null);
    video.loadedPath = "";
    logln("✓ unloaded · VRAM freed");
  }

  async function generate() {
    if (!loaded || video.generating || !video.prompt.trim()) return;
    video.generating = true; video.error = ""; video.resultB64 = ""; video.enhanced = false;
    video.progress = 0; video.progressTotal = video.steps;
    video.loadStatus = "Preparing…";
    logln(`🎬 generate · ${video.width}×${video.height} · ${video.numFrames}f · ${video.steps} steps`);
    // Phase messages (text-encoder load, then denoise) come over vidload-progress;
    // step progress comes over video_progress. The first real step clears the phase text.
    const unStatus = await listen<string>("vidload-progress", (e) => { video.loadStatus = e.payload; logln(e.payload); });
    const un = await listen<{ step: number; total: number }>("video_progress", (e) => {
      video.progress = e.payload.step; video.progressTotal = e.payload.total;
      video.loadStatus = "";
      logln(`step ${e.payload.step}/${e.payload.total}`);
    });
    try {
      const r = await T.videoGenerate({
        prompt: video.prompt, neg_prompt: video.negPrompt, model_path: video.modelPath,
        num_frames: video.numFrames, steps: video.steps, cfg_scale: video.cfg,
        width: video.width, height: video.height, fps: video.fps, seed: video.seed,
        image_b64: video.imageB64 || undefined,
      });
      video.resultB64 = r.base64_mp4; video.frames = r.frames; video.elapsed = r.elapsed;
      logln(`✓ done · ${r.frames} frames in ${r.elapsed}s`);
    } catch (e) { video.error = String(e); logln(`✗ ${e}`); }
    finally { un(); unStatus(); video.generating = false; video.loadStatus = ""; }
  }

  async function enhance() {
    if (!video.resultB64 || video.enhancing) return;
    const stages: string[] = [];
    if (video.doRefine) stages.push("refine");
    if (video.doFace) stages.push("face");
    if (video.doUpscale) stages.push("upscale");
    if (video.doInterpolate) stages.push("interpolate");
    if (stages.length === 0) return;
    video.enhancing = true; video.error = ""; video.loadStatus = "Preparing…";
    video.progress = 0; video.progressTotal = 1;
    logln(`✨ quality pass: ${stages.join(" → ")}`);
    const unStatus = await listen<string>("vidload-progress", (e) => { video.loadStatus = e.payload; logln(e.payload); });
    const un = await listen<{ step: number; total: number }>("video_progress", (e) => {
      video.progress = e.payload.step; video.progressTotal = e.payload.total;
    });
    try {
      const r = await T.videoEnhance({
        video_b64: video.resultB64, fps: video.fps, stages, model_path: video.modelPath,
        prompt: video.prompt, neg_prompt: video.negPrompt, cfg_scale: video.cfg,
        refine_strength: video.refineStrength, refine_steps: video.refineSteps, interp_factor: 2,
      });
      video.resultB64 = r.enhanced_b64; video.frames = r.frames; video.enhanced = true;
      video.loadedPath = "";   // the generator was unloaded to free VRAM for the pass
      logln(`✓ enhanced · ${r.width}×${r.height} · ${r.frames} frames in ${r.elapsed}s`);
    } catch (e) { video.error = String(e); logln(`✗ ${e}`); }
    finally { un(); unStatus(); video.enhancing = false; video.loadStatus = ""; }
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
    <select class="vsel" bind:value={video.modelPath} onchange={autoSet}>
      <option value="">— select a video model —</option>
      {#each video.models as m}<option value={m.path}>{m.label}</option>{/each}
    </select>
    <button class="vbtn-ghost" onclick={refresh}>⟳ Refresh</button>
    <button class="load-btn" onclick={load} disabled={!video.modelPath || video.loading}>
      {video.loading ? `⟳ ${video.loadStatus || "Loading…"}` : loaded ? "✓ Loaded · Reload" : "▶ Load Model"}
    </button>
    {#if loaded && !video.loading}
      <button class="vbtn-ghost" onclick={unload} disabled={video.generating || video.enhancing}>⏏ Unload · Free VRAM</button>
    {/if}
    {#if video.models.length === 0}
      <div class="vhint">No video models found. Drop a diffusers Wan/LTX folder into your models dir, then Refresh.</div>
    {/if}

    <div class="vlabel" style="margin-top:14px">LoRA <span style="font-weight:400;text-transform:none;color:var(--text3)">(optional, applied at load)</span></div>
    <select class="vsel" bind:value={video.loraPath} disabled={video.loading}>
      <option value="">— none —</option>
      {#each video.loras as l}<option value={l.path}>{l.label}</option>{/each}
    </select>
    {#if video.loraPath}
      <div class="vrow"><span>Strength</span><input type="number" bind:value={video.loraStrength} min="0" max="1.5" step="0.05" disabled={video.loading} /></div>
      <div class="vhint">LoRA forces bf16 transformer. Re-Load after changing it.</div>
    {/if}

    <div class="vlabel" style="margin-top:14px">Quality <span style="font-weight:400;text-transform:none;color:var(--text3)">(applied at load)</span></div>
    <label class="vrow" style="cursor:pointer">
      <span>bf16 transformer <span style="font-size:0.85em;color:var(--accent2,#e0a060)">· experimental</span></span>
      <input type="checkbox" bind:checked={video.qualityMode} disabled={video.loading} />
    </label>
    <div class="vhint">
      {video.qualityMode
        ? "Experimental: full-precision transformer streamed from RAM (sharper, slower). Not all models support it — if a gen errors, untick and Re-Load to fall back to fast."
        : "Fast 4-bit transformer (recommended). Toggle on for higher fidelity (experimental), then Re-Load."}
    </div>

    <div class="vlabel" style="margin-top:14px; display:flex; justify-content:space-between; align-items:center">
      <span>Params</span>
      <button class="vauto" onclick={autoSet} title="Snap all params to this model's best/required settings">⚙ Auto</button>
    </div>
    <div class="vrow"><span>Frames</span><input type="number" bind:value={video.numFrames} min="9" max="161" step="4" /></div>
    <div class="vrow"><span>Steps</span><input type="number" bind:value={video.steps} min="1" max="60" /></div>
    <div class="vrow"><span>CFG</span><input type="number" bind:value={video.cfg} min="1" max="15" step="0.5" /></div>
    <div class="vrow"><span>Size</span>
      <div class="vsize">
        <button class="vsize-btn" class:on={!video.resLocked && video.width === 832 && video.height === 480}
                disabled={video.resLocked} onclick={() => setRes(832, 480)}
                title="832×480 — fastest and rock-solid stable">SD 480p</button>
        <button class="vsize-btn" class:on={!video.resLocked && video.width === 1280 && video.height === 704}
                disabled={video.resLocked} onclick={() => setRes(1280, 704)}
                title="1280×704 native HD — best on the Wan2.2-5B (~14 GB, fits a clean card). Slower.">HD 720p</button>
      </div>
    </div>
    <div class="vrow"><span>Width</span><input type="number" bind:value={video.width} min="256" max="1360" step="16" disabled={video.resLocked} /></div>
    <div class="vrow"><span>Height</span><input type="number" bind:value={video.height} min="256" max="768" step="16" disabled={video.resLocked} /></div>
    <div class="vrow"><span>FPS</span><input type="number" bind:value={video.fps} min="6" max="30" /></div>
    <div class="vrow"><span>Seed</span><input type="number" bind:value={video.seed} /></div>
    <div class="vhint" style="margin-top:4px">
      {#if video.resLocked}🔒 This model locks resolution — Width/Height fixed.{:else}−1 seed = random. ~49 frames @ 16 fps ≈ 3 s clip.{/if}
    </div>

    <div class="vlabel" style="margin-top:14px; display:flex; justify-content:space-between; align-items:center">
      <span>Activity</span>
      {#if (video.loading || video.generating)}<span class="vdot"></span>{/if}
    </div>
    <div class="vterm" bind:this={termEl} aria-readonly="true">
      {#if video.log.length === 0}
        <div class="vterm-line vterm-idle">— idle — load a model to begin</div>
      {:else}
        {#each video.log as line}<div class="vterm-line">{line}</div>{/each}
      {/if}
    </div>
  </div>

  <div class="ig-main">
    <textarea class="vprompt" placeholder="Describe the video…" bind:value={video.prompt}></textarea>
    <textarea class="vprompt vneg" placeholder="Negative prompt (what to avoid)" bind:value={video.negPrompt}></textarea>

    <div class="i2v-row">
      <label class="i2v-pick">
        🖼 {video.imageB64 ? "Change image" : "Add image → animate (i2v)"}
        <input type="file" accept="image/*" onchange={pickImage} hidden />
      </label>
      {#if video.imageB64}
        <img class="i2v-thumb" src="data:image/*;base64,{video.imageB64}" alt="i2v source" />
        <span class="i2v-name">{video.imageName}</span>
        <button class="vbtn-ghost i2v-clear" onclick={clearImage}>✕</button>
        <span class="i2v-tag">image-to-video</span>
      {/if}
    </div>

    <button class="vgen" onclick={generate} disabled={!loaded || video.generating || !video.prompt.trim()}>
      {video.generating ? (video.progress > 0 ? `Generating… ${pct}%` : video.loadStatus || "Preparing…") : loaded ? (video.imageB64 ? "🖼 Animate image" : "🎬 Generate") : "Load a model first"}
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
        <div class="vmeta">
          {video.frames ?? ""} frames · generated in {video.elapsed}s
          {#if video.enhanced}<span class="qpass-badge">✨ enhanced</span>{/if}
        </div>
        <button class="vbtn-ghost" onclick={saveMp4}>💾 Save MP4</button>

        <div class="qpass">
          <div class="qpass-head">✨ Quality Pass <span class="qpass-sub">— frees the generator, runs each stage on its own</span></div>
          <div class="qpass-toggles">
            <label><input type="checkbox" bind:checked={video.doRefine} disabled={video.enhancing} /> Refine <span class="qpass-tag">bf16 v2v</span></label>
            <label><input type="checkbox" bind:checked={video.doFace} disabled={video.enhancing} /> Restore faces <span class="qpass-tag">CodeFormer</span></label>
            <label><input type="checkbox" bind:checked={video.doUpscale} disabled={video.enhancing} /> Upscale <span class="qpass-tag">2× ESRGAN</span></label>
            <label><input type="checkbox" bind:checked={video.doInterpolate} disabled={video.enhancing} /> Interpolate <span class="qpass-tag">2× flow</span></label>
          </div>
          {#if video.doRefine}
            <div class="vrow"><span>Strength</span><input type="number" bind:value={video.refineStrength} min="0.1" max="0.9" step="0.05" disabled={video.enhancing} /></div>
          {/if}
          <button class="qpass-run" onclick={enhance} disabled={video.enhancing || (!video.doRefine && !video.doFace && !video.doUpscale && !video.doInterpolate)}>
            {video.enhancing ? (video.loadStatus || "Enhancing…") : "✨ Run Quality Pass"}
          </button>
          {#if video.enhancing}
            <div class="vbar vbar-indet"><div class="vbar-fill"></div></div>
          {/if}
          <div class="vhint">Unloads the generator first (all VRAM goes to the pass). You'll re-Load the model to generate again.</div>
        </div>
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
  .vauto { font-size: 10px; font-weight: 600; background: var(--bg3); border: 1px solid var(--border); color: var(--text2); border-radius: var(--radius-sm); padding: 2px 8px; cursor: pointer; text-transform: none; letter-spacing: 0; }
  .vauto:hover { background: var(--accent); color: #fff; border-color: var(--accent); }
  .vhint { font-size: 9px; color: var(--text3); margin-top: 4px; line-height: 1.5; }
  .vrow { display: flex; align-items: center; gap: 6px; margin-bottom: 6px; }
  .vrow span { font-size: 11px; color: var(--text2); min-width: 48px; }
  .vrow input { flex: 1; font-family: var(--mono); }
  .vsize { flex: 1; display: flex; gap: 6px; }
  .vsize-btn { flex: 1; font-size: 11px; font-weight: 600; background: var(--bg3); border: 1px solid var(--border); color: var(--text2); border-radius: var(--radius-sm); padding: 4px 0; cursor: pointer; }
  .vsize-btn:hover:not(:disabled) { border-color: var(--accent); color: var(--text); }
  .vsize-btn.on { background: var(--accent); color: #fff; border-color: var(--accent); }
  .vsize-btn:disabled { opacity: 0.4; cursor: not-allowed; }

  /* Mini non-typable terminal — live activity log. */
  .vterm {
    background: #0a0c10; border: 1px solid var(--border); border-radius: var(--radius-sm);
    font-family: var(--mono); font-size: 10.5px; line-height: 1.5; color: #8fb98f;
    padding: 6px 8px; height: 150px; overflow-y: auto; user-select: text;
    white-space: pre-wrap; word-break: break-word;
  }
  .vterm-line { opacity: 0.92; }
  .vterm-line:last-child { color: #b6e3b6; opacity: 1; }   /* highlight newest */
  .vterm-idle { color: var(--text3); font-style: italic; }
  .vdot { width: 7px; height: 7px; border-radius: 50%; background: var(--accent); animation: vdot-pulse 1s ease-in-out infinite; }
  @keyframes vdot-pulse { 0%,100% { opacity: 0.3; } 50% { opacity: 1; } }

  .ig-main { gap: 10px; }
  .vprompt { width: 100%; min-height: 70px; resize: vertical; font-family: var(--sans); font-size: 13px; line-height: 1.6; padding: 10px 12px; }
  .vneg { min-height: 42px; color: var(--text2); }

  .i2v-row { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
  .i2v-pick { font-size: 12px; color: var(--text2); border: 1px dashed var(--border); border-radius: var(--radius-sm); padding: 6px 10px; cursor: pointer; }
  .i2v-pick:hover { background: var(--bg3); color: var(--text); border-color: var(--accent); }
  .i2v-thumb { height: 38px; border-radius: var(--radius-sm); border: 1px solid var(--border); }
  .i2v-name { font-size: 11px; color: var(--text3); font-family: var(--mono); max-width: 180px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .i2v-clear { width: auto; padding: 2px 8px; }
  .i2v-tag { font-size: 9px; text-transform: uppercase; letter-spacing: 0.05em; color: #c4a3fb; background: rgba(168,85,247,0.18); border-radius: 4px; padding: 2px 6px; }
  .vgen { align-self: flex-start; padding: 9px 22px; font-size: 14px; font-weight: 600; background: var(--accent); border: none; color: #fff; border-radius: var(--radius); cursor: pointer; }
  .vgen:hover:not(:disabled) { background: #7a9cf7; }
  .vgen:disabled { opacity: 0.5; cursor: not-allowed; }

  .vbar { height: 5px; background: var(--bg3); border-radius: 3px; overflow: hidden; }
  .vbar-fill { height: 100%; background: var(--accent); border-radius: 3px; transition: width 0.3s; }
  /* Indeterminate (no step count yet — e.g. text-encoder loading): sliding sweep. */
  .vbar-indet .vbar-fill { width: 35%; transition: none; animation: vbar-sweep 1.1s ease-in-out infinite; }
  @keyframes vbar-sweep { 0% { margin-left: -35%; } 100% { margin-left: 100%; } }
  .vbar-label { font-size: 11px; color: var(--text3); font-family: var(--mono); }
  .verr { background: rgba(248,113,113,0.08); border: 1px solid rgba(248,113,113,0.3); color: var(--red); font-size: 12px; padding: 10px 12px; border-radius: var(--radius-sm); line-height: 1.5; white-space: pre-wrap; max-height: 30vh; overflow-y: auto; }

  .vresult { display: flex; flex-direction: column; gap: 8px; align-items: flex-start; }

  /* Quality Pass panel */
  .qpass { width: 100%; margin-top: 8px; padding: 12px; border: 1px solid var(--border); border-radius: var(--radius); background: var(--bg2); display: flex; flex-direction: column; gap: 8px; }
  .qpass-head { font-size: 12px; font-weight: 700; color: var(--text); }
  .qpass-sub { font-weight: 400; color: var(--text3); font-size: 11px; }
  .qpass-toggles { display: flex; flex-wrap: wrap; gap: 14px; }
  .qpass-toggles label { display: flex; align-items: center; gap: 6px; font-size: 12px; color: var(--text2); cursor: pointer; }
  .qpass-tag { font-size: 9px; text-transform: uppercase; letter-spacing: 0.04em; color: var(--text3); border: 1px solid var(--border); border-radius: 4px; padding: 1px 4px; }
  .qpass-run { align-self: flex-start; padding: 8px 18px; font-size: 13px; font-weight: 600; background: linear-gradient(135deg, #a855f7, #6366f1); border: none; color: #fff; border-radius: var(--radius); cursor: pointer; }
  .qpass-run:hover:not(:disabled) { filter: brightness(1.1); }
  .qpass-run:disabled { opacity: 0.5; cursor: not-allowed; }
  .qpass-badge { font-size: 10px; background: rgba(168,85,247,0.18); color: #c4a3fb; border-radius: 4px; padding: 1px 6px; margin-left: 6px; }
  .vplayer { max-width: 100%; max-height: 60vh; border-radius: var(--radius); border: 1px solid var(--border); background: #000; }
  .vmeta { font-size: 11px; color: var(--text3); font-family: var(--mono); }
  .vplaceholder { flex: 1; display: flex; flex-direction: column; align-items: center; justify-content: center; min-height: 280px; }
</style>
