<script lang="ts">
  import { onMount } from "svelte";
  import { save } from "@tauri-apps/plugin-dialog";
  import { tts } from "../../lib/state.svelte.js";
  import * as T from "../../lib/tauri.js";

  const DEFAULT_VOICES = [
    { id: "af_heart",   label: "Heart (F · American)",    lang: "a" },
    { id: "af_bella",   label: "Bella (F · American)",    lang: "a" },
    { id: "af_sarah",   label: "Sarah (F · American)",    lang: "a" },
    { id: "af_nicole",  label: "Nicole (F · American)",   lang: "a" },
    { id: "am_adam",    label: "Adam (M · American)",     lang: "a" },
    { id: "am_michael", label: "Michael (M · American)",  lang: "a" },
    { id: "bf_emma",    label: "Emma (F · British)",      lang: "b" },
    { id: "bf_isabella",label: "Isabella (F · British)",  lang: "b" },
    { id: "bm_george",  label: "George (M · British)",    lang: "b" },
    { id: "bm_lewis",   label: "Lewis (M · British)",     lang: "b" },
  ];

  const voices = $derived(tts.voices.length > 0 ? tts.voices : DEFAULT_VOICES);

  onMount(async () => {
    tts.voices = await T.ttsFetchVoices().catch(() => []);
  });

  async function generate() {
    if (!tts.text.trim() || tts.generating) return;
    tts.generating = true; tts.error = ""; tts.resultB64 = "";
    try {
      const r = await T.runTts({ voice: tts.voice, speed: tts.speed, text: tts.text });
      tts.resultB64 = r.base64_wav;
      tts.duration = r.duration;
    } catch (e) { tts.error = String(e); }
    finally { tts.generating = false; }
  }

  async function saveWav() {
    const path = await save({ filters: [{ name: "WAV", extensions: ["wav"] }], defaultPath: "output.wav" }).catch(() => null);
    if (!path) return;
    // base64 → binary write via Tauri fs
    const bytes = atob(tts.resultB64);
    const arr = new Uint8Array(bytes.length);
    for (let i = 0; i < bytes.length; i++) arr[i] = bytes.charCodeAt(i);
    // TODO: write via fs plugin
    console.log("save", path, arr.length, "bytes");
  }
</script>

<div class="ig-layout">
  <div class="ig-sidebar">
    <div class="ig-section-label">Voice</div>
    <select class="ig-select" bind:value={tts.voice}>
      {#each voices as v}
        <option value={v.id}>{v.label}</option>
      {/each}
    </select>

    <div class="ig-section-label" style="margin-top:14px">Speed</div>
    <div class="speed-row">
      <input type="range" min="0.5" max="2.0" step="0.05" bind:value={tts.speed} />
      <span>{tts.speed.toFixed(2)}×</span>
    </div>
    <div class="speed-hint">0.5× = slow · 1.0× = normal · 2.0× = fast</div>
  </div>

  <div class="ig-main">
    <div class="ig-prompt-area">
      <textarea
        class="ig-prompt"
        placeholder="Type or paste text to synthesise…"
        bind:value={tts.text}
      ></textarea>
      <button class="ig-generate-btn" onclick={generate} disabled={tts.generating}>
        {tts.generating ? `Synthesising… ${tts.progress}%` : "🔊 Synthesise"}
      </button>
    </div>

    {#if tts.generating}
      <div class="ig-progress-bar">
        <div class="ig-progress-fill" style="width:{tts.progress}%"></div>
      </div>
    {/if}

    {#if tts.error}
      <div class="ig-error">{tts.error}</div>
    {:else if tts.resultB64}
      <div class="tts-result">
        <div class="tts-meta">🔊 {tts.duration.toFixed(1)}s of audio</div>
        <audio controls src="data:audio/wav;base64,{tts.resultB64}" style="width:100%;margin-top:8px;"></audio>
        <button class="tab-action" onclick={saveWav}>💾 Save WAV</button>
      </div>
    {:else if !tts.generating}
      <div class="ig-placeholder">
        <div style="font-size:48px;opacity:0.15">🔊</div>
        <div style="color:var(--text3);font-size:13px;margin-top:8px">Audio will appear here</div>
      </div>
    {/if}
  </div>
</div>

<style>
  .ig-layout { display: flex; flex: 1; overflow: hidden; }
  .ig-sidebar { width: 220px; flex-shrink: 0; border-right: 1px solid var(--border); padding: 16px; background: var(--bg2); overflow-y: auto; }
  .ig-section-label { font-size: 10px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; color: var(--text3); margin-bottom: 8px; }
  .ig-select { width: 100%; }
  .speed-row { display: flex; align-items: center; gap: 8px; }
  .speed-row span { font-size: 11px; font-family: var(--mono); color: var(--text2); width: 40px; }
  .speed-hint { font-size: 9px; color: var(--text3); margin-top: 4px; line-height: 1.6; }
  .ig-main { flex: 1; display: flex; flex-direction: column; padding: 16px; gap: 12px; overflow-y: auto; }
  .ig-prompt-area { display: flex; flex-direction: column; gap: 8px; }
  .ig-prompt { width: 100%; min-height: 120px; resize: vertical; font-family: var(--sans); font-size: 13px; line-height: 1.6; }
  .ig-generate-btn { padding: 10px; font-size: 13px; font-weight: 600; background: rgba(108,142,245,0.12); border-color: rgba(108,142,245,0.4); color: var(--accent); border-radius: var(--radius); }
  .ig-generate-btn:hover { background: rgba(108,142,245,0.22); }
  .ig-progress-bar { height: 4px; background: var(--border); border-radius: 2px; overflow: hidden; }
  .ig-progress-fill { height: 100%; background: var(--accent); transition: width 0.3s; }
  .ig-error { color: var(--red); font-size: 12px; padding: 8px 10px; background: rgba(248,113,113,0.07); border: 1px solid rgba(248,113,113,0.25); border-radius: var(--radius-sm); }
  .ig-placeholder { display: flex; flex-direction: column; align-items: center; justify-content: center; flex: 1; }
  .tts-result { display: flex; flex-direction: column; gap: 8px; }
  .tts-meta { font-size: 12px; color: var(--text2); }
</style>
