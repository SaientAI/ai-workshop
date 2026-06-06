<script lang="ts">
  import { onMount } from "svelte";
  import { vision } from "../../lib/state.svelte.js";
  import * as T from "../../lib/tauri.js";

  const PRESETS = [
    "Describe this image in detail.",
    "What text appears in this image?",
    "How many people are in this image?",
    "What is the mood and art style?",
    "List the main objects you can see.",
  ];

  let fileInput = $state<HTMLInputElement | null>(null);

  onMount(async () => {
    vision.loaded = await T.visionLoaded().catch(() => false);
  });

  function setImageFromFile(file: File | null | undefined) {
    if (!file || !file.type.startsWith("image/")) return;
    const r = new FileReader();
    r.onload = () => {
      vision.imageB64 = String(r.result).split(",")[1] || "";
      vision.imageMime = file.type || "image/png";
      vision.imageName = file.name || "pasted image";
      vision.answer = "";
      vision.error = "";
    };
    r.readAsDataURL(file);
  }

  function pickImage(e: Event) {
    setImageFromFile((e.target as HTMLInputElement).files?.[0]);
  }

  function onPaste(e: ClipboardEvent) {
    const item = Array.from(e.clipboardData?.items ?? []).find((i) => i.type.startsWith("image/"));
    if (item) {
      e.preventDefault();
      setImageFromFile(item.getAsFile());
    }
  }

  function onDrop(e: DragEvent) {
    e.preventDefault();
    setImageFromFile(e.dataTransfer?.files?.[0]);
  }

  function clearImage() {
    vision.imageB64 = ""; vision.imageName = ""; vision.answer = ""; vision.error = "";
  }

  async function analyze() {
    if (!vision.imageB64 || vision.analyzing) return;
    vision.analyzing = true; vision.error = ""; vision.answer = "";
    try {
      const r = await T.visionDescribe(vision.imageB64, vision.question);
      vision.answer = r.answer;
      vision.elapsed = r.elapsed;
      vision.device = r.device;
      vision.loaded = true;
    } catch (e) {
      vision.error = String(e);
    } finally {
      vision.analyzing = false;
    }
  }

  async function unload() {
    await T.visionUnload().catch(() => null);
    vision.loaded = false;
  }
</script>

<svelte:window onpaste={onPaste} />

<div class="v-layout">
  <div class="v-sidebar">
    <div class="v-section-label">Model</div>
    {#if vision.loaded}
      <div class="hot-badge">🔥 Loaded{vision.device ? ` · ${vision.device}` : ""}</div>
      <button class="v-unload-btn" onclick={unload}>Unload / Free VRAM</button>
    {:else}
      <div class="cold-badge">Moondream2 · loads on first analyze</div>
    {/if}

    <div class="v-section-label" style="margin-top:14px">Quick questions</div>
    <div class="presets">
      {#each PRESETS as p}
        <button class="preset" class:sel={vision.question === p} onclick={() => (vision.question = p)} title={p}>
          {p}
        </button>
      {/each}
    </div>

    <div class="v-note">
      Tip: <b>paste</b> a screenshot (Ctrl/Cmd+V) or drop an image to analyze it. Leave the question blank for a general caption.
    </div>
  </div>

  <div class="v-main">
    <!-- svelte-ignore a11y_no_static_element_interactions -->
    <div
      class="v-drop"
      class:has-img={!!vision.imageB64}
      ondragover={(e) => e.preventDefault()}
      ondrop={onDrop}
    >
      {#if vision.imageB64}
        <img class="v-thumb" src="data:{vision.imageMime};base64,{vision.imageB64}" alt={vision.imageName} />
        <div class="v-img-bar">
          <span class="v-img-name" title={vision.imageName}>{vision.imageName}</span>
          <span class="spacer"></span>
          <button class="tab-action" onclick={() => fileInput?.click()}>Change</button>
          <button class="tab-action" onclick={clearImage}>Clear</button>
        </div>
      {:else}
        <button class="v-pick" onclick={() => fileInput?.click()}>
          <div style="font-size:42px;opacity:0.2">👁</div>
          <div class="v-pick-title">Choose, paste, or drop an image</div>
          <div class="v-pick-sub">PNG · JPEG · WebP</div>
        </button>
      {/if}
      <input bind:this={fileInput} type="file" accept="image/*" onchange={pickImage} style="display:none" />
    </div>

    <textarea
      class="v-question"
      placeholder="Ask about the image… (blank = describe it)"
      bind:value={vision.question}
      rows="2"
    ></textarea>

    <button class="v-analyze-btn" onclick={analyze} disabled={!vision.imageB64 || vision.analyzing}>
      {vision.analyzing
        ? (vision.loaded ? "👁 Analyzing…" : "Loading model… (first run downloads it)")
        : !vision.imageB64 ? "Add an image first" : "👁 Analyze"}
    </button>

    <div class="v-output">
      {#if vision.error}
        <div class="v-error">{vision.error}</div>
      {:else if vision.answer}
        <div class="v-answer">{vision.answer}</div>
        <div class="v-answer-meta">{vision.elapsed.toFixed(1)}s{vision.device ? ` · ${vision.device}` : ""}</div>
      {:else}
        <div class="v-placeholder">The model's answer will appear here</div>
      {/if}
    </div>
  </div>
</div>

<style>
  .v-layout { display: flex; flex: 1; overflow: hidden; }
  .v-sidebar { width: 220px; flex-shrink: 0; border-right: 1px solid var(--border); padding: 16px; background: var(--bg2); overflow-y: auto; }
  .v-section-label { font-size: 10px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; color: var(--text3); margin-bottom: 8px; }
  .hot-badge { font-size: 11px; color: var(--amber); margin-bottom: 6px; font-weight: 600; }
  .cold-badge { font-size: 11px; color: var(--text3); }
  .v-unload-btn { width: 100%; padding: 4px; font-size: 10px; background: rgba(248,113,113,0.08); border-color: rgba(248,113,113,0.3); color: var(--red); border-radius: var(--radius); }
  .presets { display: flex; flex-direction: column; gap: 5px; }
  .preset { text-align: left; font-size: 11px; padding: 6px 8px; line-height: 1.3; white-space: normal; }
  .preset.sel { border-color: var(--accent); color: var(--accent); background: rgba(108,142,245,0.1); }
  .v-note { margin-top: 16px; font-size: 11px; color: var(--text3); line-height: 1.5; }
  .v-note b { color: var(--text2); }

  .v-main { flex: 1; display: flex; flex-direction: column; padding: 16px; gap: 12px; overflow: hidden; }
  .v-drop { border: 1px dashed var(--border); border-radius: var(--radius); background: var(--bg2); flex-shrink: 0; display: flex; flex-direction: column; }
  .v-drop.has-img { border-style: solid; }
  .v-pick { display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 4px; width: 100%; padding: 36px 16px; background: transparent; border: 0; color: var(--text3); cursor: pointer; }
  .v-pick:hover { color: var(--text2); }
  .v-pick-title { font-size: 13px; }
  .v-pick-sub { font-size: 11px; color: var(--text3); font-family: var(--mono); }
  .v-thumb { width: 100%; max-height: 320px; object-fit: contain; display: block; border-radius: var(--radius) var(--radius) 0 0; background: #000; }
  .v-img-bar { display: flex; align-items: center; gap: 8px; padding: 8px 10px; border-top: 1px solid var(--border); }
  .v-img-name { font-size: 11px; color: var(--text3); font-family: var(--mono); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; max-width: 50%; }
  .spacer { flex: 1; }

  .v-question { width: 100%; resize: none; font-family: var(--sans); font-size: 13px; line-height: 1.6; flex-shrink: 0; }
  .v-analyze-btn { padding: 10px; font-size: 13px; font-weight: 600; background: rgba(108,142,245,0.12); border-color: rgba(108,142,245,0.4); color: var(--accent); border-radius: var(--radius); flex-shrink: 0; }

  .v-output { flex: 1; min-height: 0; overflow-y: auto; }
  .v-error { color: var(--red); font-size: 12px; padding: 10px 12px; background: rgba(248,113,113,0.07); border: 1px solid rgba(248,113,113,0.25); border-radius: var(--radius-sm); }
  .v-answer { font-size: 14px; line-height: 1.65; color: var(--text); white-space: pre-wrap; padding: 12px 14px; background: var(--bg2); border: 1px solid var(--border); border-radius: var(--radius); }
  .v-answer-meta { font-size: 11px; color: var(--text3); font-family: var(--mono); margin-top: 6px; }
  .v-placeholder { display: flex; align-items: center; justify-content: center; height: 100%; color: var(--text3); font-size: 13px; }
</style>
