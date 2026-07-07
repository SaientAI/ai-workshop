<script lang="ts">
  import { listen } from "@tauri-apps/api/event";
  import { toast } from "../lib/state.svelte.js";
  import * as T from "../lib/tauri.js";

  // target: where the file lands ("checkpoint" | "lora"); filter: HF pipeline tag
  // for search (e.g. "text-to-image"); exts: file types to offer.
  let {
    target,
    filter = "",
    exts = [".safetensors"],
    title = "Find a model",
    suggestions = [],
    onClose = () => {},
    onDone = () => {},
  }: {
    target: string;
    filter?: string;
    exts?: string[];
    title?: string;
    suggestions?: { label: string; repo: string }[];
    onClose?: () => void;
    onDone?: () => void;
  } = $props();

  let query = $state("");
  let results = $state<T.HfRepo[]>([]);
  let repo = $state("");
  let files = $state<T.HfFile[]>([]);
  let busy = $state(false);
  let error = $state("");
  let downloading = $state("");
  let prog = $state<{ downloaded: number; total: number }>({ downloaded: 0, total: 0 });
  let token = $state(localStorage.getItem("hf_token") ?? "");
  let showToken = $state(false);
  $effect(() => { localStorage.setItem("hf_token", token); });

  const pct = $derived(prog.total > 0 ? Math.round((prog.downloaded / prog.total) * 100) : 0);
  const fmtGB = (b: number) => (b / 1e9).toFixed(2) + " GB";
  const repoRe = /^[\w.-]+\/[\w.-]+$/;

  function parseRepo(s: string): string | null {
    s = s.trim();
    if (/^https?:\/\//i.test(s)) {
      try { const p = new URL(s).pathname.replace(/^\/+|\/+$/g, "").split("/"); return p.length >= 2 ? `${p[0]}/${p[1]}` : null; }
      catch { return null; }
    }
    return repoRe.test(s) ? s : null;
  }

  async function go() {
    const r = parseRepo(query);
    if (r) { await listFiles(r); return; }   // looks like a repo/URL → jump to files
    busy = true; error = ""; results = []; files = [];
    try {
      results = await T.hfSearch(query, filter, token);
      if (!results.length) error = "No models found — try different words.";
    } catch (e) { error = String(e); }
    finally { busy = false; }
  }

  async function listFiles(r: string) {
    repo = r; busy = true; error = ""; files = [];
    try {
      files = await T.hfListFiles(r, exts, token);
    } catch (e) { error = String(e); }
    finally { busy = false; }
  }

  async function download(file: string) {
    if (downloading) return;
    downloading = file.split("/").pop() ?? file;
    error = ""; prog = { downloaded: 0, total: 0 };
    const un = await listen<{ downloaded: number; total: number }>("model-progress", (e) => { prog = e.payload; });
    try {
      await T.downloadHfFile(repo, file, target, token);
      toast(`Downloaded ${downloading} — it's now in your ${target} list.`, "success");
      onDone();
      onClose();
    } catch (e) {
      error = String(e);
      toast("Download failed — see the panel.", "error");
    } finally {
      un(); downloading = "";
    }
  }

  function backToResults() { files = []; repo = ""; }
</script>

<div class="hb-backdrop" role="dialog" aria-modal="true" aria-label={title}
  onclick={(e) => { if (e.target === e.currentTarget && !downloading) onClose(); }}
  onkeydown={(e) => e.key === "Escape" && !downloading && onClose()} tabindex="-1">
  <div class="hb-card">
    <div class="hb-head">
      <span class="hb-title">{title}</span>
      <button class="hb-x" onclick={onClose} disabled={!!downloading}>✕</button>
    </div>

    {#if downloading}
      <div class="hb-dl">
        <div class="hb-dl-name">{downloading}</div>
        <div class="hb-bar"><div class="hb-fill" style="width:{pct}%"></div></div>
        <div class="hb-meta">{pct}% · {fmtGB(prog.downloaded)}{prog.total ? ` / ${fmtGB(prog.total)}` : ""}</div>
      </div>
    {:else}
      <div class="hb-search">
        <input
          class="hb-input"
          placeholder="Search Hugging Face, or paste owner/name"
          bind:value={query}
          onkeydown={(e) => e.key === "Enter" && go()}
          spellcheck="false"
          autocomplete="off"
        />
        <button class="hb-go" onclick={go} disabled={busy || !query.trim()}>{busy ? "…" : "Search"}</button>
      </div>

      <button class="hb-token-toggle" onclick={() => (showToken = !showToken)}>
        {showToken ? "▾" : "▸"} HF access token (for gated models){token ? " ✓" : ""}
      </button>
      {#if showToken}
        <input class="hb-input" type="password" placeholder="hf_… (stored locally)" bind:value={token} spellcheck="false" />
      {/if}

      {#if error}<div class="hb-err">{error}</div>{/if}

      {#if files.length}
        <div class="hb-sub">
          <button class="hb-back" onclick={backToResults}>← results</button>
          <span class="hb-repo">{repo}</span>
        </div>
        <div class="hb-list">
          {#each files as f}
            <button class="hb-item" onclick={() => download(f.file)}>
              <span class="hb-item-name">{f.file}</span>
              <span class="hb-item-size">⬇ {fmtGB(f.size)}</span>
            </button>
          {/each}
        </div>
      {:else if results.length}
        <div class="hb-list">
          {#each results as r}
            <button class="hb-item" onclick={() => listFiles(r.id)}>
              <span class="hb-item-name">{r.id}</span>
              <span class="hb-item-size">↓{r.downloads.toLocaleString()} ♥{r.likes}</span>
            </button>
          {/each}
        </div>
      {:else if !busy}
        {#if suggestions.length}
          <div class="hb-suggest-label">Suggested models — one click</div>
          <div class="hb-suggest">
            {#each suggestions as s}
              <button class="hb-chip" onclick={() => listFiles(s.repo)}>{s.label}</button>
            {/each}
          </div>
        {/if}
        <div class="hb-hint">Or search by name (e.g. <b>anime</b>, <b>realistic</b>) and pick a model — it downloads straight into the right folder, no setup.</div>
      {/if}
    {/if}
  </div>
</div>

<style>
  .hb-backdrop { position: fixed; inset: 0; z-index: 220; background: rgba(8,10,14,0.78); backdrop-filter: blur(4px); display: flex; align-items: center; justify-content: center; }
  .hb-card { width: 460px; max-width: 92vw; max-height: 80vh; display: flex; flex-direction: column; padding: 20px; background: #15181e; border: 1px solid #2a2f39; border-radius: 14px; box-shadow: 0 24px 60px rgba(0,0,0,0.5); }
  .hb-head { display: flex; align-items: center; justify-content: space-between; margin-bottom: 14px; }
  .hb-title { font-weight: 700; color: #e6e9ef; font-size: 14px; }
  .hb-x { background: none; border: 0; color: #8a93a3; cursor: pointer; font-size: 14px; }
  .hb-x:hover:not(:disabled) { color: #e6e9ef; }
  .hb-search { display: flex; gap: 8px; }
  .hb-input { flex: 1; padding: 9px 11px; border-radius: 9px; border: 1px solid #2a2f39; background: #0e1116; color: #e6e9ef; font-size: 13px; }
  .hb-input:focus { outline: none; border-color: #5b8cff; }
  .hb-go { padding: 0 16px; border: 1px solid #2a2f39; border-radius: 9px; background: #222732; color: #e6e9ef; cursor: pointer; font-size: 13px; }
  .hb-go:hover:not(:disabled) { border-color: #5b8cff; }
  .hb-go:disabled { opacity: 0.5; cursor: default; }
  .hb-token-toggle { background: none; border: 0; color: #6b7280; font-size: 11px; cursor: pointer; text-align: left; margin-top: 8px; padding: 2px 0; }
  .hb-token-toggle:hover { color: #9aa3b2; }
  .hb-suggest-label { font-size: 11px; color: #8a93a3; text-transform: uppercase; letter-spacing: 0.05em; margin: 14px 0 8px; }
  .hb-suggest { display: flex; flex-wrap: wrap; gap: 6px; }
  .hb-chip { padding: 7px 12px; border-radius: 8px; background: rgba(108,142,245,0.1); border: 1px solid #2a2f39; color: #cdd6f5; font-size: 12.5px; cursor: pointer; }
  .hb-chip:hover { border-color: #5b8cff; background: rgba(108,142,245,0.18); }
  .hb-hint { font-size: 12px; color: #8a93a3; line-height: 1.5; margin-top: 14px; }
  .hb-hint b { color: #aeb6c2; }
  .hb-err { font-size: 12px; color: #ff8080; margin-top: 10px; line-height: 1.4; }
  .hb-sub { display: flex; align-items: center; gap: 10px; margin: 12px 0 6px; }
  .hb-back { background: none; border: 0; color: #5b8cff; cursor: pointer; font-size: 12px; padding: 0; }
  .hb-repo { font-size: 11px; color: #8a93a3; font-family: ui-monospace, monospace; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .hb-list { margin-top: 10px; overflow-y: auto; display: flex; flex-direction: column; gap: 5px; }
  .hb-item { display: flex; align-items: center; justify-content: space-between; gap: 10px; text-align: left; padding: 9px 11px; border-radius: 8px; background: #1a1d24; border: 1px solid #2a2f39; cursor: pointer; }
  .hb-item:hover { border-color: #5b8cff; }
  .hb-item-name { font-size: 12.5px; color: #e6e9ef; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .hb-item-size { font-size: 11px; color: #7aa2ff; font-family: ui-monospace, monospace; flex-shrink: 0; }
  .hb-dl { padding: 20px 4px; }
  .hb-dl-name { font-size: 12px; color: #aeb6c2; font-family: ui-monospace, monospace; word-break: break-all; margin-bottom: 8px; }
  .hb-bar { height: 6px; background: #0e1116; border-radius: 3px; overflow: hidden; }
  .hb-fill { height: 100%; background: #5b8cff; border-radius: 3px; transition: width 0.3s; }
  .hb-meta { font-size: 11px; color: #8a93a3; font-family: ui-monospace, monospace; margin-top: 6px; }
</style>
