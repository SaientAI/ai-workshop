<script lang="ts">
  import { listen } from "@tauri-apps/api/event";
  import { toast } from "../lib/state.svelte.js";
  import * as T from "../lib/tauri.js";

  // Opened by a `saient://models/huggingface/<owner>/<name>` link — i.e. by a
  // web page, not by a click inside Saient. So it always asks first: it names
  // the repo, and nothing is fetched until the user picks a specific file.
  let {
    repo,
    modelsDir,
    onClose = () => {},
    onDone = async (_path: string) => {},
  }: {
    repo: string;
    modelsDir: string;
    onClose?: () => void;
    onDone?: (installedPath: string) => void | Promise<void>;
  } = $props();

  let files = $state<T.HfFile[]>([]);
  let busy = $state(false);
  let error = $state("");
  let downloading = $state("");
  let prog = $state<{ downloaded: number; total: number }>({ downloaded: 0, total: 0 });
  let token = $state("");
  let showToken = $state(false);

  const pct = $derived(prog.total > 0 ? Math.round((prog.downloaded / prog.total) * 100) : 0);
  const fmtGB = (b: number) => (b / 1e9).toFixed(2) + " GB";

  async function listFiles() {
    busy = true; error = ""; files = [];
    try {
      files = await T.hfListGguf(repo, token);
    } catch (e) { error = String(e); }
    finally { busy = false; }
  }

  async function download(file: string) {
    if (downloading) return;
    downloading = file.split("/").pop() ?? file;
    error = ""; prog = { downloaded: 0, total: 0 };
    const un = await listen<{ downloaded: number; total: number }>("model-progress", (e) => { prog = e.payload; });
    try {
      const installedPath = await T.downloadStarterModel(repo, file, modelsDir, token);
      await onDone(installedPath);
      toast(`Imported ${downloading} from Hugging Face.`, "success");
      onClose();
    } catch (e) {
      error = String(e);
      toast("Import failed — see the panel.", "error");
    } finally {
      un(); downloading = "";
    }
  }
</script>

<div class="hi-backdrop" role="dialog" aria-modal="true" aria-label="Import a model from Hugging Face"
  onclick={(e) => { if (e.target === e.currentTarget && !downloading) onClose(); }}
  onkeydown={(e) => e.key === "Escape" && !downloading && onClose()} tabindex="-1">
  <div class="hi-card">
    <div class="hi-head">
      <span class="hi-title">Import from Hugging Face</span>
      <button class="hi-x" onclick={onClose} disabled={!!downloading} aria-label="Close Hugging Face import">✕</button>
    </div>

    {#if downloading}
      <div class="hi-dl">
        <div class="hi-dl-name">{downloading}</div>
        <div class="hi-bar"><div class="hi-fill" style="width:{pct}%"></div></div>
        <div class="hi-meta">
          {prog.total ? `${pct}% · ${fmtGB(prog.downloaded)} / ${fmtGB(prog.total)}` : "Starting…"}
        </div>
      </div>
    {:else}
      <div class="hi-hint">A web page asked Saient to open this model:</div>
      <div class="hi-repo">{repo}</div>

      <button class="hi-token-toggle" onclick={() => (showToken = !showToken)}>
        {showToken ? "▾" : "▸"} HF access token (for gated models){token ? " ✓" : ""}
      </button>
      {#if showToken}
        <input class="hi-input" type="password" placeholder="hf_… (stored locally)" bind:value={token} spellcheck="false" />
      {/if}

      {#if error}<div class="hi-err">{error}</div>{/if}

      {#if files.length}
        <div class="hi-sub">Choose a quantisation to download:</div>
        <div class="hi-list">
          {#each files as f}
            <button class="hi-item" onclick={() => download(f.file)}>
              <span class="hi-item-name">{f.file}</span>
              <span class="hi-item-size">⬇ {fmtGB(f.size)}</span>
            </button>
          {/each}
        </div>
      {:else}
        <div class="hi-actions">
          <button class="hi-cancel" onclick={onClose}>Not now</button>
          <button class="hi-go" onclick={listFiles} disabled={busy}>{busy ? "…" : "Show files"}</button>
        </div>
        <div class="hi-foot">Nothing is downloaded until you pick a file.</div>
      {/if}
    {/if}
  </div>
</div>

<style>
  .hi-backdrop { position: fixed; inset: 0; z-index: 230; background: rgba(8,10,14,0.78); backdrop-filter: blur(4px); display: flex; align-items: center; justify-content: center; }
  .hi-card { width: 460px; max-width: 92vw; max-height: 80vh; display: flex; flex-direction: column; padding: 20px; background: #15181e; border: 1px solid #2a2f39; border-radius: 14px; box-shadow: 0 24px 60px rgba(0,0,0,0.5); }
  .hi-head { display: flex; align-items: center; justify-content: space-between; margin-bottom: 14px; }
  .hi-title { font-weight: 700; color: #e6e9ef; font-size: 14px; }
  .hi-x { background: none; border: 0; color: #8a93a3; cursor: pointer; font-size: 14px; }
  .hi-x:hover:not(:disabled) { color: #e6e9ef; }
  .hi-hint { font-size: 12px; color: #8a93a3; }
  .hi-repo { margin-top: 6px; padding: 9px 11px; border-radius: 9px; background: #0e1116; border: 1px solid #2a2f39; color: #cdd6f5; font-size: 13px; font-family: ui-monospace, monospace; word-break: break-all; }
  .hi-input { margin-top: 6px; width: 100%; padding: 9px 11px; border-radius: 9px; border: 1px solid #2a2f39; background: #0e1116; color: #e6e9ef; font-size: 13px; }
  .hi-input:focus { outline: none; border-color: #5b8cff; }
  .hi-token-toggle { background: none; border: 0; color: #6b7280; font-size: 11px; cursor: pointer; text-align: left; margin-top: 10px; padding: 2px 0; }
  .hi-token-toggle:hover { color: #9aa3b2; }
  .hi-err { font-size: 12px; color: #ff8080; margin-top: 10px; line-height: 1.4; }
  .hi-actions { display: flex; gap: 8px; justify-content: flex-end; margin-top: 16px; }
  .hi-cancel { padding: 8px 14px; border: 1px solid #2a2f39; border-radius: 9px; background: none; color: #8a93a3; cursor: pointer; font-size: 13px; }
  .hi-cancel:hover { color: #e6e9ef; border-color: #3a4150; }
  .hi-go { padding: 8px 16px; border: 0; border-radius: 9px; background: var(--accent, #5b8cff); color: #0a0a12; font-weight: 700; cursor: pointer; font-size: 13px; }
  .hi-go:disabled { opacity: 0.5; cursor: default; }
  .hi-foot { font-size: 11px; color: #6b7280; margin-top: 10px; text-align: right; }
  .hi-sub { font-size: 11px; color: #8a93a3; text-transform: uppercase; letter-spacing: 0.05em; margin: 14px 0 0; }
  .hi-list { margin-top: 10px; overflow-y: auto; display: flex; flex-direction: column; gap: 5px; }
  .hi-item { display: flex; align-items: center; justify-content: space-between; gap: 10px; text-align: left; padding: 9px 11px; border-radius: 8px; background: #1a1d24; border: 1px solid #2a2f39; cursor: pointer; }
  .hi-item:hover { border-color: #5b8cff; }
  .hi-item-name { font-size: 12.5px; color: #e6e9ef; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .hi-item-size { font-size: 11px; color: #7aa2ff; font-family: ui-monospace, monospace; flex-shrink: 0; }
  .hi-dl { padding: 20px 4px; }
  .hi-dl-name { font-size: 12px; color: #aeb6c2; font-family: ui-monospace, monospace; word-break: break-all; margin-bottom: 8px; }
  .hi-bar { height: 6px; background: #0e1116; border-radius: 3px; overflow: hidden; }
  .hi-fill { height: 100%; background: #5b8cff; border-radius: 3px; transition: width 0.3s; }
  .hi-meta { font-size: 11px; color: #8a93a3; font-family: ui-monospace, monospace; margin-top: 6px; }
</style>
