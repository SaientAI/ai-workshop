<script lang="ts">
  import { open } from "@tauri-apps/plugin-shell";
  import { update, toast } from "../lib/state.svelte.js";
  import * as T from "../lib/tauri.js";

  let { onClose = () => {} }: { onClose?: () => void } = $props();

  async function check() {
    if (update.checking) return;
    update.checking = true; update.error = "";
    try {
      const u = await T.checkUpdate();
      update.checked = true;
      update.available = u.update_available;
      update.current = u.current;
      update.latest = u.latest;
      update.url = u.url;
      update.notes = u.notes;
      update.dismissed = false;
    } catch (e) {
      update.error = "Couldn't reach the update server. Check your connection and try again.";
    } finally {
      update.checking = false;
    }
  }

  async function getUpdate() {
    try { await open(update.url); } catch { window.open(update.url, "_blank"); }
  }

  async function copyDiagnostics() {
    try {
      const text = await T.diagnostics();
      await navigator.clipboard.writeText(text);
      toast("Diagnostics copied — paste them into your support email", "success");
    } catch {
      toast("Couldn't copy diagnostics", "error");
    }
  }
</script>

<div class="ub-backdrop" role="dialog" aria-modal="true" aria-label="Updates"
  onclick={(e) => { if (e.target === e.currentTarget) onClose(); }}
  onkeydown={(e) => e.key === "Escape" && onClose()} tabindex="-1">
  <div class="ub-card">
    <div class="ub-logo">Saient</div>

    {#if update.current}
      <div class="ub-ver">You're on <b>v{update.current}</b></div>
    {/if}

    {#if update.checking}
      <div class="ub-state">Checking for updates…</div>
    {:else if update.error}
      <div class="ub-err">{update.error}</div>
    {:else if update.checked && update.available}
      <div class="ub-state ub-yes">⬆ Version <b>{update.latest}</b> is available</div>
      {#if update.notes}<div class="ub-notes">{update.notes}</div>{/if}
      <button class="ub-get" onclick={getUpdate}>Get the latest version</button>
    {:else if update.checked}
      <div class="ub-state ub-ok">✓ You're up to date</div>
    {:else}
      <div class="ub-state">Check whether a newer version of Saient is available.</div>
    {/if}

    <div class="ub-row">
      <button class="ub-check" onclick={check} disabled={update.checking}>
        {update.checking ? "…" : "Check now"}
      </button>
      <button class="ub-site" onclick={getUpdate}>Visit site</button>
    </div>

    <button class="ub-diag" onclick={copyDiagnostics}>Copy diagnostics for support</button>
    <button class="ub-close" onclick={onClose}>Close</button>
  </div>
</div>

<style>
  .ub-backdrop {
    position: fixed; inset: 0; z-index: 210;
    background: rgba(8, 10, 14, 0.78); backdrop-filter: blur(4px);
    display: flex; align-items: center; justify-content: center;
  }
  .ub-card {
    width: 380px; max-width: 92vw; padding: 28px 26px;
    background: #15181e; border: 1px solid #2a2f39; border-radius: 16px;
    box-shadow: 0 24px 60px rgba(0,0,0,0.5); text-align: center;
  }
  .ub-logo { font-weight: 700; letter-spacing: 0.5px; color: #cdd3df; margin-bottom: 14px; }
  .ub-ver { font-size: 12px; color: #8a93a3; margin-bottom: 14px; }
  .ub-ver b { color: #e6e9ef; }
  .ub-state { font-size: 14px; color: #aeb6c2; line-height: 1.5; margin-bottom: 16px; }
  .ub-yes { color: #7aa2ff; } .ub-yes b { color: #fff; }
  .ub-ok { color: #00d68f; }
  .ub-notes { font-size: 12.5px; color: #98a0ad; line-height: 1.55; margin: -8px 0 16px; }
  .ub-err { font-size: 13px; color: #ff8080; margin-bottom: 16px; line-height: 1.5; }
  .ub-get {
    width: 100%; padding: 12px; border: 0; border-radius: 10px; cursor: pointer;
    background: linear-gradient(135deg, #5b8cff, #6a5bff); color: #fff;
    font-size: 14px; font-weight: 600; margin-bottom: 14px;
  }
  .ub-get:hover { filter: brightness(1.08); }
  .ub-row { display: flex; gap: 8px; }
  .ub-check, .ub-site {
    flex: 1; padding: 9px; border: 1px solid #2a2f39; border-radius: 9px; cursor: pointer;
    background: #222732; color: #e6e9ef; font-size: 13px;
  }
  .ub-check:hover, .ub-site:hover { border-color: #5b8cff; }
  .ub-check:disabled { opacity: 0.5; cursor: default; }
  .ub-diag {
    margin-top: 16px; background: none; border: 0; color: #8a93a3; cursor: pointer; font-size: 12px; display: block; width: 100%;
  }
  .ub-diag:hover { color: #cdd6f5; }
  .ub-close {
    margin-top: 6px; background: none; border: 0; color: #6b7280; cursor: pointer; font-size: 12px;
  }
  .ub-close:hover { color: #9aa3b2; }
</style>
