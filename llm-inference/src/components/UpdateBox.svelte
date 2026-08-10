<script lang="ts">
  import { listen } from "@tauri-apps/api/event";
  import { open } from "@tauri-apps/plugin-shell";
  import { update, toast } from "../lib/state.svelte.js";
  import * as T from "../lib/tauri.js";

  let { onClose = () => {} }: { onClose?: () => void } = $props();
  let installing = $state(false);
  let installMessage = $state("");
  let downloaded = $state(0);
  let total = $state(0);

  const percent = $derived(total > 0 ? Math.min(100, Math.round(downloaded * 100 / total)) : 0);

  async function check() {
    if (update.checking) return;
    update.checking = true; update.error = "";
    try {
      const u = await T.checkUpdate();
      update.checked = true;
      update.available = u.update_available;
      update.installSupported = u.install_supported;
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
    if (installing) return;
    if (!update.installSupported) {
      try { await open(update.url); } catch { window.open(update.url, "_blank"); }
      return;
    }

    installing = true;
    update.error = "";
    installMessage = "Checking the signed release manifest…";
    downloaded = 0;
    total = 0;
    let unlisten: (() => void) | undefined;
    try {
      unlisten = await listen<T.UpdateProgress>("update-progress", (event) => {
        installMessage = event.payload.message;
        downloaded = event.payload.downloaded;
        total = event.payload.total;
      });
      const result = await T.installUpdate(update.latest);
      installMessage = `Version ${result.version} installed. Restarting Saient…`;
      await new Promise((resolve) => setTimeout(resolve, 600));
      await T.relaunchAfterUpdate();
    } catch (e) {
      update.error = `Update failed: ${String(e)}`;
      installing = false;
    } finally {
      unlisten?.();
    }
  }

  async function visitSite() {
    const url = "https://saient.co.uk/#download";
    try { await open(url); } catch { window.open(url, "_blank"); }
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

    {#if installing}
      <div class="ub-state ub-yes">{installMessage}</div>
      {#if total > 0}
        <div class="ub-progress" aria-label="Update download progress">
          <div style:width={`${percent}%`}></div>
        </div>
        <div class="ub-progress-label">{percent}%</div>
      {/if}
    {:else if update.checking}
      <div class="ub-state">Checking for updates…</div>
    {:else if update.error}
      <div class="ub-err">{update.error}</div>
    {:else if update.checked && update.available}
      <div class="ub-state ub-yes">⬆ Version <b>{update.latest}</b> is available</div>
      {#if update.notes}<div class="ub-notes">{update.notes}</div>{/if}
      <button class="ub-get" onclick={getUpdate}>
        {update.installSupported ? "Install update and restart" : "Download update"}
      </button>
      {#if update.installSupported}
        <div class="ub-auth-note">The update is signed and verified first. Windows installs it for the current user; Linux may ask for administrator approval if the machine has not already granted it.</div>
      {/if}
    {:else if update.checked}
      <div class="ub-state ub-ok">✓ You're up to date</div>
    {:else}
      <div class="ub-state">Check whether a newer version of Saient is available.</div>
    {/if}

    <div class="ub-row">
      <button class="ub-check" onclick={check} disabled={update.checking}>
        {update.checking ? "…" : "Check now"}
      </button>
      <button class="ub-site" onclick={visitSite} disabled={installing}>Visit site</button>
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
  .ub-auth-note { font-size: 11.5px; color: #788292; line-height: 1.45; margin: -5px 0 14px; }
  .ub-err { font-size: 13px; color: #ff8080; margin-bottom: 16px; line-height: 1.5; }
  .ub-progress { height: 7px; overflow: hidden; border-radius: 999px; background: #252a34; margin: -4px 0 8px; }
  .ub-progress > div { height: 100%; border-radius: inherit; background: linear-gradient(90deg, #5b8cff, #6a5bff); transition: width 120ms linear; }
  .ub-progress-label { color: #8a93a3; font-family: monospace; font-size: 11px; margin-bottom: 16px; }
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
