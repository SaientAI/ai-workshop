<script lang="ts">
  import QRCode from "qrcode";
  import * as T from "../lib/tauri.js";

  let { onClose, onSecurity }: { onClose: () => void; onSecurity: () => void } = $props();

  type Tab = "internet" | "security";
  let tab = $state<Tab>("internet");
  let qr = $state("");
  let url = $state("");
  let localUrl = $state("");
  let error = $state("");
  let copied = $state(false);

  async function loadPairing() {
    error = "";
    copied = false;
    try {
      const info = await T.remotePairingInfo();
      url = info.url;
      localUrl = info.local_url;
      qr = await QRCode.toDataURL(JSON.stringify(info.payload), {
        width: 260,
        margin: 1,
        color: { dark: "#0d0d0f", light: "#ffffff" },
      });
    } catch (e) {
      error = String(e);
    }
  }

  async function copyUrl() {
    if (!url) return;
    await navigator.clipboard.writeText(url);
    copied = true;
    setTimeout(() => (copied = false), 1800);
  }

  $effect(() => {
    if (tab === "internet" && !qr && !error) loadPairing();
  });
</script>

<div class="settings-backdrop" role="dialog" aria-modal="true" aria-label="Settings">
  <div class="settings-card">
    <div class="settings-head">
      <div>
        <div class="settings-title">Settings</div>
        <div class="settings-sub">Desktop controls and phone pairing</div>
      </div>
      <button class="settings-x" onclick={onClose} aria-label="Close">×</button>
    </div>

    <div class="settings-body">
      <nav class="settings-nav" aria-label="Settings sections">
        <button class:active={tab === "internet"} onclick={() => (tab = "internet")}>Internet</button>
        <button class:active={tab === "security"} onclick={() => (tab = "security")}>Security</button>
      </nav>

      <section class="settings-panel">
        {#if tab === "internet"}
          <div class="panel-kicker">Phone link</div>
          <h2>Scan to connect Saient Mobile</h2>
          <p class="panel-copy">
            Keep this desktop app open, then scan this code from the Studio tab on your phone.
            The phone stores the link after the first successful scan.
          </p>

          <div class="pairing-grid">
            <div class="qr-box">
              {#if qr}
                <img src={qr} alt="Saient phone pairing QR code" />
              {:else if error}
                <div class="qr-error">{error}</div>
              {:else}
                <div class="qr-loading">Generating…</div>
              {/if}
            </div>
            <div class="pairing-details">
              <div class="field-label">Desktop URL</div>
              <button class="url-chip" onclick={copyUrl} title="Copy URL">
                <span>{url || "Detecting LAN address…"}</span>
                <strong>{copied ? "Copied" : "Copy"}</strong>
              </button>
              <div class="hint">
                If the phone cannot connect, make sure both devices are on the same Wi-Fi and that the desktop firewall allows port 18788.
              </div>
              {#if localUrl && localUrl !== url}
                <div class="local-note">Local: {localUrl}</div>
              {/if}
              <button class="refresh" onclick={loadPairing}>Refresh code</button>
            </div>
          </div>
        {:else}
          <div class="panel-kicker">Access</div>
          <h2>Launch password</h2>
          <p class="panel-copy">
            Require a password when opening Saient on this desktop.
          </p>
          <button class="security-btn" onclick={onSecurity}>Open password settings</button>
        {/if}
      </section>
    </div>
  </div>
</div>

<style>
  .settings-backdrop {
    position: fixed; inset: 0; z-index: 150;
    background: rgba(8,10,14,0.72); backdrop-filter: blur(3px);
    display: flex; align-items: center; justify-content: center;
  }
  .settings-card {
    width: 680px; max-width: 94vw; max-height: 90vh;
    background: #15181e; border: 1px solid #2a2f39; border-radius: 12px;
    box-shadow: 0 24px 60px rgba(0,0,0,0.55); overflow: hidden;
  }
  .settings-head {
    display: flex; align-items: center; justify-content: space-between;
    padding: 16px 18px; border-bottom: 1px solid #2a2f39;
  }
  .settings-title { color: #eef1f6; font-size: 15px; font-weight: 700; }
  .settings-sub { color: #8790a0; font-size: 12px; margin-top: 3px; }
  .settings-x { background: none; border: 0; color: #6b7280; font-size: 22px; cursor: pointer; line-height: 1; }
  .settings-x:hover { color: #cdd3df; }
  .settings-body { display: grid; grid-template-columns: 150px 1fr; min-height: 390px; }
  .settings-nav {
    background: #101319; border-right: 1px solid #2a2f39; padding: 12px;
    display: flex; flex-direction: column; gap: 6px;
  }
  .settings-nav button {
    height: 36px; border: 1px solid transparent; border-radius: 8px; cursor: pointer;
    background: transparent; color: #98a0ad; text-align: left; padding: 0 10px; font-weight: 600;
  }
  .settings-nav button:hover { background: #1a1f28; color: #e6e9ef; }
  .settings-nav button.active { background: rgba(108,142,245,0.14); border-color: rgba(108,142,245,0.35); color: #cdd6f5; }
  .settings-panel { padding: 22px; overflow: auto; }
  .panel-kicker { color: #6c8ef5; font-size: 11px; font-weight: 800; text-transform: uppercase; margin-bottom: 6px; }
  h2 { color: #eef1f6; font-size: 20px; margin: 0 0 8px; }
  .panel-copy { color: #98a0ad; font-size: 13px; line-height: 1.5; margin: 0 0 18px; max-width: 430px; }
  .pairing-grid { display: grid; grid-template-columns: 280px 1fr; gap: 20px; align-items: start; }
  .qr-box {
    width: 280px; height: 280px; border-radius: 10px; border: 1px solid #2a2f39;
    background: #fff; display: flex; align-items: center; justify-content: center; padding: 10px;
  }
  .qr-box img { width: 260px; height: 260px; display: block; }
  .qr-loading, .qr-error { color: #111827; font-size: 12px; text-align: center; }
  .qr-error { color: #9b1c1c; padding: 16px; line-height: 1.4; }
  .field-label { color: #707989; font-size: 11px; text-transform: uppercase; font-weight: 800; margin-bottom: 6px; }
  .url-chip {
    width: 100%; min-height: 42px; border-radius: 9px; border: 1px solid #2a2f39; cursor: pointer;
    background: #0e1116; color: #dce2ee; display: flex; align-items: center; gap: 10px;
    padding: 8px 10px; text-align: left;
  }
  .url-chip span { flex: 1; font-family: var(--mono); font-size: 12px; overflow-wrap: anywhere; }
  .url-chip strong { color: #6c8ef5; font-size: 12px; }
  .hint { color: #8790a0; font-size: 12px; line-height: 1.5; margin-top: 12px; }
  .local-note { color: #606b7c; font-size: 11px; margin-top: 10px; font-family: var(--mono); overflow-wrap: anywhere; }
  .refresh, .security-btn {
    margin-top: 16px; height: 38px; padding: 0 14px; border-radius: 8px; cursor: pointer;
    border: 1px solid rgba(108,142,245,0.35); background: rgba(108,142,245,0.12);
    color: #cdd6f5; font-weight: 700;
  }
  .refresh:hover, .security-btn:hover { background: rgba(108,142,245,0.2); }
  @media (max-width: 640px) {
    .settings-body { grid-template-columns: 1fr; }
    .settings-nav { flex-direction: row; border-right: 0; border-bottom: 1px solid #2a2f39; }
    .settings-nav button { flex: 1; text-align: center; }
    .pairing-grid { grid-template-columns: 1fr; }
    .qr-box { width: 100%; max-width: 280px; }
  }
</style>
