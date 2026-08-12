<script lang="ts">
  import { onMount } from "svelte";
  import * as T from "../lib/tauri.js";
  import type { SystemInfo } from "../lib/tauri.js";

  let { onClose, onSecurity, onSetup }: {
    onClose: () => void;
    onSecurity: () => void;
    onSetup: () => void | Promise<void>;
  } = $props();

  type Tab = "internet" | "setup" | "security";
  let tab = $state<Tab>("internet");
  let internetError = $state("");
  let internetEnabled = $state(false);
  let internetLoaded = $state(false);
  let internetSaving = $state(false);
  let setupInfo = $state<SystemInfo | null>(null);
  let setupLoaded = $state(false);
  let setupStarting = $state(false);
  let setupError = $state("");

  onMount(() => {
    void loadInternetState();
    void loadSetupState();
  });

  async function loadInternetState() {
    internetError = "";
    try {
      internetEnabled = await T.getInternetEnabled();
    } catch (e) {
      internetError = String(e);
    } finally {
      internetLoaded = true;
    }
  }

  async function toggleInternet() {
    const next = !internetEnabled;
    internetSaving = true;
    internetError = "";
    try {
      await T.setInternetEnabled(next);
      internetEnabled = next;
    } catch (e) {
      internetError = String(e);
    } finally {
      internetSaving = false;
    }
  }

  async function loadSetupState() {
    setupError = "";
    try {
      setupInfo = await T.detectSystem();
    } catch (e) {
      setupError = String(e);
    } finally {
      setupLoaded = true;
    }
  }

  async function runSetupAgain() {
    if (setupStarting) return;
    setupStarting = true;
    setupError = "";
    try {
      await onSetup();
    } catch (e) {
      setupError = String(e);
      setupStarting = false;
    }
  }

  const profileLabel = $derived.by(() => {
    if (!setupLoaded) return "Checking…";
    if (!setupInfo?.setup_done) return "Setup not completed";
    if (setupInfo.setup_profile === "full") return "Full setup selected";
    if (setupInfo.setup_profile === "fast") return "Fast setup selected";
    if (setupInfo.setup_profile === "skipped") return "Setup was skipped";
    return "Setup marker found";
  });
</script>

<div class="settings-backdrop" role="dialog" aria-modal="true" aria-label="Settings">
  <div class="settings-card">
    <div class="settings-head">
      <div>
        <div class="settings-title">Settings</div>
        <div class="settings-sub">Local runtime controls</div>
      </div>
      <button class="settings-x" onclick={onClose} aria-label="Close">×</button>
    </div>

    <div class="settings-body">
      <nav class="settings-nav" aria-label="Settings sections">
        <button class:active={tab === "internet"} onclick={() => (tab = "internet")}>Internet</button>
        <button class:active={tab === "setup"} onclick={() => (tab = "setup")}>Setup</button>
        <button class:active={tab === "security"} onclick={() => (tab = "security")}>Security</button>
      </nav>

      <section class="settings-panel">
        {#if tab === "internet"}
          <div class="panel-kicker">Network access</div>
          <h2>Internet access</h2>
          <p class="panel-copy">
            Keep this off for fully local work. Turn it on only when you want Saient to reach Hugging Face,
            check for updates, or download optional runtime assets.
          </p>

          <div class="internet-row">
            <div>
              <div class="field-label">Current state</div>
              <div class:online={internetEnabled} class="internet-state">
                {internetLoaded ? (internetEnabled ? "On" : "Off") : "Checking…"}
              </div>
            </div>
            <button class:online={internetEnabled} class="internet-toggle" onclick={toggleInternet} disabled={!internetLoaded || internetSaving}>
              {internetSaving ? "Saving…" : internetEnabled ? "Turn off" : "Turn on"}
            </button>
          </div>

          {#if internetError}<div class="panel-error">{internetError}</div>{/if}

          {#if !internetEnabled}
            <div class="offline-note">
              Automatic update checks, Hugging Face, and optional asset downloads are blocked.
              A manual update check can request temporary update-only access; local models and
              normal Saient runtime remain available.
            </div>
          {/if}
        {:else if tab === "setup"}
          <div class="panel-kicker">Runtime repair</div>
          <h2>Setup &amp; repair</h2>
          <p class="panel-copy">
            Reopen the setup wizard to install or repair the managed Python environment used by
            Image Gen, Video, Vision, TTS, and LoRA.
          </p>

          <div class="setup-status">
            <div class="setup-status-row">
              <span>Setup state</span>
              <strong class:ready={setupInfo?.setup_done}>{profileLabel}</strong>
            </div>
            <div class="setup-status-row">
              <span>Creative tools</span>
              <strong class:ready={setupInfo?.creative_ready}>
                {setupLoaded ? (setupInfo?.creative_ready ? "Ready" : "Not installed") : "Checking…"}
              </strong>
            </div>
            <div class="setup-status-row">
              <span>Managed Python</span>
              <strong class:ready={setupInfo?.venv_ready}>
                {setupLoaded ? (setupInfo?.venv_ready ? "Found" : "Not found") : "Checking…"}
              </strong>
            </div>
          </div>

          <div class="setup-note">
            This keeps existing models, settings, downloads, and environment files. It only clears
            the saved setup choice and lets you run Full setup again.
          </div>

          {#if setupError}<div class="panel-error">{setupError}</div>{/if}

          <button class="setup-btn" onclick={runSetupAgain} disabled={!setupLoaded || setupStarting}>
            {setupStarting ? "Opening setup…" : "Run setup again"}
          </button>
        {:else}
          <div class="panel-kicker">Access</div>
          <h2>Launch password</h2>
          <p class="panel-copy">Require a password when opening Saient on this desktop.</p>
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
  .internet-row {
    display: flex; align-items: center; justify-content: space-between; gap: 16px;
    border: 1px solid #2a2f39; border-radius: 10px; background: #101319;
    padding: 12px; margin-bottom: 14px;
  }
  .field-label { color: #707989; font-size: 11px; text-transform: uppercase; font-weight: 800; margin-bottom: 6px; }
  .internet-state { color: #f5a623; font-size: 15px; font-weight: 800; margin-top: 4px; }
  .internet-state.online { color: #00d68f; }
  .internet-toggle {
    min-width: 106px; height: 38px; padding: 0 14px; border-radius: 8px; cursor: pointer;
    border: 1px solid rgba(0,214,143,0.28); background: rgba(0,214,143,0.12);
    color: #b9f6dc; font-weight: 800;
  }
  .internet-toggle.online {
    border-color: rgba(245,166,35,0.36); background: rgba(245,166,35,0.1); color: #f7d08a;
  }
  .internet-toggle:disabled { opacity: 0.55; cursor: not-allowed; }
  .panel-error {
    color: #f87171; border: 1px solid rgba(248,113,113,0.35);
    background: rgba(248,113,113,0.08); border-radius: 9px; padding: 10px 12px;
    font-size: 12px; line-height: 1.4; margin-bottom: 14px;
  }
  .offline-note {
    color: #98a0ad; border: 1px solid #2a2f39; background: #0e1116;
    border-radius: 10px; padding: 14px; font-size: 13px; line-height: 1.5;
  }
  .setup-status {
    border: 1px solid #2a2f39; border-radius: 10px; background: #101319;
    padding: 4px 12px; margin-bottom: 14px;
  }
  .setup-status-row {
    min-height: 42px; display: flex; align-items: center; justify-content: space-between; gap: 16px;
    border-bottom: 1px solid #242a33; color: #98a0ad; font-size: 12px;
  }
  .setup-status-row:last-child { border-bottom: 0; }
  .setup-status-row strong { color: #f5a623; font-size: 12px; text-align: right; }
  .setup-status-row strong.ready { color: #00d68f; }
  .setup-note {
    color: #98a0ad; border: 1px solid #2a2f39; background: #0e1116;
    border-radius: 10px; padding: 12px; font-size: 12px; line-height: 1.5; margin-bottom: 14px;
  }
  .setup-btn {
    height: 40px; padding: 0 16px; border-radius: 8px; cursor: pointer;
    border: 1px solid rgba(108,142,245,0.45); background: rgba(108,142,245,0.16);
    color: #dce4ff; font-weight: 800;
  }
  .setup-btn:hover { background: rgba(108,142,245,0.24); }
  .setup-btn:disabled { opacity: 0.55; cursor: not-allowed; }
  .security-btn {
    margin-top: 16px; height: 38px; padding: 0 14px; border-radius: 8px; cursor: pointer;
    border: 1px solid rgba(108,142,245,0.35); background: rgba(108,142,245,0.12);
    color: #cdd6f5; font-weight: 700;
  }
  .security-btn:hover { background: rgba(108,142,245,0.2); }
  @media (max-width: 640px) {
    .settings-body { grid-template-columns: 1fr; }
    .settings-nav { flex-direction: row; border-right: 0; border-bottom: 1px solid #2a2f39; }
    .settings-nav button { flex: 1; text-align: center; }
  }
</style>
