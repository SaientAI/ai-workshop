<script lang="ts">
  import { onMount, onDestroy } from "svelte";
  import { getCurrentWebviewWindow } from "@tauri-apps/api/webviewWindow";
  import { setupEvents } from "./lib/events.js";
  import { ui, model, agent, license, update, toast } from "./lib/state.svelte.js";
  import { handleKey } from "./lib/shortcuts.js";
  import * as T from "./lib/tauri.js";
  import TitleBar from "./components/TitleBar.svelte";
  import ShortcutsHelp from "./components/ShortcutsHelp.svelte";
  import SetupWizard from "./components/SetupWizard.svelte";
  import Paywall from "./components/Paywall.svelte";
  import LockScreen from "./components/LockScreen.svelte";
  import SecuritySettings from "./components/SecuritySettings.svelte";
  import UpdateBox from "./components/UpdateBox.svelte";
  import Toasts from "./components/Toasts.svelte";
  import IconRail from "./components/IconRail.svelte";
  import ChatScreen from "./components/screens/ChatScreen.svelte";
  import AgentScreen from "./components/screens/AgentScreen.svelte";
  import ImageGenScreen from "./components/screens/ImageGenScreen.svelte";
  import VideoScreen from "./components/screens/VideoScreen.svelte";
  import VisionScreen from "./components/screens/VisionScreen.svelte";
  import TTSScreen from "./components/screens/TTSScreen.svelte";
  import LoRAScreen from "./components/screens/LoRAScreen.svelte";
  import MergeScreen from "./components/screens/MergeScreen.svelte";

  const appWindow = getCurrentWebviewWindow();

  let showSetup = $state(false);
  let locked = $state(false);

  onMount(async () => {
    // Launch password gate — check first so we lock before content is usable.
    locked = await T.passwordIsSet().catch(() => false);

    // Capture-phase so app shortcuts win over the xterm terminal / inputs.
    window.addEventListener("keydown", handleKey, true);

    // First-run setup wizard — show until the user completes or skips it.
    const sys = await T.detectSystem().catch(() => null);
    if (sys && !sys.setup_done) showSetup = true;

    // Licensing — trial countdown / paywall on expiry.
    const lic = await T.licenseStatus().catch(() => null);
    if (lic) {
      license.status = lic.status;
      license.daysLeft = lic.days_left;
      license.trialDays = lic.trial_days;
      license.tier = lic.tier;
    }

    // Update check — best-effort, silent on failure (offline is fine).
    T.checkUpdate().then((u) => {
      update.checked = true;
      update.available = u.update_available;
      update.current = u.current;
      update.latest = u.latest;
      update.url = u.url;
      update.notes = u.notes;
      if (u.update_available) toast(`Saient ${u.latest} is available — click ⬆ to update`, "info", 6000);
    }).catch(() => {});

    await setupEvents();

    // Sync write mode to backend
    await T.setAgentWriteMode(ui.agentWriteMode).catch(() => {});

    // Dependency check
    model.depReport = await T.checkDependencies().catch(() => null);

    // Scan models
    model.modelsDir = await T.getModelsDir().catch(() => "");
    model.models = await T.scanModelsDir().catch(() => []);
    if (model.models.length === 1) {
      model.selectedModel = model.models[0];
      model.path = model.models[0].gguf_path;
    }

    // Sandbox root
    agent.sandboxRoot = await T.getSandboxRoot().catch(() => "");

    // GPU poll
    setInterval(async () => {
      model.gpu = await T.getGpuStats().catch(() => null);
    }, 3000);
  });

  onDestroy(() => window.removeEventListener("keydown", handleKey, true));

  async function aw(action: "min" | "max" | "close") {
    if (action === "min") await appWindow.minimize();
    else if (action === "max") await appWindow.toggleMaximize();
    else await appWindow.close();
  }
</script>

<TitleBar {aw} />
<div class="layout">
  <IconRail />
  {#if ui.screen === "chat"}
    <ChatScreen />
  {:else if ui.screen === "agent"}
    <AgentScreen />
  {:else if ui.screen === "imggen"}
    <ImageGenScreen />
  {:else if ui.screen === "video"}
    <VideoScreen />
  {:else if ui.screen === "vision"}
    <VisionScreen />
  {:else if ui.screen === "tts"}
    <TTSScreen />
  {:else if ui.screen === "lora"}
    <LoRAScreen />
  {:else if ui.screen === "merge"}
    <MergeScreen />
  {/if}
</div>

{#if ui.showShortcuts}
  <ShortcutsHelp />
{/if}

{#if showSetup}
  <SetupWizard onDone={async () => {
    showSetup = false;
    // Re-scan now that deps may have been installed.
    model.modelsDir = await T.getModelsDir().catch(() => model.modelsDir);
    model.models = await T.scanModelsDir().catch(() => model.models);
    model.depReport = await T.checkDependencies().catch(() => model.depReport);
  }} />
{/if}

<!-- Trial countdown strip (non-blocking); click to unlock early -->
{#if license.status === "trial" && !showSetup}
  <button class="trial-strip" onclick={() => (license.showUnlock = true)}>
    Trial · {license.daysLeft} {license.daysLeft === 1 ? "day" : "days"} left — Unlock full version
  </button>
{/if}

<!-- Paywall: blocking when the trial has expired, dismissible when opened manually -->
{#if license.status === "expired"}
  <Paywall blocking />
{:else if license.showUnlock}
  <Paywall onClose={() => (license.showUnlock = false)} />
{/if}

<!-- Launch password gate (covers everything via z-index) -->
{#if locked}
  <LockScreen onUnlock={() => (locked = false)} />
{/if}

<!-- Security settings (set/change/remove the launch password) -->
{#if ui.showSecurity}
  <SecuritySettings onClose={() => (ui.showSecurity = false)} />
{/if}

<Toasts />

<!-- Update box (manual check + link to site) -->
{#if ui.showUpdate}
  <UpdateBox onClose={() => (ui.showUpdate = false)} />
{/if}

<!-- Auto update banner when a newer version is live -->
{#if update.available && !update.dismissed && !showSetup}
  <div class="update-bar">
    <span>Saient {update.latest} is available{update.notes ? ` — ${update.notes}` : ""}</span>
    <button class="ub-cta" onclick={() => (ui.showUpdate = true)}>Get it</button>
    <button class="ub-x" onclick={() => (update.dismissed = true)} aria-label="Dismiss">✕</button>
  </div>
{/if}

<style>
  .update-bar {
    position: fixed; top: 36px; left: 0; right: 0; z-index: 95;
    display: flex; align-items: center; gap: 10px;
    padding: 6px 12px;
    background: rgba(108, 142, 245, 0.16); color: #cdd6f5;
    font-size: 11px; border-bottom: 1px solid rgba(108,142,245,0.3);
  }
  .update-bar > span { flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .update-bar .ub-cta {
    flex-shrink: 0; padding: 3px 10px; border-radius: 6px;
    background: var(--accent); border: 0; color: #0a0a12; font-weight: 700; cursor: pointer;
  }
  .update-bar .ub-cta:hover { background: #82a0ff; }
  .update-bar .ub-x {
    flex-shrink: 0; background: none; border: 0; color: #8a93a3; cursor: pointer; font-size: 12px; padding: 2px 4px;
  }
  .update-bar .ub-x:hover { color: #cdd6f5; }
  .trial-strip {
    position: fixed; bottom: 0; left: 0; right: 0; z-index: 90;
    border: 0; cursor: pointer; padding: 6px 12px;
    background: rgba(108, 142, 245, 0.12); color: #aab6e8;
    font-size: 11px; text-align: center; border-top: 1px solid rgba(108,142,245,0.25);
  }
  .trial-strip:hover { background: rgba(108, 142, 245, 0.2); color: #cdd6f5; }
  :global(*) { box-sizing: border-box; margin: 0; padding: 0; }
  :global(body) {
    background: var(--bg);
    color: var(--text);
    font-family: var(--sans);
    font-size: 13px;
    height: 100vh;
    overflow: hidden;
    user-select: none;
  }
  :global(:root) {
    --bg: #0d0d0f;
    --bg2: #141418;
    --bg3: #1a1a20;
    --border: #2a2a35;
    --text: #e8e8f0;
    --text2: #a0a0b8;
    --text3: #606078;
    --accent: #6c8ef5;
    --green: #00d68f;
    --amber: #f5a623;
    --red: #f87171;
    --sans: -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif;
    --mono: "JetBrains Mono", "Fira Code", "Cascadia Code", Consolas, monospace;
    --radius: 8px;
    --radius-sm: 5px;
  }
  .layout {
    display: flex;
    height: calc(100vh - 36px);
    overflow: hidden;
  }

  /* ── Global shared styles ─────────────────────────────────────────────── */
  :global(.sidebar) {
    width: 240px;
    min-width: 200px;
    background: var(--bg2);
    border-right: 1px solid var(--border);
    display: flex;
    flex-direction: column;
    overflow-y: auto;
    flex-shrink: 0;
  }
  :global(.sidebar-section) {
    padding: 14px 14px 0;
    border-bottom: 1px solid var(--border);
    padding-bottom: 14px;
  }
  :global(.section-label) {
    font-size: 10px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--text3);
    margin-bottom: 10px;
  }
  :global(.field) { display: flex; flex-direction: column; gap: 5px; margin-bottom: 10px; }
  :global(.field label) { font-size: 11px; color: var(--text2); }
  :global(input[type="text"]), :global(input[type="number"]), :global(textarea), :global(select) {
    background: var(--bg3);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    color: var(--text);
    font-family: var(--mono);
    font-size: 11px;
    padding: 5px 8px;
    outline: none;
    width: 100%;
  }
  :global(input:focus), :global(textarea:focus), :global(select:focus) {
    border-color: var(--accent);
  }
  :global(button) {
    background: var(--bg3);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    color: var(--text2);
    cursor: pointer;
    font-size: 11px;
    padding: 5px 10px;
    transition: border-color 0.15s, color 0.15s;
  }
  :global(button:hover) { border-color: var(--accent); color: var(--text); }
  :global(button:disabled) { opacity: 0.4; cursor: not-allowed; }
  :global(.btn-primary) {
    background: var(--accent);
    border-color: var(--accent);
    color: #fff;
    font-weight: 600;
  }
  :global(.btn-primary:hover) { background: #7a9cf7; border-color: #7a9cf7; color: #fff; }
  :global(.btn-danger) { color: var(--red); border-color: rgba(248,113,113,0.3); }
  :global(.perf-row) {
    display: flex;
    justify-content: space-between;
    font-size: 11px;
    color: var(--text2);
    margin-bottom: 4px;
    font-family: var(--mono);
  }
  :global(.perf-row span) { color: var(--text); }
  :global(.tab-action) {
    font-size: 11px;
    padding: 4px 8px;
    border-radius: 4px;
  }
  :global(.load-btn) {
    width: 100%;
    padding: 8px;
    font-size: 12px;
    font-weight: 600;
    background: rgba(108,142,245,0.12);
    border-color: rgba(108,142,245,0.4);
    color: var(--accent);
    border-radius: var(--radius);
  }
  :global(.load-btn:hover) {
    background: rgba(108,142,245,0.22);
    border-color: var(--accent);
    color: #fff;
  }
  :global(.load-btn.loading) { opacity: 0.7; cursor: not-allowed; }
  :global(.model-card) {
    padding: 7px 9px;
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    cursor: pointer;
    font-size: 11px;
    transition: border-color 0.15s;
  }
  :global(.model-card:hover) { border-color: var(--accent); }
  :global(.model-card-sel) {
    border-color: var(--green) !important;
    background: rgba(0,214,143,0.06);
  }
  :global(.sl-field) { margin-bottom: 10px; }
  :global(.sl-row) {
    display: flex;
    justify-content: space-between;
    font-size: 11px;
    color: var(--text2);
    margin-bottom: 4px;
  }
  :global(.sl-row span) { color: var(--text); font-family: var(--mono); }
  :global(input[type="range"]) {
    width: 100%;
    accent-color: var(--accent);
    height: 2px;
  }
</style>
