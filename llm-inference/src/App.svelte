<script lang="ts">
  import { onMount, onDestroy } from "svelte";
  import { getCurrentWebviewWindow } from "@tauri-apps/api/webviewWindow";
  import { setupEvents } from "./lib/events.js";
  import { ui, model, agent, update, toast, projects } from "./lib/state.svelte.js";
  import { handleKey } from "./lib/shortcuts.js";
  import * as T from "./lib/tauri.js";
  import TitleBar from "./components/TitleBar.svelte";
  import ShortcutsHelp from "./components/ShortcutsHelp.svelte";
  import SetupWizard from "./components/SetupWizard.svelte";
  import LockScreen from "./components/LockScreen.svelte";
  import SettingsModal from "./components/SettingsModal.svelte";
  import SecuritySettings from "./components/SecuritySettings.svelte";
  import UpdateBox from "./components/UpdateBox.svelte";
  import Toasts from "./components/Toasts.svelte";
  import IconRail from "./components/IconRail.svelte";
  import SaientPulse from "./components/SaientPulse.svelte";
  import CheckpointBar from "./components/CheckpointBar.svelte";
  import ProjectPicker from "./components/ProjectPicker.svelte";
  import AutonomyConfirm from "./components/AutonomyConfirm.svelte";
  import { effectiveAgiLevel, needsConfirm, needsLoop } from "./lib/agiLevel.js";
  import ChatScreen from "./components/screens/ChatScreen.svelte";
  import AgentScreen from "./components/screens/AgentScreen.svelte";
  import ImageGenScreen from "./components/screens/ImageGenScreen.svelte";
  import AssetBuilderScreen from "./components/screens/AssetBuilderScreen.svelte";
  import VideoScreen from "./components/screens/VideoScreen.svelte";
  import VisionScreen from "./components/screens/VisionScreen.svelte";
  import TTSScreen from "./components/screens/TTSScreen.svelte";
  import LoRAScreen from "./components/screens/LoRAScreen.svelte";
  import MergeScreen from "./components/screens/MergeScreen.svelte";

  const appWindow = getCurrentWebviewWindow();

  let showSetup = $state(false);
  // Session-scoped: re-confirming autonomy is a reminder, not a gate to
  // pass on every turn.
  let autonomyConfirmed = $state(false);
  let autonomyDialogRequested = $state(false);
  let locked = $state(false);

  onMount(async () => {
    // ── Global safety net ──────────────────────────────────────────────────
    // An uncaught error or promise rejection must NEVER white-screen the app —
    // route anything that slips past a feature's own try/catch into a dismissible
    // toast instead of freezing the webview (prod) or raising the dev overlay (dev).
    // This is the backstop behind every screen's error handling.
    const benign = /ResizeObserver/i;
    const surfaceErr = (msg: string) => {
      if (!msg || benign.test(msg)) return;
      console.error("[uncaught]", msg);
      toast(msg.length > 220 ? msg.slice(0, 220) + "…" : msg, "error", 8000);
    };
    window.addEventListener("error", (e: ErrorEvent) => surfaceErr(e.message || String(e.error ?? e)));
    window.addEventListener("unhandledrejection", (e: PromiseRejectionEvent) => { surfaceErr(String(e.reason)); e.preventDefault(); });

    // Launch password gate — check first so we lock before content is usable.
    locked = await T.passwordIsSet().catch(() => false);

    // Capture-phase so app shortcuts win over the xterm terminal / inputs.
    window.addEventListener("keydown", handleKey, true);

    // First-run setup wizard — show until the user completes or skips it.
    const sys = await T.detectSystem().catch(() => null);
    if (sys && !sys.setup_done) showSetup = true;

    // Update check — best-effort, silent on failure (offline is fine).
    T.checkUpdate().then((u) => {
      update.checked = true;
      update.available = u.update_available;
      update.installSupported = u.install_supported;
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

    // Active project. When one is remembered, point the agent at it; otherwise
    // the picker asks before any work lands in a shared heap.
    // Project opening always pauses the heartbeat first. Only wake it after
    // combining the remembered master switch with this project's level.
    projects.active = await T.projectActive().catch(() => null);
    if (projects.active) {
      await T.projectOpen(projects.active.name).catch(() => {});
      agent.sandboxRoot = projects.active.path;
    } else {
      agent.sandboxRoot = await T.getSandboxRoot().catch(() => "");
    }
    await T.saientSetEnabled(
      !showSetup && needsLoop(effectiveAgiLevel(ui.saientEnabled, projects.active?.agi_level)),
    ).catch(() => {});

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

  async function reopenSetup() {
    await T.resetSetup();
    // Setup may replace the creative runtime under an active process. Pause the
    // background loop and present the wizard before any more work is started.
    await T.saientSetEnabled(false).catch(() => {});
    ui.showSettings = false;
    showSetup = true;
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
  {:else if ui.screen === "assets"}
    <AssetBuilderScreen />
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

<!-- Always mounted, showing "Idle" at rest, so the bar never appears or vanishes
     under the screen it sits beneath. -->
<SaientPulse onLevelRequest={() => (autonomyDialogRequested = true)} />

<!-- Ctrl+S and the end-of-turn save prompt. App level so the shortcut works
     everywhere and a prompt cannot be dismissed by navigating away. -->
<CheckpointBar />

{#if ui.showShortcuts}
  <ShortcutsHelp />
{/if}

<!-- Ask only when the agent is opened. Chat writes nothing to disk, so blocking
     it behind a project choice is a dialog for no reason. Suppressed during
     first-run setup and while locked so dialogs cannot stack. -->
{#if ui.screen === "agent" && !projects.active && !agent.sandboxRoot && !showSetup && !locked}
  <ProjectPicker />
{/if}

<!-- The autonomy choice lives inside ProjectPicker, which only opens when no
     project or explicit external workspace is active. A managed project is restored on every
     launch. So the level was chosen once, on first run, and never revisited:
     this project has been running "autonomous" ever since with nothing saying
     so. Re-confirm at the levels that give something up; stay quiet at the ones
     that do not. Once per session, not once per turn. -->
{#if projects.active && !showSetup && !locked && (
       autonomyDialogRequested || (
         ui.screen === "agent" && ui.saientEnabled
         && needsConfirm(projects.active.agi_level) && !autonomyConfirmed
       )
     )}
  <AutonomyConfirm
    level={effectiveAgiLevel(ui.saientEnabled, projects.active.agi_level)}
    project={projects.active.name}
    onConfirm={() => {
      autonomyConfirmed = true;
      autonomyDialogRequested = false;
    }}
    onDismiss={() => {
      // Closing is deliberately non-mutating: an accidental visit to Agent
      // should not silently change this project's saved autonomy level.
      autonomyConfirmed = true;
      autonomyDialogRequested = false;
    }}
    onChange={async (level) => {
      const name = projects.active?.name;
      if (name) projects.active = await T.projectSetLevel(name, level);
      await T.saientSetEnabled(
        needsLoop(effectiveAgiLevel(ui.saientEnabled, projects.active?.agi_level)),
      ).catch(() => {});
      autonomyConfirmed = true;
      autonomyDialogRequested = false;
    }} />
{/if}

{#if showSetup}
  <SetupWizard onDone={async () => {
    showSetup = false;
    // Re-scan now that deps may have been installed.
    model.modelsDir = await T.getModelsDir().catch(() => model.modelsDir);
    model.models = await T.scanModelsDir().catch(() => model.models);
    model.depReport = await T.checkDependencies().catch(() => model.depReport);
    await T.saientSetEnabled(
      needsLoop(effectiveAgiLevel(ui.saientEnabled, projects.active?.agi_level)),
    ).catch(() => {});
  }} />
{/if}

<!-- Launch password gate (covers everything via z-index) -->
{#if locked}
  <LockScreen onUnlock={() => (locked = false)} />
{/if}

<!-- Security settings (set/change/remove the launch password) -->
{#if ui.showSettings}
  <SettingsModal
    onClose={() => (ui.showSettings = false)}
    onSecurity={() => {
      ui.showSettings = false;
      ui.showSecurity = true;
    }}
    onSetup={reopenSetup}
  />
{/if}

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
    /* 36px title bar + 32px Saient Pulse. The Pulse is always mounted, so this
       reservation is unconditional and the screens never shift under it. */
    height: calc(100vh - 36px - 32px);
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
