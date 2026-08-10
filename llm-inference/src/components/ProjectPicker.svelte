<script lang="ts">
  // Shown when no project is open, and reachable afterwards from the Files tab.
  //
  // Before projects, every task wrote into one shared agent-workspace, so a
  // snake game, a fibonacci script and a stray log all piled up together. That
  // also made checkpoints misleading — a snapshot of "the workspace" captured
  // every unrelated file too.

  import { agent, projects, toast, ui } from "../lib/state.svelte.js";
  import * as T from "../lib/tauri.js";
  import {
    AGI_LEVELS, AGI_LEVEL_INFO, DEFAULT_AGI_LEVEL, CONDUCT_NOTE, MODEL_VOICE_NOTE,
    effectiveAgiLevel, needsLoop,
    type AgiLevel,
  } from "../lib/agiLevel.js";

  let { onDone }: { onDone?: () => void } = $props();

  let name = $state("");
  let level = $state<AgiLevel>(DEFAULT_AGI_LEVEL);
  const chosen = $derived(AGI_LEVEL_INFO[level]);
  let busy = $state(false);
  let error = $state("");

  async function refresh() {
    projects.list = await T.projectList().catch(() => []);
  }
  refresh();

  async function create() {
    if (!name.trim()) return;
    busy = true;
    error = "";
    try {
      const info = await T.projectCreate(name.trim(), level);
      await adopt(info);
    } catch (e) {
      // The Rust side explains exactly why a name was refused; show that rather
      // than a generic failure.
      error = String(e).replace(/^Error:\s*/, "");
    } finally {
      busy = false;
    }
  }

  async function openProject(projectName: string) {
    busy = true;
    error = "";
    try {
      await adopt(await T.projectOpen(projectName));
    } catch (e) {
      error = String(e).replace(/^Error:\s*/, "");
    } finally {
      busy = false;
    }
  }

  async function adopt(info: T.ProjectInfo) {
    projects.active = info;
    agent.sandboxRoot = info.path;
    agent.workspaceEpoch += 1;
    // The file tree and checkpoint list both belong to the old project; clear
    // them so nothing from the previous one lingers on screen.
    agent.tree = [];
    agent.selPath = null;
    agent.content = "";
    name = "";
    await T.saientSetEnabled(
      needsLoop(effectiveAgiLevel(ui.saientEnabled, info.agi_level)),
    ).catch(() => {});
    await refresh();
    toast(`Project “${info.name}” open`, "success");
    onDone?.();
  }

  function when(secs: number): string {
    if (!secs) return "";
    const days = Math.floor((Date.now() / 1000 - secs) / 86400);
    if (days <= 0) return "today";
    if (days === 1) return "yesterday";
    return `${days} days ago`;
  }
</script>

<div class="pp-backdrop" role="presentation"></div>
<div class="pp" role="dialog" aria-label="Choose a project">
  <h2>Which project?</h2>
  <p class="pp-sub">
    Each project is its own folder, with its own files and checkpoints. Keeping
    them apart means saving one piece of work never picks up another's files.
  </p>

  <div class="pp-new">
    <input
      bind:value={name}
      placeholder="New project name…"
      onkeydown={(e) => e.key === "Enter" && create()}
    />
    <button class="pp-btn primary" onclick={create} disabled={busy || !name.trim()}>
      Create
    </button>
  </div>

  <!-- Chosen per project, at creation, because it changes what the agent may do
       and one project may want it where another does not. -->
  <div class="pp-label">How much of Saient runs here?</div>
  <div class="pp-levels">
    {#each AGI_LEVELS as l}
      <button class="pp-level" class:sel={level === l} onclick={() => (level = l)}>
        <span class="pp-level-title">{AGI_LEVEL_INFO[l].title}</span>
        <span class="pp-level-sum">{AGI_LEVEL_INFO[l].summary}</span>
      </button>
    {/each}
  </div>

  <div class="pp-detail">
    <p>{chosen.detail}</p>
    {#if chosen.tradeoff}
      <p class="pp-tradeoff">{chosen.tradeoff}</p>
    {/if}
    {#if needsLoop(level)}
      <p class="pp-voice">{MODEL_VOICE_NOTE}</p>
      <p class="pp-conduct">{CONDUCT_NOTE}</p>
    {/if}
  </div>

  {#if error}<div class="pp-error">{error}</div>{/if}

  {#if projects.list.length}
    <div class="pp-label">Or open an existing one</div>
    <div class="pp-list">
      {#each projects.list as p}
        <button class="pp-item" onclick={() => openProject(p.name)} disabled={busy}>
          <span class="pp-item-name">{p.name}</span>
          <span class="pp-item-meta">
            {p.entry_count === 0 ? "empty" : `${p.entry_count} item${p.entry_count === 1 ? "" : "s"}`}
            {#if when(p.modified)}· {when(p.modified)}{/if}
          </span>
        </button>
      {/each}
    </div>
  {/if}

  {#if onDone}
    <button class="pp-btn pp-later" onclick={onDone}>Not now</button>
  {/if}
</div>

<style>
  .pp-backdrop { position: fixed; inset: 0; background: rgba(0,0,0,0.55); z-index: 50; }
  .pp {
    position: fixed; z-index: 51; left: 50%; top: 50%; transform: translate(-50%, -50%);
    width: 460px; max-width: calc(100vw - 40px);
    background: var(--bg2); border: 1px solid var(--border); border-radius: var(--radius);
    padding: 20px; display: flex; flex-direction: column; gap: 12px;
    box-shadow: 0 16px 48px rgba(0,0,0,0.55);
  }
  .pp h2 { margin: 0; font-size: 16px; color: var(--text); }
  .pp-sub { margin: 0; font-size: 11px; color: var(--text3); line-height: 1.55; }
  .pp-new { display: flex; gap: 8px; }
  .pp-new input {
    flex: 1; background: var(--bg); border: 1px solid var(--border);
    border-radius: var(--radius-sm); color: var(--text); padding: 7px 9px; font-size: 12px;
  }
  .pp-error { font-size: 11px; color: #d0553a; border-left: 2px solid #d0553a; padding-left: 8px; }
  .pp-label { font-size: 10px; text-transform: uppercase; letter-spacing: 0.07em; color: var(--text3); }
  .pp-list { display: flex; flex-direction: column; gap: 4px; max-height: 240px; overflow-y: auto; }
  .pp-item {
    display: flex; justify-content: space-between; align-items: baseline; gap: 10px;
    background: var(--bg); border: 1px solid var(--border); border-radius: var(--radius-sm);
    padding: 8px 10px; cursor: pointer; text-align: left; width: 100%;
  }
  .pp-item:hover { border-color: var(--accent); }
  .pp-item-name { font-size: 12px; color: var(--text); }
  .pp-item-meta { font-size: 10px; color: var(--text3); white-space: nowrap; }
  .pp-btn {
    background: transparent; border: 1px solid var(--border); color: var(--text2);
    border-radius: var(--radius-sm); font-size: 12px; padding: 7px 14px; cursor: pointer;
  }
  .pp-btn:hover { color: var(--text); border-color: var(--text3); }
  .pp-btn.primary { background: var(--accent); border-color: var(--accent); color: #fff; }
  .pp-later { align-self: flex-start; font-size: 11px; padding: 4px 10px; }
  .pp-levels { display: grid; grid-template-columns: 1fr 1fr; gap: 6px; }
  .pp-level {
    display: flex; flex-direction: column; gap: 2px; text-align: left;
    background: var(--bg); border: 1px solid var(--border);
    border-radius: var(--radius-sm); padding: 7px 9px; cursor: pointer;
  }
  .pp-level:hover { border-color: var(--text3); }
  .pp-level.sel { border-color: var(--accent); background: rgba(108,142,245,0.08); }
  .pp-level-title { font-size: 12px; color: var(--text); font-weight: 600; }
  .pp-level-sum { font-size: 10px; color: var(--text3); line-height: 1.4; }
  .pp-detail { display: flex; flex-direction: column; gap: 6px; }
  .pp-detail p { margin: 0; font-size: 11px; color: var(--text2); line-height: 1.55; }
  .pp-tradeoff { color: #d08a3a !important; }
  .pp-voice { color: var(--text3) !important; border-left: 2px solid var(--border); padding-left: 8px; }
  .pp-conduct {
    color: var(--text3) !important; border-left: 2px solid var(--border);
    padding-left: 8px;
  }
</style>
