<script lang="ts">
  import { onMount, onDestroy } from "svelte";
  import * as T from "../../lib/tauri.js";
  import { toast } from "../../lib/state.svelte.js";

  let scan = $state<T.AssetScan | null>(null);
  let loading = $state(false);
  let running = $state(false);
  let mode = $state<"idle" | "dry" | "build" | "local3d">("idle");
  let output = $state("");
  let error = $state("");
  let selectedSources = $state<string[]>([]);
  let refreshTimer: number | null = null;

  const hasSources = $derived((scan?.sources.length ?? 0) > 0);
  const selectedCount = $derived(selectedSources.length);
  const canBuild = $derived(hasSources && selectedCount > 0 && !running);

  function fileSize(bytes: number) {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
  }

  function reconcileSelection(next: T.AssetScan) {
    const available = new Set(next.sources.map((file) => file.name));
    selectedSources = selectedSources.filter((name) => available.has(name));
    if (selectedSources.length === 0 && next.sources.length === 1) {
      selectedSources = [next.sources[0].name];
    }
  }

  async function refresh(showLoading = true) {
    if (showLoading) loading = true;
    error = "";
    try {
      const next = await T.assetBuilderScan();
      scan = next;
      reconcileSelection(next);
    } catch (e) {
      error = String(e);
    } finally {
      if (showLoading) loading = false;
    }
  }

  async function openDir(kind: "source" | "output") {
    await T.assetBuilderOpenDir(kind).catch((e) => {
      error = String(e);
      toast(error, "error");
    });
  }

  async function run(dryRun: boolean, builder: "relief" | "local3d" = "relief") {
    if (running) return;
    if (selectedSources.length === 0) {
      error = "Select one or more PNG relief sources first.";
      toast(error, "error");
      return;
    }
    running = true;
    mode = dryRun ? "dry" : builder === "local3d" ? "local3d" : "build";
    output = "";
    error = "";
    try {
      const result = await T.assetBuilderRun(dryRun, selectedSources, builder);
      const action = dryRun ? "Dry run" : builder === "local3d" ? "Local 3D" : "Relief build";
      const sourceType = builder === "local3d" ? "local 3D source" : "PNG relief source";
      const header = `${action} ${result.ok ? "finished" : "failed"} for ${selectedSources.length} selected ${sourceType}${selectedSources.length === 1 ? "" : "s"} (exit ${result.code}).`;
      output = [header, result.stdout.trim(), result.stderr.trim()].filter(Boolean).join("\n\n");
      if (!result.ok) {
        error = `Asset builder exited with code ${result.code}`;
        toast(error, "error");
      } else {
        toast(dryRun ? "Dry run complete" : builder === "local3d" ? "Local 3D build complete" : "Relief GLB build complete", "success");
      }
      await refresh();
    } catch (e) {
      error = String(e);
      toast(error, "error");
    } finally {
      running = false;
      mode = "idle";
    }
  }

  function isSelected(name: string) {
    return selectedSources.includes(name);
  }

  function setSelected(name: string, checked: boolean) {
    selectedSources = checked
      ? Array.from(new Set([...selectedSources, name]))
      : selectedSources.filter((item) => item !== name);
  }

  function selectAll() {
    selectedSources = scan?.sources.map((file) => file.name) ?? [];
  }

  function clearSelection() {
    selectedSources = [];
  }

  onMount(() => {
    refresh();
    refreshTimer = window.setInterval(() => {
      if (!running && document.visibilityState === "visible") {
        refresh(false);
      }
    }, 2500);
  });

  onDestroy(() => {
    if (refreshTimer !== null) {
      window.clearInterval(refreshTimer);
    }
  });
</script>

<div class="assets-layout">
  <aside class="assets-sidebar">
    <div class="section-label">Game Assets</div>

    <div class="path-block">
      <div class="path-label">PNG Relief Sources</div>
      <button class="path-btn" onclick={() => openDir("source")}>{scan?.source_dir ?? "Loading..."}</button>
    </div>

    <div class="path-block">
      <div class="path-label">Production GLBs</div>
      <button class="path-btn" onclick={() => openDir("output")}>{scan?.output_dir ?? "Loading..."}</button>
    </div>

    <div class="status-row">
      <span>Blender</span>
      {#if scan?.blender_path}
        <strong class="ok">Ready</strong>
      {:else}
        <strong class="warn">Missing</strong>
      {/if}
    </div>
    {#if scan?.blender_path}
      <div class="small-path">{scan.blender_path}</div>
    {:else}
      <div class="hint">Install Blender or set <code>BLENDER_BIN</code> before building relief GLBs.</div>
    {/if}

    <div class="button-stack">
      <button onclick={() => refresh()} disabled={loading || running}>{loading ? "Refreshing..." : "Refresh"}</button>
      <button onclick={selectAll} disabled={!hasSources || running}>Select All</button>
      <button onclick={clearSelection} disabled={selectedCount === 0 || running}>Clear Selection</button>
      <button onclick={() => run(true)} disabled={!canBuild}>Dry Run</button>
      <button onclick={() => run(false, "relief")} disabled={!canBuild}>
        {running && mode === "build" ? "Building..." : selectedCount > 0 ? `Build Relief GLBs (${selectedCount})` : "Select PNG Relief Sources"}
      </button>
      <button class="primary" onclick={() => run(false, "local3d")} disabled={!canBuild}>
        {running && mode === "local3d" ? "Generating..." : selectedCount > 0 ? `Run Local 3D (${selectedCount})` : "Select Local 3D Sources"}
      </button>
    </div>
  </aside>

  <main class="assets-main">
    <div class="files-grid">
      <section>
        <div class="panel-head">
          <h2>PNG Relief Sources</h2>
          <div class="panel-actions">
            <button onclick={selectAll} disabled={!hasSources || running}>All</button>
            <button onclick={clearSelection} disabled={selectedCount === 0 || running}>None</button>
            <span>{selectedCount}/{scan?.sources.length ?? 0}</span>
          </div>
        </div>
        <div class="file-list">
          {#if scan && scan.sources.length > 0}
            {#each scan.sources as file}
              <label class="file-row selectable" class:selected={isSelected(file.name)}>
                <input
                  type="checkbox"
                  checked={isSelected(file.name)}
                  onchange={(e) => setSelected(file.name, e.currentTarget.checked)}
                  disabled={running}
                />
                <span>{file.name}</span>
                <em>{fileSize(file.size)}</em>
              </label>
            {/each}
          {:else}
            <div class="empty">No PNG relief sources.</div>
          {/if}
        </div>
      </section>

      <section>
        <div class="panel-head">
          <h2>Production GLBs</h2>
          <span>{scan?.outputs.length ?? 0}</span>
        </div>
        <div class="file-list">
          {#if scan && scan.outputs.length > 0}
            {#each scan.outputs as file}
              <div class="file-row">
                <span>{file.name}</span>
                <em>{fileSize(file.size)}</em>
              </div>
            {/each}
          {:else}
            <div class="empty">No production GLBs.</div>
          {/if}
        </div>
      </section>
    </div>

    <section class="log-panel">
      <div class="panel-head">
        <h2>Relief Build Log</h2>
        {#if running}<span>{mode === "dry" ? "Dry run" : mode === "local3d" ? "Local 3D" : "Building"}</span>{:else}<span>{selectedCount} selected</span>{/if}
      </div>
      {#if error}
        <div class="error">{error}</div>
      {/if}
      <pre>{output || "Run a relief dry run or build to see output here."}</pre>
    </section>
  </main>
</div>

<style>
  .assets-layout { display: flex; flex: 1; min-width: 0; overflow: hidden; }
  .assets-sidebar {
    width: 280px;
    flex-shrink: 0;
    border-right: 1px solid var(--border);
    background: var(--bg2);
    padding: 16px;
    overflow-y: auto;
  }
  .path-block { margin-bottom: 12px; }
  .path-label {
    font-size: 10px;
    color: var(--text3);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 5px;
  }
  .path-btn {
    width: 100%;
    text-align: left;
    font-family: var(--mono);
    font-size: 10px;
    line-height: 1.35;
    overflow-wrap: anywhere;
    padding: 7px 8px;
  }
  .status-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    border-top: 1px solid var(--border);
    padding-top: 12px;
    margin-top: 12px;
    font-size: 12px;
    color: var(--text2);
  }
  .ok { color: var(--green); }
  .warn { color: var(--amber); }
  .small-path {
    margin-top: 5px;
    font-family: var(--mono);
    font-size: 10px;
    color: var(--text3);
    overflow-wrap: anywhere;
  }
  .hint {
    margin-top: 6px;
    font-size: 11px;
    color: var(--text3);
    line-height: 1.45;
  }
  .hint code { font-family: var(--mono); color: var(--text2); }
  .button-stack { display: flex; flex-direction: column; gap: 8px; margin-top: 16px; }
  .button-stack button { padding: 8px 10px; }
  .button-stack .primary {
    background: rgba(108,142,245,0.14);
    border-color: rgba(108,142,245,0.42);
    color: var(--accent);
    font-weight: 700;
  }
  .assets-main {
    flex: 1;
    min-width: 0;
    padding: 16px;
    display: flex;
    flex-direction: column;
    gap: 14px;
    overflow: hidden;
  }
  .files-grid {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 14px;
    min-height: 180px;
  }
  section {
    min-width: 0;
    border: 1px solid var(--border);
    border-radius: var(--radius);
    background: rgba(255,255,255,0.015);
    overflow: hidden;
  }
  .panel-head {
    height: 38px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 12px;
    border-bottom: 1px solid var(--border);
    background: rgba(255,255,255,0.018);
  }
  .panel-head h2 {
    font-size: 12px;
    color: var(--text);
    font-weight: 700;
  }
  .panel-head span {
    font-size: 10px;
    color: var(--text3);
    font-family: var(--mono);
  }
  .panel-actions {
    display: flex;
    align-items: center;
    gap: 6px;
  }
  .panel-actions button {
    padding: 3px 7px;
    font-size: 10px;
    line-height: 1.2;
  }
  .file-list {
    max-height: 260px;
    overflow-y: auto;
  }
  .file-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
    padding: 8px 12px;
    border-bottom: 1px solid rgba(255,255,255,0.035);
    font-size: 12px;
  }
  .file-row.selectable {
    cursor: pointer;
    user-select: none;
  }
  .file-row.selectable:hover {
    background: rgba(108,142,245,0.06);
  }
  .file-row.selected {
    background: rgba(108,142,245,0.11);
  }
  .file-row input {
    flex-shrink: 0;
  }
  .file-row span {
    min-width: 0;
    flex: 1;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .file-row em {
    color: var(--text3);
    font-style: normal;
    font-family: var(--mono);
    font-size: 10px;
    flex-shrink: 0;
  }
  .empty {
    padding: 18px 12px;
    color: var(--text3);
    font-size: 12px;
  }
  .log-panel {
    flex: 1;
    min-height: 0;
    display: flex;
    flex-direction: column;
  }
  .error {
    margin: 10px 12px 0;
    color: var(--red);
    font-size: 12px;
    padding: 8px 10px;
    background: rgba(248,113,113,0.07);
    border: 1px solid rgba(248,113,113,0.25);
    border-radius: var(--radius-sm);
  }
  pre {
    flex: 1;
    min-height: 0;
    margin: 0;
    padding: 12px;
    overflow: auto;
    color: var(--text2);
    font: 11px/1.55 var(--mono);
    white-space: pre-wrap;
    user-select: text;
  }
  @media (max-width: 900px) {
    .assets-layout { flex-direction: column; }
    .assets-sidebar { width: 100%; border-right: 0; border-bottom: 1px solid var(--border); }
    .files-grid { grid-template-columns: 1fr; }
  }
</style>
