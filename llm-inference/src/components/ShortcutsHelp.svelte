<script lang="ts">
  import { ui } from "../lib/state.svelte.js";
  import { SHORTCUTS } from "../lib/shortcuts.js";

  // Group shortcuts in declaration order.
  const groups = $derived.by(() => {
    const m = new Map<string, typeof SHORTCUTS>();
    for (const s of SHORTCUTS) {
      if (!m.has(s.group)) m.set(s.group, []);
      m.get(s.group)!.push(s);
    }
    return [...m.entries()];
  });

  function close() { ui.showShortcuts = false; }
</script>

<!-- Backdrop — close only when the backdrop itself is clicked (Esc also closes,
     handled by the global shortcut handler). -->
<!-- svelte-ignore a11y_click_events_have_key_events -->
<div class="sc-backdrop" role="presentation" onclick={(e) => { if (e.target === e.currentTarget) close(); }}>
  <div class="sc-panel" role="dialog" aria-modal="true" aria-label="Keyboard shortcuts" tabindex="-1">
    <div class="sc-head">
      <span class="sc-title">⌨ Keyboard shortcuts</span>
      <button class="sc-close" onclick={close} title="Close (Esc)">✕</button>
    </div>
    <div class="sc-body">
      {#each groups as [name, items]}
        <div class="sc-group">
          <div class="sc-group-name">{name}</div>
          {#each items as s}
            <div class="sc-row">
              <span class="sc-keys">
                {#each s.keys as key}<kbd>{key}</kbd>{/each}
              </span>
              <span class="sc-desc">{s.desc}</span>
            </div>
          {/each}
        </div>
      {/each}
    </div>
    <div class="sc-foot">Press <kbd>?</kbd> or <kbd>Ctrl</kbd><kbd>/</kbd> anytime · <kbd>Esc</kbd> to close</div>
  </div>
</div>

<style>
  .sc-backdrop {
    position: fixed; inset: 0; z-index: 1000;
    background: rgba(0,0,0,0.55);
    display: flex; align-items: center; justify-content: center;
    animation: sc-fade 0.12s ease;
  }
  @keyframes sc-fade { from { opacity: 0; } to { opacity: 1; } }
  .sc-panel {
    width: min(560px, 92vw); max-height: 82vh; overflow: hidden;
    display: flex; flex-direction: column;
    background: var(--bg2); border: 1px solid var(--border);
    border-radius: var(--radius); box-shadow: 0 16px 48px rgba(0,0,0,0.5);
  }
  .sc-head {
    display: flex; align-items: center; justify-content: space-between;
    padding: 12px 16px; border-bottom: 1px solid var(--border);
    background: var(--bg3);
  }
  .sc-title { font-size: 13px; font-weight: 700; color: var(--text); letter-spacing: 0.02em; }
  .sc-close {
    font-size: 13px; padding: 2px 8px; border-radius: 4px;
    color: var(--text3); border-color: transparent; background: transparent;
  }
  .sc-close:hover { color: var(--red); border-color: rgba(248,113,113,0.3); }
  .sc-body { padding: 8px 16px 14px; overflow-y: auto; }
  .sc-group { margin-top: 12px; }
  .sc-group-name {
    font-size: 10px; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.08em; color: var(--accent); margin-bottom: 6px;
  }
  .sc-row {
    display: flex; align-items: center; gap: 12px;
    padding: 4px 0; font-size: 12px;
  }
  .sc-keys { flex: 0 0 130px; display: flex; gap: 3px; flex-wrap: wrap; }
  .sc-desc { color: var(--text2); line-height: 1.4; }
  .sc-foot {
    padding: 10px 16px; border-top: 1px solid var(--border);
    font-size: 11px; color: var(--text3); background: var(--bg3);
    display: flex; align-items: center; gap: 5px;
  }
  kbd {
    display: inline-block; min-width: 16px; text-align: center;
    padding: 2px 6px; font-family: var(--mono); font-size: 11px;
    color: var(--text); background: var(--bg); line-height: 1.2;
    border: 1px solid var(--border2); border-bottom-width: 2px;
    border-radius: 4px;
  }
</style>
