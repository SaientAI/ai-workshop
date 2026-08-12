<script lang="ts">
  /**
   * A reminder of what Saient is currently allowed to do, and a way to change it.
   *
   * The level is chosen inside ProjectPicker, which opens only when no project is
   * active — and the active project is restored from disk every launch. So the
   * choice was made once, on first run, and then became invisible: this project
   * had been running "autonomous" for weeks with nothing in the UI saying so, no
   * way to see it, and no way to change it. `projectSetLevel` existed in
   * `lib/tauri.ts` and in Rust the whole time and nothing called it.
   *
   * Shown once per session, and only for the levels that give something up.
   */
  import { untrack } from "svelte";
  import { AGI_LEVELS, AGI_LEVEL_INFO, type AgiLevel } from "../lib/agiLevel.js";

  let { level, project, onConfirm, onChange, onDismiss }: {
    level: AgiLevel;
    project: string;
    onConfirm: () => void;
    onChange: (level: AgiLevel) => void;
    onDismiss: () => void;
  } = $props();

  // This is intentionally the opening value. The dialog is unmounted after a
  // keep/change/dismiss action, so a later opening starts with a fresh value.
  let picked = $state<AgiLevel>(untrack(() => level));
  const info = $derived(AGI_LEVEL_INFO[picked]);
  const changed = $derived(picked !== level);
</script>

<div class="ac-scrim" role="presentation">
  <div class="ac" role="dialog" aria-modal="true" aria-labelledby="ac-title">
    <button class="ac-close" onclick={onDismiss} aria-label="Close autonomy settings" title="Close">×</button>
    <h2 id="ac-title">Saient is set to “{AGI_LEVEL_INFO[level].title}” in {project}</h2>
    <p class="ac-sub">{AGI_LEVEL_INFO[level].summary}</p>

    <div class="ac-levels">
      {#each AGI_LEVELS as l}
        <button class="ac-level" class:ac-on={picked === l} onclick={() => (picked = l)}>
          <span class="ac-level-title">{AGI_LEVEL_INFO[l].title}</span>
          <span class="ac-level-sum">{AGI_LEVEL_INFO[l].summary}</span>
        </button>
      {/each}
    </div>

    <p class="ac-detail">{info.detail}</p>
    {#if info.tradeoff}
      <p class="ac-tradeoff"><strong>What this gives up:</strong> {info.tradeoff}</p>
    {/if}

    <div class="ac-actions">
      {#if changed}
        <button class="ac-primary" onclick={() => onChange(picked)}>
          Change to {info.title}
        </button>
        <button class="ac-ghost" onclick={() => (picked = level)}>Cancel</button>
      {:else}
        <button class="ac-primary" onclick={onConfirm}>
          Keep {AGI_LEVEL_INFO[level].title}
        </button>
      {/if}
    </div>
  </div>
</div>
