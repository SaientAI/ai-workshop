<script lang="ts">
  import { ui } from "../lib/state.svelte.js";
  import type { Screen } from "../lib/types.js";

  const items: Array<{ id: Screen; icon: string; label: string }> = [
    { id: "chat",   icon: "💬", label: "Chat" },
    { id: "agent",  icon: "🖥",  label: "Agent" },
    { id: "imggen", icon: "🖼",  label: "Image Gen" },
    { id: "assets", icon: "🧱",  label: "Game Assets" },
    { id: "video",  icon: "🎬",  label: "Video Gen" },
    { id: "tts",    icon: "🔊",  label: "TTS" },
    { id: "lora",   icon: "🎛",  label: "LoRA" },
    { id: "merge",  icon: "🔀",  label: "Merge" },
  ];
</script>

<nav class="rail">
  {#each items as item}
    <button
      class="rail-btn"
      class:active={ui.screen === item.id}
      onclick={() => (ui.screen = item.id)}
      title={item.label}
    >
      {item.icon}
    </button>
  {/each}
  <div class="rail-spacer"></div>
  <button
    class="rail-btn"
    onclick={() => (ui.showSettings = true)}
    title="Settings"
    aria-label="Settings"
  >
    ⚙
  </button>
</nav>

<style>
  .rail {
    width: 48px;
    background: var(--bg2);
    border-right: 1px solid var(--border);
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 8px 0;
    gap: 4px;
    flex-shrink: 0;
  }
  .rail-btn {
    width: 36px; height: 36px;
    border-radius: var(--radius-sm);
    border: 1px solid transparent;
    font-size: 16px;
    display: flex; align-items: center; justify-content: center;
    color: var(--text3);
    background: transparent;
    padding: 0;
  }
  .rail-btn:hover { color: var(--text); background: var(--bg3); }
  .rail-btn.active {
    color: var(--accent);
    background: rgba(108,142,245,0.12);
    border-color: rgba(108,142,245,0.3);
  }
  .rail-spacer { flex: 1; }
</style>
