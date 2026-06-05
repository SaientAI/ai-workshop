<script lang="ts">
  import { chat, model } from "../../lib/state.svelte.js";
  import Sidebar from "../chat/Sidebar.svelte";
  import InputBar from "../chat/InputBar.svelte";
  import Message from "../chat/Message.svelte";
  import ArtifactPane from "../chat/ArtifactPane.svelte";

  let messagesEl = $state<HTMLDivElement | null>(null);

  // Auto-scroll when messages change or content updates
  $effect(() => {
    // Touch streaming content so this re-runs on each token
    void chat.streamBuffer;
    void chat.messages.length;
    if (messagesEl) messagesEl.scrollTop = messagesEl.scrollHeight;
  });

  // Tab lives in shared state so the Ctrl+Tab shortcut can switch it.

  // Saient mascot — sleepy [-_-] when no model is loaded, wide-awake [>_<] when ready.
  const mascot = $derived(
    (model.loaded ? " [>_<]" : " [-_-]") + "\n /|#|\\\n  / \\"
  );

  const PRESETS = [
    { name: "Assistant", p: "You are a helpful, accurate, and concise assistant." },
    { name: "Coder", p: "You are an expert programmer. Write clean, well-commented code. Explain your reasoning." },
    { name: "Analyst", p: "You are a data analyst. Be precise, cite limitations, think step by step." },
    { name: "Terse", p: "Be extremely concise. Answer in as few words as possible. No filler." },
    { name: "None", p: "" },
  ];
</script>

<Sidebar />

<div class="chat-area">
  <!-- Tab bar -->
  <div class="tabbar">
    <button class="tab" class:active={chat.tab === "chat"} onclick={() => (chat.tab = "chat")}>Chat</button>
    <button class="tab" class:active={chat.tab === "system"} onclick={() => (chat.tab = "system")}>System</button>
  </div>

  {#if chat.tab === "chat"}
    <div class="panel-wrap">
      <div class="panel-chat">
        <!-- Messages -->
        <div class="messages" bind:this={messagesEl}>
          {#if chat.messages.length === 0}
            <div class="empty">
              <pre class="mascot" class:awake={model.loaded}>{mascot}</pre>
              <div class="empty-title">{model.loaded ? "Saient is ready" : "Saient"}</div>
              <div class="empty-sub">
                {model.loaded ? "Start typing to chat" : "Load a GGUF model to begin"}
              </div>
            </div>
          {:else}
            {#each chat.messages as msg, i (i)}
              <Message {msg} />
            {/each}
          {/if}
        </div>

        <InputBar />
      </div>

      <ArtifactPane />
    </div>

  {:else}
    <div class="system-panel">
      <div class="sys-header">
        <span class="section-label">System prompt</span>
        <div class="preset-row">
          {#each PRESETS as p}
            <button class="tab-action" onclick={() => (chat.systemPrompt = p.p)}>{p.name}</button>
          {/each}
        </div>
      </div>
      <textarea
        class="sys-input"
        bind:value={chat.systemPrompt}
        placeholder="No system prompt"
        rows="12"
      ></textarea>
    </div>
  {/if}
</div>

<style>
  .chat-area { flex: 1; display: flex; flex-direction: column; overflow: hidden; }
  .tabbar { display: flex; border-bottom: 1px solid var(--border); background: var(--bg2); flex-shrink: 0; }
  .tab { padding: 8px 16px; font-size: 12px; font-weight: 500; color: var(--text3); border: none; background: transparent; border-bottom: 2px solid transparent; border-radius: 0; }
  .tab.active { color: var(--accent); border-bottom-color: var(--accent); }
  .tab:hover { color: var(--text2); }
  .panel-wrap { flex: 1; display: flex; overflow: hidden; }
  .panel-chat { flex: 1; display: flex; flex-direction: column; overflow: hidden; }
  .messages { flex: 1; overflow-y: auto; scroll-behavior: smooth; }
  .empty { display: flex; flex-direction: column; align-items: center; justify-content: center; height: 100%; color: var(--text3); }
  .mascot {
    font-family: var(--mono);
    font-size: 22px;
    line-height: 1.1;
    white-space: pre;
    margin: 0 0 16px;
    color: var(--text3);
    transition: color 0.3s ease;
  }
  /* Wakes up + bobs gently when a model is loaded. */
  .mascot.awake {
    color: var(--accent);
    animation: mascot-bob 2.6s ease-in-out infinite;
  }
  @keyframes mascot-bob {
    0%, 100% { transform: translateY(0); }
    50%      { transform: translateY(-5px); }
  }
  .empty-title { font-size: 18px; font-weight: 700; color: var(--text2); margin-bottom: 6px; }
  .empty-sub { font-size: 13px; }
  .system-panel { flex: 1; display: flex; flex-direction: column; padding: 16px; gap: 10px; overflow: auto; }
  .sys-header { display: flex; justify-content: space-between; align-items: center; }
  .preset-row { display: flex; gap: 4px; flex-wrap: wrap; }
  .sys-input { flex: 1; resize: none; font-size: 12px; font-family: var(--mono); line-height: 1.6; padding: 10px 12px; min-height: 200px; }
</style>
