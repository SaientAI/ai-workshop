<script lang="ts">
  import { onMount, onDestroy } from "svelte";
  import { chat, model } from "../../lib/state.svelte.js";
  import { buildSrcdoc } from "../../lib/artifact.js";

  let frameEl = $state<HTMLIFrameElement | null>(null);
  let collapsed = $state(false);
  let copied = $state(false);

  // ── Error toast ──────────────────────────────────────────────────────────────
  interface ArtifactError { message: string; stack: string; }
  let errorToast = $state<ArtifactError | null>(null);
  let dismissTimer: ReturnType<typeof setTimeout> | null = null;

  function onFrameMessage(ev: MessageEvent) {
    if (ev.data?.type !== "artifact-error") return;
    // Only handle messages from our own iframe, not other windows
    if (frameEl && ev.source !== frameEl.contentWindow) return;
    const { message, stack } = ev.data as ArtifactError & { type: string };
    // First line only in the toast — stack goes to the fix prompt
    const firstLine = (message || "Unknown error").split("\n")[0].slice(0, 200);
    errorToast = { message: firstLine, stack: stack || "" };
    if (dismissTimer) clearTimeout(dismissTimer);
    dismissTimer = setTimeout(() => { errorToast = null; }, 12_000);
  }

  function dismissError() {
    errorToast = null;
    if (dismissTimer) { clearTimeout(dismissTimer); dismissTimer = null; }
  }

  function fixError() {
    if (!errorToast) return;
    const err = errorToast.stack
      ? `${errorToast.message}\n\nStack:\n${errorToast.stack}`
      : errorToast.message;
    chat.pendingRetry = `Fix this JavaScript error that just occurred in the artifact:\n\n\`\`\`\n${err}\n\`\`\`\n\nPlease identify and fix the bug in the code you generated.`;
    dismissError();
  }

  onMount(() => { window.addEventListener("message", onFrameMessage); });
  onDestroy(() => {
    window.removeEventListener("message", onFrameMessage);
    if (dismissTimer) clearTimeout(dismissTimer);
  });

  // Reactively update srcdoc whenever artifact content changes
  $effect(() => {
    if (frameEl && chat.artifact.active && chat.artifact.content) {
      frameEl.srcdoc = buildSrcdoc(chat.artifact.content);
    }
  });

  function close() {
    chat.artifact = { active: false, title: "", type: "html", content: "", complete: false };
  }

  function copySource() {
    navigator.clipboard.writeText(chat.artifact.content).catch(() => {});
    copied = true;
    setTimeout(() => { copied = false; }, 1500);
  }

  function reload() {
    if (frameEl && chat.artifact.content) {
      frameEl.srcdoc = buildSrcdoc(chat.artifact.content);
    }
  }

  function fullscreen() {
    frameEl?.requestFullscreen?.();
  }

  const streaming = $derived(chat.streaming && !chat.artifact.complete);
</script>

{#if chat.artifact.active}
  <div class="pane" class:collapsed>
    <div class="header">
      <span class="badge">{chat.artifact.type.toUpperCase()}</span>
      <span class="title">{chat.artifact.title}</span>
      {#if streaming}<span class="stream-tag">streaming…</span>{/if}
      <div class="actions">
        <button onclick={copySource} title="Copy source">{copied ? "✓" : "⎘"}</button>
        <button onclick={reload} title="Reload" disabled={streaming}>↺</button>
        <button onclick={() => (collapsed = !collapsed)} title={collapsed ? "Expand" : "Collapse"}>
          {collapsed ? "▶" : "◀"}
        </button>
        <button onclick={fullscreen} title="Fullscreen">⛶</button>
        <button class="close" onclick={close} title="Close">✕</button>
      </div>
    </div>
    {#if !collapsed}
      <div class="frame-wrap">
        <iframe
          bind:this={frameEl}
          class="frame"
          sandbox="allow-scripts allow-forms"
          referrerpolicy="no-referrer"
          title="Artifact preview"
        ></iframe>
        {#if streaming && !chat.artifact.content}
          <div class="stream-overlay">
            <div class="stream-dots"><span></span><span></span><span></span></div>
            <div class="stream-label">Generating…</div>
          </div>
        {/if}
        {#if errorToast}
          <div class="err-toast">
            <div class="err-toast-msg">⚠ {errorToast.message}</div>
            <div class="err-toast-actions">
              <button class="err-fix-btn" onclick={fixError} disabled={chat.streaming || !model.loaded}>Fix it</button>
              <button class="err-dismiss-btn" onclick={dismissError}>✕</button>
            </div>
          </div>
        {/if}
      </div>
    {/if}
  </div>
{/if}

<style>
  .pane {
    flex: 1;
    display: flex;
    flex-direction: column;
    border-left: 1px solid var(--border);
    background: #111;
    min-width: 300px;
    max-width: 60%;
  }
  .pane.collapsed { flex: 0; min-width: 32px; max-width: 32px; }
  .header {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 8px 10px;
    background: var(--bg2);
    border-bottom: 1px solid var(--border);
    flex-shrink: 0;
  }
  .badge {
    font-size: 10px; font-weight: 700;
    background: rgba(108,142,245,0.15);
    color: var(--accent);
    padding: 1px 6px; border-radius: 3px;
    font-family: var(--mono);
  }
  .title { font-size: 12px; font-weight: 600; flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .actions { display: flex; gap: 4px; flex-shrink: 0; }
  .actions button { font-size: 11px; padding: 2px 7px; }
  .actions .close:hover { color: var(--red); border-color: rgba(248,113,113,0.3); }
  .stream-tag { font-size: 10px; color: var(--amber); font-family: var(--mono); animation: pulse 1.2s ease-in-out infinite; }
  @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.4; } }
  .frame-wrap { flex: 1; position: relative; display: flex; overflow: hidden; }
  .frame { flex: 1; width: 100%; border: none; background: #111; color-scheme: dark; }
  .stream-overlay {
    position: absolute; inset: 0;
    display: flex; flex-direction: column; align-items: center; justify-content: center;
    gap: 12px; background: #111;
  }
  .stream-dots { display: flex; gap: 8px; }
  .stream-dots span {
    width: 8px; height: 8px; border-radius: 50%; background: var(--accent);
    animation: bounce 1s ease-in-out infinite;
  }
  .stream-dots span:nth-child(2) { animation-delay: 0.15s; }
  .stream-dots span:nth-child(3) { animation-delay: 0.3s; }
  @keyframes bounce { 0%, 100% { transform: translateY(0); opacity: 0.5; } 50% { transform: translateY(-8px); opacity: 1; } }
  .stream-label { font-size: 12px; color: var(--text3); }
  /* Error toast — transform forces own compositing layer so it renders above WebKit iframes */
  .err-toast {
    position: absolute; bottom: 12px; left: 12px; right: 12px;
    background: #450a0a; border: 1px solid #7f1d1d;
    border-radius: var(--radius); padding: 10px 12px;
    display: flex; align-items: center; gap: 10px;
    animation: toast-in 0.2s ease;
    z-index: 10;
    transform: translateZ(0);
    max-height: 80px; overflow: hidden;
  }
  @keyframes toast-in { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: none; } }
  .err-toast-msg { flex: 1; font-size: 11px; font-family: var(--mono); color: #fca5a5; line-height: 1.4; word-break: break-word; }
  .err-toast-actions { display: flex; gap: 6px; flex-shrink: 0; align-items: flex-start; }
  .err-fix-btn { font-size: 11px; font-weight: 700; padding: 3px 10px; background: #dc2626; border: none; color: #fff; border-radius: var(--radius-sm); cursor: pointer; white-space: nowrap; }
  .err-fix-btn:hover:not(:disabled) { background: #ef4444; }
  .err-fix-btn:disabled { opacity: 0.5; cursor: not-allowed; }
  .err-dismiss-btn { font-size: 11px; padding: 3px 7px; background: transparent; border: 1px solid #7f1d1d; color: #fca5a5; border-radius: var(--radius-sm); cursor: pointer; }
  .err-dismiss-btn:hover { background: rgba(239,68,68,0.1); }
</style>
