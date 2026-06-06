<script lang="ts">
  import { toasts, dismissToast } from "../lib/state.svelte.js";
</script>

<div class="toast-stack">
  {#each toasts as t (t.id)}
    <button class="toast {t.kind}" onclick={() => dismissToast(t.id)} title="Dismiss">
      <span class="ic">{t.kind === "success" ? "✓" : t.kind === "error" ? "⚠" : "›"}</span>
      <span class="msg">{t.msg}</span>
    </button>
  {/each}
</div>

<style>
  .toast-stack {
    position: fixed; bottom: 16px; right: 16px; z-index: 300;
    display: flex; flex-direction: column; gap: 8px; align-items: flex-end;
    pointer-events: none;
  }
  .toast {
    pointer-events: auto; cursor: pointer; text-align: left;
    display: flex; align-items: flex-start; gap: 9px;
    max-width: 340px; padding: 10px 13px;
    background: #1a1d24; border: 1px solid #2a2f39; border-radius: 10px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.45);
    color: #dfe4ec; font-size: 12.5px; line-height: 1.45;
    animation: toast-in 0.18s ease;
  }
  .toast:hover { border-color: #3a414d; }
  .toast .ic { flex-shrink: 0; font-family: var(--mono); }
  .toast.success { border-left: 3px solid var(--green); }
  .toast.success .ic { color: var(--green); }
  .toast.error { border-left: 3px solid var(--red); }
  .toast.error .ic { color: var(--red); }
  .toast.info { border-left: 3px solid var(--accent); }
  .toast.info .ic { color: var(--accent); }
  .toast .msg { flex: 1; }
  @keyframes toast-in { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: none; } }
</style>
