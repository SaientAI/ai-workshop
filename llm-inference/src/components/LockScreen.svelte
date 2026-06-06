<script lang="ts">
  import * as T from "../lib/tauri.js";

  let { onUnlock }: { onUnlock: () => void } = $props();

  let password = $state("");
  let error = $state("");
  let busy = $state(false);
  let input: HTMLInputElement | null = $state(null);

  $effect(() => { input?.focus(); });

  async function submit() {
    if (!password || busy) return;
    busy = true; error = "";
    const ok = await T.passwordVerify(password).catch(() => false);
    busy = false;
    if (ok) onUnlock();
    else { error = "Incorrect password."; password = ""; }
  }
</script>

<div class="lock-backdrop">
  <div class="lock-card">
    <div class="lock-glyph">🔒</div>
    <div class="lock-logo">Saient</div>
    <p class="lock-sub">Enter your password to continue</p>
    <input
      bind:this={input}
      class="lock-input"
      type="password"
      placeholder="Password"
      bind:value={password}
      onkeydown={(e) => e.key === "Enter" && submit()}
    />
    <button class="lock-btn" onclick={submit} disabled={!password || busy}>
      {busy ? "…" : "Unlock"}
    </button>
    {#if error}<div class="lock-err">{error}</div>{/if}
  </div>
</div>

<style>
  .lock-backdrop {
    position: fixed; inset: 0; z-index: 300;
    background: #0b0d11; display: flex; align-items: center; justify-content: center;
  }
  .lock-card { width: 320px; max-width: 90vw; padding: 36px 28px; text-align: center; }
  .lock-glyph { font-size: 30px; margin-bottom: 12px; opacity: 0.85; }
  .lock-logo { font-weight: 700; letter-spacing: 0.5px; color: #cdd3df; font-size: 18px; }
  .lock-sub { margin: 6px 0 22px; color: #8b93a1; font-size: 12px; }
  .lock-input {
    width: 100%; padding: 11px 13px; border-radius: 10px;
    border: 1px solid #2a2f39; background: #14181e; color: #e6e9ef; font-size: 14px;
  }
  .lock-input:focus { outline: none; border-color: #6c8ef5; }
  .lock-btn {
    width: 100%; margin-top: 12px; padding: 11px; border: 0; border-radius: 10px; cursor: pointer;
    background: linear-gradient(135deg, #5b8cff, #6a5bff); color: #fff; font-size: 14px; font-weight: 600;
  }
  .lock-btn:disabled { opacity: 0.5; cursor: default; }
  .lock-err { margin-top: 12px; color: #ff8080; font-size: 12px; }
</style>
