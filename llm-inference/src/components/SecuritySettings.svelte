<script lang="ts">
  import * as T from "../lib/tauri.js";

  let { onClose }: { onClose: () => void } = $props();

  let isSet = $state(false);
  let current = $state("");
  let next = $state("");
  let confirm = $state("");
  let error = $state("");
  let ok = $state("");
  let busy = $state(false);

  $effect(() => { T.passwordIsSet().then((v) => (isSet = v)).catch(() => {}); });

  async function save() {
    error = ""; ok = "";
    if (next.length < 4) { error = "Password must be at least 4 characters."; return; }
    if (next !== confirm) { error = "Passwords don't match."; return; }
    busy = true;
    try {
      await T.passwordSet(next, isSet ? current : undefined);
      ok = isSet ? "Password changed." : "Password set.";
      isSet = true; current = ""; next = ""; confirm = "";
    } catch (e) {
      error = typeof e === "string" ? e : "Couldn't save.";
    } finally { busy = false; }
  }

  async function remove() {
    error = ""; ok = ""; busy = true;
    try {
      await T.passwordClear(current);
      ok = "Password removed."; isSet = false; current = "";
    } catch (e) {
      error = typeof e === "string" ? e : "Couldn't remove.";
    } finally { busy = false; }
  }
</script>

<div class="sec-backdrop" role="dialog" aria-modal="true" aria-label="Security">
  <div class="sec-card">
    <div class="sec-head">
      <span class="sec-title">🔒 Launch password</span>
      <button class="sec-x" onclick={onClose} aria-label="Close">×</button>
    </div>
    <p class="sec-sub">
      {isSet ? "A password is required each time Saient opens." : "Require a password to open Saient. Stored only on this device."}
    </p>

    {#if isSet}
      <input class="sec-input" type="password" placeholder="Current password" bind:value={current} />
    {/if}
    <input class="sec-input" type="password" placeholder={isSet ? "New password" : "Password"} bind:value={next} />
    <input class="sec-input" type="password" placeholder="Confirm" bind:value={confirm}
           onkeydown={(e) => e.key === 'Enter' && save()} />

    {#if error}<div class="sec-msg err">{error}</div>{/if}
    {#if ok}<div class="sec-msg ok">{ok}</div>{/if}

    <div class="sec-row">
      <button class="sec-save" onclick={save} disabled={busy || !next}>
        {isSet ? "Change password" : "Set password"}
      </button>
      {#if isSet}
        <button class="sec-remove" onclick={remove} disabled={busy || !current}>Remove</button>
      {/if}
    </div>
  </div>
</div>

<style>
  .sec-backdrop {
    position: fixed; inset: 0; z-index: 150;
    background: rgba(8,10,14,0.7); backdrop-filter: blur(3px);
    display: flex; align-items: center; justify-content: center;
  }
  .sec-card {
    width: 380px; max-width: 92vw; padding: 22px;
    background: #15181e; border: 1px solid #2a2f39; border-radius: 14px;
    box-shadow: 0 20px 50px rgba(0,0,0,0.5);
  }
  .sec-head { display: flex; align-items: center; justify-content: space-between; margin-bottom: 6px; }
  .sec-title { font-weight: 600; color: #eef1f6; font-size: 14px; }
  .sec-x { background: none; border: 0; color: #6b7280; font-size: 20px; cursor: pointer; line-height: 1; }
  .sec-x:hover { color: #cdd3df; }
  .sec-sub { margin: 0 0 16px; color: #98a0ad; font-size: 12px; line-height: 1.5; }
  .sec-input {
    width: 100%; margin-bottom: 9px; padding: 10px 12px; border-radius: 9px;
    border: 1px solid #2a2f39; background: #0e1116; color: #e6e9ef; font-size: 13px;
  }
  .sec-input:focus { outline: none; border-color: #6c8ef5; }
  .sec-msg { font-size: 12px; margin: 4px 0 10px; }
  .sec-msg.err { color: #ff8080; }
  .sec-msg.ok { color: #00d68f; }
  .sec-row { display: flex; gap: 8px; margin-top: 8px; }
  .sec-save {
    flex: 1; padding: 10px; border: 0; border-radius: 9px; cursor: pointer;
    background: linear-gradient(135deg, #5b8cff, #6a5bff); color: #fff; font-size: 13px; font-weight: 600;
  }
  .sec-save:disabled { opacity: 0.5; cursor: default; }
  .sec-remove {
    padding: 0 16px; border: 1px solid #3a2a2a; border-radius: 9px; cursor: pointer;
    background: #221717; color: #e6a0a0; font-size: 13px;
  }
  .sec-remove:disabled { opacity: 0.5; cursor: default; }
</style>
