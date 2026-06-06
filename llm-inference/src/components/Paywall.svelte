<script lang="ts">
  import { open } from "@tauri-apps/plugin-shell";
  import { license, toast } from "../lib/state.svelte.js";
  import * as T from "../lib/tauri.js";

  // `blocking` = trial has ended; the app is locked until activated.
  let { blocking = false, onClose = () => {} }: { blocking?: boolean; onClose?: () => void } = $props();

  const BUY_URL = "https://buy.stripe.com/14AaEWeuscbl3oDb2BeQM00";

  let key = $state("");
  let busy = $state(false);
  let error = $state("");

  async function activate() {
    if (!key.trim() || busy) return;
    busy = true; error = "";
    try {
      const s = await T.licenseActivate(key.trim());
      license.status = s.status;
      license.tier = s.tier;
      license.daysLeft = s.days_left;
      license.trialDays = s.trial_days;
      license.showUnlock = false;
      toast("Unlocked — thank you! Saient is yours forever.", "success", 5000);
      onClose();
    } catch (e) {
      error = typeof e === "string" ? e : "Couldn't activate that key.";
    } finally {
      busy = false;
    }
  }

  async function buy() {
    try { await open(BUY_URL); } catch { window.open(BUY_URL, "_blank"); }
  }
</script>

<div class="pw-backdrop" role="dialog" aria-modal="true" aria-label="Unlock Saient">
  <div class="pw-card">
    <div class="pw-logo">Saient</div>

    {#if blocking}
      <h2 class="pw-title">Your free month is up</h2>
      <p class="pw-sub">Hope you liked it. Unlock the full app — one payment, yours forever.</p>
    {:else}
      <h2 class="pw-title">{license.daysLeft} {license.daysLeft === 1 ? "day" : "days"} left in your trial</h2>
      <p class="pw-sub">Unlock now and never see this again. One payment, yours forever.</p>
    {/if}

    <button class="pw-buy" onclick={buy}>Unlock full version — £20</button>

    <div class="pw-or"><span>already bought? paste your key</span></div>

    <div class="pw-key">
      <input
        class="pw-input"
        placeholder="SAIENT license key"
        bind:value={key}
        onkeydown={(e) => e.key === "Enter" && activate()}
        spellcheck="false"
        autocomplete="off"
      />
      <button class="pw-activate" onclick={activate} disabled={!key.trim() || busy}>
        {busy ? "…" : "Activate"}
      </button>
    </div>
    {#if error}<div class="pw-err">{error}</div>{/if}

    {#if !blocking}
      <button class="pw-later" onclick={onClose}>Maybe later</button>
    {/if}
  </div>
</div>

<style>
  .pw-backdrop {
    position: fixed; inset: 0; z-index: 200;
    background: rgba(8, 10, 14, 0.78); backdrop-filter: blur(4px);
    display: flex; align-items: center; justify-content: center;
  }
  .pw-card {
    width: 420px; max-width: 92vw; padding: 32px 28px;
    background: #15181e; border: 1px solid #2a2f39; border-radius: 16px;
    box-shadow: 0 24px 60px rgba(0,0,0,0.5); text-align: center;
  }
  .pw-logo { font-weight: 700; letter-spacing: 0.5px; color: #cdd3df; margin-bottom: 18px; }
  .pw-title { font-size: 20px; margin: 0 0 6px; color: #f2f4f8; }
  .pw-sub { margin: 0 0 22px; color: #98a0ad; font-size: 13px; line-height: 1.5; }
  .pw-buy {
    width: 100%; padding: 13px; border: 0; border-radius: 10px; cursor: pointer;
    background: linear-gradient(135deg, #5b8cff, #6a5bff); color: #fff;
    font-size: 15px; font-weight: 600;
  }
  .pw-buy:hover { filter: brightness(1.08); }
  .pw-or { display: flex; align-items: center; gap: 10px; margin: 20px 0 12px; color: #5f6773; font-size: 11px; }
  .pw-or::before, .pw-or::after { content: ""; flex: 1; height: 1px; background: #262b34; }
  .pw-key { display: flex; gap: 8px; }
  .pw-input {
    flex: 1; padding: 10px 12px; border-radius: 9px; border: 1px solid #2a2f39;
    background: #0e1116; color: #e6e9ef; font-size: 13px; font-family: ui-monospace, monospace;
  }
  .pw-input:focus { outline: none; border-color: #5b8cff; }
  .pw-activate {
    padding: 0 16px; border: 1px solid #2a2f39; border-radius: 9px; cursor: pointer;
    background: #222732; color: #e6e9ef; font-size: 13px;
  }
  .pw-activate:disabled { opacity: 0.5; cursor: default; }
  .pw-err { margin-top: 10px; color: #ff8080; font-size: 12px; }
  .pw-later {
    margin-top: 18px; background: none; border: 0; color: #6b7280; cursor: pointer; font-size: 12px;
  }
  .pw-later:hover { color: #9aa3b2; }
</style>
