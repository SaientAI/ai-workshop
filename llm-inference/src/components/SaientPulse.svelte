<script lang="ts">
  // Saient Pulse — a persistent bottom bar reporting what Saient is doing.
  //
  // Always mounted rather than conditionally rendered: it shows "Idle" when idle,
  // which means the bar never appears or vanishes and its layout can live in a
  // component-scoped style block safely (see the note atop global.css).
  // (Avoid writing that tag name literally in here — Svelte's parser reads it as
  // a real tag opening even inside a comment, and the script block never closes.)
  //
  // Everything shown here is derived from real state. There is no timer that
  // animates on its own — if the robot moves, work is happening.

  import { agent, pulse } from "../lib/state.svelte.js";
  import { animationFor, activityLine, flavourFor, formatElapsed } from "../lib/pulse.js";
  import { activityText, isWorking } from "../lib/turnState.js";

  const anim = $derived(animationFor(agent.turn, pulse.step ?? undefined));
  const working = $derived(isWorking(agent.turn) || agent.continuing);
  const line = $derived(activityLine(agent.turn, pulse.step ?? undefined, activityText(agent.turn)));
  const flavour = $derived(flavourFor(anim, pulse.startedAt));

  // Elapsed clock. A single interval for the whole bar, and only while something
  // is actually running — an idle app should not tick.
  let now = $state(Date.now());
  $effect(() => {
    if (!working) return;
    const id = setInterval(() => (now = Date.now()), 500);
    return () => clearInterval(id);
  });
  const elapsed = $derived(formatElapsed(now - pulse.startedAt));

  function togglePause() {
    agent.paused = !agent.paused;
  }
</script>

<div class="pulse" class:pulse-working={working} class:pulse-failed={anim === "failed"}>
  <!-- The robot. Each group is shown only for the animation it belongs to, so a
       gesture cannot outlive the work that justified it. -->
  <svg class="bot bot-{anim}" viewBox="0 0 32 24" aria-hidden="true">
    <!-- Body: common to every state -->
    <rect class="bot-head" x="9" y="4" width="14" height="11" rx="3.5" />
    <line class="bot-ant" x1="16" y1="4" x2="16" y2="1.5" />
    <circle class="bot-ant-tip" cx="16" cy="1.2" r="1.2" />
    <rect class="bot-base" x="11" y="16.5" width="10" height="4" rx="1.5" />

    {#if anim === "failed"}
      <!-- Eyes become crosses, sparks flash once and stop -->
      <g class="bot-eyes-x">
        <line x1="11.8" y1="8" x2="14.2" y2="10.4" /><line x1="14.2" y1="8" x2="11.8" y2="10.4" />
        <line x1="17.8" y1="8" x2="20.2" y2="10.4" /><line x1="20.2" y1="8" x2="17.8" y2="10.4" />
      </g>
      <g class="spark">
        <line x1="25" y1="6" x2="28" y2="3" /><line x1="26" y1="9" x2="29.5" y2="9" />
        <line x1="7" y1="6" x2="4" y2="3" />
      </g>
    {:else if anim === "completed"}
      <!-- Content eyes, and a tick that draws itself once -->
      <g class="bot-eyes-happy">
        <path d="M11.6 9.6 q1.4 -1.8 2.8 0" /><path d="M17.6 9.6 q1.4 -1.8 2.8 0" />
      </g>
      <path class="tick" d="M24 11 l2.4 2.4 L31 8" />
    {:else}
      <g class="bot-eyes">
        <circle cx="13" cy="9.4" r="1.5" /><circle cx="19" cy="9.4" r="1.5" />
      </g>
    {/if}

    {#if anim === "thinking"}
      <!-- Notes either side; the eyes look between them -->
      <rect class="note note-l" x="1.5" y="7" width="5" height="7" rx="1" />
      <rect class="note note-r" x="25.5" y="7" width="5" height="7" rx="1" />
    {:else if anim === "typing"}
      <!-- Hands tapping, alternating -->
      <rect class="hand hand-l" x="7" y="17" width="3" height="2.4" rx="1.2" />
      <rect class="hand hand-r" x="22" y="17" width="3" height="2.4" rx="1.2" />
    {:else if anim === "reading"}
      <!-- A page with a scan line sweeping down it -->
      <rect class="page" x="24" y="5" width="7" height="10" rx="1" />
      <line class="page-line" x1="25.4" y1="7.4" x2="29.6" y2="7.4" />
      <line class="page-line" x1="25.4" y1="10" x2="29.6" y2="10" />
      <line class="page-line" x1="25.4" y1="12.6" x2="29.6" y2="12.6" />
      <line class="scan" x1="24" y1="5" x2="31" y2="5" />
    {:else if anim === "scanning"}
      <!-- Shield, pulsing -->
      <path class="shield" d="M27.5 4.5 l3.5 1.4 v3.6 q0 3.4 -3.5 5 q-3.5 -1.6 -3.5 -5 v-3.6 z" />
    {:else if anim === "verifying"}
      <!-- Magnifying glass sweeping across -->
      <g class="glass">
        <circle cx="27" cy="8.5" r="3.2" /><line x1="29.4" y1="10.9" x2="31.4" y2="12.9" />
      </g>
    {:else if anim === "saving"}
      <!-- A file dropping into a drawer -->
      <rect class="file" x="24.5" y="3" width="5.5" height="6.5" rx="0.8" />
      <path class="drawer" d="M23.5 13.5 h8 v5 h-8 z" />
      <line class="drawer-handle" x1="26" y1="16" x2="29" y2="16" />
    {/if}
  </svg>

  <div class="pulse-text">
    <!-- Factual. Always. -->
    <span class="pulse-line">{line}</span>
    <!-- Flavour, clearly subordinate and never load-bearing. -->
    {#if working && flavour}<span class="pulse-flavour">{flavour}</span>{/if}
  </div>

  {#if working}<span class="pulse-clock">{elapsed}</span>{/if}

  <button class="pulse-btn" onclick={() => (pulse.expanded = !pulse.expanded)}>
    {pulse.expanded ? "Hide" : "Details"}
  </button>
  {#if working}
    <button class="pulse-btn" class:on={agent.paused} onclick={togglePause}>
      {agent.paused ? "Pausing…" : "Pause"}
    </button>
  {/if}
</div>

{#if pulse.expanded}
  <div class="pulse-drawer">
    {#if pulse.log.length === 0}
      <div class="pulse-drawer-empty">No activity yet.</div>
    {:else}
      {#each pulse.log.slice().reverse() as entry}
        <div class="pulse-entry">
          <span class="pulse-at">{new Date(entry.at).toLocaleTimeString()}</span>
          <span class="pulse-what">{entry.text}</span>
          {#if entry.detail}<span class="pulse-detail">{entry.detail}</span>{/if}
        </div>
      {/each}
    {/if}
  </div>
{/if}

<style>
  .pulse {
    height: 32px;
    flex-shrink: 0;
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 0 12px;
    background: var(--bg2);
    border-top: 1px solid var(--border);
    font-size: 11px;
    color: var(--text2);
  }
  .pulse-working { background: var(--bg3); }
  .pulse-failed { border-top-color: #d0553a; }

  .pulse-text { display: flex; align-items: baseline; gap: 10px; min-width: 0; flex: 1; }
  .pulse-line { color: var(--text); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .pulse-flavour { color: var(--text3); font-style: italic; white-space: nowrap;
                   overflow: hidden; text-overflow: ellipsis; }
  .pulse-clock { font-family: var(--mono); color: var(--text3); font-variant-numeric: tabular-nums; }

  .pulse-btn {
    background: transparent; border: 1px solid var(--border); color: var(--text2);
    border-radius: var(--radius-sm); font-size: 10px; padding: 2px 8px; cursor: pointer;
  }
  .pulse-btn:hover { color: var(--text); border-color: var(--text3); }
  .pulse-btn.on { color: var(--accent); border-color: var(--accent); }

  .pulse-drawer {
    max-height: 190px; overflow-y: auto; background: var(--bg2);
    border-top: 1px solid var(--border); padding: 6px 12px; flex-shrink: 0;
  }
  .pulse-drawer-empty { font-size: 11px; color: var(--text3); }
  .pulse-entry { display: flex; gap: 10px; font-size: 11px; padding: 2px 0; }
  .pulse-at { font-family: var(--mono); color: var(--text3); flex-shrink: 0; }
  .pulse-what { color: var(--text2); }
  .pulse-detail { color: var(--text3); font-style: italic; overflow: hidden; text-overflow: ellipsis; }

  /* ── Robot ────────────────────────────────────────────────────────────── */
  .bot { width: 30px; height: 22px; flex-shrink: 0; overflow: visible; }
  .bot-head, .bot-base, .note, .page, .file, .drawer, .shield {
    fill: none; stroke: var(--text3); stroke-width: 1.2;
  }
  .bot-ant, .page-line, .drawer-handle { stroke: var(--text3); stroke-width: 1.1; }
  .bot-ant-tip { fill: var(--text3); }
  .bot-eyes circle { fill: var(--text2); }
  .pulse-working .bot-head, .pulse-working .bot-base { stroke: var(--accent); }
  .pulse-working .bot-eyes circle { fill: var(--accent); }
  .pulse-working .bot-ant-tip { fill: var(--accent); }

  /* Antenna blinks only while working — a still robot means a still app. */
  .pulse-working .bot-ant-tip { animation: blink 1.4s ease-in-out infinite; }
  @keyframes blink { 0%, 100% { opacity: 1; } 50% { opacity: 0.25; } }

  /* Thinking: eyes track between the two notes */
  .bot-thinking .bot-eyes { animation: look 2.4s ease-in-out infinite; }
  @keyframes look {
    0%, 100% { transform: translateX(-1.4px); }
    50% { transform: translateX(1.4px); }
  }
  .note-l { animation: nudge 2.4s ease-in-out infinite; }
  .note-r { animation: nudge 2.4s ease-in-out infinite reverse; }
  @keyframes nudge { 0%, 100% { opacity: 0.35; } 50% { opacity: 1; } }

  /* Running a command: hands tap, out of phase */
  .hand { fill: var(--accent); stroke: none; }
  .hand-l { animation: tap 0.5s ease-in-out infinite; }
  .hand-r { animation: tap 0.5s ease-in-out infinite 0.25s; }
  @keyframes tap { 0%, 100% { transform: translateY(0); } 50% { transform: translateY(-1.6px); } }

  /* Reading: a scan line runs down the page */
  .scan { stroke: var(--accent); stroke-width: 1.2; animation: sweep 1.6s linear infinite; }
  @keyframes sweep { 0% { transform: translateY(0.5px); } 100% { transform: translateY(9.5px); } }

  /* Security scan: the shield pulses */
  .shield { stroke: var(--accent); animation: pulseShield 1.3s ease-in-out infinite; }
  @keyframes pulseShield {
    0%, 100% { opacity: 0.4; stroke-width: 1.2; }
    50% { opacity: 1; stroke-width: 1.8; }
  }

  /* Verifying: the glass sweeps back and forth */
  .glass { fill: none; stroke: var(--accent); stroke-width: 1.2;
           animation: hunt 2s ease-in-out infinite; }
  @keyframes hunt {
    0%, 100% { transform: translate(0, 0); }
    50% { transform: translate(-3.5px, 2px); }
  }

  /* Saving: the file drops into the drawer, repeatedly */
  .file { stroke: var(--accent); animation: drop 1.8s ease-in-out infinite; }
  @keyframes drop {
    0% { transform: translateY(0); opacity: 1; }
    60% { transform: translateY(9px); opacity: 1; }
    75% { transform: translateY(9px); opacity: 0; }
    100% { transform: translateY(0); opacity: 0; }
  }

  /* Failed: sparks flash once, then everything stops */
  .bot-eyes-x line { stroke: #d0553a; stroke-width: 1.4; }
  .spark line { stroke: #d0553a; stroke-width: 1.3;
                animation: flash 0.45s steps(2) 3; opacity: 0; }
  @keyframes flash { 0% { opacity: 1; } 100% { opacity: 0; } }

  /* Completed: the tick draws itself once and stays */
  .bot-eyes-happy path { fill: none; stroke: #4ba36a; stroke-width: 1.4; stroke-linecap: round; }
  .tick { fill: none; stroke: #4ba36a; stroke-width: 1.8; stroke-linecap: round;
          stroke-linejoin: round; stroke-dasharray: 12; stroke-dashoffset: 12;
          animation: draw 0.4s ease-out forwards; }
  @keyframes draw { to { stroke-dashoffset: 0; } }

  @media (prefers-reduced-motion: reduce) {
    .bot *, .pulse * { animation: none !important; }
  }
</style>
