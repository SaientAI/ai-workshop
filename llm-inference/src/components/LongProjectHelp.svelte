<script lang="ts">
  import { buildLongProjectCommand } from "../lib/longProject.js";
  let goal = $state("");
  let hours = $state(24);
  let steps = $state(10000);
  let toolTimeout = $state(3600);
  let allowWrite = $state(false);
  let allowShell = $state(false);
  let verifyCommand = $state("");
  let copied = $state("");
  const generated = $derived.by(() => {
    try {
      return { command: buildLongProjectCommand({ goal, hours, steps, toolTimeout, allowWrite, allowShell, verifyCommand }), error: "" };
    } catch (error) {
      return { command: "", error: String(error instanceof Error ? error.message : error) };
    }
  });
  async function copyCommand() {
    copied = "";
    try {
      await navigator.clipboard.writeText(generated.command);
      copied = "Copied. Paste inside the Saient agent prompt and review its confirmation.";
    } catch {
      copied = "Clipboard unavailable. Select and copy the command below manually.";
    }
  }
</script>

<details class="long-project">
  <summary>Long Project mode — opt-in, saved progress and adjustable limits</summary>
  <div class="project-options">
    <p>Run <code>saient</code> first. Paste the generated command at its <code>❯</code> prompt,
      not directly into Bash or PowerShell. This mode continues beyond ordinary terminal turns.
      The separate Planner Auto controls are unchanged.</p>
    <label>Project goal <textarea rows="2" bind:value={goal} placeholder="Describe the outcome and the files to work on"></textarea></label>
    <div class="limits">
      <label>Deadline (hours) <input type="number" min="1" max="8784" bind:value={hours} /></label>
      <label>Model calls <input type="number" min="1" max="10000000" bind:value={steps} /></label>
      <label>Command timeout (seconds) <input type="number" min="1" max="604800" bind:value={toolTimeout} /></label>
    </div>
    <label>Acceptance command (optional) <input type="text" bind:value={verifyCommand} placeholder="e.g. npm test — an explicitly authorized command" /></label>
    <p>Without an acceptance command, Saient pauses for your review instead of marking the goal verified.
      These are time and model-call limits, not a paid-token spending meter. The deadline includes paused time.</p>
    <label class="permission"><input type="checkbox" bind:checked={allowWrite} /> Allow unattended workspace file writes</label>
    <label class="permission"><input type="checkbox" bind:checked={allowShell} /> Allow unattended host shell commands</label>
    <p class:warning={allowShell}>Host shell access is <strong>not an OS sandbox</strong>: commands have your account's access,
      including outside this folder. Keep it off unless you accept that scope. This does not authorize unrelated deletion,
      purchases or publishing. These permissions are separate from Planner Write mode and <code>/yolo</code>.</p>
    {#if generated.command}
      <label>Command to review and copy <textarea class="command" readonly rows="3" value={generated.command}></textarea></label>
      <button type="button" onclick={copyCommand}>Copy project command</button>
    {:else if goal}
      <p role="alert">{generated.error}</p>
    {/if}
    <p role="status">{copied}</p>
    <p><code>/project status</code> · <code>/project pause</code> · <code>/project stop</code> · <code>/project resume</code></p>
    <p>Keep the desktop and model available. Saved progress survives restart, but execution never starts silently:
      use <code>/project resume</code>. Unknown interrupted commands require inspection before acknowledgment.
      <code>/project help</code> explains job inspection, cancellation, budget extensions and archival.</p>
  </div>
</details>

<style>
  .long-project { flex: 0 0 auto; border-bottom: 1px solid var(--border); padding: 10px 14px; font-size: 12px; }
  summary { cursor: pointer; color: var(--text); }
  .project-options { max-height: 45vh; overflow: auto; padding-top: 8px; }
  p { color: var(--text3); line-height: 1.5; margin: 8px 0; }
  label { display: flex; flex-direction: column; gap: 4px; color: var(--text2); margin: 8px 0; }
  input, textarea { background: var(--bg); color: var(--text); border: 1px solid var(--border); border-radius: 4px; padding: 6px; min-width: 0; }
  textarea { resize: vertical; }
  .limits { display: flex; flex-wrap: wrap; gap: 12px; }
  .limits label { flex: 1 1 120px; }
  .permission { flex-direction: row; align-items: center; }
  .command, code { font-family: monospace; }
  .warning { color: #e7ae63; }
  button { background: var(--accent); color: var(--text); border: 1px solid var(--border); border-radius: 4px; padding: 7px 12px; cursor: pointer; }
</style>
