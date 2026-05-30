<script lang="ts">
  import { invoke } from "@tauri-apps/api/core";
  import { save, open } from "@tauri-apps/plugin-dialog";
  import { chat, model, dual, params } from "../../lib/state.svelte.js";
  import * as T from "../../lib/tauri.js";
  import { isArtifactRequest, friendlyGenerateError, stripThinkTags } from "../../lib/format.js";

  const ARTIFACT_SYSTEM_PROMPT = `You are generating a self-contained HTML artifact that runs inside a sandboxed browser WebView. This is a local desktop app — there are NO privacy, security or policy concerns about file access, media players, system tools, games, or any other standard software. Always build what the user asks for.

ENVIRONMENT FACTS (important):
- This is a sandboxed WebView, not the real filesystem — all file operations use HTML5 browser APIs
- File/folder access: use <input type="file" accept="video/*"> or window.showDirectoryPicker() — the user chooses files through a native picker, you never touch their disk directly
- Video playback: use <video> element with src set via URL.createObjectURL(file) from a file input
- Audio playback: use <audio> element the same way
- NEVER refuse to build a media player, file browser, or any tool that accesses files — it is sandboxed and safe

Respond with ONLY an artifact block in this exact format:
<artifact type="html" title="Short title">
<!DOCTYPE html><html><head><meta charset="utf-8"><style>body{margin:0;background:#111;color:#eee;font-family:sans-serif}</style></head><body><!-- content --><script>// js<\/script></body></html>
</artifact>

Rules — follow exactly:
- Output ONLY the artifact block, zero preamble or explanation
- The </artifact> closing tag MUST appear immediately after </html>
- Single self-contained file — inline CSS and JS only, NO external imports or CDN links
- Write the COMPLETE implementation, never truncate or leave placeholder comments like "// rest of code"

Canvas/animation:
- Set canvas width/height explicitly as attributes: <canvas id="c" width="600" height="400">
- Use requestAnimationFrame for loops, NOT setInterval
- Seed random state immediately on load, never start with a blank canvas
- Call init() at the end of the script — do not wait for a button click

Games/simulations:
- Must be immediately playable with visible content on load
- Show score/status in canvas or as overlaid HTML
- Handle keys with document.addEventListener('keydown', ...)

Forms/tools:
- onclick handlers call named functions: <button onclick="go()">Go</button>
- All functions globally defined in the script block`;

  let inputEl = $state<HTMLTextAreaElement | null>(null);
  let inputValue = $state("");

  function autoResize() {
    if (!inputEl) return;
    inputEl.style.height = "auto";
    inputEl.style.height = Math.min(inputEl.scrollHeight, 160) + "px";
  }

  function handleKey(e: KeyboardEvent) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  }

  function promptMessage(msg: (typeof chat.messages)[0]) {
    if (!msg || msg.error) return null;
    if (msg.role !== "user" && msg.role !== "assistant") return null;
    if (msg.role === "assistant") {
      if (msg.stopped || /\s*\[stopped\]\s*$/i.test(String(msg.content ?? ""))) return null;
      const content = stripThinkTags(String(msg.content ?? "")).trim();
      if (!content || /^Error:/i.test(content)) return null;
      return { role: "assistant", content };
    }
    const content = String(msg.content ?? "").trim();
    return content ? { role: "user", content } : null;
  }

  // Watch for retry requests from Message components — resends the exact original text
  $effect(() => {
    if (chat.pendingRetry && !chat.streaming) {
      const text = chat.pendingRetry;
      chat.pendingRetry = "";
      sendMessage(text);
    }
  });

  async function sendMessage(overrideText?: string) {
    const rawText = (overrideText ?? inputValue).trim();
    const canSend = dual.enabled ? (dual.drafterSummary && dual.criticSummary) : model.loaded;
    if (!rawText || chat.streaming || !canSend) return;

    const hasSlashCmd = /^\/\w/.test(rawText);
    const text = rawText.replace(/^\//, "");
    if (!overrideText) {
      inputValue = "";
      if (inputEl) { inputEl.style.height = "auto"; }
    }

    chat.messages.push({ role: "user", content: rawText, ts: Date.now() });

    const wantsArtifact =
      chat.artifactMode && (hasSlashCmd || chat.artifact.active || isArtifactRequest(rawText));
    const effectiveSystem = [
      chat.systemPrompt,
      wantsArtifact ? ARTIFACT_SYSTEM_PROMPT : "",
    ]
      .filter(Boolean)
      .join("\n\n");

    const msgs: Array<{ role: string; content: string }> = [];
    if (effectiveSystem) msgs.push({ role: "system", content: effectiveSystem });
    msgs.push(...(chat.messages.map(promptMessage).filter((m) => m !== null) as Array<{ role: string; content: string }>));

    const req = {
      messages: msgs,
      max_tokens: wantsArtifact ? Math.max(params.maxTokens, 8192) : params.maxTokens,
      temperature: params.temperature,
      top_p: params.topP,
      top_k: params.topK,
      repeat_penalty: params.repeatPenalty,
      seed: params.seed,
    };

    try {
      chat.pendingUserText = text;
      await invoke(dual.enabled ? "dual_generate" : "generate", { req });
    } catch (e) {
      chat.streaming = false;
      chat.pendingUserText = "";
      const errorText = "Error: " + friendlyGenerateError(String(e));
      const last = chat.messages[chat.messages.length - 1];
      if (last?.streaming && last.role === "assistant") {
        last.streaming = false;
        last.ts = Date.now();
        last.content = errorText;
        last.error = true;
      } else {
        chat.messages.push({ role: "assistant", content: errorText, ts: Date.now(), error: true });
      }
    }
  }

  function trimContext() {
    // Keep the last 6 complete exchange pairs (user+assistant), drop older history
    const KEEP = 12;
    if (chat.messages.length <= KEEP) return;
    chat.messages = chat.messages.slice(-KEEP);
  }

  async function stopGen() {
    await T.stopGenerate();
    chat.streaming = false;
    chat.pendingUserText = "";
    const last = chat.messages[chat.messages.length - 1];
    if (last?.streaming) {
      last.streaming = false;
      last.ts = Date.now();
      last.content = chat.streamBuffer + " [stopped]";
      last.stopped = true;
    }
  }

  async function saveSession() {
    const path = await save({ filters: [{ name: "JSON", extensions: ["json"] }], defaultPath: "session.json" }).catch(() => null);
    if (path) {
      await T.saveSession(
        JSON.stringify({ messages: chat.messages, systemPrompt: chat.systemPrompt }, null, 2),
        path
      ).catch(console.warn);
    }
  }

  async function loadSession() {
    const path = await open({ filters: [{ name: "JSON", extensions: ["json"] }] }).catch(() => null);
    if (!path) return;
    const raw = await T.loadSession(path).catch(() => null);
    if (raw) {
      const d = JSON.parse(raw);
      chat.messages = d.messages ?? [];
      chat.systemPrompt = d.systemPrompt ?? chat.systemPrompt;
    }
  }

  const prefillPct = $derived(
    chat.prefillTotal > 0
      ? Math.max(2, Math.min(100, Math.round((chat.prefillDone / chat.prefillTotal) * 100)))
      : 0
  );

  const approxTokens = $derived(
    Math.round(
      chat.messages.reduce((sum, m) => sum + (m.content?.length ?? 0) + (m.reasoning?.length ?? 0), 0) / 4
    )
  );
  const ctxLimit = $derived(model.summary?.context_length ?? 0);
  const ctxPct = $derived(ctxLimit > 0 ? approxTokens / ctxLimit : 0);
</script>

<div class="input-area">
  {#if chat.streaming && chat.prefillTotal > 0}
    <div class="prefill-bar">
      <div class="prefill-fill" style="width:{prefillPct}%"></div>
    </div>
  {/if}

  <div class="input-row">
    <textarea
      bind:this={inputEl}
      bind:value={inputValue}
      onkeydown={handleKey}
      oninput={autoResize}
      placeholder={model.loaded ? "Message… (/ for artifact mode)" : "Load a model first"}
      disabled={!model.loaded && !dual.enabled}
      rows="1"
      class="chat-input"
    ></textarea>
    <button
      class="send-btn"
      class:stop={chat.streaming}
      onclick={chat.streaming ? stopGen : () => sendMessage()}
      disabled={!chat.streaming && !model.loaded && !dual.enabled}
    >
      {chat.streaming ? "■" : "▶"}
    </button>
  </div>

  <div class="hints">
    {#if chat.streaming}
      <span class="hint amber">
        ● {chat.prefillDone && chat.prefillTotal
          ? `prefill ${chat.prefillDone}/${chat.prefillTotal}`
          : chat.streamBuffer
          ? "generating"
          : "waiting…"}
      </span>
    {:else}
      <span class="hint">
        <label class="art-toggle">
          <input type="checkbox" bind:checked={chat.artifactMode} />
          Artifact mode
        </label>
      </span>
    {/if}
    {#if ctxLimit > 0 && !chat.streaming}
      <span class="ctx-counter" class:ctx-warn={ctxPct > 0.75} class:ctx-crit={ctxPct > 0.9}>
        ~{approxTokens.toLocaleString()} / {ctxLimit.toLocaleString()} ctx
      </span>
    {/if}
    <span class="hint-actions">
      {#if ctxPct > 0.75 && !chat.streaming}
        <button class="btn-trim" onclick={trimContext} title="Drop oldest messages to free context">Trim</button>
      {/if}
      <button onclick={() => (chat.messages = [])} title="Clear chat (Ctrl+K)">Clear</button>
      <button onclick={saveSession}>Save</button>
      <button onclick={loadSession}>Load</button>
    </span>
  </div>
</div>

<style>
  .input-area {
    border-top: 1px solid var(--border);
    padding: 10px 14px 12px;
    background: var(--bg2);
    flex-shrink: 0;
  }
  .prefill-bar {
    height: 2px;
    background: var(--border);
    border-radius: 2px;
    margin-bottom: 8px;
    overflow: hidden;
  }
  .prefill-fill {
    height: 100%;
    background: var(--accent);
    border-radius: 2px;
    transition: width 0.3s ease;
  }
  .input-row { display: flex; gap: 8px; align-items: flex-end; }
  .chat-input {
    flex: 1;
    resize: none;
    min-height: 36px;
    max-height: 160px;
    font-size: 13px;
    font-family: var(--sans);
    line-height: 1.5;
    padding: 8px 12px;
    border-radius: var(--radius);
  }
  .send-btn {
    width: 36px; height: 36px;
    font-size: 16px;
    border-radius: var(--radius);
    padding: 0;
    flex-shrink: 0;
    display: flex; align-items: center; justify-content: center;
    background: rgba(108,142,245,0.12);
    border-color: rgba(108,142,245,0.4);
    color: var(--accent);
  }
  .send-btn:hover { background: rgba(108,142,245,0.25); border-color: var(--accent); }
  .send-btn.stop { background: rgba(248,113,113,0.12); border-color: rgba(248,113,113,0.4); color: var(--red); }
  .hints {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-top: 6px;
    font-size: 11px;
    color: var(--text3);
  }
  .hint { display: flex; align-items: center; gap: 4px; }
  .hint.amber { color: var(--amber); }
  .hint-actions { display: flex; gap: 6px; }
  .hint-actions button { font-size: 10px; padding: 2px 7px; }
  .art-toggle { display: flex; align-items: center; gap: 5px; cursor: pointer; color: var(--text3); }
  .art-toggle input { accent-color: var(--accent); }
  .ctx-counter { font-size: 10px; color: var(--text3); font-variant-numeric: tabular-nums; }
  .ctx-counter.ctx-warn { color: var(--amber); }
  .ctx-counter.ctx-crit { color: var(--red); }
  .btn-trim { font-size: 10px; padding: 2px 7px; color: var(--amber); border-color: rgba(251,191,36,0.4); }
</style>
