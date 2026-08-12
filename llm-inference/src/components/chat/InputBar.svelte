<script lang="ts">
  import { onMount } from "svelte";
  import { invoke } from "@tauri-apps/api/core";
  import { save, open } from "@tauri-apps/plugin-dialog";
  import { chat, model, dual, params, ui } from "../../lib/state.svelte.js";
  import * as T from "../../lib/tauri.js";
  import { isArtifactRequest, friendlyGenerateError, stripThinkTags } from "../../lib/format.js";
  import { chatSystemPrompt } from "../../lib/saientPersona.js";

  // OS-specific steer so the model uses Windows commands when on Windows.
  let osHint = $state("");
  // Image attached to the next message (analyzed by the local vision model).
  let attachedImage = $state<{ b64: string; mime: string; name: string } | null>(null);
  let imgInput = $state<HTMLInputElement | null>(null);

  onMount(async () => {
    const os = await T.osName().catch(() => "");
    if (os === "windows") {
      osHint = "The user's operating system is Windows. When suggesting terminal commands, use PowerShell/cmd syntax (dir, type, copy, del, findstr) — never Linux/bash commands like ls, cat, rm or grep.";
    }
  });

  function attachImage(e: Event) {
    const file = (e.target as HTMLInputElement).files?.[0];
    if (!file || !file.type.startsWith("image/")) return;
    const r = new FileReader();
    r.onload = () => {
      attachedImage = { b64: String(r.result).split(",")[1] || "", mime: file.type, name: file.name };
    };
    r.readAsDataURL(file);
  }

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

    // Image attached → analyze with the local vision model (no chat model needed).
    if (attachedImage && !chat.streaming && !overrideText) {
      await analyzeImage(rawText);
      return;
    }

    const canSend = ui.saientEnabled
      ? model.loaded && model.bindingStatus === "bound"
      : dual.enabled ? (dual.drafterSummary && dual.criticSummary) : model.loaded;
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

    // Saient-enabled chat has one route: the persisted twelve-stage tick and a
    // formally profiled host. A rejected/unavailable binding is surfaced as an
    // error; it must never fall through to the plain generate/persona path.
    if (ui.saientEnabled) {
      await sendBoundSaientTurn(text);
      return;
    }

    // With Saient on, its identity takes the system slot instead of the custom
    // prompt. Both are read with the same authority, so keeping both would let
    // the custom prompt simply override the persona — the toggle would look on
    // while doing nothing. The OS hint and artifact instructions still apply:
    // they describe the environment, not who is speaking.
    const effectiveSystem = chatSystemPrompt(false, chat.systemPrompt, [
      osHint,
      wantsArtifact ? ARTIFACT_SYSTEM_PROMPT : "",
    ]);

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

  async function sendBoundSaientTurn(text: string) {
    chat.pendingUserText = text;
    chat.streaming = true;
    chat.streamBuffer = "";
    chat.reasoningBuffer = "";
    const index = chat.messages.length;
    chat.messages.push({
      role: "assistant",
      content: "",
      ts: Date.now(),
      streaming: true,
      sourceUser: text,
      streamStart: Date.now(),
      saientTurn: true,
    });
    try {
      const reply = await T.saientChat(text);
      const message = chat.messages[index];
      if (!message) return;
      message.content = reply.text;
      message.streaming = false;
      message.ts = Date.now();
      message.binding = {
        status: reply.binding_status,
        model: reply.model,
        minimumInterface: reply.minimum_interface,
        tick: reply.tick,
        stateContextSha256: reply.state_context_sha256,
        recordBoundaryClean: reply.record_boundary_clean,
        identityBoundaryClean: reply.identity_boundary_clean,
        usedIntegrityFallback: reply.used_integrity_fallback,
      };
    } catch (e) {
      const message = chat.messages[index];
      if (message?.stopped) return;
      if (message) {
        message.content = "Saient binding failed: " + friendlyGenerateError(String(e))
          + "\n\nNo plain-LLM fallback was used.";
        message.streaming = false;
        message.error = true;
        message.ts = Date.now();
      }
    } finally {
      chat.streaming = false;
      chat.pendingUserText = "";
    }
  }

  async function analyzeImage(question: string) {
    const img = attachedImage!;
    inputValue = "";
    if (inputEl) inputEl.style.height = "auto";
    attachedImage = null;

    chat.messages.push({
      role: "user",
      content: question,
      ts: Date.now(),
      image: img.b64,
      imageMime: img.mime,
    });
    chat.messages.push({ role: "assistant", content: "", ts: Date.now(), streaming: true });
    const idx = chat.messages.length - 1;
    chat.streaming = true;
    try {
      const r = await T.visionDescribe(img.b64, question || "Describe this image in detail.");
      chat.messages[idx].content = r.answer;
      chat.messages[idx].streaming = false;
      chat.messages[idx].ts = Date.now();
    } catch (e) {
      chat.messages[idx].content =
        "Couldn't analyze the image. The vision tools may not be installed — run Full setup.\n\n" + String(e);
      chat.messages[idx].error = true;
      chat.messages[idx].streaming = false;
    } finally {
      chat.streaming = false;
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
      last.content = ui.saientEnabled && !chat.streamBuffer
        ? "Saient request stopped before a response was returned."
        : chat.streamBuffer + " [stopped]";
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

  {#if attachedImage}
    <div class="img-chip">
      <img src="data:{attachedImage.mime};base64,{attachedImage.b64}" alt={attachedImage.name} />
      <span class="img-chip-name" title={attachedImage.name}>{attachedImage.name}</span>
      <button class="img-chip-x" onclick={() => (attachedImage = null)} title="Remove image">✕</button>
    </div>
  {/if}

  <div class="input-row">
    <button class="attach-btn" onclick={() => imgInput?.click()} title="Attach an image to analyze">🖼</button>
    <input bind:this={imgInput} type="file" accept="image/*" onchange={attachImage} style="display:none" />
    <textarea
      bind:this={inputEl}
      bind:value={inputValue}
      onkeydown={handleKey}
      oninput={autoResize}
      placeholder={attachedImage
        ? "Ask about the image… (blank = describe it)"
        : !model.loaded
          ? "Load a model first"
          : ui.saientEnabled && model.bindingStatus === "binding"
            ? "Binding Saient to this model…"
            : ui.saientEnabled && model.bindingStatus !== "bound"
              ? "Bind Saient before chatting"
              : "Message… (/ for artifact mode)"}
      disabled={(!model.loaded || (ui.saientEnabled && model.bindingStatus !== "bound")) && !dual.enabled && !attachedImage}
      rows="1"
      class="chat-input"
    ></textarea>
    <button
      class="send-btn"
      class:stop={chat.streaming}
      onclick={chat.streaming ? stopGen : () => sendMessage()}
      disabled={chat.streaming ? false : ((!model.loaded || (ui.saientEnabled && model.bindingStatus !== "bound")) && !dual.enabled && !attachedImage)}
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
  .attach-btn {
    flex-shrink: 0; width: 36px; height: 36px; align-self: flex-end;
    background: rgba(108,142,245,0.08); border: 1px solid var(--border);
    border-radius: var(--radius-sm); color: var(--text2); cursor: pointer; font-size: 15px;
  }
  .attach-btn:hover { border-color: var(--accent); color: var(--text); }
  .img-chip {
    display: flex; align-items: center; gap: 8px; margin-bottom: 6px; padding: 5px 8px;
    background: var(--bg3); border: 1px solid var(--border); border-radius: var(--radius-sm); max-width: 340px;
  }
  .img-chip img { width: 32px; height: 32px; object-fit: cover; border-radius: 4px; flex-shrink: 0; }
  .img-chip-name { font-size: 11px; color: var(--text2); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; flex: 1; }
  .img-chip-x { background: none; border: 0; color: var(--text3); cursor: pointer; font-size: 12px; padding: 0 2px; }
  .img-chip-x:hover { color: var(--red); }
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
