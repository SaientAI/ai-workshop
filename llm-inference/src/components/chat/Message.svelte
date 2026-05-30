<script lang="ts">
  import type { ChatMessage } from "../../lib/types.js";
  import { splitArtifact } from "../../lib/artifact.js";
  import { formatElapsed, isRefusal } from "../../lib/format.js";
  import { renderMarkdown } from "../../lib/markdown.js";
  import { chat } from "../../lib/state.svelte.js";

  let { msg }: { msg: ChatMessage } = $props();

  function stripThinkTags(t: string) {
    return t.replace(/<think>[\s\S]*?<\/think>/gi,"").replace(/<thinking>[\s\S]*?<\/thinking>/gi,"").trim();
  }

  function splitLeakedThinking(text: string): [string, string] {
    if (!text) return ["", ""];
    text = stripThinkTags(text);
    const answerStart = /(\n#{1,3} |\n\|[-|]+\||\n```|\*\*(?:A |An |The )[A-Z]|(?:^|\n)(?:Here(?:'s| is)|Let(?:'s| me give you|me provide)|Below (?:is|are)|The following|Sure[,!]|Of course))/;
    const m = answerStart.exec(text);
    if (!m || m.index < 20) return ["", text];
    const leaked = text.slice(0, m.index).trim();
    const answer = text.slice(m.index).trim();
    if (leaked.length < 20 || leaked.split("\n").length > 15) return ["", text];
    return [leaked, answer];
  }

  function splitReasoningAnswer(text: string): [string, string] {
    if (!text) return ["", ""];
    const transitions = [
      /\n+(?:Final answer|My answer|Answer):\s*/i,
      /\n+(?:So(?:,| the answer is)|Therefore(?:,| the answer is))\s*/i,
    ];
    for (const re of transitions) {
      const m = re.exec(text);
      if (m && m.index > 30) return [text.slice(0, m.index).trim(), text.slice(m.index).replace(re, "").trim()];
    }
    return ["", text];
  }

  function cleanDisplayText(text: string) {
    return String(text ?? "").replace(/\s*\[stopped\]\s*$/i, "").trim();
  }

  const time = $derived(new Date(msg.ts).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }));
  const meta = $derived(
    msg.perf
      ? `${time} · ${msg.perf.tokens_generated} tok · ${msg.perf.tokens_per_sec.toFixed(1)} tok/s`
      : time
  );

  // Pre-compute body parts reactively
  const body = $derived.by(() => {
    if (msg.role === "user") return null; // rendered as plain text
    if (msg.dual) return "dual";

    if (msg.streaming && !msg.content && !msg.reasoning) return "generating";

    // Fast path while streaming: skip splitArtifact, renderMarkdown, and all
    // regex-heavy parsing. Show plain text — full processing runs once on done.
    if (msg.streaming) {
      return {
        type: "streaming" as const,
        text: stripThinkTags(String(msg.content ?? "")),
        hasReasoning: !!msg.reasoning,
      };
    }

    const { chatText, artifact } = splitArtifact(msg.content ?? "");
    const visible = cleanDisplayText(chatText);

    if (msg.reasoning && !visible && !artifact) {
      const [trace, finalAnswer] = splitReasoningAnswer(msg.reasoning);
      return { type: "reasoning-only" as const, trace: trace || msg.reasoning, finalAnswer };
    }

    const [leaked, answer] = splitLeakedThinking(visible);
    const combinedReasoning = [msg.reasoning, leaked].filter(Boolean).join("\n\n---\n\n");
    return { type: "normal" as const, reasoning: combinedReasoning, answer, artifact };
  });

  const showRetry = $derived(
    !msg.streaming &&
    msg.role === "assistant" &&
    !!msg.sourceUser &&
    isRefusal(String(msg.content ?? ""))
  );

  // Elapsed timer for streaming messages
  let elapsed = $state("");
  $effect(() => {
    if (!msg.streaming) { elapsed = ""; return; }
    const start = msg.streamStart ?? Date.now();
    const id = setInterval(() => { elapsed = formatElapsed(Date.now() - start); }, 1000);
    return () => clearInterval(id);
  });
</script>

<div class="msg {msg.role}" class:error={msg.error}>
  <div class="avatar">{msg.role === "user" ? "U" : "AI"}</div>
  <div class="body">
    <div class="role">{msg.role}</div>
    <div class="content">
      {#if msg.role === "user"}
        <div class="user-text">{msg.content}</div>

      {:else if body === "generating"}
        <div class="gen-status">
          <div class="gen-head">
            <span class="gen-dot"></span>
            <span>
              {msg.prefillTotal ? "Processing prompt" : "Request sent"}
            </span>
            <span class="gen-elapsed">{elapsed}</span>
          </div>
          {#if msg.prefillTotal}
            <div class="gen-detail">{msg.prefillDone} / {msg.prefillTotal} prompt tokens</div>
            <div class="bar-track">
              <div class="bar-fill" style="width:{Math.max(2, Math.min(100, Math.round((msg.prefillDone ?? 0) / msg.prefillTotal * 100)))}%"></div>
            </div>
          {:else}
            <div class="gen-detail">Waiting for first token…</div>
            <div class="bar-track pending"><div class="bar-fill" style="width:38%"></div></div>
          {/if}
        </div>

      {:else if msg.dual}
        <div class="agent-slot">
          <div class="agent-label a">◈ Drafter</div>
          {#if msg.drafterReasoning}
            <details class="reasoning">
              <summary>💭 Thinking</summary>
              <div class="reasoning-body">{msg.drafterReasoning}</div>
            </details>
          {/if}
          {#if msg.drafterContent}
            <div class="answer">{@html renderMarkdown(msg.drafterContent)}</div>
          {/if}
        </div>
        {#if msg.phase === "critiquing" || msg.phase === "done"}
          <div class="agent-divider">
            <span class="divider-line"></span>
            <span class="agent-label b">◈ Critic{msg.phase === "critiquing" ? " reviewing…" : ""}</span>
            <span class="divider-line"></span>
          </div>
          <div class="agent-slot">
            {#if msg.reasoning}
              <details class="reasoning">
                <summary>💭 Thinking</summary>
                <div class="reasoning-body">{msg.reasoning}</div>
              </details>
            {/if}
            {#if msg.content}
              <div class="answer">{@html renderMarkdown(msg.content)}</div>
            {/if}
          </div>
        {/if}

      {:else if body && body !== "dual"}
        {#if body.type === "streaming"}
          {#if body.hasReasoning}
            <details class="reasoning" open>
              <summary>💭 Thinking…</summary>
              <div class="reasoning-body">{msg.reasoning}</div>
            </details>
          {/if}
          {#if body.text}
            <div class="answer streaming-plain">{body.text}</div>
          {/if}

        {:else if body.type === "reasoning-only"}
          <details class="reasoning">
            <summary>💭 Thinking <span class="tok-count">{body.trace.split(/\s+/).filter(Boolean).length} words</span></summary>
            <div class="reasoning-body">{body.trace}</div>
          </details>
          {#if body.finalAnswer}
            <div class="answer">{@html renderMarkdown(body.finalAnswer)}</div>
          {:else}
            <div class="answer muted">No final answer returned.</div>
          {/if}
        {:else if body.type === "normal"}
          {#if body.reasoning}
            <details class="reasoning">
              <summary>💭 Thinking <span class="tok-count">{body.reasoning.split(/\s+/).filter(Boolean).length} words</span></summary>
              <div class="reasoning-body">{body.reasoning}</div>
            </details>
          {/if}
          {#if body.answer}
            <div class="answer">{@html renderMarkdown(body.answer)}</div>
          {/if}
          {#if body.artifact}
            <div class="artifact-ref">
              <span class="artifact-badge">{body.artifact.type.toUpperCase()}</span>
              {body.artifact.title}
              {#if !body.artifact.complete}<span class="streaming-tag">streaming…</span>{/if}
            </div>
          {/if}
        {/if}
      {/if}

      {#if msg.streaming}<span class="cursor"></span>{/if}
    </div>
    <div class="meta">
      {meta}
      {#if showRetry}
        <button
          class="retry-btn"
          onclick={() => { chat.pendingRetry = msg.sourceUser ?? ""; }}
          title="Resend your exact message — no changes made to it"
        >↻ Retry</button>
      {/if}
    </div>
  </div>
</div>

<style>
  .msg {
    display: flex;
    gap: 12px;
    padding: 14px 16px;
    border-bottom: 1px solid var(--border);
  }
  .msg.user { background: var(--bg3); }
  .msg.error { background: rgba(248,113,113,0.05); }
  .avatar {
    width: 28px; height: 28px;
    border-radius: 50%;
    background: var(--bg3);
    border: 1px solid var(--border);
    font-size: 11px;
    font-weight: 700;
    color: var(--text2);
    display: flex; align-items: center; justify-content: center;
    flex-shrink: 0;
  }
  .msg.user .avatar { background: rgba(108,142,245,0.15); border-color: rgba(108,142,245,0.3); color: var(--accent); }
  .body { flex: 1; min-width: 0; }
  .role { font-size: 10px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.06em; color: var(--text3); margin-bottom: 6px; }
  .content { line-height: 1.6; }
  .user-text { color: var(--text); font-size: 13px; white-space: pre-wrap; word-break: break-word; }
  .answer { font-size: 13px; color: var(--text); }
  .answer :global(code) { font-family: var(--mono); font-size: 12px; background: var(--bg3); padding: 1px 4px; border-radius: 3px; }
  .answer :global(pre) { background: var(--bg3); border: 1px solid var(--border); border-radius: var(--radius-sm); padding: 10px 12px; margin: 8px 0; overflow-x: auto; font-size: 12px; font-family: var(--mono); }
  .answer :global(pre code) { background: none; padding: 0; }
  .answer :global(strong) { color: var(--text); font-weight: 600; }
  .answer.muted { color: var(--text3); font-size: 12px; }
  .answer.streaming-plain { white-space: pre-wrap; word-break: break-word; }
  .meta { font-size: 10px; color: var(--text3); margin-top: 8px; font-family: var(--mono); display: flex; align-items: center; gap: 10px; }
  .retry-btn { font-size: 10px; padding: 2px 8px; color: var(--amber); border-color: rgba(251,191,36,0.4); background: rgba(251,191,36,0.07); border-radius: var(--radius-sm); cursor: pointer; font-family: var(--sans); }
  .cursor { display: inline-block; width: 2px; height: 14px; background: var(--accent); margin-left: 2px; animation: blink 1s infinite; vertical-align: middle; }
  @keyframes blink { 0%,100% { opacity:1 } 50% { opacity:0 } }

  /* Reasoning block */
  .reasoning { margin-bottom: 8px; }
  .reasoning summary {
    font-size: 11px; color: var(--text3); cursor: pointer;
    display: flex; align-items: center; gap: 6px;
    padding: 4px 0;
  }
  .reasoning summary:hover { color: var(--text2); }
  .tok-count { font-family: var(--mono); font-size: 10px; }
  .reasoning-body { font-size: 12px; color: var(--text3); padding: 8px 12px; border-left: 2px solid var(--border); margin-top: 6px; white-space: pre-wrap; font-family: var(--mono); line-height: 1.5; }

  /* Generation status */
  .gen-status { padding: 10px 12px; background: var(--bg3); border: 1px solid var(--border); border-radius: var(--radius-sm); }
  .gen-head { display: flex; align-items: center; gap: 8px; font-size: 12px; margin-bottom: 6px; }
  .gen-dot { width: 6px; height: 6px; border-radius: 50%; background: var(--amber); animation: pulse 1.2s infinite; }
  @keyframes pulse { 0%,100% { opacity:1 } 50% { opacity:0.3 } }
  .gen-elapsed { margin-left: auto; font-family: var(--mono); font-size: 11px; color: var(--text3); }
  .gen-detail { font-size: 11px; color: var(--text3); font-family: var(--mono); margin-bottom: 6px; }
  .bar-track { height: 3px; background: var(--border); border-radius: 2px; overflow: hidden; }
  .bar-fill { height: 100%; background: var(--accent); border-radius: 2px; transition: width 0.5s ease; }
  .bar-track.pending .bar-fill { animation: shimmer 1.5s infinite; }
  @keyframes shimmer { 0% { transform: translateX(-100%) } 100% { transform: translateX(300%) } }

  /* Dual agent */
  .agent-slot { margin-bottom: 8px; }
  .agent-label { font-size: 10px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 6px; }
  .agent-label.a { color: var(--accent); }
  .agent-label.b { color: var(--amber); }
  .agent-divider { display: flex; align-items: center; gap: 8px; margin: 12px 0; }
  .divider-line { flex: 1; height: 1px; background: var(--border); }

  /* Artifact reference inline */
  .artifact-ref { display: flex; align-items: center; gap: 8px; font-size: 11px; color: var(--text3); margin-top: 8px; padding: 6px 10px; background: var(--bg3); border: 1px solid var(--border); border-radius: var(--radius-sm); }
  .artifact-badge { background: rgba(108,142,245,0.15); color: var(--accent); padding: 1px 6px; border-radius: 3px; font-size: 10px; font-weight: 700; }
  .streaming-tag { color: var(--amber); font-family: var(--mono); }

  /* ── Markdown prose styles ──────────────────────────────────────────────── */
  .answer :global(p)          { margin: 0 0 10px; }
  .answer :global(p:last-child) { margin-bottom: 0; }
  .answer :global(h1), .answer :global(h2), .answer :global(h3),
  .answer :global(h4), .answer :global(h5), .answer :global(h6) {
    color: var(--text); font-weight: 700; margin: 16px 0 6px;
    line-height: 1.3;
  }
  .answer :global(h1) { font-size: 16px; }
  .answer :global(h2) { font-size: 14px; }
  .answer :global(h3) { font-size: 13px; }
  .answer :global(h4), .answer :global(h5), .answer :global(h6) { font-size: 12px; }
  .answer :global(ul), .answer :global(ol) {
    margin: 6px 0 10px 18px; padding: 0;
  }
  .answer :global(li) { margin-bottom: 4px; }
  .answer :global(li > p) { margin: 0; }
  .answer :global(blockquote) {
    border-left: 3px solid var(--border); margin: 8px 0; padding: 4px 12px;
    color: var(--text2); font-style: italic;
  }
  .answer :global(hr) { border: none; border-top: 1px solid var(--border); margin: 12px 0; }
  .answer :global(a) { color: var(--accent); text-decoration: underline; }
  .answer :global(table) { border-collapse: collapse; margin: 10px 0; font-size: 12px; width: 100%; }
  .answer :global(th), .answer :global(td) {
    border: 1px solid var(--border); padding: 5px 10px; text-align: left;
  }
  .answer :global(th) { background: var(--bg3); font-weight: 600; color: var(--text2); }
  .answer :global(tr:nth-child(even)) { background: var(--bg3); }

  /* ── Code block wrapper ──────────────────────────────────────────────────── */
  .answer :global(.code-block) {
    position: relative; margin: 8px 0;
    border: 1px solid var(--border); border-radius: var(--radius-sm); overflow: hidden;
  }
  .answer :global(.code-lang) {
    display: block; font-size: 10px; font-family: var(--mono);
    color: var(--text3); background: var(--bg3);
    padding: 3px 10px; border-bottom: 1px solid var(--border);
    text-transform: lowercase; letter-spacing: 0.04em;
  }
  .answer :global(.code-block pre) { margin: 0; border: none; border-radius: 0; }

  /* ── Highlight.js dark theme (minimal, matches app palette) ─────────────── */
  .answer :global(.hljs) {
    background: #0d1117; color: #c9d1d9;
    padding: 10px 12px; font-size: 12px; font-family: var(--mono);
    line-height: 1.55; overflow-x: auto; display: block;
  }
  .answer :global(.hljs-keyword),
  .answer :global(.hljs-operator)          { color: #ff7b72; }
  .answer :global(.hljs-string),
  .answer :global(.hljs-template-string),
  .answer :global(.hljs-regexp)            { color: #a5d6ff; }
  .answer :global(.hljs-comment),
  .answer :global(.hljs-quote)             { color: #6e7681; font-style: italic; }
  .answer :global(.hljs-number),
  .answer :global(.hljs-literal)           { color: #79c0ff; }
  .answer :global(.hljs-type),
  .answer :global(.hljs-class .hljs-title),
  .answer :global(.hljs-title.class_)     { color: #ffa657; }
  .answer :global(.hljs-function),
  .answer :global(.hljs-title.function_),
  .answer :global(.hljs-title)            { color: #d2a8ff; }
  .answer :global(.hljs-variable),
  .answer :global(.hljs-name)             { color: #c9d1d9; }
  .answer :global(.hljs-built_in),
  .answer :global(.hljs-builtin-name)     { color: #ffa657; }
  .answer :global(.hljs-attr),
  .answer :global(.hljs-attribute)        { color: #79c0ff; }
  .answer :global(.hljs-params)           { color: #c9d1d9; }
  .answer :global(.hljs-meta),
  .answer :global(.hljs-meta .hljs-keyword) { color: #f2cc60; }
  .answer :global(.hljs-tag),
  .answer :global(.hljs-name),
  .answer :global(.hljs-selector-tag)     { color: #7ee787; }
  .answer :global(.hljs-deletion)         { color: #ffa198; background: #490202; }
  .answer :global(.hljs-addition)         { color: #56d364; background: #0f5323; }
  .answer :global(.hljs-emphasis)         { font-style: italic; }
  .answer :global(.hljs-strong)           { font-weight: bold; }
</style>
