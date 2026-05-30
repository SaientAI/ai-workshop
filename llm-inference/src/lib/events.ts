// Tauri event listeners — wires backend events into reactive state.
// Call once at app startup.

import { listen } from "@tauri-apps/api/event";
import { model, chat, dual, agent, tts, lora, merge, params } from "./state.svelte.js";
import { splitArtifact } from "./artifact.js";
import { agentRun, checkGoalCompletion } from "./tauri.js";
import type { LoadPhase } from "./types.js";

const LOAD_PHASE_LABELS: Record<string, string> = {
  stopping: "⬛ Stopping server",
  freeing_vram: "🧹 Freeing VRAM",
  launching: "🚀 Launching tinyq4",
  health_check: "⏳ Waiting for health",
  ready: "✓ Ready",
};

export async function setupEvents() {
  // ── Model load phases ──────────────────────────────────────────────────────
  await listen("model-loading", (e) => {
    model.loading = true;
    model.loadStatus = (e.payload as string) || "Loading…";
  });

  await listen<{ step: string; detail: string }>("load-phase", (e) => {
    const { step, detail } = e.payload;
    const label = LOAD_PHASE_LABELS[step] ?? detail;
    model.loadPhases = model.loadPhases.filter((p: LoadPhase) => p.step !== step);
    model.loadPhases.push({ step, label, done: step === "ready", ts: Date.now() });
    model.loadStatus = label;
  });

  await listen("model-loaded", (e) => {
    model.loaded = true;
    model.loading = false;
    model.summary = e.payload as typeof model.summary;
    model.loadPhases = [];
    // Auto-set max tokens to half the context window on model load
    if (model.summary?.context_length) {
      params.maxTokens = Math.min(Math.floor(model.summary.context_length / 2), 8192);
    }
  });

  // ── Chat streaming ─────────────────────────────────────────────────────────

  // Watchdog: if no token arrives for 30 s the model or tinyq4 likely crashed.
  // Force-close the streaming state so the UI doesn't hang indefinitely.
  const STREAM_TIMEOUT_MS = 30_000;
  let streamWatchdog: ReturnType<typeof setTimeout> | null = null;
  function kickWatchdog() {
    if (streamWatchdog) clearTimeout(streamWatchdog);
    streamWatchdog = setTimeout(() => {
      if (!chat.streaming) return;
      chat.streaming = false;
      chat.pendingUserText = "";
      const last = chat.messages[chat.messages.length - 1];
      if (last?.streaming) {
        last.streaming = false;
        last.ts = Date.now();
        last.content = (last.content || "") +
          "\n\n*[No response from model for 30 s — connection may have dropped.]*";
        last.error = true;
      }
    }, STREAM_TIMEOUT_MS);
  }
  function clearWatchdog() {
    if (streamWatchdog) { clearTimeout(streamWatchdog); streamWatchdog = null; }
  }

  // ── Token batching ─────────────────────────────────────────────────────────
  // Accumulate tokens between animation frames so the DOM renders at most 60fps
  // instead of once per token (which can be 15–50+ updates/sec).
  let pendingTokens = "";
  let pendingReasoning = "";
  let batchScheduled = false;

  function flushBatch() {
    batchScheduled = false;
    const tokens = pendingTokens;
    const reasoning = pendingReasoning;
    pendingTokens = "";
    pendingReasoning = "";
    if (!tokens && !reasoning) return;

    const last = chat.messages[chat.messages.length - 1];
    if (!last) return;

    if (reasoning) {
      chat.reasoningBuffer += reasoning;
      last.reasoning = chat.reasoningBuffer;
    }

    if (tokens) {
      chat.streamBuffer += tokens;
      // Only parse for artifact changes; Message.svelte uses its own fast-path.
      const { artifact } = splitArtifact(chat.streamBuffer);
      if (artifact) {
        const now = Date.now();
        const updateContent =
          artifact.content.length > 200 && now - chat.lastArtifactPreview > 2000;
        if (updateContent) chat.lastArtifactPreview = now;
        chat.artifact = {
          active: true,
          ...artifact,
          content: updateContent
            ? artifact.content
            : chat.artifact.active
            ? chat.artifact.content
            : artifact.content,
        };
      }
      last.content = chat.streamBuffer;
    }
  }

  function scheduleBatch() {
    if (!batchScheduled) {
      batchScheduled = true;
      requestAnimationFrame(flushBatch);
    }
  }

  await listen("stream-start", () => {
    pendingTokens = "";
    pendingReasoning = "";
    batchScheduled = false;
    chat.streaming = true;
    chat.streamBuffer = "";
    chat.reasoningBuffer = "";
    chat.streamStart = Date.now();
    chat.lastArtifactPreview = 0;
    chat.prefillDone = 0;
    chat.prefillTotal = 0;
    chat.artifact = { active: false, title: "", type: "html", content: "", complete: false };
    chat.messages.push({
      role: "assistant",
      content: "",
      reasoning: "",
      ts: Date.now(),
      streaming: true,
      sourceUser: chat.pendingUserText,
      streamStart: chat.streamStart,
      prefillDone: 0,
      prefillTotal: 0,
    });
    kickWatchdog();
  });

  await listen<{ done: number; total: number }>("prefill-progress", (e) => {
    kickWatchdog();
    chat.prefillDone = e.payload.done;
    chat.prefillTotal = e.payload.total;
    const last = chat.messages[chat.messages.length - 1];
    if (last?.streaming) {
      last.prefillDone = e.payload.done;
      last.prefillTotal = e.payload.total;
    }
  });

  await listen<string>("stream-reasoning", (e) => {
    kickWatchdog();
    pendingReasoning += e.payload;
    scheduleBatch();
  });

  await listen<string>("stream-token", (e) => {
    kickWatchdog();
    pendingTokens += e.payload;
    scheduleBatch();
  });

  await listen("stream-done", (e) => {
    // Drain any buffered tokens before finalizing so nothing is lost.
    flushBatch();
    clearWatchdog();
    chat.streaming = false;
    chat.lastPerf = e.payload as typeof chat.lastPerf;
    chat.pendingUserText = "";
    const last = chat.messages[chat.messages.length - 1];
    if (last?.streaming) {
      last.streaming = false;
      last.ts = Date.now();
      last.content = chat.streamBuffer;
      last.reasoning = chat.reasoningBuffer;
      last.perf = chat.lastPerf ?? undefined;
      const { artifact } = splitArtifact(chat.streamBuffer);
      if (artifact) {
        chat.artifact = { active: true, ...artifact };
      }
    }
  });

  // ── Dual agent ─────────────────────────────────────────────────────────────
  await listen("drafter-loading", () => {
    dual.drafterLoading = true;
    dual.drafterError = "";
    dual.drafterBuffer = "";
  });
  await listen("drafter-loaded", (e) => {
    dual.drafterLoading = false;
    dual.drafterSummary = e.payload as typeof dual.drafterSummary;
  });
  await listen("critic-loading", () => {
    dual.criticLoading = true;
    dual.criticError = "";
  });
  await listen("critic-loaded", (e) => {
    dual.criticLoading = false;
    dual.criticSummary = e.payload as typeof dual.criticSummary;
  });
  await listen("dual-start", () => {
    chat.streaming = true;
    dual.drafterBuffer = "";
    dual.drafterReasoningBuffer = "";
    chat.streamBuffer = "";
    chat.reasoningBuffer = "";
    chat.streamStart = Date.now();
    chat.messages.push({
      role: "assistant",
      dual: true,
      phase: "drafting",
      drafterContent: "",
      drafterReasoning: "",
      content: "",
      reasoning: "",
      ts: Date.now(),
      streaming: true,
    });
  });
  await listen<string>("drafter-token", (e) => {
    dual.drafterBuffer += e.payload;
    const last = chat.messages[chat.messages.length - 1];
    if (last?.streaming) last.drafterContent = dual.drafterBuffer;
  });
  await listen<string>("drafter-reasoning", (e) => {
    dual.drafterReasoningBuffer += e.payload;
    const last = chat.messages[chat.messages.length - 1];
    if (last?.streaming) last.drafterReasoning = dual.drafterReasoningBuffer;
  });
  await listen("drafter-done", () => {
    const last = chat.messages[chat.messages.length - 1];
    if (last?.dual) last.phase = "critiquing";
  });
  await listen<string>("critic-token", (e) => {
    chat.streamBuffer += e.payload;
    const last = chat.messages[chat.messages.length - 1];
    if (last?.streaming) last.content = chat.streamBuffer;
  });
  await listen<string>("critic-reasoning", (e) => {
    chat.reasoningBuffer += e.payload;
    const last = chat.messages[chat.messages.length - 1];
    if (last?.streaming) last.reasoning = chat.reasoningBuffer;
  });
  await listen("dual-done", (e) => {
    chat.streaming = false;
    chat.lastPerf = e.payload as typeof chat.lastPerf;
    const last = chat.messages[chat.messages.length - 1];
    if (last?.streaming) {
      last.streaming = false;
      last.phase = "done";
      last.drafterContent = dual.drafterBuffer;
      last.drafterReasoning = dual.drafterReasoningBuffer;
      last.content = chat.streamBuffer;
      last.reasoning = chat.reasoningBuffer;
    }
  });

  // ── TTS ────────────────────────────────────────────────────────────────────
  await listen<number>("tts_progress", (e) => {
    tts.progress = e.payload;
  });

  // ── LoRA ───────────────────────────────────────────────────────────────────
  // Rust emits a single "lora-event" with { type, msg?, step?, total_steps?,
  // epoch?, total_epochs?, loss?, output? } — dispatch here by type.
  await listen<Record<string, unknown>>("lora-event", (e) => {
    const ev = e.payload;
    switch (ev.type) {
      case "log":
      case "warn":
        lora.log.push({ type: "out", text: String(ev.msg ?? "") });
        break;
      case "progress":
        lora.step        = Number(ev.step ?? 0);
        lora.totalSteps  = Number(ev.total_steps ?? 0);
        lora.epoch       = Number(ev.epoch ?? 0);
        lora.totalEpochs = Number(ev.total_epochs ?? 0);
        lora.loss        = ev.loss != null ? Number(ev.loss) : null;
        break;
      case "done":
        if (ev.output) {
          // Training done — has an output path
          lora.training   = false;
          lora.done       = true;
          lora.outputPath = String(ev.output);
          lora.log.push({ type: "out", text: `Saved: ${ev.output}` });
        } else {
          // Dataset cleaning done — no output path
          lora.cleaning = false;
        }
        break;
      case "error":
        lora.log.push({ type: "err", text: String(ev.msg ?? "") });
        lora.error    = String(ev.msg ?? "");
        lora.training = false;
        lora.cleaning = false;
        break;
    }
  });

  // ── Merge ──────────────────────────────────────────────────────────────────
  // Rust emits a single "merge-event" with { type, msg?, done?, total?, output? }.
  await listen<Record<string, unknown>>("merge-event", (e) => {
    const ev = e.payload;
    switch (ev.type) {
      case "log":
        merge.log.push({ type: "out", text: String(ev.msg ?? "") });
        break;
      case "progress":
        merge.progress = Number(ev.done ?? 0);
        merge.total    = Number(ev.total ?? 0);
        break;
      case "done":
        merge.running    = false;
        merge.done       = true;
        merge.outputPath = String(ev.output ?? "");
        merge.log.push({ type: "out", text: `Saved: ${ev.output}` });
        break;
      case "error":
        merge.error   = String(ev.msg ?? "");
        merge.running = false;
        break;
    }
  });

  // ── Agent exec ─────────────────────────────────────────────────────────────
  // exec-stdout / exec-stderr are routed to the xterm terminal in AgentScreen.svelte.
  // Only lifecycle events are handled here.
  await listen<string>("exec-start", (e) => {
    agent.lastExecId = e.payload;
  });
  await listen("exec-done", () => {
    agent.termRunning = false;
  });

  // ── Agent plan ─────────────────────────────────────────────────────────────
  await listen<string>("agent-planning", () => {
    agent.planRunning = true;
    agent.planPhase = "generating";
    agent.planJson = "";
    agent.plan = null;
  });
  await listen<string>("agent-plan-token", (e) => {
    agent.planJson += e.payload;
  });
  await listen<string>("agent-plan-ready", (e) => {
    agent.planJson = e.payload;
    agent.planPhase = "executing";
  });
  await listen("plan-start", (e) => {
    agent.plan = e.payload as typeof agent.plan;
  });
  await listen("plan-step-start", () => {});
  await listen("plan-step-done", (e) => {
    agent.plan = e.payload as typeof agent.plan;
  });
  await listen("plan-step-failed", () => {});
  await listen("plan-done", async (e) => {
    agent.plan = e.payload as typeof agent.plan;
    agent.planRunning = false;
    agent.planPhase = "idle";

    // ── Autonomous loop ─────────────────────────────────────────────────────
    if (!agent.autoMode) return;

    if (agent.autoIteration >= agent.autoMaxIter) {
      agent.autoMode = false;
      agent.autoStatus = `Reached max iterations (${agent.autoMaxIter}) — review progress in the Memory tab`;
      return;
    }

    // Ask the LLM: is the goal achieved?
    agent.autoStatus = "Evaluating progress…";
    try {
      const verdict = await checkGoalCompletion(agent.planGoal);
      if (verdict.complete) {
        agent.autoMode = false;
        agent.autoGoalDone = true;
        agent.autoStatus = verdict.reason;
      } else {
        agent.autoIteration += 1;
        agent.autoStatus = verdict.reason;
        // Re-run — episodic memory tells the LLM what's already been done
        await agentRun(agent.planGoal).catch((err: unknown) => {
          agent.termLines.push({ type: "err", text: `Auto-run failed: ${String(err)}` });
          agent.autoMode = false;
          agent.autoStatus = "Run error — autonomous mode stopped";
        });
      }
    } catch (err) {
      agent.autoStatus = `Evaluation failed: ${String(err)}`;
      agent.autoMode = false;
    }
  });
}
