// Tauri event listeners — wires backend events into reactive state.
// Call once at app startup.

import { listen } from "@tauri-apps/api/event";
import { model, chat, dual, agent, tts, lora, merge, params, toast, video, pulse } from "./state.svelte.js";
import { splitArtifact } from "./artifact.js";
import { agentRun, checkGoalCompletion } from "./tauri.js";
import type { LoadPhase } from "./types.js";
import { animationFor, activityLine, pushActivity } from "./pulse.js";
import { activityText } from "./turnState.js";

/**
 * Record a change of activity: reset the elapsed clock, point the robot at the
 * new work, and append a timeline entry.
 *
 * Called from the event handlers rather than derived on a timer, so the bar can
 * only ever show work that actually started. `text` overrides the derived line
 * for things the turn state alone cannot name.
 */
function beginActivity(
  step: { tool?: string; target?: string | null } | null,
  text?: string,
  detail?: string,
) {
  pulse.step = step;
  pulse.startedAt = Date.now();
  const line = text ?? activityLine(agent.turn, step ?? undefined, activityText(agent.turn));
  pushActivity(pulse.log, {
    at: pulse.startedAt,
    text: line,
    animation: animationFor(agent.turn, step ?? undefined),
    detail: detail?.trim() || undefined,
  });
}

const LOAD_PHASE_LABELS: Record<string, string> = {
  stopping: "⬛ Stopping server",
  freeing_vram: "🧹 Freeing VRAM",
  launching: "🚀 Launching tinyq4",
  health_check: "⏳ Waiting for health",
  ready: "✓ Ready",
};

type VideoProgressEvent = {
  step: number;
  total: number;
  step_seconds?: number;
  elapsed_seconds?: number;
};

type VideoPreviewEvent = {
  base64_jpeg: string;
  step: number;
  total: number;
  frames: number[];
  decode_seconds?: number;
};

function appendVideoLog(message: unknown) {
  const line = String(message ?? "").trim();
  if (!line) return;
  const last = video.log[video.log.length - 1];
  if (last?.endsWith(line)) return;
  const time = new Date().toLocaleTimeString([], { hour12: false });
  video.log = [...video.log, `${time}  ${line}`].slice(-300);
}

export async function setupEvents() {
  await listen<string>("vidload-progress", (e) => {
    video.loadStatus = e.payload || "Working…";
    appendVideoLog(e.payload);
  });

  await listen<VideoProgressEvent>("video_progress", (e) => {
    video.generating = true;
    video.progress = e.payload.step;
    video.progressTotal = e.payload.total;
    video.loadStatus = "";
    const timing = typeof e.payload.step_seconds === "number"
      ? ` · ${e.payload.step_seconds.toFixed(1)}s step · ${Math.round(e.payload.elapsed_seconds ?? 0)}s elapsed`
      : "";
    appendVideoLog(`step ${e.payload.step}/${e.payload.total}${timing}`);
  });

  await listen<VideoPreviewEvent>("video-preview", (e) => {
    video.previewB64 = e.payload.base64_jpeg;
    video.previewStep = e.payload.step;
    video.previewFrames = e.payload.frames;
    const timing = typeof e.payload.decode_seconds === "number"
      ? ` · ${e.payload.decode_seconds.toFixed(2)}s decode`
      : "";
    appendVideoLog(`preview · step ${e.payload.step}/${e.payload.total} · ${e.payload.frames.length} frames${timing}`);
  });

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

  await listen<{ sample?: number; total?: number; rung?: string }>("saient-binding-progress", (e) => {
    if (model.bindingStatus !== "binding") return;
    model.bindingSample = Number(e.payload.sample ?? model.bindingSample);
    model.bindingRung = String(e.payload.rung ?? model.bindingRung);
  });

  // ── Chat streaming ─────────────────────────────────────────────────────────

  // Watchdog: declare the stream dead only after a long silence. CPU inference is
  // much slower (the first token after prefill can take a while), so the timeout
  // is adaptive — generous when there's no GPU — to avoid false "dropped" alarms.
  const STREAM_TIMEOUT_GPU_MS = 45_000;
  const STREAM_TIMEOUT_CPU_MS = 240_000;
  // A gentle, time-based "still working" nudge — fires only when a response is
  // actually slow, so fast machines never see it (no blanket "you're on CPU" nag).
  const SLOW_HINT_MS = 18_000;

  const onCpu = () => (model.gpu as { available?: boolean } | null)?.available === false;
  const watchdogMs = () => (onCpu() ? STREAM_TIMEOUT_CPU_MS : STREAM_TIMEOUT_GPU_MS);

  let streamWatchdog: ReturnType<typeof setTimeout> | null = null;
  let slowHint: ReturnType<typeof setTimeout> | null = null;
  let gotFirstToken = false;

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
        last.content = (last.content || "") + (onCpu()
          ? "\n\n*[The model went quiet for a while. On CPU this can just be slow — try again, or load a smaller / more-quantized model.]*"
          : "\n\n*[No response from the model — it may have crashed. Try reloading the model.]*");
        last.error = true;
      }
    }, watchdogMs());
  }
  function clearWatchdog() {
    if (streamWatchdog) { clearTimeout(streamWatchdog); streamWatchdog = null; }
    if (slowHint) { clearTimeout(slowHint); slowHint = null; }
  }
  // Cancel the slow-hint timer the moment real output starts.
  function sawOutput() {
    if (!gotFirstToken) {
      gotFirstToken = true;
      if (slowHint) { clearTimeout(slowHint); slowHint = null; }
    }
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
    gotFirstToken = false;
    if (slowHint) clearTimeout(slowHint);
    slowHint = setTimeout(() => {
      if (chat.streaming && !gotFirstToken) {
        toast("Still generating — the first response can take a moment on CPU.", "info", 5000);
      }
    }, SLOW_HINT_MS);
    kickWatchdog();
  });

  await listen<{ done: number; total: number }>("prefill-progress", (e) => {
    kickWatchdog();
    sawOutput();   // prefill bar is visible progress — no need for the slow hint
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
    sawOutput();
    pendingReasoning += e.payload;
    scheduleBatch();
  });

  await listen<string>("stream-token", (e) => {
    kickWatchdog();
    sawOutput();
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
  await listen<string>("agent-planning", (e) => {
    agent.planRunning = true;
    agent.planPhase = "generating";
    agent.turn = "SAIENT_THINKING";
    agent.retry = null;
    beginActivity(null, "Planning", String(e.payload ?? ""));
    agent.planJson = "";
    agent.plan = null;
    agent.planPrefill = null;
    agent.planReasoning = "";
    agent.planAbandoned = [];
  });
  await listen<{ done: number; total: number }>("agent-plan-prefill", (e) => {
    // Reading the prompt is the bulk of the wait on a local model. Without this
    // the panel sat blank until the first token, which reads as a hang.
    agent.planPrefill = e.payload;
  });
  await listen<string>("agent-plan-reasoning", (e) => {
    agent.planReasoning += e.payload;
  });
  await listen<string>("agent-plan-token", (e) => {
    agent.planPrefill = null;        // tokens are flowing; the prompt is read
    agent.planJson += e.payload;
  });
  await listen<string>("agent-plan-ready", (e) => {
    agent.planJson = e.payload;
    agent.planPrefill = null;
    agent.planPhase = "executing";
  });
  await listen<{ count: number; descriptions: string[] }>("plan-steps-abandoned", (e) => {
    agent.planAbandoned = e.payload.descriptions;
  });
  await listen("plan-start", (e) => {
    agent.plan = e.payload as typeof agent.plan;
    agent.turn = "SAIENT_ACTING";
  });
  await listen<{ tool: string; description: string; target?: string | null }>(
    "plan-step-start",
    (e) => {
      // Shell-outs are the slow, quiet ones — name them separately so a long build
      // doesn't look like a stall.
      agent.turn = e.payload.tool?.startsWith("exec") ? "WAITING_FOR_TOOL" : "SAIENT_ACTING";
      agent.retry = null;
      beginActivity(
        { tool: e.payload.tool, target: e.payload.target ?? null },
        undefined,
        e.payload.description,
      );
    },
  );
  await listen<{ passed: boolean; reason: string }>("plan-step-verify", (e) => {
    agent.turn = "VERIFYING";
    pulse.step = null;
    if (!e.payload.passed) {
      beginActivity(null, "Verification failed", e.payload.reason);
    }
  });
  await listen<{ step: number; total: number; reason: string }>("plan-step-retry", (e) => {
    agent.turn = "RETRYING";
    agent.retry = e.payload;
    beginActivity(
      null,
      `Retrying step ${e.payload.step} of ${e.payload.total}`,
      e.payload.reason,
    );
  });
  await listen("plan-step-done", (e) => {
    agent.plan = e.payload as typeof agent.plan;
  });
  await listen("plan-step-failed", () => {});
  await listen("plan-done", async (e) => {
    agent.plan = e.payload as typeof agent.plan;

    // One plan finished. That is NOT the same as Saient having stopped: when the
    // autonomous loop is on, a goal-completion inference and often a whole second
    // run still follow. This used to set planRunning=false and planPhase="idle"
    // right here, handing the keyboard back while all of that was still running.
    // Ownership is now released in exactly one place — settle() below.
    const planFailed = (agent.plan?.status ?? "").toLowerCase().includes("fail");

    /** Release the turn. The single exit from working states. */
    const settle = (state: "COMPLETED" | "FAILED" | "INTERRUPTED", why: string) => {
      agent.autoMode = false;
      agent.continuing = false;
      agent.planRunning = false;
      agent.planPhase = "idle";
      agent.turn = state;
      agent.retry = null;
      if (why) agent.autoStatus = why;
      // The robot stops with the work, not on a timer of its own.
      beginActivity(null, activityText(state), why);
    };

    if (!agent.autoMode) {
      settle(planFailed ? "FAILED" : "COMPLETED", "");
      return;
    }

    if (agent.autoIteration >= agent.autoMaxIter) {
      settle(
        "COMPLETED",
        `Reached max iterations (${agent.autoMaxIter}) — review progress in the Memory tab`,
      );
      return;
    }

    // A pause takes effect between iterations, so the current step is never cut
    // in half. Saient is stopping, so the turn does return to the user.
    if (agent.paused) {
      settle("INTERRUPTED", "Paused — press Auto to resume");
      agent.paused = false;
      return;
    }

    // Still working: evaluating counts as working, and the loop intends to run
    // again, so the input stays with Saient across the gap.
    agent.continuing = true;
    agent.turn = "VERIFYING";
    agent.autoStatus = "Evaluating progress…";
    try {
      const verdict = await checkGoalCompletion(agent.planGoal);
      if (verdict.complete) {
        agent.autoGoalDone = true;
        settle("COMPLETED", verdict.reason);
        return;
      }

      agent.autoIteration += 1;
      agent.autoStatus = verdict.reason;
      agent.turn = "SAIENT_THINKING";

      // Anything typed while Saient was working joins the next iteration rather
      // than being lost or silently ignored.
      let goal = agent.planGoal;
      if (agent.pendingInstructions.length) {
        goal = `${goal}\n\nAdditional instructions from the user:\n` +
          agent.pendingInstructions.map((s) => `- ${s}`).join("\n");
        agent.pendingInstructions = [];
      }

      // Re-run — episodic memory tells the LLM what's already been done
      await agentRun(goal).catch((err: unknown) => {
        agent.termLines.push({ type: "err", text: `Auto-run failed: ${String(err)}` });
        settle("FAILED", "Run error — autonomous mode stopped");
      });
    } catch (err) {
      settle("FAILED", `Evaluation failed: ${String(err)}`);
    }
  });
}
