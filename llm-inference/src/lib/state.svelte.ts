// Central reactive state — Svelte 5 runes.
// Import and mutate directly; Svelte tracks dependencies automatically.

import type {
  Screen, AgentTab, ChatTab, ChatMessage, SamplingParams, Artifact,
  LoadPhase, DepReport, ModelEntry, ModelSummary, RunningServer, PerfResult,
  IgState, TtsState, LoraState, MergeState, TreeEntry, TermLine, Plan, MemoryFact,
} from "./types.js";

// ── UI ────────────────────────────────────────────────────────────────────────

export const ui = $state({
  screen: "chat" as Screen,
  saientEnabled: localStorage.getItem("saient_enabled") === "true",
  agentWriteMode: localStorage.getItem("agent_write_mode") === "true",
  showShortcuts: false,
  showSettings: false,
  showSecurity: false,
  showUpdate: false,
});

// ── Toasts (transient tips / confirmations) ─────────────────────────────────────
export type ToastKind = "info" | "success" | "error";
export interface Toast { id: number; msg: string; kind: ToastKind }
export const toasts = $state<Toast[]>([]);
let _toastId = 0;

/** Show a transient toast. Auto-dismisses; errors linger a little longer. */
export function toast(msg: string, kind: ToastKind = "info", ms?: number) {
  const id = ++_toastId;
  toasts.push({ id, msg, kind });
  const ttl = ms ?? (kind === "error" ? 6000 : 3500);
  setTimeout(() => {
    const i = toasts.findIndex((t) => t.id === id);
    if (i >= 0) toasts.splice(i, 1);
  }, ttl);
}

export function dismissToast(id: number) {
  const i = toasts.findIndex((t) => t.id === id);
  if (i >= 0) toasts.splice(i, 1);
}

// ── Updates (best-effort version check against the site) ─────────────────────────
export const update = $state({
  checking: false,
  checked: false,
  available: false,
  current: "",
  latest: "",
  url: "https://saient.co.uk/#download",
  notes: "",
  error: "",
  dismissed: false,
});

// ── Licensing (30-day trial → £20 unlock) ───────────────────────────────────────
export const license = $state({
  status: "licensed" as "trial" | "expired" | "licensed",  // optimistic until checked
  daysLeft: 30,
  trialDays: 30,
  tier: null as string | null,
  showUnlock: false,   // paywall opened manually (vs. forced on expiry)
});

// ── Model / server ────────────────────────────────────────────────────────────

export const model = $state({
  loaded: false,
  loading: false,
  path: "",
  summary: null as ModelSummary | null,
  loadError: "",
  loadStatus: "",
  loadPhases: [] as LoadPhase[],
  activeServerPort: null as number | null,
  gpuLayers: -1,
  ctxSize: 4096,
  modelsDir: "",
  models: [] as ModelEntry[],
  selectedModel: null as ModelEntry | null,
  runningServers: [] as RunningServer[],
  runningServersScanned: false,
  depReport: null as DepReport | null,
  gpu: null as Record<string, unknown> | null,
});

// ── Chat ──────────────────────────────────────────────────────────────────────

export const chat = $state({
  tab: "chat" as ChatTab,
  messages: [] as ChatMessage[],
  systemPrompt:
    "You are a helpful, accurate assistant running inside Saient, a local desktop application. You can help with coding, writing, analysis, and building interactive HTML tools. Always fulfil the user's request directly and completely — never refuse standard software tasks like media players, file browsers, games, or utilities. If the user only greets you, reply briefly and ask how you can help.",
  artifactMode: true,
  artifact: { active: false, title: "", type: "html", content: "", complete: false } as Artifact,
  streaming: false,
  streamBuffer: "",
  reasoningBuffer: "",
  streamStart: 0,
  lastArtifactPreview: 0,
  prefillDone: 0,
  prefillTotal: 0,
  pendingUserText: "",
  lastPerf: null as PerfResult | null,
  pendingRetry: "",
});

// ── Dual agent ────────────────────────────────────────────────────────────────

export const dual = $state({
  enabled: false,
  drafterPath: "",
  criticPath: "",
  selectedDrafter: null as ModelEntry | null,
  selectedCritic: null as ModelEntry | null,
  drafterSummary: null as ModelSummary | null,
  criticSummary: null as ModelSummary | null,
  drafterLoading: false,
  criticLoading: false,
  drafterError: "",
  criticError: "",
  drafterBuffer: "",
  drafterReasoningBuffer: "",
});

// ── Sampling params ───────────────────────────────────────────────────────────

export const params = $state<SamplingParams>({
  maxTokens: 2048,
  temperature: 0.3,
  topP: 0.95,
  topK: 40,
  repeatPenalty: 1.1,
  seed: 42,
});

// ── Agent ─────────────────────────────────────────────────────────────────────

export const agent = $state({
  tab: "files" as AgentTab,
  sandboxRoot: "",
  tree: [] as TreeEntry[],
  selPath: null as string | null,
  content: "",
  dirty: false,
  termLines: [] as TermLine[],
  termRunning: false,
  lastExecId: "",
  termCmd: "ls",
  termArgs: "",
  termCwd: "",
  planGoal: localStorage.getItem("saient_goal") ?? "",
  planJson: localStorage.getItem("saient_plan_json") ?? "",
  plan: null as Plan | null,
  planRunning: false,
  planPhase: "idle" as "idle" | "generating" | "executing",
  memFacts: [] as MemoryFact[],
  memQuery: "",
  // Autonomous loop
  autoMode: false,
  autoIteration: 0,
  autoMaxIter: 5,
  autoStatus: "",
  autoGoalDone: false,
});

// ── Image gen ─────────────────────────────────────────────────────────────────

export const ig = $state<IgState>({
  models: [],
  checkpoints: [],
  loras: [],
  modelPath: "",
  loraPath: "",
  prompt: "",
  negPrompt: "lowres, bad anatomy, blurry, watermark, monochrome, grayscale, black and white, sepia, pencil sketch, line art, duplicate, two characters, multiple views, character sheet, color palette, swatches, text, cropped",
  steps: 20,
  cfg: 7.0,
  seed: 42,
  width: 1024,
  height: 1024,
  device: "auto",
  scheduler: "auto",
  faceDetail: true,   // ADetailer-style hi-res face pass (auto-skips when no small face)
  assetGuard: true,
  assetKind: "humanoid",
  generating: false,
  progress: 0,
  progressTotal: 20,
  resultB64: "",
  error: "",
  elapsed: 0,
  vramFreed: false,
});

// ── Video gen ───────────────────────────────────────────────────────────────

export const video = $state({
  models: [] as Array<{ path: string; label: string; pipeline: string }>,
  loras: [] as Array<{ path: string; label: string }>,
  loraPath: "",
  loraStrength: 1.0,
  loraProfile: "single" as "single" | "high_low",
  loraHighStrength: 2.2,
  loraLowStrength: 0.8,
  loraSplitStep: 4,
  qualityMode: false,    // false = fast 4-bit transformer (default); true = bf16 transformer
                         // streamed from RAM (higher fidelity, ~10 GB PCIe round-trip/gen)
  lowVramMode: false,
  blockOffload: false,   // park transformer to RAM, stream per-step → fits native 5s@720p (slow ~17min)
  imageB64: "",          // i2v input still (base64, no data: prefix)
  imageName: "",         // display name of the picked image
  resLocked: false,      // model requires a fixed resolution (e.g. CogVideoX)
  modelPath: "",
  loadedPath: "",
  loading: false,
  loadStatus: "",
  prompt: "",
  // Base quality + anatomy anti-confusion. Wan/SVI often maps vulva↔mouth/lips or
  // fuses sex organs; keep those negatives resident so every gen gets the guard.
  negPrompt: "blurry, distorted, low quality, static, watermark, bad anatomy, deformed hands, extra fingers, deformed genitals, fused genitals, ambiguous genitals, hermaphrodite, mouth between legs, lips instead of vagina, oral opening as genitals, teeth on crotch, face on genitals, penis on vagina, inverted genitals, malformed labia, anatomically incorrect genitals",
  // Storyboard for segmented long videos: separate prompts per ~5s chunk.
  // If filled, a storyboard generate can chain them with auto-extend/stitch.
  storyboardPrompts: ["", "", "", ""],
  // When true (default), explicit prompts get positive anatomy lock + merged neg guards.
  anatomyLock: true,
  numFrames: 49,
  steps: 30,
  cfg: 6.0,
  scheduler: "auto" as "auto" | "euler_beta",
  shift: 5,
  width: 832,
  height: 480,
  fps: 16,
  seed: -1,
  generating: false,
  progress: 0,
  progressTotal: 30,
  resultB64: "",
  frames: 0,
  elapsed: 0,
  error: "",
  log: [] as string[],   // rolling, read-only activity log shown in the sidebar
  // ── Quality pass (separate, unloaded step) ──
  enhancing: false,
  enhanced: false,           // current resultB64 is an enhanced result
  doRefine: true,
  doUpscale: true,
  doInterpolate: false,      // RIFE not wired yet
  doFace: false,             // CodeFormer face restoration — fixes melted faces on animated people
  refineStrength: 0.35,
  refineSteps: 20,
});

// ── Vision (local image understanding) ──────────────────────────────────────────

export const vision = $state({
  imageB64: "",        // base64 of the picked image (no data: prefix)
  imageMime: "image/png",
  imageName: "",       // display name of the picked file
  question: "",        // empty → general caption
  loaded: false,       // model warm in VRAM
  analyzing: false,
  answer: "",
  elapsed: 0,
  device: "",
  error: "",
});

// ── TTS ───────────────────────────────────────────────────────────────────────

export const tts = $state<TtsState>({
  voices: [],
  voice: "af_heart",
  speed: 1.0,
  text: "",
  generating: false,
  progress: 0,
  resultB64: "",
  duration: 0,
  error: "",
});

// ── LoRA ──────────────────────────────────────────────────────────────────────

export const lora = $state<LoraState>({
  modelPath: "",
  datasetDir: "",
  outputName: "my_lora",
  outputDir: "",
  rank: 16,
  alpha: 16,
  lr: 1e-4,
  epochs: 10,
  batchSize: 1,
  resolution: 1024,
  cleaning: false,
  training: false,
  step: 0,
  totalSteps: 0,
  epoch: 0,
  totalEpochs: 0,
  loss: null,
  log: [],
  error: "",
  outputPath: "",
  done: false,
});

// ── Merge ─────────────────────────────────────────────────────────────────────

export const merge = $state<MergeState>({
  modelA: "",
  modelB: "",
  modelC: "",
  method: "weighted_sum",
  weight: 0.5,
  outputDir: "",
  outputName: "merged_model",
  running: false,
  done: false,
  error: "",
  outputPath: "",
  log: [],
  progress: 0,
  total: 0,
});
