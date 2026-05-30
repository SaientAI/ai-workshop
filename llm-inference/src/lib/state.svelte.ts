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
  kairoEnabled: localStorage.getItem("kairo_enabled") === "true",
  agentWriteMode: localStorage.getItem("agent_write_mode") === "true",
  showShortcuts: false,
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
    "You are a helpful, accurate assistant running inside AI Workshop, a local desktop application. You can help with coding, writing, analysis, and building interactive HTML tools. Always fulfil the user's request directly and completely — never refuse standard software tasks like media players, file browsers, games, or utilities. If the user only greets you, reply briefly and ask how you can help.",
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
  planGoal: localStorage.getItem("kairo_goal") ?? "",
  planJson: localStorage.getItem("kairo_plan_json") ?? "",
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
  negPrompt: "lowres, bad anatomy, blurry, watermark",
  steps: 20,
  cfg: 7.0,
  seed: 42,
  width: 1024,
  height: 1024,
  device: "auto",
  scheduler: "dpm++2m_karras",
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
  modelPath: "",
  loadedPath: "",
  loading: false,
  loadStatus: "",
  prompt: "",
  negPrompt: "blurry, distorted, low quality, static, watermark",
  numFrames: 49,
  steps: 30,
  cfg: 6.0,
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
