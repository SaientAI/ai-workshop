// Shared TypeScript types for AI Workshop

export type Screen = "chat" | "agent" | "imggen" | "tts" | "lora" | "merge";
export type AgentTab = "files" | "terminal" | "planner" | "memory";
export type ChatTab = "chat" | "system";
export type StepStatus = "pending" | "running" | "done" | "failed" | "skipped" | "retrying";

export interface ModelSummary {
  name: string;
  architecture: string;
  quant: string;
  block_count: number;
  context_length: number;
  size_gb?: number;
  gguf_path?: string;
}

export interface ModelEntry {
  name: string;
  gguf_path: string;
  size_gb: number;
}

export interface RunningServer {
  port: number;
  model_id: string;
  pid?: number;
  model_path?: string;
  source?: string;
}

export interface PerfResult {
  tokens_generated: number;
  tokens_per_sec: number;
  elapsed_ms: number;
  total_tokens?: number;
}

export interface Artifact {
  active: boolean;
  title: string;
  type: string;
  content: string;
  complete: boolean;
}

export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
  reasoning?: string;
  ts: number;
  streaming?: boolean;
  error?: boolean;
  stopped?: boolean;
  perf?: PerfResult;
  sourceUser?: string;
  streamStart?: number;
  prefillDone?: number;
  prefillTotal?: number;
  // dual-agent fields
  dual?: boolean;
  phase?: "drafting" | "critiquing" | "done";
  drafterContent?: string;
  drafterReasoning?: string;
}

export interface SamplingParams {
  maxTokens: number;
  temperature: number;
  topP: number;
  topK: number;
  repeatPenalty: number;
  seed: number;
}

export interface LoadPhase {
  step: string;
  label: string;
  done: boolean;
  ts: number;
}

export interface DepStatus {
  name: string;
  ok: boolean;
  detail: string;
}

export interface DepReport {
  deps: DepStatus[];
  all_ok: boolean;
}

// Agent types
export interface FileEntry {
  path: string;
  name: string;
  is_dir: boolean;
  size: number;
  modified: string;
  extension: string;
}

export interface TreeEntry {
  path: string;
  name: string;
  is_dir: boolean;
  depth: number;
  children: TreeEntry[];
}

export interface TermLine {
  type: "out" | "err" | "cmd" | "info";
  text: string;
}

export interface PlanStep {
  id: string;
  index: number;
  description: string;
  status: StepStatus;
  output?: unknown;
  error?: string;
  depends_on: string[];
  retry_count: number;
  max_retries: number;
}

export interface Plan {
  id: string;
  goal: string;
  steps: PlanStep[];
  status: StepStatus;
  created_at: number;
}

export interface MemoryFact {
  id: string;
  key: string;
  value: string;
  category: string;
  confidence: number;
  source: string;
  created_at: number;
  accessed_at: number;
  access_count: number;
}

// Image gen types
export interface IgState {
  models: ModelEntry[];
  checkpoints: ModelEntry[];
  loras: ModelEntry[];
  modelPath: string;
  loraPath: string;
  prompt: string;
  negPrompt: string;
  steps: number;
  cfg: number;
  seed: number;
  width: number;
  height: number;
  device: "auto" | "cuda" | "cpu";
  scheduler: string;
  generating: boolean;
  progress: number;
  progressTotal: number;
  resultB64: string;
  error: string;
  elapsed: number;
  vramFreed: boolean;
}

export interface TtsState {
  voices: Array<{ id: string; label: string; lang: string }>;
  voice: string;
  speed: number;
  text: string;
  generating: boolean;
  progress: number;
  resultB64: string;
  duration: number;
  error: string;
}

export interface LoraState {
  modelPath: string;
  datasetDir: string;
  outputName: string;
  outputDir: string;
  rank: number;
  alpha: number;
  lr: number;
  epochs: number;
  batchSize: number;
  resolution: number;
  cleaning: boolean;
  training: boolean;
  step: number;
  totalSteps: number;
  epoch: number;
  totalEpochs: number;
  loss: number | null;
  log: TermLine[];
  error: string;
  outputPath: string;
  done: boolean;
}

export interface MergeState {
  modelA: string;
  modelB: string;
  modelC: string;
  method: "weighted_sum" | "add_difference";
  weight: number;
  outputDir: string;
  outputName: string;
  running: boolean;
  done: boolean;
  error: string;
  outputPath: string;
  log: TermLine[];
  progress: number;
  total: number;
}
