// Typed wrappers around Tauri invoke calls.
// Command names and argument shapes match Rust #[command] definitions exactly.
// Rust param names are snake_case; Tauri deserializes them from camelCase automatically,
// but struct fields must be passed as snake_case object keys.

import { invoke } from "@tauri-apps/api/core";
import type { ModelSummary, ModelEntry, RunningServer, DepReport, PerfResult } from "./types.js";

// ── Model ─────────────────────────────────────────────────────────────────────

export const loadModel = (modelPath: string, gpuLayers: number, ctxSize: number) =>
  invoke<ModelSummary>("load_model", { modelPath, gpuLayers, ctxSize });

export const unloadModel = () => invoke<void>("unload_model");

export const stopModelServer = (port: number | null) =>
  invoke<void>("stop_model_server", { port });

export const attachModel = (modelPath: string, port: number) =>
  invoke<ModelSummary>("attach_model", { modelPath, port });

export const currentModelPort = () => invoke<number | null>("current_model_port");

export const scanRunningServers = () => invoke<RunningServer[]>("scan_running_servers");

export const scanModelsDir = () => invoke<ModelEntry[]>("scan_models_dir");

export const getModelsDir = () => invoke<string>("get_models_dir");

export const setModelsDir = (path: string) => invoke<void>("set_models_dir", { path });

export const openModelsDir = () => invoke<void>("open_models_dir");

// ── Inference ─────────────────────────────────────────────────────────────────

export interface GenerateRequest {
  messages: Array<{ role: string; content: string }>;
  max_tokens: number;
  temperature: number;
  top_p: number;
  top_k: number;
  repeat_penalty: number;
  seed: number;
}

export const generate = (req: GenerateRequest) => invoke<PerfResult>("generate", { req });

export const dualGenerate = (req: GenerateRequest) => invoke<unknown>("dual_generate", { req });

export const stopGenerate = () => invoke<void>("stop_generate");

// Dual agent
export const loadDrafter = (modelPath: string, gpuLayers: number, ctxSize: number) =>
  invoke<ModelSummary>("load_drafter", { modelPath, gpuLayers, ctxSize });

export const loadCritic = (modelPath: string, gpuLayers: number, ctxSize: number) =>
  invoke<ModelSummary>("load_critic", { modelPath, gpuLayers, ctxSize });

export interface DualAgentStatus {
  drafter: ModelSummary | null;
  critic: ModelSummary | null;
}
export const dualAgentStatus = () => invoke<DualAgentStatus>("dual_agent_status");

// ── Session ───────────────────────────────────────────────────────────────────

export const saveSession = (json: string, path: string) =>
  invoke<void>("save_session", { json, path });

export const loadSession = (path: string) => invoke<string>("load_session", { path });

// ── GPU ───────────────────────────────────────────────────────────────────────

export const getGpuStats = () => invoke<Record<string, unknown>>("get_gpu_stats");

// ── Agent: filesystem ─────────────────────────────────────────────────────────

export const fsTree = (path: string, maxDepth: number) =>
  invoke<unknown[]>("fs_tree", { path, maxDepth });

export const fsRead = (path: string) => invoke<{ content: string }>("fs_read", { path });

export const fsWrite = (path: string, content: string) =>
  invoke<void>("fs_write", { path, content });

export const fsDelete = (path: string) => invoke<void>("fs_delete", { path });

export const fsSearch = (path: string, pattern: string, context: number) =>
  invoke<unknown[]>("fs_search", { path, pattern, context });

export const fsMkdir = (path: string) => invoke<void>("fs_mkdir", { path });

export const getSandboxRoot = () => invoke<string>("get_sandbox_root");

export const setSandboxRoot = (path: string) => invoke<void>("set_sandbox_root", { path });

// ── Agent: patch ──────────────────────────────────────────────────────────────

export const applyPatch = (path: string, content: string, description: string) =>
  invoke<unknown>("apply_patch", { path, content, description });

export const diffProposed = (path: string, newContent: string) =>
  invoke<{ unified: string }>("diff_proposed", { path, newContent });

export const undoPatch = (path: string) => invoke<unknown>("undo_patch", { path });

// ── Agent: exec ───────────────────────────────────────────────────────────────

// ExecRequest mirrors src-tauri/src/tools/sandbox.rs::ExecRequest
export interface ExecRequest {
  command: string;
  args: string[];
  cwd: string | null;
  env: Record<string, string>;
  timeout_secs: number;
  stdin_data: string | null;
  capture_output: boolean;
}

export const execCommand = (
  command: string,
  args: string[],
  cwd: string | null,
  timeoutSecs: number
) =>
  invoke<unknown>("exec_command", {
    req: {
      command,
      args,
      cwd,
      env: {},
      timeout_secs: timeoutSecs,
      stdin_data: null,
      capture_output: true,
    } satisfies ExecRequest,
  });

// kill_process takes the exec ID assigned when exec_command was called.
// The frontend must track the last ID from the "exec-start" event.
export const killProcess = (id: string) => invoke<boolean>("kill_process", { id });

// ── Agent: plan ───────────────────────────────────────────────────────────────

export const agentRun = (goal: string) => invoke<void>("agent_run", { goal });

export const checkGoalCompletion = (goal: string) =>
  invoke<{ complete: boolean; reason: string }>("check_goal_completion", { goal });

// plan_execute requires both the JSON and the original goal text
export const executePlan = (planJson: string, goal: string) =>
  invoke<void>("plan_execute", { planJson, goal });

export const genPlanPrompt = (goal: string, memoryContext: string) =>
  invoke<string>("plan_prompt_template", { goal, memoryContext });

// ── Agent: memory ─────────────────────────────────────────────────────────────

export const memorySearch = (query: string) => invoke<unknown[]>("mem_recall", { query });

export const memoryAll = () => invoke<unknown>("mem_store");

export const memoryForget = (id: string) => invoke<boolean>("mem_forget", { id });

// ── Agent: write mode ─────────────────────────────────────────────────────────

export const getAgentWriteMode = () => invoke<boolean>("get_agent_write_mode");

export const setAgentWriteMode = (enabled: boolean) =>
  invoke<void>("set_agent_write_mode", { enabled });

export const writeBinaryB64 = (path: string, b64: string) =>
  invoke<void>("write_binary_b64", { path, b64 });

// ── Privacy / data management ─────────────────────────────────────────────────

export const clearUserData = (opts: {
  clearMemory: boolean;
  clearAudit: boolean;
  clearLogs: boolean;
}) =>
  invoke<string[]>("clear_user_data", {
    clearMemory: opts.clearMemory,
    clearAudit: opts.clearAudit,
    clearLogs: opts.clearLogs,
  });

// ── Dep check ─────────────────────────────────────────────────────────────────

export const checkDependencies = () => invoke<DepReport>("check_dependencies");

// ── Image gen ─────────────────────────────────────────────────────────────────

// ImgGenPayload mirrors src-tauri/src/imggen.rs::ImgGenPayload
export interface ImgGenPayload {
  prompt: string;
  neg_prompt?: string;
  model_path: string;
  lora_path?: string;
  steps?: number;
  cfg_scale?: number;
  seed?: number;
  width?: number;
  height?: number;
  device?: string;
  scheduler?: string;
}

// ImgGenResult mirrors src-tauri/src/imggen.rs::ImgGenResult
export interface ImgGenResult {
  base64_png: string;
  device: string;
  elapsed: number;
}

export const imggenScanModels = () => invoke<unknown[]>("imggen_scan_models");

export const imggenScanCheckpoints = () => invoke<unknown[]>("imggen_scan_checkpoints");

export const imggenScanLoras = () => invoke<unknown[]>("imggen_scan_loras");

export const runImggen = (payload: ImgGenPayload) =>
  invoke<ImgGenResult>("imggen_generate", { payload });

export const imggenLoad = (modelPath: string, loraPath: string, device: string) =>
  invoke<string>("imggen_load", { modelPath, loraPath, device });

export const imggenUnload = () => invoke<void>("imggen_unload");

export const imggenLoadedModel = () => invoke<string | null>("imggen_loaded_model");

// ── Video gen ───────────────────────────────────────────────────────────────

export interface VideoModelEntry { path: string; label: string; pipeline: string }
export interface VideoPayload {
  prompt: string;
  neg_prompt?: string;
  model_path: string;
  num_frames?: number;
  steps?: number;
  cfg_scale?: number;
  width?: number;
  height?: number;
  fps?: number;
  seed?: number;
  image_b64?: string;   // optional still → image-to-video
}
export interface VideoResult { base64_mp4: string; frames: number; elapsed: number }

export interface EnhancePayload {
  video_b64: string; fps: number; stages: string[]; model_path: string;
  prompt: string; neg_prompt?: string; cfg_scale?: number;
  refine_strength?: number; refine_steps?: number; interp_factor?: number;
}
export interface EnhanceResult { enhanced_b64: string; frames: number; width: number; height: number; elapsed: number }

export interface LoraEntry { path: string; label: string }
export const videoScanModels  = () => invoke<VideoModelEntry[]>("video_scan_models");
export const videoScanLoras   = () => invoke<LoraEntry[]>("video_scan_loras");
export const videoLoad        = (modelPath: string, loraPath?: string, loraStrength?: number, frames?: number) =>
  invoke<string>("video_load", { modelPath, loraPath: loraPath || null, loraStrength: loraStrength ?? 1.0, frames: frames ?? 49 });
export const videoUnload      = () => invoke<void>("video_unload");
export const videoLoadedModel = () => invoke<string | null>("video_loaded_model");
export const videoGenerate    = (payload: VideoPayload) => invoke<VideoResult>("video_generate", { payload });
export const videoEnhance     = (payload: EnhancePayload) => invoke<EnhanceResult>("video_enhance", { payload });

// ── TTS ───────────────────────────────────────────────────────────────────────

// TtsPayload mirrors src-tauri/src/tts.rs::TtsPayload
export interface TtsPayload {
  text: string;
  voice?: string;
  speed?: number;
  lang?: string;
}

// TtsResult mirrors src-tauri/src/tts.rs::TtsResult
export interface TtsResult {
  base64_wav: string;
  duration: number;
  sample_rate: number;
}

export const ttsFetchVoices = () =>
  invoke<Array<{ id: string; label: string; lang: string }>>("tts_voices");

export const runTts = (payload: TtsPayload) =>
  invoke<TtsResult>("tts_generate", { payload });

// ── LoRA ──────────────────────────────────────────────────────────────────────

// LoraTrainPayload mirrors src-tauri/src/lora.rs::LoraTrainPayload
export interface LoraTrainPayload {
  model_path: string;
  dataset_dir: string;
  output_name: string;
  output_dir?: string;
  rank?: number;
  alpha?: number;
  lr?: number;
  epochs?: number;
  batch_size?: number;
  resolution?: number;
}

export const loraCleanDataset = (datasetDir: string) =>
  invoke<void>("lora_clean_dataset", { datasetDir });

export const loraStart = (payload: LoraTrainPayload) =>
  invoke<void>("lora_start_training", { payload });

export const loraStop = () => invoke<void>("lora_stop_training");

// ── Merge ─────────────────────────────────────────────────────────────────────

// MergePayload mirrors src-tauri/src/merge.rs::MergePayload
export interface MergePayload {
  model_a: string;
  model_b: string;
  model_c?: string;
  method: string;
  weight: number;
  output_path: string;
}

export const mergeRun = (payload: MergePayload) =>
  invoke<void>("merge_start", { payload });

export const mergeCancel = () => invoke<void>("merge_cancel");

// ── Setup wizard ──────────────────────────────────────────────────────────────

export interface SystemInfo {
  os: string;
  gpu_name: string | null;
  driver_version: string | null;
  cuda_version: string | null;
  vram_gb: number | null;
  torch_index: string;
  torch_index_url: string;
  system_python: string | null;
  python_version: string | null;
  venv_ready: boolean;
  tinyq4_ready: boolean;
  disk_free_gb: number;
  setup_done: boolean;
}

export const detectSystem = () => invoke<SystemInfo>("detect_system");

/** profile: "full" (creative + core) or "fast" (core only). Streams "setup-log" / "setup-step". */
export const runSetup = (profile: "full" | "fast") => invoke<void>("run_setup", { profile });

export const skipSetup  = () => invoke<void>("skip_setup");
export const resetSetup = () => invoke<void>("reset_setup");

/** Download a starter GGUF from HuggingFace into the models dir. Streams "model-progress". */
export const downloadStarterModel = (repo: string, file: string, modelsDir: string) =>
  invoke<string>("download_starter_model", { repo, file, modelsDir });

// ── PTY terminal ──────────────────────────────────────────────────────────────

/** Spawn $SHELL at cwd in a real PTY. Emits "pty-data" events for output. */
export const ptySpawn  = (cwd: string, cols: number, rows: number) =>
  invoke<void>("pty_spawn", { cwd, cols, rows });

/** Forward xterm.js onData keystrokes to the PTY master. */
export const ptyWrite  = (data: string) => invoke<void>("pty_write", { data });

/** Sync kernel PTY dimensions after xterm resize so apps line-wrap correctly. */
export const ptyResize = (cols: number, rows: number) =>
  invoke<void>("pty_resize", { cols, rows });

/** Kill the shell process and clean up the PTY session. */
export const ptyKill   = () => invoke<void>("pty_kill");
