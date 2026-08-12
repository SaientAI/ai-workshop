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

export interface SaientBindingReply {
  text: string;
  tick: number;
  action: string | null;
  conscience: string | null;
  refused: boolean;
  redirected: boolean;
  success: boolean;
  verified: boolean;
  guarantees: Record<string, boolean>;
  binding_status: "bound";
  minimum_interface: string;
  model: string;
  manifest: string;
  state_tick_before: number;
  state_tick_after: number;
  state_context_injected: boolean;
  state_context_sha256: string;
  record_boundary_clean: boolean;
  identity_boundary_clean: boolean;
  model_calls: number;
  used_integrity_fallback: boolean;
}

export const saientBind = () => invoke<Record<string, unknown>>("saient_bind");
export const saientChat = (message: string) =>
  invoke<SaientBindingReply>("saient_chat", { message });

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

/// Prefill the constant part of the planning prompt into the engine's KV cache.
/// Resolves false when no model is loaded. Safe to call more than once.
export const warmAgentCache = () => invoke<boolean>("warm_agent_cache");

// ── Projects ──────────────────────────────────────────────────────────────────
// One folder per piece of work. Opening a project repoints files, patches, the
// sandbox and checkpoints at it.

export interface ProjectInfo {
  name: string;
  path: string;
  modified: number;
  entry_count: number;
  /** off | guided | companion | autonomous — see lib/agiLevel.ts */
  agi_level: string;
}

export const projectList = () => invoke<ProjectInfo[]>("project_list");
export const projectActive = () => invoke<ProjectInfo | null>("project_active");
export const projectCreate = (name: string, agiLevel: string) =>
  invoke<ProjectInfo>("project_create", { name, agiLevel });

export const projectSetLevel = (name: string, agiLevel: string) =>
  invoke<ProjectInfo>("project_set_level", { name, agiLevel });
export const projectOpen = (name: string) => invoke<ProjectInfo>("project_open", { name });

// ── Checkpoints ───────────────────────────────────────────────────────────────
// A checkpoint saves the working state, not just the transcript: goal, step in
// flight, terminal cwd, outstanding work and the workspace files.

export type CheckpointKind = "manual" | "auto_turn" | "auto_task" | "pre_restore";

export interface CheckpointMeta {
  id: string;
  name: string;
  created_at: number;
  kind: CheckpointKind;
  parent: string | null;
  goal: string;
  turn_state: string;
  terminal_cwd: string;
  step_index: number | null;
  step_total: number | null;
  outstanding: string[];
  file_count: number;
  total_bytes: number;
}

export interface SessionState {
  goal: string;
  turn_state: string;
  terminal_cwd: string;
  step_index: number | null;
  step_total: number | null;
  outstanding: string[];
  conversation: unknown;
  plan: unknown;
  terminal: string[];
}

export interface RestoreReport {
  restored: string[];
  unchanged: string[];
  left_in_place: string[];
  undo_checkpoint: string;
}

export const checkpointSave = (
  name: string,
  kind: CheckpointKind,
  parent: string | null,
  session: SessionState,
) => invoke<CheckpointMeta>("checkpoint_save", { name, kind, parent, session });

export const checkpointList = () => invoke<CheckpointMeta[]>("checkpoint_list");

export const checkpointLoad = (id: string) => invoke<unknown>("checkpoint_load", { id });

/// Overwrites the workspace. A safety checkpoint is taken first; its id comes
/// back as `undo_checkpoint`.
export const checkpointRestore = (id: string, session: SessionState) =>
  invoke<RestoreReport>("checkpoint_restore", { id, session });

export const checkpointDelete = (id: string) => invoke<void>("checkpoint_delete", { id });

export const checkpointExport = (id: string, format: "markdown" | "json") =>
  invoke<string>("checkpoint_export", { id, format });

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

// ── Internet / network access gate ───────────────────────────────────────────

export const getInternetEnabled = () => invoke<boolean>("get_internet_enabled");

export const setInternetEnabled = (enabled: boolean) =>
  invoke<void>("set_internet_enabled", { enabled });

/** Session-only network authority for first-run setup. Never changes Settings. */
export const setSetupInternetAuthorized = (authorized: boolean) =>
  invoke<void>("set_setup_internet_authorized", { authorized });

// ── Dep check ─────────────────────────────────────────────────────────────────

export const checkDependencies = () => invoke<DepReport>("check_dependencies");

// ── Game asset builder ───────────────────────────────────────────────────────

export interface AssetFile {
  name: string;
  path: string;
  size: number;
}

export interface AssetScan {
  project_dir: string;
  source_dir: string;
  output_dir: string;
  blender_path: string | null;
  sources: AssetFile[];
  outputs: AssetFile[];
}

export interface AssetRunResult {
  ok: boolean;
  code: number;
  stdout: string;
  stderr: string;
}

export const assetBuilderScan = () => invoke<AssetScan>("asset_builder_scan");

export const assetBuilderOpenDir = (kind: "source" | "output") =>
  invoke<void>("asset_builder_open_dir", { kind });

export const assetBuilderRun = (
  dryRun: boolean,
  sources: string[] = [],
  builder: "relief" | "local3d" = "relief"
) => invoke<AssetRunResult>("asset_builder_run", { dryRun, sources, builder });

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
  face_detail?: boolean;
  asset_guard?: boolean;
  asset_kind?: string;
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
  scheduler?: string;
  shift?: number;
  lora_profile?: string;
  lora_strength_high?: number;
  lora_strength_low?: number;
  lora_split_step?: number;
  width?: number;
  height?: number;
  fps?: number;
  seed?: number;
  image_b64?: string;   // optional still → image-to-video
  previous_video_b64?: string;  // for extend: the prior full clip to append the new segment to
  force_seam_blend?: boolean;   // for storyboard on T2V: force frame-level concat with crossfade blend at seam
  low_vram?: boolean;
  block_offload?: boolean;      // park transformer to RAM, stream per-step → fits native 5s@720p (slow)
  denoise_cache?: "off" | "balanced";
  cache_threshold?: number;
  preview?: boolean;
  preview_every?: number;
  preview_max_width?: number;
}
export interface VideoResult { base64_mp4: string; frames: number; elapsed: number }

export interface EnhancePayload {
  video_b64: string; fps: number; stages: string[]; model_path: string;
  prompt: string; neg_prompt?: string; cfg_scale?: number;
  refine_strength?: number; refine_steps?: number; interp_factor?: number;
}
export interface EnhanceResult {
  enhanced_b64: string;
  frames: number;
  width: number;
  height: number;
  elapsed: number;
  completed_stages: string[];
  failed_stages: string[];
}

export interface LoraEntry { path: string; label: string }
export const videoScanModels  = () => invoke<VideoModelEntry[]>("video_scan_models");
export const videoScanLoras   = () => invoke<LoraEntry[]>("video_scan_loras");
export const videoLoad        = (modelPath: string, loraPath?: string, loraStrength?: number, frames?: number, precision?: string) =>
  invoke<string>("video_load", { modelPath, loraPath: loraPath || null, loraStrength: loraStrength ?? 1.0, frames: frames ?? 49, precision: precision ?? "fast" });
export const videoUnload      = () => invoke<void>("video_unload");
export const videoLoadedModel = () => invoke<string | null>("video_loaded_model");
export const videoLoadedLora = () => invoke<string | null>("video_loaded_lora");
export const videoGenerate    = (payload: VideoPayload) => invoke<VideoResult>("video_generate", { payload });
export const videoEnhance     = (payload: EnhancePayload) => invoke<EnhanceResult>("video_enhance", { payload });

// ── Vision (local image understanding — Moondream2 daemon) ──────────────────────

// VisionResult mirrors src-tauri/src/vision.rs::VisionResult
export interface VisionResult {
  answer: string;
  elapsed: number;
  device: string;
}

/** Describe / answer a question about an image (base64 PNG/JPEG). Loads the model on first use. */
export const visionDescribe = (imageB64: string, question: string) =>
  invoke<VisionResult>("vision_describe", { imageB64, question });

/** Free the vision model's VRAM. */
export const visionUnload = () => invoke<void>("vision_unload");

/** Whether the vision model is currently loaded. */
export const visionLoaded = () => invoke<boolean>("vision_loaded");

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

/** Download a GGUF from HuggingFace into the models dir. Streams "model-progress". */
export const downloadStarterModel = (repo: string, file: string, modelsDir: string, token?: string) =>
  invoke<string>("download_starter_model", { repo, file, modelsDir, token: token || null });

/** A .gguf file in a HuggingFace repo. Mirrors src-tauri/src/setup.rs::HfFile. */
export interface HfFile { file: string; size: number }

/** List the .gguf files (with sizes) in a HuggingFace repo's main branch. */
export const hfListGguf = (repo: string, token?: string) =>
  invoke<HfFile[]>("hf_list_gguf", { repo, token: token || null });

/** A HuggingFace model repo from search. */
export interface HfRepo { id: string; downloads: number; likes: number }

/** Search HuggingFace models. `filter` is a pipeline tag e.g. "text-to-image". */
export const hfSearch = (query: string, filter?: string, token?: string) =>
  invoke<HfRepo[]>("hf_search", { query, filter: filter || null, token: token || null });

/** List files in a repo matching the given extensions (e.g. [".safetensors"]). */
export const hfListFiles = (repo: string, exts: string[], token?: string) =>
  invoke<HfFile[]>("hf_list_files", { repo, exts, token: token || null });

/** Download a repo file into the managed folder for `target` ("checkpoint" | "lora"). */
export const downloadHfFile = (repo: string, file: string, target: string, token?: string) =>
  invoke<string>("download_hf_file", { repo, file, target, token: token || null });

/** Download a full diffusers repo into the managed image model folder. */
export const downloadHfRepo = (repo: string, target: string, token?: string) =>
  invoke<string>("download_hf_repo", { repo, target, token: token || null });

// ── PTY terminal ──────────────────────────────────────────────────────────────

/** Spawn the platform shell at cwd in a real PTY. Emits "pty-data" events for output. */
export const ptySpawn  = (cwd: string, cols: number, rows: number, serverPort?: number | null) =>
  invoke<void>("pty_spawn", { cwd, cols, rows, serverPort: serverPort ?? null });

/** Forward xterm.js onData keystrokes to the PTY master. */
export const ptyWrite  = (data: string) => invoke<void>("pty_write", { data });

/** Sync kernel PTY dimensions after xterm resize so apps line-wrap correctly. */
export const ptyResize = (cols: number, rows: number) =>
  invoke<void>("pty_resize", { cols, rows });

/** Kill the shell process and clean up the PTY session. */
export const ptyKill   = () => invoke<void>("pty_kill");


// ── Signed desktop updates ──────────────────────────────────────────────────────
export interface UpdateInfo {
  current: string;
  latest: string;
  update_available: boolean;
  install_supported: boolean;
  url: string;
  notes: string;
}

export interface UpdateProgress {
  phase: "checking" | "downloading" | "verifying" | "installing" | "installed";
  downloaded: number;
  total: number;
  message: string;
}

export interface InstallUpdateResult {
  version: string;
  restart_required: boolean;
}

/** Check the site for a newer version. Best-effort — throws if offline. */
export const checkUpdate = () => invoke<UpdateInfo>("check_update");

/** Download, verify and install the signed Debian update with OS authorization. */
export const installUpdate = (expectedVersion: string) =>
  invoke<InstallUpdateResult>("install_update", { expectedVersion });

/** Restart the current executable after the package manager replaces it. */
export const relaunchAfterUpdate = () => invoke<void>("relaunch_after_update");

/** Plain-text diagnostics (version, OS, GPU, paths) for support. No telemetry. */
export const diagnostics = () => invoke<string>("diagnostics");

/** OS name ("windows" | "linux" | "macos"). */
export const osName = () => invoke<string>("os_name");

// ── Launch password (argon2) ────────────────────────────────────────────────────
export const passwordIsSet  = () => invoke<boolean>("password_is_set");
export const passwordVerify = (password: string) => invoke<boolean>("password_verify", { password });
export const passwordSet    = (next: string, current?: string) =>
  invoke<void>("password_set", { new: next, current: current ?? null });
export const passwordClear  = (current: string) => invoke<void>("password_clear", { current });

/** Write the effective loop gate (master switch combined with project level). */
export const saientSetEnabled = (enabled: boolean) =>
  invoke<void>("saient_set_enabled", { enabled });
export const saientIsEnabled = () => invoke<boolean>("saient_is_enabled");
export const saientLoopRunning = () => invoke<boolean>("saient_loop_running");
