//! Project-local runtime paths for Saient.
//!
//! Local builds keep app-owned mutable data under the checkout's `data/`
//! directory. `SAIENT_DATA_DIR` can override that root for packaging or tests.

use std::path::{Path, PathBuf};
use std::process::Command;

/// Resolve an existing directory once before it becomes a process or tool root.
/// Callers keep this returned path; display names and persisted labels must never
/// be used to reconstruct it downstream.
pub fn canonical_existing_dir(path: &Path, label: &str) -> Result<PathBuf, String> {
    if path.as_os_str().is_empty() {
        return Err(format!("{label} path is empty"));
    }
    if !path.is_dir() {
        return Err(format!("{label} is not a directory: {}", path.display()));
    }
    std::fs::canonicalize(path)
        .map_err(|e| format!("could not resolve {label} '{}': {e}", path.display()))
}

pub const DATA_DIR_ENV: &str = "SAIENT_DATA_DIR";
pub const CONFIG_DIR_ENV: &str = "SAIENT_CONFIG_DIR";
pub const SHARE_DIR_ENV: &str = "SAIENT_SHARE_DIR";
pub const MODELS_DIR_ENV: &str = "SAIENT_MODELS_DIR";
pub const HF_HOME_ENV: &str = "HF_HOME";
pub const HF_HUB_CACHE_ENV: &str = "HF_HUB_CACHE";
pub const HF_DATASETS_CACHE_ENV: &str = "HF_DATASETS_CACHE";
pub const RUNTIME_ASSETS_ENV: &str = "SAIENT_RUNTIME_ASSETS_DIR";

pub fn repo_root() -> Option<PathBuf> {
    if let Some(path) = env_path("SAIENT_REPO_ROOT") {
        if is_repo_root(&path) {
            return Some(path);
        }
    }

    if let Ok(exe) = std::env::current_exe() {
        if let Some(root) = find_repo_ancestor(&exe) {
            return Some(root);
        }
    }

    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    if let Some(root) = find_repo_ancestor(&manifest) {
        return Some(root);
    }

    std::env::current_dir()
        .ok()
        .and_then(|cwd| find_repo_ancestor(&cwd))
}

pub fn data_dir() -> PathBuf {
    if let Some(path) = env_path(DATA_DIR_ENV) {
        return path;
    }
    if let Some(root) = repo_root() {
        return root.join("data");
    }
    platform_data_dir()
}

pub fn config_dir() -> PathBuf {
    if let Some(path) = env_path(CONFIG_DIR_ENV) {
        return path;
    }
    data_dir().join("config").join(config_dir_name())
}

pub fn config_base_dir() -> PathBuf {
    data_dir().join("config")
}

pub fn share_dir() -> PathBuf {
    if let Some(path) = env_path(SHARE_DIR_ENV) {
        return path;
    }
    data_dir().join("share").join("saient")
}

pub fn models_dir() -> PathBuf {
    if let Some(path) = env_path(MODELS_DIR_ENV) {
        return path;
    }
    data_dir().join("models")
}

pub fn checkpoints_dir() -> PathBuf {
    models_dir().join("checkpoints")
}

pub fn loras_dir() -> PathBuf {
    models_dir().join("lora")
}

// ── Per-category model folders ──────────────────────────────────────────────
// Saient owns its model storage under `models_dir()`. Each generation subsystem
// reads from its own dedicated folder so the app never scans random locations
// elsewhere on the machine. Diffusers models pulled by the HF stack still live
// in `hf_home()` addressed by repo-id; these folders are for loose files the
// user (or the in-app downloader) drops in: GGUF text models, single-file SDXL
// checkpoints, and any locally-exported diffusers/LoRA directories.

/// Text LLM (GGUF) models — the in-app downloader lands here and it is scanned
/// for chat/coder models.
pub fn llm_models_dir() -> PathBuf {
    models_dir().join("llm")
}

/// Image-generation models: single-file SDXL checkpoints and locally-exported
/// diffusers pipelines.
pub fn image_models_dir() -> PathBuf {
    models_dir().join("image")
}

/// Video-generation models: locally-exported Wan/video diffusers pipelines.
pub fn video_models_dir() -> PathBuf {
    models_dir().join("video")
}

/// Create every app-owned model folder so they exist (and are visible to the
/// user in a file manager) even before anything is downloaded. Best-effort:
/// failures are ignored so startup never blocks on a read-only volume.
pub fn ensure_model_dirs() {
    for d in [
        models_dir(),
        llm_models_dir(),
        image_models_dir(),
        image_models_dir().join("lora"),
        video_models_dir(),
        video_models_dir().join("loras"),
        checkpoints_dir(),
        loras_dir(),
    ] {
        std::fs::create_dir_all(&d).ok();
    }
}

pub fn agent_workspace_dir() -> PathBuf {
    data_dir().join("agent-workspace")
}

pub fn llm_runtime_dir() -> PathBuf {
    data_dir().join("llm-runtime")
}

/// Explicitly downloaded, user-visible voice and vision files.
pub fn runtime_assets_dir() -> PathBuf {
    data_dir().join("runtime-assets")
}

/// Compatibility cache created by releases up to 1.0.17. New code never writes
/// here; Settings can remove it after reporting its exact size.
pub fn legacy_hf_cache_dir() -> PathBuf {
    data_dir().join("huggingface")
}

/// Throw-away state required by some Hugging Face libraries even when loading
/// local files. It is removed on every Saient start so prompts/datasets cannot
/// accumulate between runs.
pub fn hf_home() -> PathBuf {
    data_dir().join("runtime-tmp").join("huggingface")
}

pub fn hf_hub_cache() -> PathBuf {
    hf_home().join("hub")
}

pub fn path_is_inside_data(path: &Path) -> bool {
    let data = data_dir().canonicalize().unwrap_or_else(|_| data_dir());
    let candidate = path.canonicalize().unwrap_or_else(|_| path.to_path_buf());
    candidate.starts_with(data)
}

pub fn apply_child_env(cmd: &mut Command) -> &mut Command {
    cmd.env(DATA_DIR_ENV, data_dir())
        .env(CONFIG_DIR_ENV, config_dir())
        .env(SHARE_DIR_ENV, share_dir())
        .env(MODELS_DIR_ENV, models_dir())
        .env(RUNTIME_ASSETS_ENV, runtime_assets_dir())
        .env(HF_HOME_ENV, hf_home())
        .env(HF_HUB_CACHE_ENV, hf_hub_cache())
        .env(HF_DATASETS_CACHE_ENV, hf_home().join("datasets"))
        // Runtime inference is always local-only. The general Internet switch
        // must never let a model library upload input or grow a durable cache.
        // Explicit setup/model-download commands remove these flags themselves.
        .env("HF_HUB_OFFLINE", "1")
        .env("TRANSFORMERS_OFFLINE", "1")
        .env("DIFFUSERS_OFFLINE", "1")
        .env("HF_DATASETS_OFFLINE", "1")
        .env("HF_HUB_DISABLE_TELEMETRY", "1")
        .env("DO_NOT_TRACK", "1")
        .env("HF_HUB_DISABLE_XET", "1");
    cmd
}

pub fn legacy_config_dir(name: &str) -> Option<PathBuf> {
    #[cfg(target_os = "windows")]
    {
        std::env::var_os("APPDATA").map(PathBuf::from).map(|p| p.join(name))
    }
    #[cfg(not(target_os = "windows"))]
    {
        home_dir().map(|h| h.join(".config").join(name))
    }
}

pub fn legacy_share_dir(name: &str) -> Option<PathBuf> {
    #[cfg(target_os = "windows")]
    {
        std::env::var_os("APPDATA").map(PathBuf::from).map(|p| p.join(name))
    }
    #[cfg(not(target_os = "windows"))]
    {
        home_dir().map(|h| h.join(".local").join("share").join(name))
    }
}

fn config_dir_name() -> &'static str {
    if cfg!(debug_assertions) { "saient-dev" } else { "saient" }
}

fn env_path(name: &str) -> Option<PathBuf> {
    std::env::var_os(name)
        .filter(|v| !v.is_empty())
        .map(PathBuf::from)
}

fn find_repo_ancestor(path: &Path) -> Option<PathBuf> {
    for ancestor in path.ancestors() {
        if is_repo_root(ancestor) {
            return Some(ancestor.to_path_buf());
        }
    }
    None
}

fn is_repo_root(path: &Path) -> bool {
    path.join("llm-inference").join("src-tauri").is_dir()
        && path.join("scripts").is_dir()
}

fn platform_data_dir() -> PathBuf {
    #[cfg(target_os = "windows")]
    {
        if let Some(appdata) = std::env::var_os("APPDATA") {
            return PathBuf::from(appdata).join("saient").join("data");
        }
    }
    #[cfg(target_os = "macos")]
    {
        if let Some(home) = home_dir() {
            return home.join("Library").join("Application Support").join("saient");
        }
    }
    home_dir()
        .map(|h| h.join(".local").join("share").join("saient"))
        .unwrap_or_else(|| std::env::temp_dir().join("saient"))
}

fn home_dir() -> Option<PathBuf> {
    std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))
        .map(PathBuf::from)
}
