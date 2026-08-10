use serde::{Deserialize, Serialize};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::{Arc, Mutex, TryLockError};
use tauri::{Emitter, State, WebviewWindow};
use crate::resolve;
use crate::resolve::NoConsole;

// ── Public types ──────────────────────────────────────────────────────────────

#[derive(Serialize, Clone)]
pub struct ModelEntry {
    pub path:  String,
    pub label: String,
    pub kind:  String,
}

#[derive(Deserialize)]
pub struct ImgGenPayload {
    pub prompt:     String,
    pub neg_prompt: Option<String>,
    pub model_path: String,
    pub lora_path:  Option<String>,
    pub steps:      Option<u32>,
    pub cfg_scale:  Option<f32>,
    pub seed:       Option<i64>,
    pub width:      Option<u32>,
    pub height:     Option<u32>,
    pub device:     Option<String>,
    pub scheduler:  Option<String>,
    pub face_detail: Option<bool>,
    pub asset_guard: Option<bool>,
    pub asset_kind: Option<String>,
}

#[derive(Serialize, Clone)]
pub struct ImgGenProgress {
    pub step:  u32,
    pub total: u32,
}

#[derive(Serialize)]
pub struct ImgGenResult {
    pub base64_png: String,
    pub device:     String,
    pub elapsed:    f64,
}

// ── Hot daemon ────────────────────────────────────────────────────────────────

pub struct ImgGenDaemon {
    _child:     Child,
    stdin:      BufWriter<ChildStdin>,
    stdout:     BufReader<ChildStdout>,
    pub model_path: String,
    pub lora_path:  String,
    pub device:     String,
}

impl ImgGenDaemon {
    fn is_running(&mut self) -> bool {
        matches!(self._child.try_wait(), Ok(None))
    }
}

impl Drop for ImgGenDaemon {
    fn drop(&mut self) {
        let _ = self._child.kill();
        let _ = self._child.wait();
    }
}

pub type DaemonHandle = Arc<Mutex<Option<ImgGenDaemon>>>;
pub(crate) type LoadProgress = Arc<dyn Fn(String) + Send + Sync>;

pub fn new_daemon_handle() -> DaemonHandle {
    Arc::new(Mutex::new(None))
}

pub(crate) fn loaded_matches(
    daemon: &DaemonHandle,
    model_path: &str,
    lora_path: &str,
    device: &str,
) -> Result<bool, String> {
    let mut guard = daemon.lock().map_err(|e| e.to_string())?;
    Ok(guard.as_mut().is_some_and(|d| {
        d.is_running()
            && d.model_path == model_path
            && d.lora_path == lora_path
            && (device == "auto" || d.device == device)
    }))
}

pub(crate) fn loaded_model_from_handle(daemon: &DaemonHandle) -> Result<Option<String>, String> {
    let mut guard = daemon.lock().map_err(|e| e.to_string())?;
    if guard.as_mut().is_some_and(|d| !d.is_running()) {
        *guard = None;
    }
    Ok(guard.as_ref().map(|d| d.model_path.clone()))
}

/// Returns `Ok(None)` when generation currently owns the daemon mutex.
pub(crate) fn try_loaded_model_from_handle(
    daemon: &DaemonHandle,
) -> Result<Option<Option<String>>, String> {
    match daemon.try_lock() {
        Ok(mut guard) => {
            if guard.as_mut().is_some_and(|d| !d.is_running()) {
                *guard = None;
            }
            Ok(Some(guard.as_ref().map(|d| d.model_path.clone())))
        }
        Err(TryLockError::WouldBlock) => Ok(None),
        Err(TryLockError::Poisoned(error)) => Err(error.to_string()),
    }
}

pub(crate) fn load_blocking(
    daemon: DaemonHandle,
    vid: crate::video::VideoHandle,
    window: WebviewWindow,
    model_path: String,
    lora_path: String,
    device: String,
) -> Result<String, String> {
    if let Ok(mut g) = vid.lock() { *g = None; }
    do_load(daemon, window, model_path, lora_path, device, None)
}

pub(crate) fn load_blocking_with_progress(
    daemon: DaemonHandle,
    vid: crate::video::VideoHandle,
    window: WebviewWindow,
    model_path: String,
    lora_path: String,
    device: String,
    progress: LoadProgress,
) -> Result<String, String> {
    if let Ok(mut g) = vid.lock() { *g = None; }
    do_load(daemon, window, model_path, lora_path, device, Some(progress))
}

pub(crate) fn generate_blocking(
    daemon: DaemonHandle,
    payload: ImgGenPayload,
    window: WebviewWindow,
) -> Result<ImgGenResult, String> {
    do_generate(daemon, payload, window)
}

// ── Commands: scan ────────────────────────────────────────────────────────────

#[tauri::command]
pub fn imggen_scan_models() -> Vec<ModelEntry> {
    let mut entries = Vec::new();
    for dir in resolve::model_scan_dirs() {
        scan_diffusers_dir(dir, 4, &mut entries);
    }
    entries
}

#[tauri::command]
pub fn imggen_scan_checkpoints() -> Vec<ModelEntry> {
    scan_safetensors_dirs(&resolve::checkpoint_scan_dirs(), "checkpoint")
}

#[tauri::command]
pub fn imggen_scan_loras() -> Vec<ModelEntry> {
    scan_safetensors_dirs(&resolve::lora_scan_dirs(), "lora")
}

// ── Commands: daemon lifecycle ────────────────────────────────────────────────

/// Load (or reload) the hot daemon with the given model. Blocks until ready.
#[tauri::command]
pub async fn imggen_load(
    daemon: State<'_, DaemonHandle>,
    vid: State<'_, crate::video::VideoHandle>,
    window: tauri::WebviewWindow,
    model_path: String,
    lora_path: String,
    device: String,
) -> Result<String, String> {
    // Free a resident video daemon first — image + video models can't co-exist on a 16 GB
    // card (mirrors video_load freeing the image daemon).
    if let Ok(mut g) = vid.lock() { *g = None; }
    let arc = daemon.inner().clone();
    tokio::task::spawn_blocking(move || do_load(arc, window, model_path, lora_path, device, None))
        .await
        .map_err(|e| format!("task join: {e}"))?
}

/// Kill the daemon and free VRAM.
#[tauri::command]
pub async fn imggen_unload(daemon: State<'_, DaemonHandle>) -> Result<(), String> {
    *daemon.lock().map_err(|e| e.to_string())? = None;
    Ok(())
}

/// Returns the loaded model path, or null if nothing is loaded.
#[tauri::command]
pub fn imggen_loaded_model(daemon: State<'_, DaemonHandle>) -> Result<Option<String>, String> {
    Ok(daemon.lock().map_err(|e| e.to_string())?
        .as_ref()
        .map(|d| d.model_path.clone()))
}

// ── Commands: generate ────────────────────────────────────────────────────────

#[tauri::command]
pub async fn imggen_generate(
    daemon: State<'_, DaemonHandle>,
    payload: ImgGenPayload,
    window: WebviewWindow,
) -> Result<ImgGenResult, String> {
    let arc = daemon.inner().clone();
    tokio::task::spawn_blocking(move || do_generate(arc, payload, window))
        .await
        .map_err(|e| format!("task join: {e}"))?
}

// ── Internal: load ────────────────────────────────────────────────────────────

fn do_load(
    arc: DaemonHandle,
    window: tauri::WebviewWindow,
    model_path: String,
    lora_path: String,
    device: String,
    progress: Option<LoadProgress>,
) -> Result<String, String> {
    let mut guard = arc.lock().map_err(|e| e.to_string())?;
    let already_loaded = guard.as_mut().and_then(|daemon| {
        (daemon.is_running()
            && daemon.model_path == model_path
            && daemon.lora_path == lora_path
            && (device == "auto" || daemon.device == device))
            .then(|| daemon.device.clone())
    });
    if let Some(actual_device) = already_loaded {
        emit_igload_progress(
            &window,
            format!("Already loaded on {actual_device}"),
            progress.as_ref(),
        );
        return Ok(actual_device);
    }
    *guard = None; // kill any existing daemon

    let python = resolve::find_python().map_err(|e: anyhow::Error| e.to_string())?;
    let script = resolve::find_script("generate_sdxl.py").map_err(|e: anyhow::Error| e.to_string())?;

    emit_igload_progress(&window, "Starting Python…", progress.as_ref());

    let mut cmd = Command::new(python);
    crate::paths::apply_child_env(&mut cmd);
    let mut child = cmd
        .arg(script)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .no_console()
        .spawn()
        .map_err(|e| format!("Failed to spawn Python: {e}"))?;

    let raw_stdin  = child.stdin.take().ok_or("no stdin")?;
    let raw_stdout = child.stdout.take().ok_or("no stdout")?;
    let mut stdin  = BufWriter::new(raw_stdin);
    let mut stdout = BufReader::new(raw_stdout);

    // Send load config
    let cfg = serde_json::json!({
        "model_path": model_path,
        "lora_path":  lora_path,
        "device":     device,
    });
    writeln!(stdin, "{cfg}").map_err(|e| format!("stdin write: {e}"))?;
    stdin.flush().map_err(|e| format!("stdin flush: {e}"))?;

    // Read lines until {"ready": true} — intermediate {"loading_status": "..."} lines are forwarded
    let mut loaded_arch: Option<serde_json::Value> = None;
    let actual_device = loop {
        let mut line = String::new();
        let n = stdout.read_line(&mut line).map_err(|e| format!("read ready: {e}"))?;
        if n == 0 {
            return Err("Python process exited before model was ready".into());
        }
        let trimmed = line.trim();
        if trimmed.is_empty() { continue; }

        let v: serde_json::Value = serde_json::from_str(trimmed)
            .map_err(|e| format!("Bad JSON from daemon: {e} (line: {trimmed})"))?;

        if let Some(err) = v["error"].as_str() {
            return Err(err.to_string());
        }
        if v["ready"].as_bool() == Some(true) {
            loaded_arch = v.get("arch").cloned();
            break v["device"].as_str().unwrap_or("unknown").to_string();
        }
        if let Some(status) = v["loading_status"].as_str() {
            emit_igload_progress(&window, status, progress.as_ref());
        }
    };

    *guard = Some(ImgGenDaemon {
        _child: child,
        stdin,
        stdout,
        model_path,
        lora_path,
        device: actual_device.clone(),
    });

    // Let the UI reset CFG/steps/etc to what this specific model actually wants (e.g.
    // SD3.5 wants cfg≈4.5, not the SDXL-era 7.0 default) instead of carrying over
    // whatever was set for the previously loaded model.
    if let Some(arch) = loaded_arch {
        let _ = window.emit("igload-arch", arch);
    }

    Ok(actual_device)
}

fn emit_igload_progress(window: &WebviewWindow, message: impl Into<String>, progress: Option<&LoadProgress>) {
    let message = message.into();
    if let Some(progress) = progress {
        progress(message.clone());
    }
    let _ = window.emit("igload-progress", message);
}

// ── Internal: generate ────────────────────────────────────────────────────────

fn do_generate(
    arc: DaemonHandle,
    payload: ImgGenPayload,
    window: WebviewWindow,
) -> Result<ImgGenResult, String> {
    let mut guard = arc.lock().map_err(|e| e.to_string())?;
    let daemon = guard.as_mut()
        .ok_or("Model not loaded — click Load Model first")?;

    let req = serde_json::json!({
        "prompt":     payload.prompt,
        "neg_prompt": payload.neg_prompt.unwrap_or_default(),
        "steps":      payload.steps.unwrap_or(20),
        "cfg_scale":  payload.cfg_scale.unwrap_or(7.0_f32),
        "seed":       payload.seed.unwrap_or(42),
        "width":      payload.width.unwrap_or(1024),
        "height":     payload.height.unwrap_or(1024),
        "scheduler":  payload.scheduler.unwrap_or_else(|| "auto".into()),
        "face_detail": payload.face_detail.unwrap_or(true),
        "asset_guard": payload.asset_guard.unwrap_or(true),
        "asset_kind": payload.asset_kind.unwrap_or_else(|| "humanoid".into()),
        "model_path": daemon.model_path.clone(),
    });

    writeln!(daemon.stdin, "{req}").map_err(|e| format!("stdin write: {e}"))?;
    daemon.stdin.flush().map_err(|e| format!("stdin flush: {e}"))?;

    // Read progress lines then final result
    loop {
        let mut line = String::new();
        let n = daemon.stdout.read_line(&mut line)
            .map_err(|e| format!("stdout read: {e}"))?;
        if n == 0 {
            return Err("Image generation daemon exited unexpectedly — reload the model".into());
        }
        let trimmed = line.trim();
        if trimmed.is_empty() { continue; }

        let v: serde_json::Value = match serde_json::from_str(trimmed) {
            Ok(v) => v,
            Err(_) => continue,
        };

        if let Some(b64) = v["base64_png"].as_str() {
            return Ok(ImgGenResult {
                base64_png: b64.to_string(),
                device:     daemon.device.clone(),
                elapsed:    v["elapsed"].as_f64().unwrap_or(0.0),
            });
        }
        if let Some(err) = v["error"].as_str() {
            return Err(err.to_string());
        }
        if v["step"].is_number() {
            let prog = ImgGenProgress {
                step:  v["step"].as_u64().unwrap_or(0) as u32,
                total: v["total"].as_u64().unwrap_or(1) as u32,
            };
            let _ = window.emit("imggen_progress", prog);
        }
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Fast, local, best-effort family label for the model picker/scanner ONLY. This is a
/// coarse display classifier, not the authoritative architecture detector — it never
/// reaches generation (ImgGenPayload carries no `kind` field), so it's safe for this to
/// under-classify as "unknown" but it must never drive generation behavior. The real,
/// authoritative descriptor (family, default CFG/steps, scheduler mode, text-encoder
/// limits, Turbo/v-prediction detection via scheduler config) is computed once in Python
/// at load time — see architecture_of() in generate_sdxl.py — and read from there for
/// anything that actually affects how a model is run. Don't add generation-affecting
/// logic here; extend the Python descriptor instead.
fn detect_kind(path: &Path) -> String {
    if let Ok(s) = std::fs::read_to_string(path.join("model_index.json")) {
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(&s) {
            let cls = v["_class_name"].as_str().unwrap_or("");
            // SD3/3.5 must be checked before the generic "StableDiffusion" contains-check
            // below — "StableDiffusion3Pipeline" contains that substring too and would
            // otherwise be misdetected as plain sd15 (generate_sdxl.py's own loader gets
            // this right already via the same "StableDiffusion3" prefix check).
            if cls.contains("StableDiffusion3") { return "sd3".into(); }
            if cls.contains("XL") { return "sdxl".into(); }
            if cls.contains("StableDiffusion") { return "sd15".into(); }
        }
    }
    "unknown".into()
}

fn scan_diffusers_dir(base: PathBuf, depth: usize, out: &mut Vec<ModelEntry>) {
    let Ok(rd) = std::fs::read_dir(&base) else { return };
    for entry in rd.flatten() {
        let path = entry.path();
        if !path.is_dir() { continue; }
        if path.join("model_index.json").exists() {
            let kind = detect_kind(&path);
            // Only surface loadable image pipelines. Video models (CogVideoX, Wan,
            // …) also carry a model_index.json but have a transformer, not a unet —
            // loading them as a StableDiffusionPipeline blows up ("no unet"). They
            // have their own scanner in video.rs, so skip anything not sd15/sdxl/sd3.
            if kind == "sd15" || kind == "sdxl" || kind == "sd3" {
                out.push(ModelEntry {
                    label: path.file_name().unwrap_or_default().to_string_lossy().to_string(),
                    path:  path.to_string_lossy().to_string(),
                    kind,
                });
            }
        } else if depth > 1 {
            scan_diffusers_dir(path, depth - 1, out);
        }
    }
}

fn scan_safetensors_dirs(dirs: &[PathBuf], kind: &str) -> Vec<ModelEntry> {
    let mut entries = Vec::new();
    for dir in dirs {
        if let Ok(rd) = std::fs::read_dir(dir) {
            for entry in rd.flatten() {
                let path = entry.path();
                if path.extension().map(|e| e == "safetensors").unwrap_or(false) {
                    entries.push(ModelEntry {
                        label: path.file_name().unwrap_or_default().to_string_lossy().to_string(),
                        path:  path.to_string_lossy().to_string(),
                        kind:  kind.to_string(),
                    });
                }
            }
        }
    }
    entries
}

#[cfg(test)]
mod tests {
    use super::detect_kind;
    use std::fs;

    // Regression test: "StableDiffusion3Pipeline" contains the substring "StableDiffusion",
    // so a naive contains-check misdetects SD3/3.5 as plain sd15 — which then makes it
    // invisible to the image-tab scanner (which only allowlists sd15/sdxl/sd3).
    #[test]
    fn detect_kind_distinguishes_sd3_from_sd15() {
        let dir = std::env::temp_dir().join(format!("saient_detect_kind_test_{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();

        fs::write(dir.join("model_index.json"), r#"{"_class_name": "StableDiffusion3Pipeline"}"#).unwrap();
        assert_eq!(detect_kind(&dir), "sd3");

        fs::write(dir.join("model_index.json"), r#"{"_class_name": "StableDiffusionPipeline"}"#).unwrap();
        assert_eq!(detect_kind(&dir), "sd15");

        fs::write(dir.join("model_index.json"), r#"{"_class_name": "StableDiffusionXLPipeline"}"#).unwrap();
        assert_eq!(detect_kind(&dir), "sdxl");

        fs::remove_dir_all(&dir).ok();
    }
}
