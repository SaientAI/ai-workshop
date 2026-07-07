use serde::{Deserialize, Serialize};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::{Arc, Mutex};
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

impl Drop for ImgGenDaemon {
    fn drop(&mut self) {
        let _ = self._child.kill();
    }
}

pub type DaemonHandle = Arc<Mutex<Option<ImgGenDaemon>>>;

pub fn new_daemon_handle() -> DaemonHandle {
    Arc::new(Mutex::new(None))
}

pub(crate) fn loaded_matches(
    daemon: &DaemonHandle,
    model_path: &str,
    lora_path: &str,
    device: &str,
) -> Result<bool, String> {
    let guard = daemon.lock().map_err(|e| e.to_string())?;
    Ok(guard.as_ref().is_some_and(|d| {
        d.model_path == model_path
            && d.lora_path == lora_path
            && (device == "auto" || d.device == device)
    }))
}

pub(crate) fn loaded_model_from_handle(daemon: &DaemonHandle) -> Result<Option<String>, String> {
    Ok(daemon.lock().map_err(|e| e.to_string())?
        .as_ref()
        .map(|d| d.model_path.clone()))
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
    do_load(daemon, window, model_path, lora_path, device)
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
        scan_diffusers_dir(dir, 2, &mut entries);
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
    tokio::task::spawn_blocking(move || do_load(arc, window, model_path, lora_path, device))
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
) -> Result<String, String> {
    let mut guard = arc.lock().map_err(|e| e.to_string())?;
    *guard = None; // kill any existing daemon

    let python = resolve::find_python().map_err(|e: anyhow::Error| e.to_string())?;
    let script = resolve::find_script("generate_sdxl.py").map_err(|e: anyhow::Error| e.to_string())?;

    let _ = window.emit("igload-progress", "Starting Python…");

    let mut child = Command::new(python)
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
            break v["device"].as_str().unwrap_or("unknown").to_string();
        }
        if let Some(status) = v["loading_status"].as_str() {
            let _ = window.emit("igload-progress", status);
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

    Ok(actual_device)
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

fn detect_kind(path: &Path) -> String {
    if let Ok(s) = std::fs::read_to_string(path.join("model_index.json")) {
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(&s) {
            let cls = v["_class_name"].as_str().unwrap_or("");
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
            // have their own scanner in video.rs, so skip anything not sd15/sdxl.
            if kind == "sd15" || kind == "sdxl" {
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
