//! video.rs — text-to-video generation via a hot Python daemon.
//!
//! Mirrors imggen.rs: loads a diffusers video pipeline (Wan) once into a
//! persistent Python process, then generates clips over stdin/stdout JSON lines.
//! The model is heavy (~27 GB) so keeping it resident across clips is essential.

use serde::{Deserialize, Serialize};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::PathBuf;
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::{Arc, Mutex};
use tauri::{Emitter, State, WebviewWindow};
use crate::resolve;

// ── Types ───────────────────────────────────────────────────────────────────

#[derive(Serialize, Clone)]
pub struct VideoModelEntry {
    pub path: String,
    pub label: String,
    pub pipeline: String,
}

#[derive(Deserialize)]
pub struct VideoPayload {
    pub prompt: String,
    pub neg_prompt: Option<String>,
    pub model_path: String,
    pub num_frames: Option<u32>,
    pub steps: Option<u32>,
    pub cfg_scale: Option<f32>,
    pub width: Option<u32>,
    pub height: Option<u32>,
    pub fps: Option<u32>,
    pub seed: Option<i64>,
}

#[derive(Serialize, Clone)]
pub struct VideoProgress { pub step: u32, pub total: u32 }

#[derive(Serialize)]
pub struct VideoResult {
    pub base64_mp4: String,
    pub frames: u32,
    pub elapsed: f64,
}

#[derive(Deserialize)]
pub struct EnhancePayload {
    pub video_b64: String,
    pub fps: u32,
    pub stages: Vec<String>,        // ordered: subset of ["refine","upscale","interpolate"]
    pub model_path: String,         // for the refine (Wan v2v) stage
    pub prompt: String,
    pub neg_prompt: Option<String>,
    pub cfg_scale: Option<f32>,
    pub refine_strength: Option<f32>,
    pub refine_steps: Option<u32>,
    pub interp_factor: Option<u32>,
}

#[derive(Serialize)]
pub struct EnhanceResult {
    pub enhanced_b64: String,
    pub frames: u32,
    pub width: u32,
    pub height: u32,
    pub elapsed: f64,
}

// ── Hot daemon ──────────────────────────────────────────────────────────────

pub struct VideoDaemon {
    _child: Child,
    stdin: BufWriter<ChildStdin>,
    stdout: BufReader<ChildStdout>,
    pub model_path: String,
}
impl Drop for VideoDaemon {
    fn drop(&mut self) { let _ = self._child.kill(); }
}
pub type VideoHandle = Arc<Mutex<Option<VideoDaemon>>>;
pub fn new_video_handle() -> VideoHandle { Arc::new(Mutex::new(None)) }

// ── Scan for video models (diffusers dirs with a video pipeline) ──────────────

#[tauri::command]
pub fn video_scan_models() -> Vec<VideoModelEntry> {
    let mut out = Vec::new();
    for dir in resolve::model_scan_dirs() {
        scan(dir, 3, &mut out);
    }
    out.sort_by(|a, b| a.label.to_lowercase().cmp(&b.label.to_lowercase()));
    out.dedup_by(|a, b| a.path == b.path);
    out
}

/// True if a diffusers `_class_name` is a video pipeline we can drive.
fn is_video_pipeline(cls: &str) -> bool {
    ["Wan", "LTX", "CogVideo", "HunyuanVideo", "Mochi", "SanaVideo", "Video"]
        .iter().any(|k| cls.contains(k))
}

fn scan(base: PathBuf, depth: usize, out: &mut Vec<VideoModelEntry>) {
    let Ok(rd) = std::fs::read_dir(&base) else { return };
    for entry in rd.flatten() {
        let path = entry.path();
        if !path.is_dir() { continue; }
        let idx = path.join("model_index.json");
        if idx.exists() {
            if let Ok(s) = std::fs::read_to_string(&idx) {
                if let Ok(v) = serde_json::from_str::<serde_json::Value>(&s) {
                    let cls = v["_class_name"].as_str().unwrap_or("");
                    if is_video_pipeline(cls) {
                        out.push(VideoModelEntry {
                            label: path.file_name().unwrap_or_default().to_string_lossy().to_string(),
                            path: path.to_string_lossy().to_string(),
                            pipeline: cls.to_string(),
                        });
                    }
                }
            }
        } else if depth > 1 {
            scan(path, depth - 1, out);
        }
    }
}

// ── Lifecycle ─────────────────────────────────────────────────────────────────

#[tauri::command]
pub async fn video_load(
    handle: State<'_, VideoHandle>,
    window: WebviewWindow,
    model_path: String,
    lora_path: Option<String>,
    lora_strength: Option<f32>,
) -> Result<String, String> {
    let arc = handle.inner().clone();
    tokio::task::spawn_blocking(move || do_load(arc, window, model_path, lora_path, lora_strength))
        .await
        .map_err(|e| format!("task join: {e}"))?
}

#[derive(Serialize, Clone)]
pub struct LoraEntry { pub path: String, pub label: String }

/// Scan for LoRA .safetensors — the managed loras dir + any `loras/` folders
/// alongside the video models.
#[tauri::command]
pub fn video_scan_loras() -> Vec<LoraEntry> {
    let mut out = Vec::new();
    let mut roots: Vec<PathBuf> = Vec::new();
    if let Ok(home) = std::env::var("HOME") {
        roots.push(PathBuf::from(format!("{home}/.config/ai-workshop/loras")));
    }
    for d in resolve::model_scan_dirs() {
        roots.push(d.join("loras"));
        roots.push(d);
    }
    for root in roots {
        scan_loras(root, 2, &mut out);
    }
    out.sort_by(|a, b| a.label.to_lowercase().cmp(&b.label.to_lowercase()));
    out.dedup_by(|a, b| a.path == b.path);
    out
}

fn scan_loras(base: PathBuf, depth: usize, out: &mut Vec<LoraEntry>) {
    let Ok(rd) = std::fs::read_dir(&base) else { return };
    for entry in rd.flatten() {
        let path = entry.path();
        if path.is_dir() {
            if depth > 0 { scan_loras(path, depth - 1, out); }
        } else if path.extension().and_then(|e| e.to_str()) == Some("safetensors") {
            // label = parent-dir name if file is generic (model.safetensors), else file stem
            let stem = path.file_stem().unwrap_or_default().to_string_lossy().to_string();
            let label = if stem == "model" || stem == "pytorch_lora_weights" {
                path.parent().and_then(|p| p.file_name()).map(|n| n.to_string_lossy().to_string()).unwrap_or(stem)
            } else { stem };
            out.push(LoraEntry { label, path: path.to_string_lossy().to_string() });
        }
    }
}

#[tauri::command]
pub async fn video_unload(handle: State<'_, VideoHandle>) -> Result<(), String> {
    *handle.lock().map_err(|e| e.to_string())? = None;
    Ok(())
}

#[tauri::command]
pub fn video_loaded_model(handle: State<'_, VideoHandle>) -> Result<Option<String>, String> {
    Ok(handle.lock().map_err(|e| e.to_string())?.as_ref().map(|d| d.model_path.clone()))
}

#[tauri::command]
pub async fn video_generate(
    handle: State<'_, VideoHandle>,
    payload: VideoPayload,
    window: WebviewWindow,
) -> Result<VideoResult, String> {
    let arc = handle.inner().clone();
    tokio::task::spawn_blocking(move || do_generate(arc, payload, window))
        .await
        .map_err(|e| format!("task join: {e}"))?
}

/// Quality pass — drops the generator daemon first (frees ALL VRAM), then runs
/// the one-shot enhancer (refine / upscale / interpolate) on the produced clip.
#[tauri::command]
pub async fn video_enhance(
    handle: State<'_, VideoHandle>,
    payload: EnhancePayload,
    window: WebviewWindow,
) -> Result<EnhanceResult, String> {
    let arc = handle.inner().clone();
    tokio::task::spawn_blocking(move || do_enhance(arc, payload, window))
        .await
        .map_err(|e| format!("task join: {e}"))?
}

// ── Internal ────────────────────────────────────────────────────────────────

fn do_load(arc: VideoHandle, window: WebviewWindow, model_path: String,
           lora_path: Option<String>, lora_strength: Option<f32>) -> Result<String, String> {
    let mut guard = arc.lock().map_err(|e| e.to_string())?;
    *guard = None; // kill any existing daemon (frees VRAM)

    let python = resolve::find_python().map_err(|e: anyhow::Error| e.to_string())?;
    let script = resolve::find_script("generate_video.py").map_err(|e: anyhow::Error| e.to_string())?;

    let _ = window.emit("vidload-progress", "Starting Python…");

    let mut child = Command::new(python)
        .arg(script)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()
        .map_err(|e| format!("Failed to spawn Python: {e}"))?;

    let mut stdin = BufWriter::new(child.stdin.take().ok_or("no stdin")?);
    let mut stdout = BufReader::new(child.stdout.take().ok_or("no stdout")?);

    let cfg = serde_json::json!({
        "model_path": model_path, "device": "auto",
        "lora_path": lora_path.unwrap_or_default(),
        "lora_strength": lora_strength.unwrap_or(1.0_f32),
    });
    writeln!(stdin, "{cfg}").map_err(|e| format!("stdin write: {e}"))?;
    stdin.flush().map_err(|e| format!("stdin flush: {e}"))?;

    let device = loop {
        let mut line = String::new();
        let n = stdout.read_line(&mut line).map_err(|e| format!("read ready: {e}"))?;
        if n == 0 { return Err("Python exited before the model was ready".into()); }
        let t = line.trim();
        if t.is_empty() { continue; }
        let v: serde_json::Value = serde_json::from_str(t)
            .map_err(|e| format!("bad JSON from daemon: {e} (line: {t})"))?;
        if let Some(err) = v["error"].as_str() { return Err(err.to_string()); }
        if v["ready"].as_bool() == Some(true) {
            break v["device"].as_str().unwrap_or("unknown").to_string();
        }
        if let Some(s) = v["loading_status"].as_str() {
            let _ = window.emit("vidload-progress", s);
        }
    };

    *guard = Some(VideoDaemon { _child: child, stdin, stdout, model_path });
    Ok(device)
}

fn do_generate(arc: VideoHandle, payload: VideoPayload, window: WebviewWindow) -> Result<VideoResult, String> {
    let mut guard = arc.lock().map_err(|e| e.to_string())?;
    let daemon = guard.as_mut().ok_or("No video model loaded — click Load Model first")?;

    let req = serde_json::json!({
        "prompt":     payload.prompt,
        "neg_prompt": payload.neg_prompt.unwrap_or_default(),
        "num_frames": payload.num_frames.unwrap_or(49),
        "steps":      payload.steps.unwrap_or(30),
        "cfg_scale":  payload.cfg_scale.unwrap_or(6.0_f32),
        "width":      payload.width.unwrap_or(832),
        "height":     payload.height.unwrap_or(480),
        "fps":        payload.fps.unwrap_or(16),
        "seed":       payload.seed.unwrap_or(-1),
    });
    writeln!(daemon.stdin, "{req}").map_err(|e| format!("stdin write: {e}"))?;
    daemon.stdin.flush().map_err(|e| format!("stdin flush: {e}"))?;

    loop {
        let mut line = String::new();
        let n = daemon.stdout.read_line(&mut line).map_err(|e| format!("stdout read: {e}"))?;
        if n == 0 { return Err("Video daemon exited unexpectedly — reload the model".into()); }
        let t = line.trim();
        if t.is_empty() { continue; }
        let v: serde_json::Value = match serde_json::from_str(t) { Ok(v) => v, Err(_) => continue };

        if let Some(b64) = v["base64_mp4"].as_str() {
            return Ok(VideoResult {
                base64_mp4: b64.to_string(),
                frames: v["frames"].as_u64().unwrap_or(0) as u32,
                elapsed: v["elapsed"].as_f64().unwrap_or(0.0),
            });
        }
        if let Some(err) = v["error"].as_str() { return Err(err.to_string()); }
        // Staged generate loads the text encoder on first use, then denoises —
        // surface those phase messages so the UI isn't stuck at 0% for ~1 min.
        if let Some(s) = v["loading_status"].as_str() {
            let _ = window.emit("vidload-progress", s);
        }
        if v["step"].is_number() {
            let _ = window.emit("video_progress", VideoProgress {
                step: v["step"].as_u64().unwrap_or(0) as u32,
                total: v["total"].as_u64().unwrap_or(1) as u32,
            });
        }
    }
}

fn do_enhance(arc: VideoHandle, payload: EnhancePayload, window: WebviewWindow) -> Result<EnhanceResult, String> {
    // Free the whole GPU first: drop the resident generator daemon. The quality
    // pass then owns all VRAM (the user's "let it stop generating, free the load").
    {
        let mut guard = arc.lock().map_err(|e| e.to_string())?;
        *guard = None;
    }
    let _ = window.emit("vidload-progress", "freeing generator — handing the GPU to the quality pass…");

    let python = resolve::find_python().map_err(|e: anyhow::Error| e.to_string())?;
    let script = resolve::find_script("enhance_video.py").map_err(|e: anyhow::Error| e.to_string())?;

    // Default upscale weights live in the managed config dir.
    let upscale_model = std::env::var("HOME")
        .map(|h| format!("{h}/.config/ai-workshop/upscale/RealESRGAN_x2plus.pth"))
        .unwrap_or_default();

    let mut child = Command::new(python)
        .arg(script)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()
        .map_err(|e| format!("Failed to spawn enhancer: {e}"))?;

    let req = serde_json::json!({
        "video_b64":       payload.video_b64,
        "fps":             payload.fps,
        "stages":          payload.stages,
        "model_path":      payload.model_path,
        "prompt":          payload.prompt,
        "neg_prompt":      payload.neg_prompt.unwrap_or_default(),
        "cfg_scale":       payload.cfg_scale.unwrap_or(6.0_f32),
        "refine_strength": payload.refine_strength.unwrap_or(0.35_f32),
        "refine_steps":    payload.refine_steps.unwrap_or(20),
        "interp_factor":   payload.interp_factor.unwrap_or(2),
        "upscale_model":   upscale_model,
    });
    {
        let mut stdin = BufWriter::new(child.stdin.take().ok_or("no stdin")?);
        writeln!(stdin, "{req}").map_err(|e| format!("stdin write: {e}"))?;
        stdin.flush().map_err(|e| format!("stdin flush: {e}"))?;
        // drop stdin here → EOF for the one-shot script
    }
    let mut stdout = BufReader::new(child.stdout.take().ok_or("no stdout")?);

    loop {
        let mut line = String::new();
        let n = stdout.read_line(&mut line).map_err(|e| format!("stdout read: {e}"))?;
        if n == 0 { return Err("Enhancer exited unexpectedly".into()); }
        let t = line.trim();
        if t.is_empty() { continue; }
        let v: serde_json::Value = match serde_json::from_str(t) { Ok(v) => v, Err(_) => continue };

        if let Some(b64) = v["enhanced_b64"].as_str() {
            return Ok(EnhanceResult {
                enhanced_b64: b64.to_string(),
                frames: v["frames"].as_u64().unwrap_or(0) as u32,
                width:  v["width"].as_u64().unwrap_or(0) as u32,
                height: v["height"].as_u64().unwrap_or(0) as u32,
                elapsed: v["elapsed"].as_f64().unwrap_or(0.0),
            });
        }
        if let Some(err) = v["error"].as_str() { return Err(err.to_string()); }
        if let Some(s) = v["loading_status"].as_str() {
            let _ = window.emit("vidload-progress", s);
        }
        if v["step"].is_number() {
            let _ = window.emit("video_progress", VideoProgress {
                step: v["step"].as_u64().unwrap_or(0) as u32,
                total: v["total"].as_u64().unwrap_or(1) as u32,
            });
        }
    }
}
