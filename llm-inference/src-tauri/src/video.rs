//! video.rs — text-to-video generation via a hot Python daemon.
//!
//! Mirrors imggen.rs: loads a diffusers video pipeline (Wan) once into a
//! persistent Python process, then generates clips over stdin/stdout JSON lines.
//! The model is heavy (~27 GB) so keeping it resident across clips is essential.

use serde::{Deserialize, Serialize};
use std::io::{BufRead, BufReader, BufWriter, Read, Write};
use std::path::PathBuf;
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::{Arc, Mutex};
use tauri::{Emitter, State, WebviewWindow};
use crate::resolve;
use crate::resolve::NoConsole;

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
    pub scheduler: Option<String>,
    pub shift: Option<f32>,
    pub lora_profile: Option<String>,
    pub lora_strength_high: Option<f32>,
    pub lora_strength_low: Option<f32>,
    pub lora_split_step: Option<u32>,
    pub width: Option<u32>,
    pub height: Option<u32>,
    pub fps: Option<u32>,
    pub seed: Option<i64>,
    pub image_b64: Option<String>,   // optional still → image-to-video
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
    pub lora_path: String,
    pub lora_strength: f32,
    pub frames_hint: u32,
    pub precision: String,
}
impl Drop for VideoDaemon {
    fn drop(&mut self) {
        // Graceful shutdown: ask the daemon to quit so it tears down its (up to ~10 GB) CUDA
        // context IN-PROCESS and the driver hands VRAM back cleanly. A SIGKILL of a process
        // holding that much VRAM makes the driver reclaim the context abruptly, which on a
        // single GPU that also drives the display spikes + freezes the desktop (the "Clean
        // VRAM" lock-up). Send a quit line, wait briefly, then SIGKILL only as a fallback.
        // (The Unload button is disabled mid-generation, so the daemon is idle on stdin here
        // and reads the quit immediately.)
        use std::time::{Duration, Instant};
        let _ = self.stdin.write_all(b"{\"cmd\":\"quit\"}\n");
        let _ = self.stdin.flush();
        let deadline = Instant::now() + Duration::from_secs(6);
        loop {
            if matches!(self._child.try_wait(), Ok(Some(_))) { return; } // exited cleanly
            if Instant::now() >= deadline { break; }
            std::thread::sleep(Duration::from_millis(100));
        }
        let _ = self._child.kill();   // fallback: force it
        let _ = self._child.wait();
    }
}
pub type VideoHandle = Arc<Mutex<Option<VideoDaemon>>>;
pub fn new_video_handle() -> VideoHandle { Arc::new(Mutex::new(None)) }

pub(crate) fn loaded_matches(
    handle: &VideoHandle,
    model_path: &str,
    lora_path: &str,
    lora_strength: f32,
    frames_hint: u32,
    precision: &str,
) -> Result<bool, String> {
    let guard = handle.lock().map_err(|e| e.to_string())?;
    Ok(guard.as_ref().is_some_and(|d| {
        d.model_path == model_path
            && d.lora_path == lora_path
            && (d.lora_strength - lora_strength).abs() < f32::EPSILON
            && d.frames_hint == frames_hint
            && d.precision == precision
    }))
}

pub(crate) fn loaded_model_from_handle(handle: &VideoHandle) -> Result<Option<String>, String> {
    Ok(handle.lock().map_err(|e| e.to_string())?
        .as_ref()
        .map(|d| d.model_path.clone()))
}

pub(crate) fn load_blocking(
    handle: VideoHandle,
    img: crate::imggen::DaemonHandle,
    window: WebviewWindow,
    model_path: String,
    lora_path: Option<String>,
    lora_strength: Option<f32>,
    frames: Option<u32>,
    precision: Option<String>,
) -> Result<String, String> {
    if let Ok(mut g) = img.lock() { *g = None; }
    do_load(handle, window, model_path, lora_path, lora_strength, frames, precision)
}

pub(crate) fn generate_blocking(
    handle: VideoHandle,
    payload: VideoPayload,
    window: WebviewWindow,
) -> Result<VideoResult, String> {
    do_generate(handle, payload, window)
}

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
    img: State<'_, crate::imggen::DaemonHandle>,
    window: WebviewWindow,
    model_path: String,
    lora_path: Option<String>,
    lora_strength: Option<f32>,
    frames: Option<u32>,
    precision: Option<String>,
) -> Result<String, String> {
    // A 16 GB card can't hold an image model + a video model at once. The image (SDXL)
    // daemon lives on its own screen and stays resident in the background; free it here so
    // loading a video model doesn't OOM on top of ~6–7 GB the user can't see is in use.
    if let Ok(mut g) = img.lock() { *g = None; }
    let arc = handle.inner().clone();
    tokio::task::spawn_blocking(move || do_load(arc, window, model_path, lora_path, lora_strength, frames, precision))
        .await
        .map_err(|e| format!("task join: {e}"))?
}

#[derive(Serialize, Clone)]
pub struct LoraEntry { pub path: String, pub label: String }

/// Scan for Wan/video LoRA .safetensors. Keep this deliberately narrower than the
/// image LoRA/checkpoint scan: a full SDXL checkpoint is also a .safetensors file,
/// and offering one here makes the Wan loader try to apply a whole model as an
/// adapter.
#[tauri::command]
pub fn video_scan_loras() -> Vec<LoraEntry> {
    let mut out = Vec::new();
    let mut roots: Vec<PathBuf> = Vec::new();
    if let Ok(home) = std::env::var("HOME") {
        roots.push(PathBuf::from(format!("{home}/.config/saient/loras")));
        roots.push(PathBuf::from(format!("{home}/Saient/models/video-loras")));
        roots.push(PathBuf::from(format!("{home}/Saient/models/wan/loras")));
    }
    for d in resolve::model_scan_dirs() {
        roots.push(d.join("loras"));
        roots.push(d.join("wan").join("loras"));
        roots.push(d.join("wan").join("_distill_loras").join("loras"));
        collect_lora_dirs(d, 4, &mut roots);
    }
    roots.sort();
    roots.dedup();
    for root in roots {
        scan_loras(root, 2, &mut out);
    }
    out.sort_by(|a, b| a.label.to_lowercase().cmp(&b.label.to_lowercase()));
    out.dedup_by(|a, b| a.path == b.path);
    out
}

fn collect_lora_dirs(base: PathBuf, depth: usize, roots: &mut Vec<PathBuf>) {
    let Ok(rd) = std::fs::read_dir(&base) else { return };
    for entry in rd.flatten() {
        let path = entry.path();
        if !path.is_dir() { continue; }
        let name = path.file_name().unwrap_or_default().to_string_lossy().to_lowercase();
        if name.contains("lora") {
            roots.push(path.clone());
        }
        if depth > 0 {
            collect_lora_dirs(path, depth - 1, roots);
        }
    }
}

fn scan_loras(base: PathBuf, depth: usize, out: &mut Vec<LoraEntry>) {
    let Ok(rd) = std::fs::read_dir(&base) else { return };
    for entry in rd.flatten() {
        let path = entry.path();
        if path.is_dir() {
            if depth > 0 { scan_loras(path, depth - 1, out); }
        } else if path.extension().and_then(|e| e.to_str()) == Some("safetensors")
            && is_probably_wan_lora(&path)
        {
            // label = parent-dir name if file is generic (model.safetensors), else file stem
            let stem = path.file_stem().unwrap_or_default().to_string_lossy().to_string();
            let label = if stem == "model" || stem == "pytorch_lora_weights" {
                path.parent().and_then(|p| p.file_name()).map(|n| n.to_string_lossy().to_string()).unwrap_or(stem)
            } else { stem };
            out.push(LoraEntry { label, path: path.to_string_lossy().to_string() });
        }
    }
}

fn is_probably_wan_lora(path: &PathBuf) -> bool {
    let Ok(mut f) = std::fs::File::open(path) else { return false };
    let mut len_buf = [0_u8; 8];
    if f.read_exact(&mut len_buf).is_err() { return false; }
    let header_len = u64::from_le_bytes(len_buf);
    if header_len == 0 || header_len > 16 * 1024 * 1024 {
        return false;
    }
    let mut header = vec![0_u8; header_len as usize];
    if f.read_exact(&mut header).is_err() { return false; }
    let Ok(json) = serde_json::from_slice::<serde_json::Value>(&header) else { return false };
    let Some(obj) = json.as_object() else { return false };

    obj.keys().filter(|k| k.as_str() != "__metadata__").take(4096).any(|key| {
        let k = key.to_lowercase();
        k.contains("lora")
            && (k.starts_with("blocks.")
                || k.starts_with("diffusion_model.blocks.")
                || k.starts_with("transformer.blocks.")
                || k.starts_with("lora_unet_blocks_"))
    })
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

/// Poll free VRAM until it reaches `want_mib` or `timeout_ms` elapses; returns the last
/// reading (None only if nvidia-smi isn't available — a CPU box, nothing to wait for).
/// Used after evicting other GPU models so we don't spawn the video daemon onto memory the
/// driver hasn't reclaimed yet. Advisory — the caller proceeds either way.
fn wait_for_vram(window: &WebviewWindow, want_mib: u64, timeout_ms: u64) -> Option<u64> {
    use std::time::{Duration, Instant};
    let start = Instant::now();
    let mut last = None;
    loop {
        match crate::engine::gpu_free_mib() {
            Some(free) => {
                last = Some(free);
                if free >= want_mib {
                    return Some(free);
                }
                let _ = window.emit(
                    "vidload-progress",
                    format!("waiting for GPU memory… {:.1} GB free", free as f64 / 1024.0),
                );
            }
            None => return last,
        }
        if start.elapsed() >= Duration::from_millis(timeout_ms) {
            return last;
        }
        std::thread::sleep(Duration::from_millis(250));
    }
}

fn do_load(arc: VideoHandle, window: WebviewWindow, model_path: String,
           lora_path: Option<String>, lora_strength: Option<f32>, frames: Option<u32>,
           precision: Option<String>) -> Result<String, String> {
    let mut guard = arc.lock().map_err(|e| e.to_string())?;
    *guard = None; // kill any existing daemon (frees VRAM)

    // A 16 GB card holds exactly ONE model at a time. Native 720p denoise peaks at
    // ~13.9 GB (measured on the 5B), so a resident chat server (tinyq4) — or any server we
    // spawned — means an instant CUDA OOM the moment the user hits Generate. That is THE
    // cause of "720p never stable": the standalone daemon fits, but the app left the chat
    // model on the GPU underneath it. Kill our background servers now (the image daemon was
    // already dropped in video_load and the frontend unloads chat before calling us), then
    // WAIT for the driver to actually hand the VRAM back — kill() returns long before the
    // memory is reclaimed, so spawning immediately would race onto not-yet-freed VRAM.
    crate::engine::kill_our_stale_servers();
    if crate::engine::gpu_available() {
        let _ = window.emit("vidload-progress", "clearing GPU memory from other models…");
        let free = wait_for_vram(&window, 13_500, 8_000);
        if let Some(mib) = free {
            if mib < 12_000 {
                // Couldn't get the card clean — something we don't manage (an external
                // tinyq4/Saient server, another GPU app) is holding it. Don't swap-die or
                // OOM cryptically: tell the user plainly. (We still proceed; a 480p gen may
                // fit, and the daemon turns any residual OOM into a clean error, not a crash.)
                let _ = window.emit("vidload-progress", format!(
                    "⚠ only {:.1} GB GPU free — close other GPU apps/servers or HD may run out of memory",
                    mib as f64 / 1024.0));
            }
        }
    }

    let python = resolve::find_python().map_err(|e: anyhow::Error| e.to_string())?;
    // Pick the daemon by model family (CogVideoX is a different architecture).
    let model_index = std::fs::read_to_string(PathBuf::from(&model_path).join("model_index.json"))
        .unwrap_or_default();
    let script_name = if model_index.contains("CogVideo") {
        "generate_cogvideo.py"
    } else if model_index.contains("ImageToVideo") {
        "generate_wan_i2v.py"   // Wan2.1-I2V: native image-to-video (CLIP image encoder)
    } else {
        "generate_video.py"
    };
    let script = resolve::find_script(script_name).map_err(|e: anyhow::Error| e.to_string())?;

    let _ = window.emit("vidload-progress", "Starting Python…");

    let mut child = Command::new(python)
        .arg(script)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .no_console()
        .spawn()
        .map_err(|e| format!("Failed to spawn Python: {e}"))?;

    let mut stdin = BufWriter::new(child.stdin.take().ok_or("no stdin")?);
    let mut stdout = BufReader::new(child.stdout.take().ok_or("no stdout")?);

    let lora_path_value = lora_path.unwrap_or_default();
    let lora_strength_value = lora_strength.unwrap_or(1.0_f32);
    let frames_hint_value = frames.unwrap_or(49);
    let precision_value = precision.unwrap_or_else(|| "fast".into());

    let cfg = serde_json::json!({
        "model_path": model_path, "device": "auto",
        "lora_path": lora_path_value.clone(),
        "lora_strength": lora_strength_value,
        "frames_hint": frames_hint_value,
        "precision": precision_value.clone(),
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

    *guard = Some(VideoDaemon {
        _child: child,
        stdin,
        stdout,
        model_path,
        lora_path: lora_path_value,
        lora_strength: lora_strength_value,
        frames_hint: frames_hint_value,
        precision: precision_value,
    });
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
        "scheduler":  payload.scheduler.unwrap_or_else(|| "auto".into()),
        "shift":      payload.shift.unwrap_or(5.0_f32),
        "lora_profile": payload.lora_profile.unwrap_or_else(|| "single".into()),
        "lora_strength_high": payload.lora_strength_high.unwrap_or(1.0_f32),
        "lora_strength_low":  payload.lora_strength_low.unwrap_or(1.0_f32),
        "lora_split_step":    payload.lora_split_step.unwrap_or(4),
        "width":      payload.width.unwrap_or(832),
        "height":     payload.height.unwrap_or(480),
        "fps":        payload.fps.unwrap_or(16),
        "seed":       payload.seed.unwrap_or(-1),
        "image_b64":  payload.image_b64.unwrap_or_default(),
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

    // Same one-GPU rule as load: the enhancer loads its OWN models (refine transformer +
    // RealESRGAN), so a resident chat server left over from a chat session would OOM the
    // pass. Evict it and wait for the VRAM (the video daemon was just dropped above).
    crate::engine::kill_our_stale_servers();
    if crate::engine::gpu_available() {
        wait_for_vram(&window, 13_500, 8_000);
    }

    let python = resolve::find_python().map_err(|e: anyhow::Error| e.to_string())?;
    let script = resolve::find_script("enhance_video.py").map_err(|e: anyhow::Error| e.to_string())?;

    // Default upscale weights live in the managed config dir.
    let upscale_model = std::env::var("HOME")
        .map(|h| format!("{h}/.config/saient/upscale/RealESRGAN_x2plus.pth"))
        .unwrap_or_default();

    let mut child = Command::new(python)
        .arg(script)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .no_console()
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
