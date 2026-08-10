use base64::Engine as _;
use argon2::password_hash::rand_core::{OsRng, RngCore};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::collections::HashMap;
use std::io::{Read, Write};
use std::net::{IpAddr, Ipv4Addr, TcpListener, TcpStream, UdpSocket};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock, RwLock};
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tauri::{AppHandle, Emitter, Manager, WebviewWindow};

use crate::{imggen, video};

const PORT: u16 = 18788;
const MAX_HEADER_BYTES: usize = 32 * 1024;
const MAX_BODY_BYTES: usize = 96 * 1024 * 1024;
const PAIRING_VERSION: u8 = 2;
const PAIRING_TOKEN_BYTES: usize = 32;

static PAIRING_TOKEN: OnceLock<RwLock<String>> = OnceLock::new();

#[derive(Serialize, Deserialize)]
struct StoredPhoneBinding {
    version: u8,
    token: String,
    created_ms: u64,
}

#[derive(serde::Serialize)]
pub struct PairingInfo {
    pub name: String,
    pub port: u16,
    pub url: String,
    pub local_url: String,
    pub payload: serde_json::Value,
}

struct RemoteContext {
    app: AppHandle,
    img: imggen::DaemonHandle,
    vid: video::VideoHandle,
    op_lock: Mutex<()>,
    jobs: Mutex<HashMap<String, RemoteJob>>,
    video_results: Mutex<HashMap<String, RemoteVideoResult>>,
    loaded: Mutex<RemoteLoadedModels>,
    job_seq: AtomicU64,
}

#[derive(Clone, Default)]
struct RemoteLoadedModels {
    image: Option<String>,
    video: Option<String>,
    video_lora: Option<String>,
}

struct HttpRequest {
    method: String,
    path: String,
    authorization: Option<String>,
    body: Vec<u8>,
}

struct HttpResponse {
    status: u16,
    body: Vec<u8>,
    content_type: &'static str,
}

struct RemoteVideoResult {
    bytes: Vec<u8>,
}

#[derive(Serialize, Clone)]
struct RemotePreview {
    base64_jpeg: String,
    step: u32,
    total: u32,
    frames: Vec<u32>,
    decode_seconds: Option<f64>,
}

#[derive(Serialize, Clone)]
struct RemoteJob {
    id: String,
    target: String,
    action: String,
    status: String,
    message: String,
    logs: Vec<String>,
    started_ms: u64,
    updated_ms: u64,
    finished_ms: Option<u64>,
    model_path: Option<String>,
    device: Option<String>,
    error: Option<String>,
    progress_step: Option<u32>,
    progress_total: Option<u32>,
    step_seconds: Option<f64>,
    elapsed_seconds: Option<f64>,
    preview: Option<RemotePreview>,
    result_ready: bool,
    frames: Option<u32>,
    width: Option<u32>,
    height: Option<u32>,
    elapsed: Option<f64>,
    completed_stages: Vec<String>,
    failed_stages: Vec<String>,
}

#[derive(Deserialize)]
struct ImageLoadRequest {
    model_path: String,
    lora_path: Option<String>,
    device: Option<String>,
}

#[derive(Deserialize)]
struct ImageGenerateRequest {
    prompt: String,
    neg_prompt: Option<String>,
    model_path: Option<String>,
    lora_path: Option<String>,
    steps: Option<u32>,
    cfg_scale: Option<f32>,
    seed: Option<i64>,
    width: Option<u32>,
    height: Option<u32>,
    device: Option<String>,
    scheduler: Option<String>,
    face_detail: Option<bool>,
    asset_guard: Option<bool>,
    asset_kind: Option<String>,
}

#[derive(Deserialize)]
struct VideoLoadRequest {
    model_path: String,
    lora_path: Option<String>,
    lora_strength: Option<f32>,
    frames: Option<u32>,
    precision: Option<String>,
}

#[derive(Deserialize)]
struct VideoGenerateRequest {
    prompt: String,
    neg_prompt: Option<String>,
    model_path: Option<String>,
    num_frames: Option<u32>,
    steps: Option<u32>,
    cfg_scale: Option<f32>,
    scheduler: Option<String>,
    shift: Option<f32>,
    lora_profile: Option<String>,
    lora_strength_high: Option<f32>,
    lora_strength_low: Option<f32>,
    lora_split_step: Option<u32>,
    width: Option<u32>,
    height: Option<u32>,
    fps: Option<u32>,
    seed: Option<i64>,
    image_b64: Option<String>,
    previous_video_b64: Option<String>,
    force_seam_blend: Option<bool>,
    low_vram: Option<bool>,
    denoise_cache: Option<String>,
    cache_threshold: Option<f32>,
    preview: Option<bool>,
    preview_every: Option<u32>,
    preview_max_width: Option<u32>,
    lora_path: Option<String>,
    lora_strength: Option<f32>,
    precision: Option<String>,
}

#[derive(Deserialize)]
struct VideoPolishRequest {
    source_job_id: String,
    fps: Option<u32>,
    interp_factor: Option<u32>,
}

fn video_payload_from_request(
    req: VideoGenerateRequest,
    model_path: String,
) -> video::VideoPayload {
    video::VideoPayload {
        prompt: req.prompt,
        neg_prompt: req.neg_prompt,
        model_path,
        num_frames: req.num_frames,
        steps: req.steps,
        cfg_scale: req.cfg_scale,
        scheduler: req.scheduler,
        shift: req.shift,
        lora_profile: req.lora_profile,
        lora_strength_high: req.lora_strength_high,
        lora_strength_low: req.lora_strength_low,
        lora_split_step: req.lora_split_step,
        width: req.width,
        height: req.height,
        fps: req.fps,
        seed: req.seed,
        image_b64: req.image_b64,
        previous_video_b64: req.previous_video_b64,
        force_seam_blend: req.force_seam_blend,
        low_vram: req.low_vram,
        block_offload: None,
        denoise_cache: req.denoise_cache,
        cache_threshold: req.cache_threshold,
        preview: req.preview,
        preview_every: req.preview_every,
        preview_max_width: req.preview_max_width,
    }
}

#[derive(Deserialize)]
struct UnloadRequest {
    target: Option<String>,
}

pub fn start(app: AppHandle, img: imggen::DaemonHandle, vid: video::VideoHandle) {
    std::thread::spawn(move || {
        if let Err(error) = current_pairing_token() {
            eprintln!("Saient remote API could not initialise phone binding: {error}");
            return;
        }
        let listener = match TcpListener::bind(("127.0.0.1", PORT)) {
            Ok(listener) => listener,
            Err(e) => {
                eprintln!("Saient remote API failed to bind on port {PORT}: {e}");
                return;
            }
        };

        let local_url = format!("http://127.0.0.1:{PORT}");
        let lan_url = lan_ip()
            .map(|ip| format!("http://{ip}:{PORT}"))
            .unwrap_or_else(|| format!("http://<desktop-lan-ip>:{PORT}"));
        eprintln!("Saient remote API listening: {local_url} / {lan_url}");

        let ctx = Arc::new(RemoteContext {
            app,
            img,
            vid,
            op_lock: Mutex::new(()),
            jobs: Mutex::new(HashMap::new()),
            video_results: Mutex::new(HashMap::new()),
            loaded: Mutex::new(RemoteLoadedModels::default()),
            job_seq: AtomicU64::new(1),
        });

        for stream in listener.incoming() {
            match stream {
                Ok(stream) => {
                    let ctx = Arc::clone(&ctx);
                    std::thread::spawn(move || handle_stream(stream, ctx));
                }
                Err(e) => eprintln!("Saient remote API accept error: {e}"),
            }
        }
    });
}

#[tauri::command]
pub fn remote_pairing_info() -> Result<PairingInfo, String> {
    // Phone pairing is LAN-only (port 18788 on a private address). It makes no
    // outbound requests, so it is not gated by the Internet switch.
    pairing_info(current_pairing_token()?)
}

#[tauri::command]
pub fn remote_reset_pairing() -> Result<PairingInfo, String> {
    pairing_info(rotate_pairing_token()?)
}

fn pairing_info(token: String) -> Result<PairingInfo, String> {
    let local_url = format!("http://127.0.0.1:{PORT}");
    let url = lan_ip()
        .map(|ip| format!("http://{ip}:{PORT}"))
        .unwrap_or_else(|| local_url.clone());
    Ok(PairingInfo {
        name: "Saient desktop".into(),
        port: PORT,
        url: url.clone(),
        local_url,
        payload: json!({
            "type": "saient-desktop-remote",
            "version": PAIRING_VERSION,
            "url": url,
            "token": token,
        }),
    })
}

fn handle_stream(mut stream: TcpStream, ctx: Arc<RemoteContext>) {
    let response = match read_request(&mut stream).and_then(|req| route(req, Arc::clone(&ctx))) {
        Ok(response) => response,
        Err((status, message)) => json_response(status, json!({ "ok": false, "error": message })),
    };
    let _ = write_response(&mut stream, response);
}

fn route(req: HttpRequest, ctx: Arc<RemoteContext>) -> Result<HttpResponse, (u16, String)> {
    if req.method == "OPTIONS" {
        return Ok(HttpResponse {
            status: 204,
            body: Vec::new(),
            content_type: "application/json",
        });
    }

    authorize_request(req.authorization.as_deref())?;

    if req.method == "GET" && req.path.starts_with("/api/jobs/") {
        let path = req.path.trim_start_matches("/api/jobs/");
        if let Some(id) = path.strip_suffix("/result") {
            return api_video_job_result(&ctx, id);
        }
        let id = path;
        return api_job(&ctx, id);
    }

    match (req.method.as_str(), req.path.as_str()) {
        ("GET", "/") | ("GET", "/api/health") | ("GET", "/health") => Ok(json_response(
            200,
            json!({
                "ok": true,
                "name": "Saient desktop remote",
                "port": PORT,
                "local_url": format!("http://127.0.0.1:{PORT}"),
                "lan_url": lan_ip().map(|ip| format!("http://{ip}:{PORT}")),
            }),
        )),
        ("GET", "/api/models") => api_models(&ctx),
        ("POST", "/api/image/load_async") => api_image_load_async(Arc::clone(&ctx), parse_json(&req.body)?),
        ("POST", "/api/image/load") => api_image_load(&ctx, parse_json(&req.body)?),
        ("POST", "/api/image/generate") => api_image_generate(&ctx, parse_json(&req.body)?),
        ("POST", "/api/video/load_async") => api_video_load_async(Arc::clone(&ctx), parse_json(&req.body)?),
        ("POST", "/api/video/load") => api_video_load(&ctx, parse_json(&req.body)?),
        ("POST", "/api/video/generate_async") => api_video_generate_async(Arc::clone(&ctx), parse_json(&req.body)?),
        ("POST", "/api/video/polish_async") => api_video_polish_async(Arc::clone(&ctx), parse_json(&req.body)?),
        ("POST", "/api/video/generate") => api_video_generate(&ctx, parse_json(&req.body)?),
        ("POST", "/api/unload") => api_unload(&ctx, parse_json(&req.body)?),
        _ => Err((
            404,
            format!("No remote endpoint for {} {}", req.method, req.path),
        )),
    }
}

fn api_models(ctx: &RemoteContext) -> Result<HttpResponse, (u16, String)> {
    let loaded = remote_loaded_models(ctx)?;
    Ok(json_response(
        200,
        json!({
            "ok": true,
            "image_models": imggen::imggen_scan_models(),
            "image_checkpoints": imggen::imggen_scan_checkpoints(),
            "image_loras": imggen::imggen_scan_loras(),
            "video_models": video::video_scan_models(),
            "video_loras": video::video_scan_loras(),
            "loaded": {
                "image": loaded.image,
                "video": loaded.video,
                "video_lora": loaded.video_lora,
            },
        }),
    ))
}

fn api_job(ctx: &RemoteContext, id: &str) -> Result<HttpResponse, (u16, String)> {
    let jobs = ctx.jobs.lock().map_err(|e| internal_error(e.to_string()))?;
    let job = jobs
        .get(id)
        .cloned()
        .ok_or_else(|| (404, format!("No remote job found for {id}")))?;
    let value = serde_json::to_value(job).map_err(|e| internal_error(e.to_string()))?;
    Ok(json_response(200, value))
}

fn api_video_job_result(ctx: &RemoteContext, id: &str) -> Result<HttpResponse, (u16, String)> {
    let results = ctx
        .video_results
        .lock()
        .map_err(|e| internal_error(e.to_string()))?;
    let result = results
        .get(id)
        .ok_or_else(|| (404, format!("Video result is not ready for job {id}")))?;
    Ok(HttpResponse {
        status: 200,
        body: result.bytes.clone(),
        content_type: "video/mp4",
    })
}

fn api_image_load_async(
    ctx: Arc<RemoteContext>,
    req: ImageLoadRequest,
) -> Result<HttpResponse, (u16, String)> {
    let window = desktop_window(&ctx)?;
    let id = create_job(&ctx, "image", "load", Some(req.model_path.clone()))?;
    let model_path = req.model_path.clone();
    let lora_path = req.lora_path.unwrap_or_default();
    let device = req.device.unwrap_or_else(|| "auto".into());
    let thread_ctx = Arc::clone(&ctx);
    let thread_id = id.clone();

    std::thread::spawn(move || {
        record_job_log(&thread_ctx, &thread_id, "Waiting for desktop load slot");
        let _op = match thread_ctx.op_lock.lock() {
            Ok(guard) => guard,
            Err(e) => {
                fail_job(&thread_ctx, &thread_id, e.to_string());
                return;
            }
        };
        record_job_log(&thread_ctx, &thread_id, "Desktop load slot acquired");
        let progress_ctx = Arc::clone(&thread_ctx);
        let progress_id = thread_id.clone();
        let progress: imggen::LoadProgress = Arc::new(move |message| {
            record_job_log(&progress_ctx, &progress_id, message);
        });

        match imggen::load_blocking_with_progress(
            thread_ctx.img.clone(),
            thread_ctx.vid.clone(),
            window,
            model_path.clone(),
            lora_path,
            device,
            progress,
        ) {
            Ok(actual_device) => {
                record_loaded_image(&thread_ctx, Some(model_path.clone()));
                finish_job(&thread_ctx, &thread_id, Some(model_path), Some(actual_device));
            }
            Err(e) => fail_job(&thread_ctx, &thread_id, e),
        }
    });

    Ok(json_response(202, json!({ "ok": true, "job_id": id })))
}

fn api_video_load_async(
    ctx: Arc<RemoteContext>,
    req: VideoLoadRequest,
) -> Result<HttpResponse, (u16, String)> {
    let window = desktop_window(&ctx)?;
    let id = create_job(&ctx, "video", "load", Some(req.model_path.clone()))?;
    let model_path = req.model_path.clone();
    let lora_path = req.lora_path;
    let selected_lora = lora_path.clone().unwrap_or_default();
    let lora_strength = req.lora_strength;
    let frames = req.frames;
    let precision = req.precision;
    let thread_ctx = Arc::clone(&ctx);
    let thread_id = id.clone();
    let event_window = window.clone();

    emit_remote_video_load(
        &event_window,
        "started",
        Some(&model_path),
        Some(&selected_lora),
        None,
        "Phone requested video model load",
    );

    std::thread::spawn(move || {
        record_job_log(&thread_ctx, &thread_id, "Waiting for desktop load slot");
        let _op = match thread_ctx.op_lock.lock() {
            Ok(guard) => guard,
            Err(e) => {
                let error = e.to_string();
                fail_job(&thread_ctx, &thread_id, error.clone());
                emit_remote_video_load(
                    &event_window,
                    "error",
                    Some(&model_path),
                    Some(&selected_lora),
                    None,
                    &error,
                );
                return;
            }
        };
        record_job_log(&thread_ctx, &thread_id, "Desktop load slot acquired");
        let progress_ctx = Arc::clone(&thread_ctx);
        let progress_id = thread_id.clone();
        let progress: video::LoadProgress = Arc::new(move |message| {
            record_job_log(&progress_ctx, &progress_id, message);
        });

        match video::load_blocking_with_progress(
            thread_ctx.vid.clone(),
            thread_ctx.img.clone(),
            window,
            model_path.clone(),
            lora_path,
            lora_strength,
            frames,
            precision,
            progress,
        ) {
            Ok(device) => {
                record_loaded_video(
                    &thread_ctx,
                    Some(model_path.clone()),
                    Some(selected_lora.clone()),
                );
                finish_job(
                    &thread_ctx,
                    &thread_id,
                    Some(model_path.clone()),
                    Some(device.clone()),
                );
                emit_remote_video_load(
                    &event_window,
                    "done",
                    Some(&model_path),
                    Some(&selected_lora),
                    Some(&device),
                    "Video model loaded from phone",
                );
            }
            Err(error) => {
                fail_job(&thread_ctx, &thread_id, error.clone());
                emit_remote_video_load(
                    &event_window,
                    "error",
                    Some(&model_path),
                    Some(&selected_lora),
                    None,
                    &error,
                );
            }
        }
    });

    Ok(json_response(202, json!({ "ok": true, "job_id": id })))
}

fn create_job(
    ctx: &RemoteContext,
    target: &str,
    action: &str,
    model_path: Option<String>,
) -> Result<String, (u16, String)> {
    let now = now_ms();
    let id = format!(
        "{}-{}-{}",
        target,
        now,
        ctx.job_seq.fetch_add(1, Ordering::Relaxed)
    );
    let job = RemoteJob {
        id: id.clone(),
        target: target.to_string(),
        action: action.to_string(),
        status: "running".into(),
        message: "Queued on desktop".into(),
        logs: vec!["0s Queued on desktop".into()],
        started_ms: now,
        updated_ms: now,
        finished_ms: None,
        model_path,
        device: None,
        error: None,
        progress_step: None,
        progress_total: None,
        step_seconds: None,
        elapsed_seconds: None,
        preview: None,
        result_ready: false,
        frames: None,
        width: None,
        height: None,
        elapsed: None,
        completed_stages: Vec::new(),
        failed_stages: Vec::new(),
    };
    ctx.jobs
        .lock()
        .map_err(|e| internal_error(e.to_string()))?
        .insert(id.clone(), job);
    Ok(id)
}

fn record_job_log(ctx: &RemoteContext, id: &str, message: impl Into<String>) {
    let message = message.into();
    if message.trim().is_empty() {
        return;
    }
    let now = now_ms();
    if let Ok(mut jobs) = ctx.jobs.lock() {
        if let Some(job) = jobs.get_mut(id) {
            job.message = message.clone();
            job.updated_ms = now;
            let elapsed = now.saturating_sub(job.started_ms) / 1000;
            job.logs.push(format!("{elapsed}s {message}"));
            trim_job_logs(&mut job.logs);
        }
    }
}

fn record_job_progress(ctx: &RemoteContext, id: &str, progress: &video::VideoProgress) {
    let now = now_ms();
    if let Ok(mut jobs) = ctx.jobs.lock() {
        if let Some(job) = jobs.get_mut(id) {
            job.progress_step = Some(progress.step);
            job.progress_total = Some(progress.total);
            job.step_seconds = progress.step_seconds;
            job.elapsed_seconds = progress.elapsed_seconds;
            let phase = if job.action == "polish" { "Pass" } else { "Denoising" };
            job.message = format!("{phase} step {}/{}", progress.step, progress.total);
            job.updated_ms = now;
            let elapsed = now.saturating_sub(job.started_ms) / 1000;
            let timing = progress
                .step_seconds
                .map(|seconds| format!(" · {seconds:.1}s"))
                .unwrap_or_default();
            job.logs.push(format!(
                "{elapsed}s step {}/{}{timing}",
                progress.step, progress.total
            ));
            trim_job_logs(&mut job.logs);
        }
    }
}

fn record_job_preview(ctx: &RemoteContext, id: &str, preview: video::VideoPreview) {
    let now = now_ms();
    if let Ok(mut jobs) = ctx.jobs.lock() {
        if let Some(job) = jobs.get_mut(id) {
            job.preview = Some(RemotePreview {
                base64_jpeg: preview.base64_jpeg,
                step: preview.step,
                total: preview.total,
                frames: preview.frames,
                decode_seconds: preview.decode_seconds,
            });
            job.updated_ms = now;
        }
    }
}

fn finish_generate_job(ctx: &RemoteContext, id: &str, frames: u32, elapsed_seconds: f64) {
    let now = now_ms();
    if let Ok(mut jobs) = ctx.jobs.lock() {
        if let Some(job) = jobs.get_mut(id) {
            job.status = "done".into();
            job.message = format!("MP4 ready · {frames} frames");
            job.updated_ms = now;
            job.finished_ms = Some(now);
            job.error = None;
            job.preview = None;
            job.result_ready = true;
            job.frames = Some(frames);
            job.elapsed = Some(elapsed_seconds);
            let elapsed = now.saturating_sub(job.started_ms) / 1000;
            job.logs
                .push(format!("{elapsed}s MP4 ready · {frames} frames"));
            trim_job_logs(&mut job.logs);
        }
    }
}

fn finish_polish_job(
    ctx: &RemoteContext,
    id: &str,
    frames: u32,
    width: u32,
    height: u32,
    elapsed_seconds: f64,
    completed_stages: Vec<String>,
    failed_stages: Vec<String>,
) {
    let now = now_ms();
    if let Ok(mut jobs) = ctx.jobs.lock() {
        if let Some(job) = jobs.get_mut(id) {
            job.status = "done".into();
            job.message = format!("Polished MP4 ready · {width}x{height} · {frames} frames");
            job.updated_ms = now;
            job.finished_ms = Some(now);
            job.error = None;
            job.preview = None;
            job.result_ready = true;
            job.frames = Some(frames);
            job.width = Some(width);
            job.height = Some(height);
            job.elapsed = Some(elapsed_seconds);
            job.completed_stages = completed_stages;
            job.failed_stages = failed_stages;
            let elapsed = now.saturating_sub(job.started_ms) / 1000;
            job.logs.push(format!(
                "{elapsed}s Polished MP4 ready · {width}x{height} · {frames} frames"
            ));
            trim_job_logs(&mut job.logs);
        }
    }
}

fn retain_video_result(ctx: &RemoteContext, id: String, bytes: Vec<u8>) -> Result<(), String> {
    let mut results = ctx.video_results.lock().map_err(|e| e.to_string())?;
    while results.len() >= 3 {
        let Some(oldest) = results.keys().next().cloned() else {
            break;
        };
        results.remove(&oldest);
    }
    results.insert(id, RemoteVideoResult { bytes });
    Ok(())
}

fn finish_job(
    ctx: &RemoteContext,
    id: &str,
    model_path: Option<String>,
    device: Option<String>,
) {
    let now = now_ms();
    if let Ok(mut jobs) = ctx.jobs.lock() {
        if let Some(job) = jobs.get_mut(id) {
            job.status = "done".into();
            job.message = "Loaded on desktop".into();
            job.updated_ms = now;
            job.finished_ms = Some(now);
            job.model_path = model_path;
            job.device = device.clone();
            job.error = None;
            let elapsed = now.saturating_sub(job.started_ms) / 1000;
            let suffix = device.map(|d| format!(" on {d}")).unwrap_or_default();
            job.logs.push(format!("{elapsed}s Loaded on desktop{suffix}"));
            trim_job_logs(&mut job.logs);
        }
    }
}

fn fail_job(ctx: &RemoteContext, id: &str, error: impl Into<String>) {
    let error = error.into();
    let now = now_ms();
    if let Ok(mut jobs) = ctx.jobs.lock() {
        if let Some(job) = jobs.get_mut(id) {
            job.status = "error".into();
            job.message = error.clone();
            job.error = Some(error.clone());
            job.updated_ms = now;
            job.finished_ms = Some(now);
            job.preview = None;
            job.result_ready = false;
            let elapsed = now.saturating_sub(job.started_ms) / 1000;
            job.logs.push(format!("{elapsed}s ERROR: {error}"));
            trim_job_logs(&mut job.logs);
        }
    }
    if let Ok(mut results) = ctx.video_results.lock() {
        results.remove(id);
    }
}

fn trim_job_logs(logs: &mut Vec<String>) {
    if logs.len() > 200 {
        let remove = logs.len() - 200;
        logs.drain(0..remove);
    }
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

fn api_image_load(
    ctx: &RemoteContext,
    req: ImageLoadRequest,
) -> Result<HttpResponse, (u16, String)> {
    let _op = ctx.op_lock.lock().map_err(|e| internal_error(e.to_string()))?;
    let window = desktop_window(ctx)?;
    let device = req.device.unwrap_or_else(|| "auto".into());
    let actual_device = imggen::load_blocking(
        ctx.img.clone(),
        ctx.vid.clone(),
        window,
        req.model_path.clone(),
        req.lora_path.unwrap_or_default(),
        device,
    )
    .map_err(internal_error)?;

    Ok(json_response(
        200,
        json!({
            "ok": true,
            "model_path": req.model_path,
            "device": actual_device,
        }),
    ))
}

fn api_image_generate(
    ctx: &RemoteContext,
    req: ImageGenerateRequest,
) -> Result<HttpResponse, (u16, String)> {
    let _op = ctx.op_lock.lock().map_err(|e| internal_error(e.to_string()))?;
    let window = desktop_window(ctx)?;
    let device = req.device.clone().unwrap_or_else(|| "auto".into());
    let lora_path = req.lora_path.clone().unwrap_or_default();
    let model_path = match req.model_path.clone() {
        Some(model_path) => {
            if !imggen::loaded_matches(&ctx.img, &model_path, &lora_path, &device)
                .map_err(internal_error)?
            {
                imggen::load_blocking(
                    ctx.img.clone(),
                    ctx.vid.clone(),
                    window.clone(),
                    model_path.clone(),
                    lora_path.clone(),
                    device.clone(),
                )
                .map_err(internal_error)?;
            }
            model_path
        }
        None => imggen::loaded_model_from_handle(&ctx.img)
            .map_err(internal_error)?
            .ok_or_else(|| {
                (
                    400,
                    "No image model loaded; pass model_path first".to_string(),
                )
            })?,
    };

    let result = imggen::generate_blocking(
        ctx.img.clone(),
        imggen::ImgGenPayload {
            prompt: req.prompt,
            neg_prompt: req.neg_prompt,
            model_path,
            lora_path: req.lora_path,
            steps: req.steps,
            cfg_scale: req.cfg_scale,
            seed: req.seed,
            width: req.width,
            height: req.height,
            device: req.device,
            scheduler: req.scheduler,
            face_detail: req.face_detail,
            asset_guard: req.asset_guard,
            asset_kind: req.asset_kind,
        },
        window,
    )
    .map_err(internal_error)?;

    Ok(json_response(
        200,
        json!({
            "ok": true,
            "type": "image",
            "base64_png": result.base64_png,
            "device": result.device,
            "elapsed": result.elapsed,
        }),
    ))
}

fn api_video_load(
    ctx: &RemoteContext,
    req: VideoLoadRequest,
) -> Result<HttpResponse, (u16, String)> {
    let _op = ctx.op_lock.lock().map_err(|e| internal_error(e.to_string()))?;
    let window = desktop_window(ctx)?;
    let model_path = req.model_path.clone();
    let selected_lora = req.lora_path.clone().unwrap_or_default();
    emit_remote_video_load(
        &window,
        "started",
        Some(&model_path),
        Some(&selected_lora),
        None,
        "Phone requested video model load",
    );
    let device = match video::load_blocking(
        ctx.vid.clone(),
        ctx.img.clone(),
        window.clone(),
        model_path.clone(),
        req.lora_path,
        req.lora_strength,
        req.frames,
        req.precision,
    ) {
        Ok(device) => device,
        Err(error) => {
            emit_remote_video_load(
                &window,
                "error",
                Some(&model_path),
                Some(&selected_lora),
                None,
                &error,
            );
            return Err(internal_error(error));
        }
    };
    emit_remote_video_load(
        &window,
        "done",
        Some(&model_path),
        Some(&selected_lora),
        Some(&device),
        "Video model loaded from phone",
    );

    Ok(json_response(
        200,
        json!({
            "ok": true,
            "model_path": model_path,
            "device": device,
        }),
    ))
}

fn api_video_generate_async(
    ctx: Arc<RemoteContext>,
    req: VideoGenerateRequest,
) -> Result<HttpResponse, (u16, String)> {
    let window = desktop_window(&ctx)?;
    let id = create_job(&ctx, "video", "generate", req.model_path.clone())?;
    let thread_ctx = Arc::clone(&ctx);
    let thread_id = id.clone();
    emit_remote_video_generation(
        &window,
        "started",
        "Phone started video generation",
        None,
        None,
    );

    std::thread::spawn(move || {
        record_job_log(&thread_ctx, &thread_id, "Waiting for desktop generation slot");
        let _op = match thread_ctx.op_lock.lock() {
            Ok(guard) => guard,
            Err(error) => {
                fail_remote_video_generation(&thread_ctx, &thread_id, &window, error.to_string());
                return;
            }
        };
        record_job_log(&thread_ctx, &thread_id, "Desktop generation slot acquired");

        let lora_path = req.lora_path.clone().unwrap_or_default();
        let lora_strength = req.lora_strength.unwrap_or(1.0);
        let frames_hint = req.num_frames.unwrap_or(49);
        let precision = req.precision.clone().unwrap_or_else(|| "fast".into());
        let model_path = match req.model_path.clone() {
            Some(model_path) => {
                let matches = match video::loaded_matches(
                    &thread_ctx.vid,
                    &model_path,
                    &lora_path,
                    lora_strength,
                    &precision,
                ) {
                    Ok(matches) => matches,
                    Err(error) => {
                        fail_remote_video_generation(&thread_ctx, &thread_id, &window, error);
                        return;
                    }
                };
                if !matches {
                    emit_remote_video_load(
                        &window,
                        "started",
                        Some(&model_path),
                        Some(&lora_path),
                        None,
                        "Phone generation requested a video model load",
                    );
                    let load_ctx = Arc::clone(&thread_ctx);
                    let load_id = thread_id.clone();
                    let load_progress: video::LoadProgress = Arc::new(move |message| {
                        record_job_log(&load_ctx, &load_id, message);
                    });
                    let loaded_device = match video::load_blocking_with_progress(
                        thread_ctx.vid.clone(),
                        thread_ctx.img.clone(),
                        window.clone(),
                        model_path.clone(),
                        if lora_path.is_empty() {
                            None
                        } else {
                            Some(lora_path.clone())
                        },
                        Some(lora_strength),
                        Some(frames_hint),
                        Some(precision.clone()),
                        load_progress,
                    ) {
                        Ok(device) => device,
                        Err(error) => {
                            fail_remote_video_generation(
                                &thread_ctx,
                                &thread_id,
                                &window,
                                error.clone(),
                            );
                            emit_remote_video_load(
                                &window,
                                "error",
                                Some(&model_path),
                                Some(&lora_path),
                                None,
                                &error,
                            );
                            return;
                        }
                    };
                    emit_remote_video_load(
                        &window,
                        "done",
                        Some(&model_path),
                        Some(&lora_path),
                        Some(&loaded_device),
                        "Video model loaded for phone generation",
                    );
                }
                model_path
            }
            None => match video::loaded_model_from_handle(&thread_ctx.vid) {
                Ok(Some(model_path)) => model_path,
                Ok(None) => {
                    fail_remote_video_generation(
                        &thread_ctx,
                        &thread_id,
                        &window,
                        "No video model loaded; pass model_path first",
                    );
                    return;
                }
                Err(error) => {
                    fail_remote_video_generation(&thread_ctx, &thread_id, &window, error);
                    return;
                }
            },
        };

        let progress_ctx = Arc::clone(&thread_ctx);
        let progress_id = thread_id.clone();
        let progress: video::GenerateProgress = Arc::new(move |update| match update {
            video::GenerateUpdate::Status(message) => {
                record_job_log(&progress_ctx, &progress_id, message);
            }
            video::GenerateUpdate::Progress(update) => {
                record_job_progress(&progress_ctx, &progress_id, &update);
            }
            video::GenerateUpdate::Preview(update) => {
                record_job_preview(&progress_ctx, &progress_id, update);
            }
        });
        let payload = video_payload_from_request(req, model_path);
        record_loaded_video(
            &thread_ctx,
            Some(payload.model_path.clone()),
            Some(lora_path),
        );
        let result = match video::generate_blocking_with_progress(
            thread_ctx.vid.clone(),
            payload,
            window.clone(),
            progress,
        ) {
            Ok(result) => result,
            Err(error) => {
                fail_remote_video_generation(&thread_ctx, &thread_id, &window, error);
                return;
            }
        };

        let bytes = match base64::engine::general_purpose::STANDARD
            .decode(result.base64_mp4.as_bytes())
        {
            Ok(bytes) => bytes,
            Err(error) => {
                fail_remote_video_generation(
                    &thread_ctx,
                    &thread_id,
                    &window,
                    format!("Desktop returned an invalid MP4 payload: {error}"),
                );
                return;
            }
        };
        if retain_video_result(&thread_ctx, thread_id.clone(), bytes).is_err() {
            fail_remote_video_generation(
                &thread_ctx,
                &thread_id,
                &window,
                "Could not retain the generated MP4 in memory",
            );
            return;
        }
        finish_generate_job(&thread_ctx, &thread_id, result.frames, result.elapsed);
        emit_remote_video_generation(
            &window,
            "done",
            "Phone video generation complete",
            Some(result.frames),
            Some(result.elapsed),
        );
    });

    Ok(json_response(202, json!({ "ok": true, "job_id": id })))
}

fn api_video_polish_async(
    ctx: Arc<RemoteContext>,
    req: VideoPolishRequest,
) -> Result<HttpResponse, (u16, String)> {
    let window = desktop_window(&ctx)?;
    let source_job_id = req.source_job_id.trim().to_string();
    if source_job_id.is_empty() {
        return Err((400, "source_job_id is required".into()));
    }
    let fps = req.fps.unwrap_or(16);
    if !(1..=120).contains(&fps) {
        return Err((400, "fps must be between 1 and 120".into()));
    }
    let interp_factor = req.interp_factor.unwrap_or(2);
    if !(2..=4).contains(&interp_factor) {
        return Err((400, "interp_factor must be between 2 and 4".into()));
    }

    let source_bytes = {
        let results = ctx
            .video_results
            .lock()
            .map_err(|e| internal_error(e.to_string()))?;
        results
            .get(&source_job_id)
            .map(|result| result.bytes.clone())
            .ok_or_else(|| {
                (
                    404,
                    format!("Temporary source MP4 is no longer available for job {source_job_id}"),
                )
            })?
    };

    let id = create_job(&ctx, "video", "polish", None)?;
    let thread_ctx = Arc::clone(&ctx);
    let thread_id = id.clone();

    std::thread::spawn(move || {
        record_job_log(
            &thread_ctx,
            &thread_id,
            format!("Source concept {source_job_id} retained in memory"),
        );
        record_job_log(&thread_ctx, &thread_id, "Waiting for desktop quality-pass slot");
        let _op = match thread_ctx.op_lock.lock() {
            Ok(guard) => guard,
            Err(error) => {
                fail_job(&thread_ctx, &thread_id, error.to_string());
                return;
            }
        };
        record_job_log(&thread_ctx, &thread_id, "Desktop quality-pass slot acquired");
        record_job_log(
            &thread_ctx,
            &thread_id,
            "Safe pass: optical-flow interpolation, then 2x ESRGAN upscale",
        );

        let progress_ctx = Arc::clone(&thread_ctx);
        let progress_id = thread_id.clone();
        let progress: video::GenerateProgress = Arc::new(move |update| match update {
            video::GenerateUpdate::Status(message) => {
                record_job_log(&progress_ctx, &progress_id, message);
            }
            video::GenerateUpdate::Progress(update) => {
                record_job_progress(&progress_ctx, &progress_id, &update);
            }
            video::GenerateUpdate::Preview(_) => {}
        });
        let payload = video::EnhancePayload {
            video_b64: base64::engine::general_purpose::STANDARD.encode(source_bytes),
            fps,
            stages: vec!["interpolate".into(), "upscale".into()],
            model_path: String::new(),
            prompt: String::new(),
            neg_prompt: None,
            cfg_scale: None,
            refine_strength: None,
            refine_steps: None,
            interp_factor: Some(interp_factor),
        };
        let result = match video::enhance_blocking_with_progress(
            thread_ctx.vid.clone(),
            payload,
            window,
            progress,
        ) {
            Ok(result) => result,
            Err(error) => {
                fail_job(&thread_ctx, &thread_id, error);
                return;
            }
        };

        for warning in &result.failed_stages {
            record_job_log(&thread_ctx, &thread_id, format!("WARNING: {warning}"));
        }
        let bytes = match base64::engine::general_purpose::STANDARD
            .decode(result.enhanced_b64.as_bytes())
        {
            Ok(bytes) => bytes,
            Err(error) => {
                fail_job(
                    &thread_ctx,
                    &thread_id,
                    format!("Quality pass returned an invalid MP4 payload: {error}"),
                );
                return;
            }
        };
        if let Err(error) = retain_video_result(&thread_ctx, thread_id.clone(), bytes) {
            fail_job(
                &thread_ctx,
                &thread_id,
                format!("Could not retain the polished MP4 in memory: {error}"),
            );
            return;
        }
        finish_polish_job(
            &thread_ctx,
            &thread_id,
            result.frames,
            result.width,
            result.height,
            result.elapsed,
            result.completed_stages,
            result.failed_stages,
        );
    });

    Ok(json_response(202, json!({ "ok": true, "job_id": id })))
}

fn api_video_generate(
    ctx: &RemoteContext,
    req: VideoGenerateRequest,
) -> Result<HttpResponse, (u16, String)> {
    let _op = ctx.op_lock.lock().map_err(|e| internal_error(e.to_string()))?;
    let window = desktop_window(ctx)?;
    let lora_path = req.lora_path.clone().unwrap_or_default();
    let lora_strength = req.lora_strength.unwrap_or(1.0);
    let frames_hint = req.num_frames.unwrap_or(49);
    let precision = req.precision.clone().unwrap_or_else(|| "fast".into());
    let model_path = match req.model_path.clone() {
        Some(model_path) => {
            if !video::loaded_matches(
                &ctx.vid,
                &model_path,
                &lora_path,
                lora_strength,
                &precision,
            )
            .map_err(internal_error)?
            {
                emit_remote_video_load(
                    &window,
                    "started",
                    Some(&model_path),
                    Some(&lora_path),
                    None,
                    "Phone generation requested a video model load",
                );
                let loaded_device = match video::load_blocking(
                    ctx.vid.clone(),
                    ctx.img.clone(),
                    window.clone(),
                    model_path.clone(),
                    if lora_path.is_empty() {
                        None
                    } else {
                        Some(lora_path.clone())
                    },
                    Some(lora_strength),
                    Some(frames_hint),
                    Some(precision.clone()),
                ) {
                    Ok(device) => device,
                    Err(error) => {
                        emit_remote_video_load(
                            &window,
                            "error",
                            Some(&model_path),
                            Some(&lora_path),
                            None,
                            &error,
                        );
                        return Err(internal_error(error));
                    }
                };
                emit_remote_video_load(
                    &window,
                    "done",
                    Some(&model_path),
                    Some(&lora_path),
                    Some(&loaded_device),
                    "Video model loaded for phone generation",
                );
            }
            model_path
        }
        None => video::loaded_model_from_handle(&ctx.vid)
            .map_err(internal_error)?
            .ok_or_else(|| {
                (
                    400,
                    "No video model loaded; pass model_path first".to_string(),
                )
            })?,
    };

    let payload = video_payload_from_request(req, model_path);
    let result = video::generate_blocking(ctx.vid.clone(), payload, window)
    .map_err(internal_error)?;

    Ok(json_response(
        200,
        json!({
            "ok": true,
            "type": "video",
            "base64_mp4": result.base64_mp4,
            "frames": result.frames,
            "elapsed": result.elapsed,
        }),
    ))
}

fn api_unload(ctx: &RemoteContext, req: UnloadRequest) -> Result<HttpResponse, (u16, String)> {
    let _op = ctx.op_lock.lock().map_err(|e| internal_error(e.to_string()))?;
    let target = req.target.unwrap_or_else(|| "all".into());
    if target == "image" || target == "all" {
        *ctx.img.lock().map_err(|e| internal_error(e.to_string()))? = None;
        record_loaded_image(ctx, None);
    }
    if target == "video" || target == "all" {
        *ctx.vid.lock().map_err(|e| internal_error(e.to_string()))? = None;
        record_loaded_video(ctx, None, None);
        if let Ok(window) = desktop_window(ctx) {
            emit_remote_video_load(
                &window,
                "unloaded",
                None,
                None,
                None,
                "Phone unloaded the video model",
            );
        }
    }
    if target != "image" && target != "video" && target != "all" {
        return Err((400, "target must be image, video, or all".into()));
    }
    Ok(json_response(
        200,
        json!({ "ok": true, "unloaded": target }),
    ))
}

fn remote_loaded_models(ctx: &RemoteContext) -> Result<RemoteLoadedModels, (u16, String)> {
    let image = imggen::try_loaded_model_from_handle(&ctx.img).map_err(internal_error)?;
    let video = video::try_loaded_state_from_handle(&ctx.vid).map_err(internal_error)?;
    let mut loaded = ctx
        .loaded
        .lock()
        .map_err(|e| internal_error(e.to_string()))?;
    if let Some(image) = image {
        loaded.image = image;
    }
    if let Some((model, lora)) = video {
        loaded.video = model;
        loaded.video_lora = lora;
    }
    Ok(loaded.clone())
}

fn record_loaded_image(ctx: &RemoteContext, model: Option<String>) {
    if let Ok(mut loaded) = ctx.loaded.lock() {
        loaded.image = model;
        if loaded.image.is_some() {
            loaded.video = None;
            loaded.video_lora = None;
        }
    }
}

fn record_loaded_video(
    ctx: &RemoteContext,
    model: Option<String>,
    lora: Option<String>,
) {
    if let Ok(mut loaded) = ctx.loaded.lock() {
        loaded.video = model;
        loaded.video_lora = lora;
        if loaded.video.is_some() {
            loaded.image = None;
        }
    }
}

fn emit_remote_video_load(
    window: &WebviewWindow,
    status: &str,
    model_path: Option<&str>,
    lora_path: Option<&str>,
    device: Option<&str>,
    message: &str,
) {
    let _ = window.emit(
        "remote-video-load",
        json!({
            "status": status,
            "model_path": model_path,
            "lora_path": lora_path,
            "device": device,
            "message": message,
        }),
    );
}

fn emit_remote_video_generation(
    window: &WebviewWindow,
    status: &str,
    message: &str,
    frames: Option<u32>,
    elapsed: Option<f64>,
) {
    let _ = window.emit(
        "remote-video-generation",
        json!({
            "status": status,
            "message": message,
            "frames": frames,
            "elapsed": elapsed,
        }),
    );
}

fn fail_remote_video_generation(
    ctx: &RemoteContext,
    id: &str,
    window: &WebviewWindow,
    error: impl Into<String>,
) {
    let error = error.into();
    fail_job(ctx, id, error.clone());
    emit_remote_video_generation(window, "error", &error, None, None);
}

fn desktop_window(ctx: &RemoteContext) -> Result<WebviewWindow, (u16, String)> {
    ctx.app
        .get_webview_window("main")
        .ok_or_else(|| (503, "Desktop window is not ready".to_string()))
}

fn parse_json<T: DeserializeOwned>(body: &[u8]) -> Result<T, (u16, String)> {
    serde_json::from_slice(body).map_err(|e| (400, format!("Invalid JSON body: {e}")))
}

fn internal_error(message: String) -> (u16, String) {
    (500, message)
}

fn json_response(status: u16, value: serde_json::Value) -> HttpResponse {
    HttpResponse {
        status,
        body: serde_json::to_vec(&value).unwrap_or_else(|_| b"{\"ok\":false}".to_vec()),
        content_type: "application/json",
    }
}

fn read_request(stream: &mut TcpStream) -> Result<HttpRequest, (u16, String)> {
    stream
        .set_read_timeout(Some(Duration::from_secs(15)))
        .map_err(|e| internal_error(e.to_string()))?;

    let mut buf = Vec::new();
    let mut scratch = [0_u8; 4096];
    let header_end = loop {
        let n = stream
            .read(&mut scratch)
            .map_err(|e| (400, e.to_string()))?;
        if n == 0 {
            return Err((400, "Connection closed before request headers".into()));
        }
        buf.extend_from_slice(&scratch[..n]);
        if buf.len() > MAX_HEADER_BYTES {
            return Err((431, "Request headers are too large".into()));
        }
        if let Some(pos) = find_header_end(&buf) {
            break pos;
        }
    };

    let header_bytes = &buf[..header_end];
    let header_text = std::str::from_utf8(header_bytes)
        .map_err(|_| (400, "Request headers must be UTF-8".to_string()))?;
    let mut lines = header_text.split("\r\n");
    let request_line = lines
        .next()
        .ok_or_else(|| (400, "Missing request line".to_string()))?;
    let mut request_parts = request_line.split_whitespace();
    let method = request_parts
        .next()
        .ok_or_else(|| (400, "Missing HTTP method".to_string()))?
        .to_string();
    let raw_path = request_parts
        .next()
        .ok_or_else(|| (400, "Missing HTTP path".to_string()))?;
    let path = raw_path.split('?').next().unwrap_or(raw_path).to_string();

    let mut content_len = 0_usize;
    let mut authorization = None;
    for line in lines {
        let Some((key, value)) = line.split_once(':') else {
            continue;
        };
        if key.eq_ignore_ascii_case("content-length") {
            content_len = value
                .trim()
                .parse::<usize>()
                .map_err(|_| (400, "Invalid content-length".to_string()))?;
        } else if key.eq_ignore_ascii_case("authorization") {
            authorization = Some(value.trim().to_string());
        }
    }
    if content_len > MAX_BODY_BYTES {
        return Err((413, "Request body is too large".into()));
    }

    let body_start = header_end + 4;
    let mut body = buf[body_start..].to_vec();
    while body.len() < content_len {
        let n = stream
            .read(&mut scratch)
            .map_err(|e| (400, e.to_string()))?;
        if n == 0 {
            return Err((
                400,
                "Connection closed before request body completed".into(),
            ));
        }
        body.extend_from_slice(&scratch[..n]);
    }
    body.truncate(content_len);

    Ok(HttpRequest {
        method,
        path,
        authorization,
        body,
    })
}

fn write_response(stream: &mut TcpStream, response: HttpResponse) -> std::io::Result<()> {
    let reason = match response.status {
        200 => "OK",
        202 => "Accepted",
        204 => "No Content",
        401 => "Unauthorized",
        400 => "Bad Request",
        404 => "Not Found",
        413 => "Payload Too Large",
        431 => "Request Header Fields Too Large",
        500 => "Internal Server Error",
        503 => "Service Unavailable",
        _ => "OK",
    };
    let headers = format!(
        "HTTP/1.1 {} {}\r\n\
         Content-Type: {}\r\n\
         Content-Length: {}\r\n\
         Access-Control-Allow-Origin: *\r\n\
         Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n\
         Access-Control-Allow-Headers: content-type, authorization\r\n\
         Connection: close\r\n\r\n",
        response.status,
        reason,
        response.content_type,
        response.body.len()
    );
    stream.write_all(headers.as_bytes())?;
    stream.write_all(&response.body)
}

fn authorize_request(authorization: Option<&str>) -> Result<(), (u16, String)> {
    let expected = current_pairing_token().map_err(internal_error)?;
    let provided = authorization
        .and_then(bearer_token)
        .unwrap_or_default();
    if constant_time_eq(expected.as_bytes(), provided.as_bytes()) {
        return Ok(());
    }
    Err((
        401,
        "Phone binding is missing or no longer valid. Scan the current QR code in desktop Settings.".into(),
    ))
}

fn bearer_token(header: &str) -> Option<&str> {
    let (scheme, token) = header.trim().split_once(' ')?;
    if scheme.eq_ignore_ascii_case("bearer") && !token.trim().is_empty() {
        Some(token.trim())
    } else {
        None
    }
}

fn constant_time_eq(expected: &[u8], provided: &[u8]) -> bool {
    let mut difference = expected.len() ^ provided.len();
    let max_len = expected.len().max(provided.len());
    for index in 0..max_len {
        let left = expected.get(index).copied().unwrap_or(0);
        let right = provided.get(index).copied().unwrap_or(0);
        difference |= usize::from(left ^ right);
    }
    difference == 0
}

fn binding_path() -> PathBuf {
    crate::paths::config_dir().join("phone-binding.json")
}

fn current_pairing_token() -> Result<String, String> {
    if let Some(store) = PAIRING_TOKEN.get() {
        return store.read().map(|token| token.clone()).map_err(|e| e.to_string());
    }

    let token = read_stored_binding()
        .map(|binding| binding.token)
        .unwrap_or_else(generate_pairing_token);
    if read_stored_binding().is_none() {
        write_stored_binding(&token)?;
    }

    let _ = PAIRING_TOKEN.set(RwLock::new(token.clone()));
    PAIRING_TOKEN
        .get()
        .ok_or_else(|| "Phone binding store did not initialise".to_string())?
        .read()
        .map(|current| current.clone())
        .map_err(|e| e.to_string())
}

fn rotate_pairing_token() -> Result<String, String> {
    let next = generate_pairing_token();
    write_stored_binding(&next)?;
    if let Some(store) = PAIRING_TOKEN.get() {
        *store.write().map_err(|e| e.to_string())? = next.clone();
    } else {
        let _ = PAIRING_TOKEN.set(RwLock::new(next.clone()));
    }
    Ok(next)
}

fn read_stored_binding() -> Option<StoredPhoneBinding> {
    let text = std::fs::read_to_string(binding_path()).ok()?;
    let binding: StoredPhoneBinding = serde_json::from_str(&text).ok()?;
    if binding.version != PAIRING_VERSION || binding.token.len() < 32 {
        return None;
    }
    Some(binding)
}

fn write_stored_binding(token: &str) -> Result<(), String> {
    let path = binding_path();
    let parent = path
        .parent()
        .ok_or_else(|| "Phone binding path has no parent directory".to_string())?;
    std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    let binding = StoredPhoneBinding {
        version: PAIRING_VERSION,
        token: token.to_string(),
        created_ms: now_ms(),
    };
    let bytes = serde_json::to_vec(&binding).map_err(|e| e.to_string())?;
    let mut file = tempfile::NamedTempFile::new_in(parent).map_err(|e| e.to_string())?;
    file.write_all(&bytes).map_err(|e| e.to_string())?;
    file.as_file().sync_all().map_err(|e| e.to_string())?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        file.as_file()
            .set_permissions(std::fs::Permissions::from_mode(0o600))
            .map_err(|e| e.to_string())?;
    }
    file.persist(path).map_err(|e| e.error.to_string())?;
    Ok(())
}

fn generate_pairing_token() -> String {
    let mut bytes = [0_u8; PAIRING_TOKEN_BYTES];
    OsRng.fill_bytes(&mut bytes);
    base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(bytes)
}

fn find_header_end(buf: &[u8]) -> Option<usize> {
    buf.windows(4).position(|w| w == b"\r\n\r\n")
}

fn lan_ip() -> Option<String> {
    let mut ips = lan_ips();
    ips.sort_by_key(|ip| private_ip_score(ip));
    ips.into_iter().next()
}

fn lan_ips() -> Vec<String> {
    let mut out = Vec::new();

    if let Ok(hostname_ips) = std::process::Command::new("hostname").arg("-I").output() {
        if hostname_ips.status.success() {
            let text = String::from_utf8_lossy(&hostname_ips.stdout);
            for raw in text.split_whitespace() {
                if let Ok(IpAddr::V4(ip)) = raw.parse::<IpAddr>() {
                    if is_lan_ip(ip) {
                        out.push(ip.to_string());
                    }
                }
            }
        }
    }

    if let Ok(sock) = UdpSocket::bind("0.0.0.0:0") {
        if sock.connect("8.8.8.8:80").is_ok() {
            if let Ok(addr) = sock.local_addr() {
                if let IpAddr::V4(ip) = addr.ip() {
                    if is_lan_ip(ip) {
                        out.push(ip.to_string());
                    }
                }
            }
        }
    }

    out.sort();
    out.dedup();
    out
}

fn is_lan_ip(ip: Ipv4Addr) -> bool {
    ip.is_private() && !ip.is_loopback() && !ip.is_link_local()
}

fn private_ip_score(ip: &str) -> u8 {
    match ip.parse::<Ipv4Addr>() {
        Ok(ip) if ip.octets()[0] == 192 && ip.octets()[1] == 168 => 0,
        Ok(ip) if ip.octets()[0] == 172 && (16..=31).contains(&ip.octets()[1]) => 1,
        Ok(ip) if ip.octets()[0] == 10 => 2,
        _ => 3,
    }
}

#[cfg(test)]
mod tests {
    use super::{bearer_token, constant_time_eq};

    #[test]
    fn parses_bearer_tokens_case_insensitively() {
        assert_eq!(bearer_token("Bearer secret"), Some("secret"));
        assert_eq!(bearer_token("bearer secret"), Some("secret"));
        assert_eq!(bearer_token("Basic secret"), None);
        assert_eq!(bearer_token("Bearer"), None);
    }

    #[test]
    fn compares_pairing_tokens_without_early_length_exit() {
        assert!(constant_time_eq(b"same-token", b"same-token"));
        assert!(!constant_time_eq(b"same-token", b"other-token"));
        assert!(!constant_time_eq(b"same-token", b"same-token-longer"));
    }
}
