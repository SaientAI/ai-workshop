use serde::de::DeserializeOwned;
use serde::Deserialize;
use serde_json::json;
use std::io::{Read, Write};
use std::net::{IpAddr, Ipv4Addr, TcpListener, TcpStream, UdpSocket};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tauri::{AppHandle, Manager, WebviewWindow};

use crate::{imggen, video};

const PORT: u16 = 18788;
const MAX_HEADER_BYTES: usize = 32 * 1024;
const MAX_BODY_BYTES: usize = 96 * 1024 * 1024;

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
    jobs: Mutex<()>,
}

struct HttpRequest {
    method: String,
    path: String,
    body: Vec<u8>,
}

struct HttpResponse {
    status: u16,
    body: Vec<u8>,
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
    lora_path: Option<String>,
    lora_strength: Option<f32>,
    precision: Option<String>,
}

#[derive(Deserialize)]
struct UnloadRequest {
    target: Option<String>,
}

pub fn start(app: AppHandle, img: imggen::DaemonHandle, vid: video::VideoHandle) {
    std::thread::spawn(move || {
        let listener = match TcpListener::bind(("0.0.0.0", PORT)) {
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
            jobs: Mutex::new(()),
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
pub fn remote_pairing_info() -> PairingInfo {
    let local_url = format!("http://127.0.0.1:{PORT}");
    let url = lan_ip()
        .map(|ip| format!("http://{ip}:{PORT}"))
        .unwrap_or_else(|| local_url.clone());
    PairingInfo {
        name: "Saient desktop".into(),
        port: PORT,
        url: url.clone(),
        local_url,
        payload: json!({
            "type": "saient-desktop-remote",
            "version": 1,
            "url": url,
        }),
    }
}

fn handle_stream(mut stream: TcpStream, ctx: Arc<RemoteContext>) {
    let response = match read_request(&mut stream).and_then(|req| route(req, &ctx)) {
        Ok(response) => response,
        Err((status, message)) => json_response(status, json!({ "ok": false, "error": message })),
    };
    let _ = write_response(&mut stream, response);
}

fn route(req: HttpRequest, ctx: &RemoteContext) -> Result<HttpResponse, (u16, String)> {
    if req.method == "OPTIONS" {
        return Ok(HttpResponse {
            status: 204,
            body: Vec::new(),
        });
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
        ("GET", "/api/models") => api_models(ctx),
        ("POST", "/api/image/load") => api_image_load(ctx, parse_json(&req.body)?),
        ("POST", "/api/image/generate") => api_image_generate(ctx, parse_json(&req.body)?),
        ("POST", "/api/video/load") => api_video_load(ctx, parse_json(&req.body)?),
        ("POST", "/api/video/generate") => api_video_generate(ctx, parse_json(&req.body)?),
        ("POST", "/api/unload") => api_unload(ctx, parse_json(&req.body)?),
        _ => Err((
            404,
            format!("No remote endpoint for {} {}", req.method, req.path),
        )),
    }
}

fn api_models(ctx: &RemoteContext) -> Result<HttpResponse, (u16, String)> {
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
                "image": imggen::loaded_model_from_handle(&ctx.img).map_err(internal_error)?,
                "video": video::loaded_model_from_handle(&ctx.vid).map_err(internal_error)?,
            },
        }),
    ))
}

fn api_image_load(
    ctx: &RemoteContext,
    req: ImageLoadRequest,
) -> Result<HttpResponse, (u16, String)> {
    let _job = ctx.jobs.lock().map_err(|e| internal_error(e.to_string()))?;
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
    let _job = ctx.jobs.lock().map_err(|e| internal_error(e.to_string()))?;
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
    let _job = ctx.jobs.lock().map_err(|e| internal_error(e.to_string()))?;
    let window = desktop_window(ctx)?;
    let device = video::load_blocking(
        ctx.vid.clone(),
        ctx.img.clone(),
        window,
        req.model_path.clone(),
        req.lora_path,
        req.lora_strength,
        req.frames,
        req.precision,
    )
    .map_err(internal_error)?;

    Ok(json_response(
        200,
        json!({
            "ok": true,
            "model_path": req.model_path,
            "device": device,
        }),
    ))
}

fn api_video_generate(
    ctx: &RemoteContext,
    req: VideoGenerateRequest,
) -> Result<HttpResponse, (u16, String)> {
    let _job = ctx.jobs.lock().map_err(|e| internal_error(e.to_string()))?;
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
                frames_hint,
                &precision,
            )
            .map_err(internal_error)?
            {
                video::load_blocking(
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
                )
                .map_err(internal_error)?;
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

    let result = video::generate_blocking(
        ctx.vid.clone(),
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
            block_offload: None,  // remote API doesn't expose park-to-RAM (UI toggle only)
        },
        window,
    )
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
    let _job = ctx.jobs.lock().map_err(|e| internal_error(e.to_string()))?;
    let target = req.target.unwrap_or_else(|| "all".into());
    if target == "image" || target == "all" {
        *ctx.img.lock().map_err(|e| internal_error(e.to_string()))? = None;
    }
    if target == "video" || target == "all" {
        *ctx.vid.lock().map_err(|e| internal_error(e.to_string()))? = None;
    }
    if target != "image" && target != "video" && target != "all" {
        return Err((400, "target must be image, video, or all".into()));
    }
    Ok(json_response(
        200,
        json!({ "ok": true, "unloaded": target }),
    ))
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
    for line in lines {
        let Some((key, value)) = line.split_once(':') else {
            continue;
        };
        if key.eq_ignore_ascii_case("content-length") {
            content_len = value
                .trim()
                .parse::<usize>()
                .map_err(|_| (400, "Invalid content-length".to_string()))?;
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

    Ok(HttpRequest { method, path, body })
}

fn write_response(stream: &mut TcpStream, response: HttpResponse) -> std::io::Result<()> {
    let reason = match response.status {
        200 => "OK",
        204 => "No Content",
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
         Content-Type: application/json\r\n\
         Content-Length: {}\r\n\
         Access-Control-Allow-Origin: *\r\n\
         Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n\
         Access-Control-Allow-Headers: content-type, authorization\r\n\
         Connection: close\r\n\r\n",
        response.status,
        reason,
        response.body.len()
    );
    stream.write_all(headers.as_bytes())?;
    stream.write_all(&response.body)
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
