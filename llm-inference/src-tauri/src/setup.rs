//! setup.rs — first-run setup wizard backend.
//!
//! Detects the machine (OS, GPU/driver/CUDA, Python, disk), maps CUDA → the right
//! PyTorch wheel index, and runs the install (managed venv + CUDA-matched pip +
//! starter model download) streaming progress to the frontend.
//!
//! Everything installs into a self-contained managed dir so we never touch the
//! user's system Python:  ~/.config/saient/venv  (Linux/Mac)
//!                        %APPDATA%\saient\venv   (Windows)

use serde::Serialize;
use std::path::PathBuf;
use std::process::Command;
use crate::resolve::NoConsole;
use tauri::{Emitter, WebviewWindow};

// ── Paths ──────────────────────────────────────────────────────────────────────

pub fn config_dir() -> PathBuf {
    #[cfg(target_os = "windows")]
    {
        if let Ok(appdata) = std::env::var("APPDATA") {
            return PathBuf::from(appdata).join("saient");
        }
    }
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
    PathBuf::from(home).join(".config/saient")
}

pub fn managed_venv() -> PathBuf {
    config_dir().join("venv")
}

/// Path to the python executable inside the managed venv (whether or not it exists yet).
pub fn venv_python() -> PathBuf {
    #[cfg(target_os = "windows")]
    { managed_venv().join("Scripts").join("python.exe") }
    #[cfg(not(target_os = "windows"))]
    { managed_venv().join("bin").join("python") }
}

/// Path to a console-script binary inside the managed venv (e.g. "tinyq4").
pub fn venv_bin(name: &str) -> PathBuf {
    #[cfg(target_os = "windows")]
    { managed_venv().join("Scripts").join(format!("{name}.exe")) }
    #[cfg(not(target_os = "windows"))]
    { managed_venv().join("bin").join(name) }
}

fn setup_marker() -> PathBuf {
    config_dir().join("setup_done.json")
}

/// One-time rename of the pre-rebrand dirs (ai-workshop → saient) so existing installs
/// keep their venv, sessions, LoRAs, and settings. No-op on a fresh install. Run once at
/// startup, before anything touches the config/data dirs.
pub fn migrate_legacy_dirs() {
    let mut pairs: Vec<(PathBuf, PathBuf)> = Vec::new();
    #[cfg(target_os = "windows")]
    {
        if let Ok(appdata) = std::env::var("APPDATA") {
            let b = PathBuf::from(appdata);
            pairs.push((b.join("ai-workshop"), b.join("saient")));
        }
    }
    #[cfg(not(target_os = "windows"))]
    {
        let h = PathBuf::from(std::env::var("HOME").unwrap_or_else(|_| ".".into()));
        pairs.push((h.join(".config/ai-workshop"),      h.join(".config/saient")));
        pairs.push((h.join(".local/share/ai-workshop"), h.join(".local/share/saient")));
    }
    for (old, new) in pairs {
        if old.exists() && !new.exists() {
            if let Some(parent) = new.parent() { let _ = std::fs::create_dir_all(parent); }
            let _ = std::fs::rename(&old, &new);
        }
    }
}

// ── System detection ─────────────────────────────────────────────────────────

#[derive(Serialize, Clone)]
pub struct SystemInfo {
    pub os: String,                       // "linux" | "windows" | "macos"
    pub gpu_name: Option<String>,
    pub driver_version: Option<String>,
    pub cuda_version: Option<String>,     // max CUDA the driver supports (nvidia-smi header)
    pub vram_gb: Option<f64>,
    pub torch_index: String,              // "cu124" | "cu121" | "cu118" | "cpu"
    pub torch_index_url: String,          // full pip --index-url
    pub system_python: Option<String>,    // a base python3 to build the venv from
    pub python_version: Option<String>,
    pub venv_ready: bool,                  // managed venv exists
    pub tinyq4_ready: bool,                // tinyq4 resolvable
    pub disk_free_gb: f64,
    pub setup_done: bool,                  // marker present
}

/// Map the driver's max-supported CUDA version to the closest PyTorch wheel build.
/// Torch ships cu118 / cu121 / cu124 / cu126 / cu128; pick the highest the driver
/// supports. Newer (cu126+) carry Blackwell sm_120 kernels — important for the
/// RTX 50-series; older wheels would fall back / fail on those cards.
fn torch_index_for_cuda(cuda: Option<&str>) -> &'static str {
    let v = match cuda.and_then(parse_major_minor) {
        Some(v) => v,
        None => return "cpu",
    };
    if v >= (12, 8) { "cu128" }
    else if v >= (12, 6) { "cu126" }
    else if v >= (12, 4) { "cu124" }
    else if v >= (12, 1) { "cu121" }
    else if v >= (11, 8) { "cu118" }
    else { "cpu" } // very old driver — fall back to CPU torch
}

fn parse_major_minor(s: &str) -> Option<(u32, u32)> {
    let mut it = s.trim().split('.');
    let maj = it.next()?.parse::<u32>().ok()?;
    let min = it.next().unwrap_or("0").parse::<u32>().unwrap_or(0);
    Some((maj, min))
}

fn nvidia_smi_cuda_version() -> Option<String> {
    // The plain `nvidia-smi` header prints e.g. "CUDA Version: 12.4".
    let out = Command::new("nvidia-smi").no_console().output().ok()?;
    if !out.status.success() { return None; }
    let txt = String::from_utf8_lossy(&out.stdout);
    let idx = txt.find("CUDA Version:")?;
    let after = &txt[idx + "CUDA Version:".len()..];
    let ver: String = after.trim_start()
        .chars().take_while(|c| c.is_ascii_digit() || *c == '.').collect();
    if ver.is_empty() { None } else { Some(ver) }
}

fn nvidia_smi_field(query: &str) -> Option<String> {
    let out = Command::new("nvidia-smi")
        .args(["--query-gpu", query])
        .arg("--format=csv,noheader,nounits")
        .no_console()
        .output().ok()?;
    if !out.status.success() { return None; }
    let s = String::from_utf8_lossy(&out.stdout).lines().next()?.trim().to_string();
    if s.is_empty() { None } else { Some(s) }
}

fn detect_system_python() -> (Option<String>, Option<String>) {
    // Prefer an explicit base python, else python3 / python on PATH.
    let candidates = if cfg!(target_os = "windows") {
        vec!["python", "python3", "py"]
    } else {
        vec!["python3", "python"]
    };
    for c in candidates {
        if let Ok(out) = Command::new(c).arg("--version").no_console().output() {
            if out.status.success() {
                let ver = String::from_utf8_lossy(&out.stdout).trim().to_string();
                let ver = if ver.is_empty() {
                    String::from_utf8_lossy(&out.stderr).trim().to_string() // py2 prints to stderr
                } else { ver };
                return (Some(c.to_string()), Some(ver));
            }
        }
    }
    (None, None)
}

fn disk_free_gb(dir: &std::path::Path) -> f64 {
    #[cfg(unix)]
    {
        use std::ffi::CString;
        use std::os::unix::ffi::OsStrExt;
        let path = CString::new(dir.as_os_str().as_bytes()).ok();
        if let Some(path) = path {
            unsafe {
                let mut stat: libc::statvfs = std::mem::zeroed();
                if libc::statvfs(path.as_ptr(), &mut stat) == 0 {
                    let avail = stat.f_bavail as f64 * stat.f_frsize as f64;
                    return avail / 1e9;
                }
            }
        }
        0.0
    }
    #[cfg(windows)]
    {
        // Best-effort via PowerShell; non-fatal if it fails.
        let drive = dir.components().next()
            .map(|c| c.as_os_str().to_string_lossy().to_string())
            .unwrap_or_else(|| "C:".into());
        if let Ok(out) = Command::new("powershell").args([
            "-NoProfile","-Command",
            &format!("(Get-PSDrive {}).Free", drive.trim_end_matches(':').trim_end_matches('\\'))
        ]).no_console().output() {
            if let Ok(s) = String::from_utf8(out.stdout) {
                if let Ok(bytes) = s.trim().parse::<f64>() { return bytes / 1e9; }
            }
        }
        0.0
    }
    #[cfg(not(any(unix, windows)))]
    { 0.0 }
}

#[tauri::command]
pub fn detect_system() -> SystemInfo {
    let os = if cfg!(target_os = "windows") { "windows" }
             else if cfg!(target_os = "macos") { "macos" }
             else { "linux" }.to_string();

    let cuda_version = nvidia_smi_cuda_version();
    let gpu_name = nvidia_smi_field("name");
    let driver_version = nvidia_smi_field("driver_version");
    let vram_gb = nvidia_smi_field("memory.total")
        .and_then(|s| s.parse::<f64>().ok())
        .map(|mb| mb / 1024.0);

    let torch_index = torch_index_for_cuda(cuda_version.as_deref()).to_string();
    let torch_index_url = format!("https://download.pytorch.org/whl/{}", torch_index);

    let (system_python, python_version) = detect_system_python();

    SystemInfo {
        os,
        gpu_name,
        driver_version,
        cuda_version,
        vram_gb,
        torch_index,
        torch_index_url,
        system_python,
        python_version,
        venv_ready: venv_python().exists(),
        tinyq4_ready: crate::engine::find_tinyq4().is_ok(),
        disk_free_gb: disk_free_gb(&config_dir().parent().map(|p| p.to_path_buf())
            .unwrap_or_else(|| PathBuf::from("/"))),
        setup_done: setup_marker().exists(),
    }
}

// ── Install orchestration ──────────────────────────────────────────────────────

/// Python packages for the creative stack (Full setup). torch/torchvision are
/// installed separately with the CUDA-matched index URL.
const CREATIVE_PKGS: &[&str] = &[
    "diffusers", "transformers", "peft", "safetensors",
    "numpy", "pillow", "soundfile", "kokoro",
];

fn emit_log(window: &WebviewWindow, line: impl AsRef<str>) {
    let _ = window.emit("setup-log", line.as_ref().to_string());
}
fn emit_step(window: &WebviewWindow, step: &str, status: &str) {
    let _ = window.emit("setup-step", serde_json::json!({ "step": step, "status": status }));
}

/// Run a command, streaming stdout+stderr line-by-line as "setup-log" events.
/// Returns Ok(()) on exit 0, Err with the code otherwise.
fn run_streamed(window: &WebviewWindow, mut cmd: Command, label: &str) -> Result<(), String> {
    use std::io::{BufRead, BufReader};
    use std::process::Stdio;
    emit_log(window, format!("$ {label}"));
    cmd.no_console().stdout(Stdio::piped()).stderr(Stdio::piped());
    let mut child = cmd.spawn().map_err(|e| format!("failed to start: {e}"))?;
    if let Some(out) = child.stdout.take() {
        for line in BufReader::new(out).lines().map_while(Result::ok) {
            emit_log(window, line);
        }
    }
    if let Some(err) = child.stderr.take() {
        for line in BufReader::new(err).lines().map_while(Result::ok) {
            emit_log(window, line);
        }
    }
    let status = child.wait().map_err(|e| e.to_string())?;
    if status.success() { Ok(()) }
    else { Err(format!("{label} exited with {}", status.code().unwrap_or(-1))) }
}

/// Create the managed venv if missing, using the detected base python.
fn ensure_venv(window: &WebviewWindow, base_python: &str) -> Result<(), String> {
    let vp = venv_python();
    if vp.exists() {
        emit_log(window, "managed venv already present");
        return Ok(());
    }
    std::fs::create_dir_all(config_dir()).map_err(|e| e.to_string())?;
    let mut cmd = Command::new(base_python);
    cmd.arg("-m").arg("venv").arg(managed_venv());
    run_streamed(window, cmd, "python -m venv")?;
    // Upgrade pip inside the venv.
    let mut up = Command::new(venv_python());
    up.args(["-m", "pip", "install", "--upgrade", "pip", "wheel"]);
    run_streamed(window, up, "pip upgrade")?;
    Ok(())
}

fn pip_install(window: &WebviewWindow, args: &[&str], label: &str) -> Result<(), String> {
    let mut cmd = Command::new(venv_python());
    cmd.args(["-m", "pip", "install"]);
    cmd.args(args);
    run_streamed(window, cmd, label)
}

/// Full = creative + core. Fast = core only (just tinyq4).
#[tauri::command]
pub async fn run_setup(window: WebviewWindow, profile: String) -> Result<(), String> {
    let info = detect_system();
    let full = profile == "full";
    emit_log(&window, format!("Setup profile: {profile}"));
    emit_log(&window, format!("OS: {} · CUDA: {} · torch wheel: {}",
        info.os, info.cuda_version.clone().unwrap_or_else(|| "none".into()), info.torch_index));

    // ── Core: the LLM engine ───────────────────────────────────────────────────
    // tinyq4 ships bundled with the app (CUDA + CPU builds + libcudart). The right
    // one is selected at runtime — nothing to install for chat/agent.
    emit_step(&window, "engine", "running");
    match crate::engine::find_tinyq4() {
        Ok(p) => emit_log(&window, format!("engine ready (bundled): {}", p.display())),
        Err(e) => emit_log(&window, format!("engine: {e}")),
    }
    emit_step(&window, "engine", "done");

    // ── Creative stack (Full only) — Python is needed only for image/video/voice ─
    if full {
        let base = info.system_python.clone()
            .ok_or("Full setup needs Python 3.10+ for the image/video/voice tools. Install Python 3 and retry, or pick Fast.")?;
        emit_step(&window, "venv", "running");
        ensure_venv(&window, &base)?;
        emit_step(&window, "venv", "done");

        emit_step(&window, "torch", "running");
        // CUDA-matched torch. cpu index also valid (download.pytorch.org/whl/cpu).
        pip_install(&window,
            &["torch", "torchvision", "--index-url", &info.torch_index_url],
            &format!("pip install torch ({})", info.torch_index))?;
        emit_step(&window, "torch", "done");

        emit_step(&window, "creative", "running");
        let mut args = vec!["--upgrade"];
        args.extend_from_slice(CREATIVE_PKGS);
        pip_install(&window, &args, "pip install diffusers/transformers/…")?;
        emit_step(&window, "creative", "done");
    }

    // Persist the marker so the wizard doesn't reappear, recording what we did.
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64).unwrap_or(0);
    let marker = serde_json::json!({
        "profile": profile,
        "torch_index": info.torch_index,
        "venv": managed_venv().to_string_lossy(),
        "ts": ts,
    });
    std::fs::create_dir_all(config_dir()).ok();
    std::fs::write(setup_marker(), marker.to_string()).map_err(|e| e.to_string())?;

    emit_step(&window, "done", "done");
    emit_log(&window, "✓ Setup complete.");
    Ok(())
}

// ── Starter model download ─────────────────────────────────────────────────────

/// Stream-download a GGUF from HuggingFace into the models dir, emitting
/// "model-progress" {downloaded, total, done} events. Writes to a .part file and
/// atomically renames on completion. Skips if the file already exists.
#[tauri::command]
pub async fn download_starter_model(
    window: WebviewWindow,
    repo: String,
    file: String,
    models_dir: String,
    token: Option<String>,
) -> Result<String, String> {
    use futures::StreamExt;
    use tokio::io::AsyncWriteExt;

    // Each model in its own folder so scan_models_dir uses the folder as the name.
    let folder = file.strip_suffix(".gguf").unwrap_or(&file);
    let dest_dir = PathBuf::from(&models_dir).join(folder);
    std::fs::create_dir_all(&dest_dir).map_err(|e| e.to_string())?;
    let dest = dest_dir.join(&file);

    // Already downloaded? report done and return.
    if let Ok(m) = std::fs::metadata(&dest) {
        if m.len() > 0 {
            let _ = window.emit("model-progress",
                serde_json::json!({"downloaded": m.len(), "total": m.len(), "done": true}));
            return Ok(dest.to_string_lossy().into_owned());
        }
    }

    let url = format!("https://huggingface.co/{}/resolve/main/{}?download=true", repo, file);
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(3600))
        .build().map_err(|e| e.to_string())?;
    let mut req = client.get(&url).header("User-Agent", "Saient");
    if let Some(t) = token.as_deref().map(str::trim).filter(|t| !t.is_empty()) {
        req = req.bearer_auth(t);   // gated/private repos
    }
    let resp = req.send().await.map_err(|e| format!("request failed: {e}"))?;
    if resp.status() == reqwest::StatusCode::UNAUTHORIZED || resp.status() == reqwest::StatusCode::FORBIDDEN {
        return Err("This model is gated/private. Add a Hugging Face access token (and accept the model's licence on HF) to download it.".into());
    }
    if !resp.status().is_success() {
        return Err(format!("download failed: HTTP {}", resp.status()));
    }
    let total = resp.content_length().unwrap_or(0);

    // Guardrail: refuse to start if there clearly isn't room (keep ~0.5 GB headroom).
    if total > 0 {
        let free_gb = disk_free_gb(&dest_dir);
        let need_gb = total as f64 / 1e9 + 0.5;
        if free_gb > 0.0 && free_gb < need_gb {
            return Err(format!(
                "Not enough disk space: this model needs ~{:.1} GB but only {:.1} GB is free in the models folder.",
                total as f64 / 1e9, free_gb
            ));
        }
    }

    let part = dest.with_extension("part");

    // Stream to the .part file; on ANY error, clean it up so we never leave a
    // half-written stub that looks like a real model to the scanner.
    let result: Result<u64, String> = async {
        let mut out = tokio::fs::File::create(&part).await.map_err(|e| e.to_string())?;
        let mut stream = resp.bytes_stream();
        let mut downloaded: u64 = 0;
        let mut last = std::time::Instant::now();
        while let Some(chunk) = stream.next().await {
            let chunk = chunk.map_err(|e| e.to_string())?;
            out.write_all(&chunk).await.map_err(|e| e.to_string())?;
            downloaded += chunk.len() as u64;
            if last.elapsed().as_millis() >= 200 {
                let _ = window.emit("model-progress",
                    serde_json::json!({"downloaded": downloaded, "total": total}));
                last = std::time::Instant::now();
            }
        }
        out.flush().await.map_err(|e| e.to_string())?;
        drop(out);
        std::fs::rename(&part, &dest).map_err(|e| e.to_string())?;
        Ok(downloaded)
    }.await;

    let downloaded = match result {
        Ok(d) => d,
        Err(e) => {
            let _ = std::fs::remove_file(&part);   // tidy: drop the half file
            return Err(e);
        }
    };

    let _ = window.emit("model-progress",
        serde_json::json!({"downloaded": downloaded, "total": total, "done": true}));
    Ok(dest.to_string_lossy().into_owned())
}

// ── Hugging Face: list a repo's GGUF files ──────────────────────────────────────

#[derive(Serialize)]
pub struct HfFile {
    pub file: String,
    pub size: u64,
}

/// List the `.gguf` files (with sizes) in a HuggingFace repo's main branch, so the
/// UI can let the user pick a quant before downloading. Files are then fetched with
/// `download_starter_model`, which drops them in the managed models dir.
#[tauri::command]
pub async fn hf_list_gguf(repo: String, token: Option<String>) -> Result<Vec<HfFile>, String> {
    #[derive(serde::Deserialize)]
    struct Lfs {
        #[serde(default)]
        size: u64,
    }
    #[derive(serde::Deserialize)]
    struct TreeEntry {
        #[serde(rename = "type")]
        kind: String,
        path: String,
        #[serde(default)]
        size: u64,
        #[serde(default)]
        lfs: Option<Lfs>,
    }

    let repo = repo.trim().trim_matches('/');
    if repo.is_empty() || repo.split('/').count() != 2 {
        return Err("Enter a HuggingFace repo like \"owner/name\".".into());
    }

    // Non-recursive: GGUF repos keep quants at the top level. Avoids nested paths.
    let url = format!("https://huggingface.co/api/models/{repo}/tree/main");
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(30))
        .build()
        .map_err(|e| e.to_string())?;
    let mut request = client.get(&url).header("User-Agent", "Saient");
    if let Some(t) = token.as_deref().map(str::trim).filter(|t| !t.is_empty()) {
        request = request.bearer_auth(t);   // gated/private repos
    }
    let resp = request
        .send()
        .await
        .map_err(|e| format!("request failed: {e}"))?;
    if resp.status() == reqwest::StatusCode::NOT_FOUND {
        return Err(format!("Repo \"{repo}\" not found on HuggingFace."));
    }
    if resp.status() == reqwest::StatusCode::UNAUTHORIZED || resp.status() == reqwest::StatusCode::FORBIDDEN {
        return Err("This repo is gated/private. Add a Hugging Face access token (and accept its licence on HF) to use it.".into());
    }
    if !resp.status().is_success() {
        return Err(format!("HuggingFace returned HTTP {}.", resp.status()));
    }
    let entries: Vec<TreeEntry> = resp
        .json()
        .await
        .map_err(|e| format!("couldn't read repo file list: {e}"))?;

    let mut files: Vec<HfFile> = entries
        .into_iter()
        .filter(|e| e.kind == "file" && e.path.to_lowercase().ends_with(".gguf"))
        .map(|e| {
            // For LFS files the real size lives under `lfs`; fall back to `size`.
            let size = e.lfs.map(|l| l.size).filter(|&s| s > 0).unwrap_or(e.size);
            HfFile { file: e.path, size }
        })
        .collect();

    if files.is_empty() {
        return Err("No .gguf files found in that repo.".into());
    }
    files.sort_by(|a, b| a.file.to_lowercase().cmp(&b.file.to_lowercase()));
    Ok(files)
}

/// Mark setup as complete without installing (e.g. user already has everything).
#[tauri::command]
pub fn skip_setup() -> Result<(), String> {
    std::fs::create_dir_all(config_dir()).ok();
    std::fs::write(setup_marker(),
        serde_json::json!({"profile": "skipped"}).to_string())
        .map_err(|e| e.to_string())
}

/// Re-open the wizard later by clearing the marker.
#[tauri::command]
pub fn reset_setup() -> Result<(), String> {
    std::fs::remove_file(setup_marker()).ok();
    Ok(())
}
