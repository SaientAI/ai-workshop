//! setup.rs — first-run setup wizard backend.
//!
//! Detects the machine (OS, GPU/driver/CUDA, Python, disk), maps CUDA → the right
//! PyTorch wheel index, and runs the install (managed venv + CUDA-matched pip +
//! starter model download) streaming progress to the frontend.
//!
//! Everything installs into a self-contained managed dir so we never touch the
//! user's system Python:  ~/.config/ai-workshop/venv  (Linux/Mac)
//!                        %APPDATA%\ai-workshop\venv   (Windows)

use serde::Serialize;
use std::path::PathBuf;
use std::process::Command;
use tauri::{Emitter, WebviewWindow};

// ── Paths ──────────────────────────────────────────────────────────────────────

fn config_dir() -> PathBuf {
    #[cfg(target_os = "windows")]
    {
        if let Ok(appdata) = std::env::var("APPDATA") {
            return PathBuf::from(appdata).join("ai-workshop");
        }
    }
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
    PathBuf::from(home).join(".config/ai-workshop")
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
    let out = Command::new("nvidia-smi").output().ok()?;
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
        if let Ok(out) = Command::new(c).arg("--version").output() {
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
        ]).output() {
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
    cmd.stdout(Stdio::piped()).stderr(Stdio::piped());
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
    let base = info.system_python.clone()
        .ok_or("No Python 3 found. Install Python 3.10+ and retry.")?;

    let full = profile == "full";
    emit_log(&window, format!("Setup profile: {profile}"));
    emit_log(&window, format!("OS: {} · CUDA: {} · torch wheel: {}",
        info.os, info.cuda_version.clone().unwrap_or_else(|| "none".into()), info.torch_index));

    // ── Managed venv ─────────────────────────────────────────────────────────
    emit_step(&window, "venv", "running");
    ensure_venv(&window, &base)?;
    emit_step(&window, "venv", "done");

    // ── Core: tinyq4 (the LLM engine) ──────────────────────────────────────────
    emit_step(&window, "tinyq4", "running");
    pip_install(&window, &["tinyq4"], "pip install tinyq4")?;
    emit_step(&window, "tinyq4", "done");

    // ── Creative stack (Full only) ─────────────────────────────────────────────
    if full {
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
    let resp = client.get(&url).send().await.map_err(|e| format!("request failed: {e}"))?;
    if !resp.status().is_success() {
        return Err(format!("download failed: HTTP {}", resp.status()));
    }
    let total = resp.content_length().unwrap_or(0);

    let part = dest.with_extension("part");
    let mut out = tokio::fs::File::create(&part).await.map_err(|e| e.to_string())?;
    let mut stream = resp.bytes_stream();
    let mut downloaded: u64 = 0;
    let mut last = std::time::Instant::now();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|e| e.to_string())?;
        out.write_all(&chunk).await.map_err(|e| e.to_string())?;
        downloaded += chunk.len() as u64;
        // Throttle UI updates to ~5/sec.
        if last.elapsed().as_millis() >= 200 {
            let _ = window.emit("model-progress",
                serde_json::json!({"downloaded": downloaded, "total": total}));
            last = std::time::Instant::now();
        }
    }
    out.flush().await.map_err(|e| e.to_string())?;
    drop(out);
    std::fs::rename(&part, &dest).map_err(|e| e.to_string())?;
    let _ = window.emit("model-progress",
        serde_json::json!({"downloaded": downloaded, "total": total, "done": true}));
    Ok(dest.to_string_lossy().into_owned())
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
