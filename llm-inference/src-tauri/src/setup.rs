//! setup.rs — first-run setup wizard backend.
//!
//! Detects the machine (OS, GPU/driver/CUDA, Python, disk), maps CUDA → the right
//! PyTorch wheel index, and runs the install (managed venv + CUDA-matched pip +
//! starter model download) streaming progress to the frontend.
//!
//! Everything installs into a self-contained managed dir so we never touch the
//! user's system Python:  <repo>/data/config/saient/venv by default.

use serde::Serialize;
use std::path::PathBuf;
use std::process::Command;
use crate::resolve::NoConsole;
use tauri::{Emitter, WebviewWindow};

// ── Paths ──────────────────────────────────────────────────────────────────────

pub fn config_dir() -> PathBuf {
    crate::paths::config_dir()
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

/// One-time migration of old platform dirs into the repo-local data tree so
/// existing installs keep their venv, sessions, LoRAs, and settings.
pub fn migrate_legacy_dirs() {
    let mut pairs: Vec<(PathBuf, PathBuf)> = Vec::new();
    if let Some(old) = crate::paths::legacy_config_dir("ai-workshop") {
        pairs.push((old, crate::paths::config_dir()));
    }
    if let Some(old) = crate::paths::legacy_config_dir("saient") {
        pairs.push((old, crate::paths::config_base_dir().join("saient")));
    }
    if let Some(old) = crate::paths::legacy_config_dir("saient-dev") {
        pairs.push((old, crate::paths::config_base_dir().join("saient-dev")));
    }
    if let Some(old) = crate::paths::legacy_share_dir("ai-workshop") {
        pairs.push((old, crate::paths::share_dir()));
    }
    if let Some(old) = crate::paths::legacy_share_dir("saient") {
        pairs.push((old, crate::paths::share_dir()));
    }
    for (old, new) in pairs {
        if !old.exists() { continue; }
        if std::fs::symlink_metadata(&old)
            .map(|m| m.file_type().is_symlink())
            .unwrap_or(false)
        {
            continue;
        }
        if old.canonicalize().ok() == new.canonicalize().ok() {
            continue;
        }
        if !new.exists() {
            // Clean move — the new-brand dir doesn't exist yet.
            if let Some(parent) = new.parent() { let _ = std::fs::create_dir_all(parent); }
            let _ = std::fs::rename(&old, &new);
        } else {
            // Partial rebrand: new/ already exists (e.g. upscale weights got written there
            // first), so the plain rename was SKIPPED and pre-rebrand caches/sessions/venv
            // were stranded at old/. Merge each item across (skip ones already in new), then
            // drop old/ if it's now empty. Renames are instant (same filesystem).
            if let Ok(entries) = std::fs::read_dir(&old) {
                for e in entries.flatten() {
                    let dest = new.join(e.file_name());
                    if !dest.exists() { let _ = std::fs::rename(e.path(), &dest); }
                }
            }
            let _ = std::fs::remove_dir(&old);
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
    pub creative_ready: bool,              // image runtime modules resolve
    pub tinyq4_ready: bool,                // tinyq4 resolvable
    pub disk_free_gb: f64,
    pub setup_done: bool,                  // marker present
    pub setup_profile: Option<String>,     // "full" | "fast" | "skipped"
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
        // On an actual first launch neither data/config nor its Saient child
        // necessarily exists yet. statvfs rejects that path and the wizard used
        // to claim the fresh machine had 0 GB free. The containing filesystem is
        // the same, so ask the nearest existing ancestor without creating state
        // during a read-only system probe.
        let probe = dir.ancestors().find(|path| path.exists()).unwrap_or(dir);
        let path = CString::new(probe.as_os_str().as_bytes()).ok();
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

#[cfg(test)]
mod disk_tests {
    use super::disk_free_gb;

    #[test]
    fn fresh_nonexistent_config_path_uses_its_existing_filesystem() {
        let missing = std::env::temp_dir()
            .join(format!("saient-disk-probe-{}", std::process::id()))
            .join("not-created")
            .join("config");
        assert!(!missing.exists());
        assert!(disk_free_gb(&missing) > 0.0);
    }
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

    let marker = std::fs::read_to_string(setup_marker()).ok();
    let setup_profile = marker.as_deref()
        .and_then(|raw| serde_json::from_str::<serde_json::Value>(raw).ok())
        .and_then(|value| value.get("profile")?.as_str().map(str::to_owned));

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
        creative_ready: crate::resolve::image_runtime_ready(),
        tinyq4_ready: crate::engine::find_tinyq4().is_ok(),
        disk_free_gb: disk_free_gb(&config_dir().parent().map(|p| p.to_path_buf())
            .unwrap_or_else(|| PathBuf::from("/"))),
        setup_done: marker.is_some(),
        setup_profile,
    }
}

// ── Install orchestration ──────────────────────────────────────────────────────

/// Python packages for the creative stack (Full setup). torch/torchvision are
/// installed separately with the CUDA-matched index URL.
const CREATIVE_PKGS: &[&str] = &[
    // The bundled Moondream revision declares transformers 4.52.4 in its
    // config. Transformers 5.x changed PreTrainedModel's tied-weight contract
    // and fails after loading the real weights, so this compatibility pin is a
    // runtime requirement rather than a best-effort package preference.
    "diffusers", "transformers==4.52.4", "accelerate", "huggingface_hub", "peft", "safetensors",
    "numpy", "pillow", "soundfile", "kokoro", "imageio", "imageio-ffmpeg",
    // Video enhancers: spandrel (RealESRGAN upscale) + face restoration (CodeFormer, registered
    // by spandrel_extra_arches; facexlib does detect/align/paste). CodeFormer and facexlib
    // detector/parser weights are checked locally by enhance_video.py before those APIs are
    // constructed, so a missing optional asset is an explicit error rather than a runtime download.
    "spandrel", "spandrel_extra_arches", "facexlib",
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
    let stdout = child.stdout.take().ok_or("failed to capture setup stdout")?;
    let stderr = child.stderr.take().ok_or("failed to capture setup stderr")?;
    let stdout_window = window.clone();
    let stderr_window = window.clone();
    // Drain both pipes concurrently. Reading stdout to EOF before touching
    // stderr can deadlock when pip/Python fills the stderr pipe.
    let stdout_reader = std::thread::spawn(move || {
        let mut tail = std::collections::VecDeque::with_capacity(12);
        for line in BufReader::new(stdout).lines().map_while(Result::ok) {
            emit_log(&stdout_window, &line);
            if tail.len() == 12 { tail.pop_front(); }
            tail.push_back(line);
        }
        tail
    });
    let stderr_reader = std::thread::spawn(move || {
        let mut tail = std::collections::VecDeque::with_capacity(12);
        for line in BufReader::new(stderr).lines().map_while(Result::ok) {
            emit_log(&stderr_window, &line);
            if tail.len() == 12 { tail.pop_front(); }
            tail.push_back(line);
        }
        tail
    });
    let status = child.wait().map_err(|e| e.to_string())?;
    let stdout_tail = stdout_reader.join().map_err(|_| "setup stdout reader stopped unexpectedly")?;
    let stderr_tail = stderr_reader.join().map_err(|_| "setup stderr reader stopped unexpectedly")?;
    if status.success() { Ok(()) }
    else {
        let detail = stderr_tail.iter().chain(stdout_tail.iter())
            .rev().find(|line| !line.trim().is_empty())
            .map(|line| format!(": {}", line.trim()))
            .unwrap_or_default();
        Err(format!("{label} exited with {}{detail}", status.code().unwrap_or(-1)))
    }
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

fn prefetch_runtime_assets(window: &WebviewWindow) -> Result<(), String> {
    crate::internet::require_setup_enabled("Offline runtime asset download")?;

    // Download a fixed, pinned allow-list straight into named runtime folders.
    // This deliberately does not import huggingface_hub: there is no shared HF
    // cache, Xet staging area, or datasets cache to grow behind the user's back.
    let code = r#"
import json, os, shutil, time, urllib.parse, urllib.request
from pathlib import Path, PurePosixPath

ROOT = Path(os.environ["SAIENT_RUNTIME_ASSETS_DIR"])
SPECS = (
    ("hexgrad/Kokoro-82M", "f3ff3571791e39611d31c381e3a41a3af07b4987", ROOT / "voice" / "kokoro-82m",
     lambda p: p in {"config.json", "kokoro-v1_0.pth"} or (p.startswith("voices/") and p.endswith(".pt"))),
    ("vikhyatk/moondream2", "6b714b26eea5cbd9f31e4edb2541c170afa935ba", ROOT / "vision" / "moondream2",
     lambda p: p.endswith((".json", ".py", ".txt", ".safetensors"))),
    ("moondream/starmie-v1", "35192e10a54e36eabe0a7cc57a2c1aab371cafc5", ROOT / "vision" / "starmie-v1",
     lambda p: p in {"special_tokens_map.json", "tokenizer.json", "tokenizer_config.json"}),
)
HEADERS = {"User-Agent": "Saient setup/1.0"}

def open_url(url):
    return urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS), timeout=120)

def manifest(repo, revision, allowed):
    url = f"https://huggingface.co/api/models/{repo}/tree/{revision}?recursive=true&expand=true"
    with open_url(url) as response:
        rows = json.load(response)
    files = []
    for row in rows:
        name = row.get("path", "")
        rel = PurePosixPath(name)
        if row.get("type") == "file" and allowed(name):
            if rel.is_absolute() or ".." in rel.parts:
                raise RuntimeError(f"unsafe runtime asset path: {name}")
            files.append((name, int(row.get("size") or 0)))
    if not files or any(size <= 0 for _, size in files):
        raise RuntimeError(f"incomplete pinned asset manifest for {repo}")
    return files

plans = []
missing = 0
for repo, revision, dest, allowed in SPECS:
    files = manifest(repo, revision, allowed)
    plans.append((repo, revision, dest, files))
    for name, size in files:
        target = dest.joinpath(*PurePosixPath(name).parts)
        if not target.is_file() or target.stat().st_size != size:
            missing += size

ROOT.mkdir(parents=True, exist_ok=True)
free = shutil.disk_usage(ROOT).free
headroom = 1_000_000_000
if free < missing + headroom:
    raise RuntimeError(
        f"Not enough disk space for voice and vision assets: "
        f"{missing / 1e9:.1f} GB still needed plus 1.0 GB safety space, but only {free / 1e9:.1f} GB is free. "
        "Open Settings > Setup and clear the legacy Hugging Face cache, then retry."
    )

for repo, revision, dest, files in plans:
    print(f"downloading managed runtime assets: {repo}", flush=True)
    for name, size in files:
        target = dest.joinpath(*PurePosixPath(name).parts)
        if target.is_file() and target.stat().st_size == size:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        part = target.with_name(target.name + ".part")
        url = f"https://huggingface.co/{repo}/resolve/{revision}/{urllib.parse.quote(name, safe='/')}?download=true"
        last_error = None
        for attempt in range(2):
            try:
                if part.exists():
                    part.unlink()
                with open_url(url) as response, part.open("wb") as output:
                    shutil.copyfileobj(response, output, length=4 * 1024 * 1024)
                actual = part.stat().st_size
                if actual != size:
                    raise RuntimeError(f"size mismatch for {repo}/{name}: expected {size}, received {actual}")
                os.replace(part, target)
                last_error = None
                break
            except Exception as exc:
                last_error = exc
                if part.exists():
                    part.unlink()
                if attempt == 0:
                    time.sleep(1)
        if last_error is not None:
            raise last_error

required = (
    ROOT / "voice" / "kokoro-82m" / "config.json",
    ROOT / "voice" / "kokoro-82m" / "kokoro-v1_0.pth",
    ROOT / "voice" / "kokoro-82m" / "voices" / "af_heart.pt",
    ROOT / "vision" / "moondream2" / "model.safetensors",
    ROOT / "vision" / "starmie-v1" / "tokenizer.json",
)
for path in required:
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"managed runtime asset missing after download: {path}")
print("managed voice and vision assets are ready; no Hugging Face cache was created", flush=True)
"#;
    let mut assets = Command::new(venv_python());
    assets.arg("-c").arg(code)
        .env(crate::paths::RUNTIME_ASSETS_ENV, crate::paths::runtime_assets_dir());
    run_streamed(window, assets, "download managed Kokoro and Moondream assets")?;

    // misaki/spaCy otherwise starts a pip subprocess from the first TTS call.
    // Install it while the setup-scoped capability is visibly active instead.
    let mut language = Command::new(venv_python());
    language.args(["-m", "spacy", "download", "en_core_web_sm"]);
    run_streamed(window, language, "install Kokoro English language data")
}

/// Full = creative + core. Fast = core only (just tinyq4).
#[tauri::command]
pub async fn run_setup(window: WebviewWindow, profile: String) -> Result<(), String> {
    let info = detect_system();
    let full = profile == "full";
    if full {
        crate::internet::require_setup_enabled("Full setup downloads")?;
        if info.disk_free_gb < 12.0 {
            return Err(format!(
                "Full setup needs at least 12 GB free for Python and managed voice/vision assets, but only {:.1} GB is free. Open Settings > Setup and clear the legacy Hugging Face cache, then retry.",
                info.disk_free_gb
            ));
        }
    }
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
        if info.cuda_version.is_some() && info.os != "macos" {
            match pip_install(&window, &["--upgrade", "bitsandbytes"], "pip install bitsandbytes (optional SD3 quantization)") {
                Ok(()) => emit_log(&window, "bitsandbytes ready for optional SD3 quantized paths"),
                Err(e) => emit_log(&window, format!("bitsandbytes optional install skipped: {e}")),
            }
        }
        emit_step(&window, "creative", "done");

        emit_step(&window, "assets", "running");
        prefetch_runtime_assets(&window)?;
        emit_step(&window, "assets", "done");
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
    crate::internet::require_setup_enabled("Starter model download")?;

    // Text models live in Saient's dedicated LLM folder under the models root, in
    // their own subfolder so scan_models_dir uses the folder as the name.
    let folder = file.strip_suffix(".gguf").unwrap_or(&file);
    let dest_dir = PathBuf::from(&models_dir).join("llm").join(folder);
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
    crate::internet::require_enabled("Hugging Face browsing")?;

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

// ── Hugging Face: search + generic file list + download (for the creative studios) ──

#[derive(Serialize)]
pub struct HfRepo {
    pub id: String,
    pub downloads: u64,
    pub likes: u64,
}

/// Search HuggingFace for model repos. `filter` is a pipeline tag like
/// "text-to-image" (img gen) — pass None for an unfiltered search.
#[tauri::command]
pub async fn hf_search(query: String, filter: Option<String>, token: Option<String>) -> Result<Vec<HfRepo>, String> {
    crate::internet::require_enabled("Hugging Face search")?;

    #[derive(serde::Deserialize)]
    struct Raw { id: String, #[serde(default)] downloads: u64, #[serde(default)] likes: u64 }

    let q = query.trim();
    if q.is_empty() { return Err("Type something to search for.".into()); }
    let mut url = format!(
        "https://huggingface.co/api/models?search={}&sort=downloads&direction=-1&limit=20",
        urlencoding(q)
    );
    if let Some(f) = filter.as_deref().filter(|f| !f.is_empty()) {
        url.push_str(&format!("&filter={}", urlencoding(f)));
    }
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(30))
        .build().map_err(|e| e.to_string())?;
    let mut req = client.get(&url).header("User-Agent", "Saient");
    if let Some(t) = token.as_deref().map(str::trim).filter(|t| !t.is_empty()) {
        req = req.bearer_auth(t);
    }
    let resp = req.send().await.map_err(|e| format!("search failed: {e}"))?;
    if !resp.status().is_success() {
        return Err(format!("HuggingFace search returned HTTP {}.", resp.status()));
    }
    let raw: Vec<Raw> = resp.json().await.map_err(|e| format!("couldn't read results: {e}"))?;
    Ok(raw.into_iter().map(|r| HfRepo { id: r.id, downloads: r.downloads, likes: r.likes }).collect())
}

/// List files in a repo whose name ends with one of `exts` (e.g. [".safetensors"]).
#[tauri::command]
pub async fn hf_list_files(repo: String, exts: Vec<String>, token: Option<String>) -> Result<Vec<HfFile>, String> {
    crate::internet::require_enabled("Hugging Face browsing")?;

    #[derive(serde::Deserialize)]
    struct Lfs { #[serde(default)] size: u64 }
    #[derive(serde::Deserialize)]
    struct TreeEntry { #[serde(rename = "type")] kind: String, path: String, #[serde(default)] size: u64, #[serde(default)] lfs: Option<Lfs> }

    let repo = repo.trim().trim_matches('/');
    if repo.split('/').count() != 2 { return Err("Enter a repo like \"owner/name\".".into()); }
    let exts: Vec<String> = exts.iter().map(|e| e.to_lowercase()).collect();

    let url = format!("https://huggingface.co/api/models/{repo}/tree/main");
    let client = reqwest::Client::builder().timeout(std::time::Duration::from_secs(30)).build().map_err(|e| e.to_string())?;
    let mut req = client.get(&url).header("User-Agent", "Saient");
    if let Some(t) = token.as_deref().map(str::trim).filter(|t| !t.is_empty()) { req = req.bearer_auth(t); }
    let resp = req.send().await.map_err(|e| format!("request failed: {e}"))?;
    if resp.status() == reqwest::StatusCode::NOT_FOUND { return Err(format!("Repo \"{repo}\" not found.")); }
    if resp.status() == reqwest::StatusCode::UNAUTHORIZED || resp.status() == reqwest::StatusCode::FORBIDDEN {
        return Err("This repo is gated/private. Add a Hugging Face token to use it.".into());
    }
    if !resp.status().is_success() { return Err(format!("HuggingFace returned HTTP {}.", resp.status())); }
    let entries: Vec<TreeEntry> = resp.json().await.map_err(|e| format!("couldn't read file list: {e}"))?;

    let mut files: Vec<HfFile> = entries.into_iter()
        .filter(|e| e.kind == "file" && exts.iter().any(|x| e.path.to_lowercase().ends_with(x)))
        .map(|e| { let size = e.lfs.map(|l| l.size).filter(|&s| s > 0).unwrap_or(e.size); HfFile { file: e.path, size } })
        .collect();
    if files.is_empty() { return Err("No matching model files found in that repo.".into()); }
    files.sort_by(|a, b| b.size.cmp(&a.size)); // biggest first (usually the full model)
    Ok(files)
}

/// Download a single file from a repo straight into the managed folder for `target`
/// ("checkpoint" | "lora"), so it shows up in that studio with no path fiddling.
#[tauri::command]
pub async fn download_hf_file(
    window: WebviewWindow,
    repo: String,
    file: String,
    target: String,
    token: Option<String>,
) -> Result<String, String> {
    use futures::StreamExt;
    use tokio::io::AsyncWriteExt;
    crate::internet::require_enabled("Hugging Face downloads")?;

    let dest_dir = match target.as_str() {
        "lora" => crate::resolve::loras_download_dir(),
        "gguf" => crate::resolve::models_download_dir(),
        _      => crate::resolve::checkpoints_download_dir(),
    };
    std::fs::create_dir_all(&dest_dir).map_err(|e| e.to_string())?;
    let fname = std::path::Path::new(&file).file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_else(|| file.clone());
    let dest = dest_dir.join(&fname);

    if let Ok(m) = std::fs::metadata(&dest) {
        if m.len() > 0 {
            let _ = window.emit("model-progress", serde_json::json!({"downloaded": m.len(), "total": m.len(), "done": true}));
            return Ok(dest.to_string_lossy().into_owned());
        }
    }

    let url = format!("https://huggingface.co/{repo}/resolve/main/{file}?download=true");
    let client = reqwest::Client::builder().timeout(std::time::Duration::from_secs(3600)).build().map_err(|e| e.to_string())?;
    let mut rq = client.get(&url).header("User-Agent", "Saient");
    if let Some(t) = token.as_deref().map(str::trim).filter(|t| !t.is_empty()) { rq = rq.bearer_auth(t); }
    let resp = rq.send().await.map_err(|e| format!("request failed: {e}"))?;
    if resp.status() == reqwest::StatusCode::UNAUTHORIZED || resp.status() == reqwest::StatusCode::FORBIDDEN {
        return Err("This model is gated/private. Add a Hugging Face token (and accept its licence on HF) to download it.".into());
    }
    if !resp.status().is_success() { return Err(format!("download failed: HTTP {}", resp.status())); }
    let total = resp.content_length().unwrap_or(0);

    if total > 0 {
        let free_gb = disk_free_gb(&dest_dir);
        let need_gb = total as f64 / 1e9 + 0.5;
        if free_gb > 0.0 && free_gb < need_gb {
            return Err(format!("Not enough disk space: needs ~{:.1} GB but only {:.1} GB free.", total as f64/1e9, free_gb));
        }
    }

    let part = dest.with_extension("part");
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
                let _ = window.emit("model-progress", serde_json::json!({"downloaded": downloaded, "total": total}));
                last = std::time::Instant::now();
            }
        }
        out.flush().await.map_err(|e| e.to_string())?;
        drop(out);
        std::fs::rename(&part, &dest).map_err(|e| e.to_string())?;
        Ok(downloaded)
    }.await;

    let downloaded = match result { Ok(d) => d, Err(e) => { let _ = std::fs::remove_file(&part); return Err(e); } };
    let _ = window.emit("model-progress", serde_json::json!({"downloaded": downloaded, "total": total, "done": true}));
    Ok(dest.to_string_lossy().into_owned())
}

/// Download a full Hugging Face diffusers repository into the managed image-model
/// folder so multi-file models such as SD3.5 and SDXL Turbo appear under [BASE].
#[tauri::command]
pub async fn download_hf_repo(
    window: WebviewWindow,
    repo: String,
    target: String,
    token: Option<String>,
) -> Result<String, String> {
    crate::internet::require_enabled("Hugging Face downloads")?;

    let repo = repo.trim().trim_matches('/').to_string();
    if repo.split('/').count() != 2 {
        return Err("Enter a repo like \"owner/name\".".into());
    }
    if target != "model" {
        return Err("Full repo downloads are only supported for image base models.".into());
    }

    let folder = repo.split('/').next_back().unwrap_or("model");
    let dest = crate::resolve::image_models_download_dir().join(folder);
    if dest.join("model_index.json").exists() {
        let size = dir_size(&dest);
        let _ = window.emit("model-progress", serde_json::json!({"downloaded": size, "total": size, "done": true}));
        return Ok(dest.to_string_lossy().into_owned());
    }
    std::fs::create_dir_all(&dest).map_err(|e| e.to_string())?;

    let python = crate::resolve::find_python().map_err(|e| e.to_string())?;
    let script = r#"
import json, os, shutil, sys
from pathlib import Path
try:
    from huggingface_hub import snapshot_download
except Exception as e:
    print(json.dumps({"error": "huggingface_hub is not installed. Run Full setup, then retry. Details: " + str(e)}))
    sys.exit(2)

repo = os.environ["SAIENT_HF_REPO"]
dest = os.environ["SAIENT_HF_DEST"]
token = os.environ.get("HF_TOKEN") or None
low = repo.lower()

if "sdxl-turbo" in low:
    # Turbo is the 16 GB fast path. Fetch fp16 weights only; the full safetensors
    # duplicate the same model at roughly double the disk size.
    base_allow = [
        "model_index.json", "*.json", "*.txt", "*.model",
        "*.fp16.safetensors", "*/*.fp16.safetensors",
    ]
else:
    base_allow = [
        "model_index.json", "*.json", "*.txt", "*.model",
        "scheduler/*", "tokenizer/*", "tokenizer_2/*", "tokenizer_3/*",
        "text_encoder/*", "text_encoder_2/*", "text_encoder_3/*",
        "unet/*", "vae/*", "transformer/*",
    ]
ignore = ["*.bin", "*.ckpt", "*.pt", "*.pth", "*.onnx", "*.onnx_data", "*.msgpack", "*.h5", "*.gguf"]

kwargs = dict(
    repo_id=repo,
    local_dir=dest,
    token=token,
    allow_patterns=base_allow,
    ignore_patterns=ignore,
    resume_download=True,
)
try:
    try:
        snapshot_download(local_dir_use_symlinks=False, **kwargs)
    except TypeError:
        snapshot_download(**kwargs)
finally:
    cache = Path(dest) / ".cache" / "huggingface"
    if cache.exists():
        shutil.rmtree(cache)
    cache_parent = cache.parent
    if cache_parent.exists() and not any(cache_parent.iterdir()):
        cache_parent.rmdir()
print(json.dumps({"ok": True, "dest": dest}))
"#;

    let window_for_task = window.clone();
    let result = tokio::task::spawn_blocking(move || {
        let _ = window_for_task.emit("model-progress", serde_json::json!({"downloaded": 0, "total": 0}));
        let mut cmd = Command::new(python);
        crate::paths::apply_child_env(&mut cmd);
        cmd.arg("-c").arg(script)
            .env("SAIENT_HF_REPO", &repo)
            .env("SAIENT_HF_DEST", &dest)
            // This is an explicit, user-triggered model download. Generation
            // children never receive these overrides.
            .env_remove("HF_HUB_OFFLINE")
            .env_remove("TRANSFORMERS_OFFLINE")
            .env_remove("DIFFUSERS_OFFLINE")
            .env_remove("HF_DATASETS_OFFLINE");
        if let Some(t) = token.as_deref().map(str::trim).filter(|t| !t.is_empty()) {
            cmd.env("HF_TOKEN", t);
        }
        cmd.no_console();
        let out = cmd.output().map_err(|e| format!("failed to start Hugging Face download: {e}"))?;
        let stdout = String::from_utf8_lossy(&out.stdout);
        let stderr = String::from_utf8_lossy(&out.stderr);
        if !out.status.success() {
            let msg = stdout.lines()
                .filter_map(|line| serde_json::from_str::<serde_json::Value>(line).ok())
                .find_map(|v| v["error"].as_str().map(|s| s.to_string()))
                .unwrap_or_else(|| {
                    let detail = stderr.trim();
                    if detail.is_empty() { format!("Hugging Face download exited with {}", out.status.code().unwrap_or(-1)) }
                    else { detail.to_string() }
                });
            return Err(msg);
        }
        if !dest.join("model_index.json").exists() {
            return Err("Download finished but model_index.json was not found. This does not look like a diffusers image model.".into());
        }
        let size = dir_size(&dest);
        let _ = window_for_task.emit("model-progress", serde_json::json!({"downloaded": size, "total": size, "done": true}));
        Ok(dest.to_string_lossy().into_owned())
    }).await.map_err(|e| format!("download task join: {e}"))?;

    result
}

fn dir_size(path: &std::path::Path) -> u64 {
    walkdir::WalkDir::new(path)
        .follow_links(false)
        .into_iter()
        .filter_map(Result::ok)
        .filter(|entry| entry.file_type().is_file())
        .filter_map(|entry| entry.metadata().ok().map(|metadata| metadata.len()))
        .sum()
}

fn incomplete_size(path: &std::path::Path) -> u64 {
    walkdir::WalkDir::new(path)
        .follow_links(false)
        .into_iter()
        .filter_map(Result::ok)
        .filter(|entry| entry.file_type().is_file())
        .filter(|entry| entry.file_name().to_string_lossy().ends_with(".incomplete")
            || entry.file_name().to_string_lossy().ends_with(".part"))
        .filter_map(|entry| entry.metadata().ok().map(|metadata| metadata.len()))
        .sum()
}

#[derive(Serialize)]
pub struct ManagedStorageInfo {
    pub legacy_hf_bytes: u64,
    pub legacy_incomplete_bytes: u64,
    pub runtime_assets_bytes: u64,
    pub temporary_hf_bytes: u64,
}

#[tauri::command]
pub fn managed_storage_info() -> ManagedStorageInfo {
    let legacy = crate::paths::legacy_hf_cache_dir();
    ManagedStorageInfo {
        legacy_hf_bytes: dir_size(&legacy),
        legacy_incomplete_bytes: incomplete_size(&legacy),
        runtime_assets_bytes: dir_size(&crate::paths::runtime_assets_dir()),
        temporary_hf_bytes: dir_size(&crate::paths::hf_home()),
    }
}

fn remove_exact_managed_dir(parent: &std::path::Path, name: &str) -> Result<u64, String> {
    let target = parent.join(name);
    if target.parent() != Some(parent) || target.file_name().and_then(|part| part.to_str()) != Some(name) {
        return Err("refusing unsafe managed-cache path".into());
    }
    let metadata = match std::fs::symlink_metadata(&target) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(0),
        Err(error) => return Err(format!("could not inspect {}: {error}", target.display())),
    };
    if metadata.file_type().is_symlink() || !metadata.is_dir() {
        return Err(format!("refusing to remove unexpected cache path: {}", target.display()));
    }
    let bytes = dir_size(&target);
    std::fs::remove_dir_all(&target)
        .map_err(|error| format!("could not remove {}: {error}", target.display()))?;
    if target.exists() {
        return Err(format!("cache path still exists after removal: {}", target.display()));
    }
    Ok(bytes)
}

/// Remove cache/state from releases up to 1.0.17. Named runtime assets and the
/// user's models/configuration are intentionally outside both targets.
#[tauri::command]
pub fn clear_legacy_hf_cache() -> Result<u64, String> {
    let data = crate::paths::data_dir();
    let mut freed = remove_exact_managed_dir(&data, "huggingface")?;
    let runtime_tmp = data.join("runtime-tmp");
    freed += remove_exact_managed_dir(&runtime_tmp, "huggingface")?;
    Ok(freed)
}

/// Best-effort startup cleanup for prior run-only library state.
pub fn clear_ephemeral_hf_state() -> Result<u64, String> {
    remove_exact_managed_dir(&crate::paths::data_dir().join("runtime-tmp"), "huggingface")
}

/// Minimal URL-encoder for query values (avoids pulling a crate).
fn urlencoding(s: &str) -> String {
    s.bytes().map(|b| match b {
        b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => (b as char).to_string(),
        b' ' => "+".to_string(),
        _ => format!("%{b:02X}"),
    }).collect()
}

/// Mark setup as complete without installing (e.g. user already has everything).
#[tauri::command]
pub fn skip_setup() -> Result<(), String> {
    std::fs::create_dir_all(config_dir()).ok();
    std::fs::write(setup_marker(),
        serde_json::json!({"profile": "skipped"}).to_string())
        .map_err(|e| e.to_string())
}

fn remove_setup_marker(marker: &std::path::Path) -> Result<(), String> {
    match std::fs::remove_file(marker) {
        Ok(()) => Ok(()),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(e) => Err(format!("Could not reopen setup: {e}")),
    }
}

/// Re-open the wizard later by clearing only its completion marker. Existing
/// environments, models, settings, and downloaded assets remain untouched.
#[tauri::command]
pub fn reset_setup() -> Result<(), String> {
    remove_setup_marker(&setup_marker())
}

#[cfg(test)]
mod reset_setup_tests {
    use super::remove_setup_marker;

    #[test]
    fn reset_removes_only_the_marker() {
        let root = tempfile::tempdir().unwrap();
        let marker = root.path().join("setup_done.json");
        let venv_python = root.path().join("venv").join("bin").join("python");
        std::fs::create_dir_all(venv_python.parent().unwrap()).unwrap();
        std::fs::write(&venv_python, "keep").unwrap();
        std::fs::write(&marker, r#"{"profile":"skipped"}"#).unwrap();

        remove_setup_marker(&marker).unwrap();

        assert!(!marker.exists());
        assert_eq!(std::fs::read_to_string(venv_python).unwrap(), "keep");
    }

    #[test]
    fn reset_is_idempotent_when_marker_is_absent() {
        let root = tempfile::tempdir().unwrap();
        remove_setup_marker(&root.path().join("setup_done.json")).unwrap();
    }

    #[test]
    fn reset_reports_a_real_removal_error() {
        let root = tempfile::tempdir().unwrap();
        let directory_at_marker_path = root.path().join("setup_done.json");
        std::fs::create_dir(&directory_at_marker_path).unwrap();
        let error = remove_setup_marker(&directory_at_marker_path).unwrap_err();
        assert!(error.starts_with("Could not reopen setup:"));
    }
}

#[cfg(test)]
mod managed_storage_tests {
    use super::{dir_size, incomplete_size, remove_exact_managed_dir};

    #[test]
    fn removal_is_scoped_to_the_named_managed_child() {
        let root = tempfile::tempdir().unwrap();
        let cache = root.path().join("huggingface");
        let keep = root.path().join("models");
        std::fs::create_dir_all(cache.join("nested")).unwrap();
        std::fs::create_dir_all(&keep).unwrap();
        std::fs::write(cache.join("nested").join("download.incomplete"), [1_u8; 32]).unwrap();
        std::fs::write(keep.join("model.bin"), [2_u8; 16]).unwrap();

        assert_eq!(dir_size(&cache), 32);
        assert_eq!(incomplete_size(&cache), 32);
        assert_eq!(remove_exact_managed_dir(root.path(), "huggingface").unwrap(), 32);
        assert!(!cache.exists());
        assert_eq!(std::fs::read(keep.join("model.bin")).unwrap(), [2_u8; 16]);
    }

    #[cfg(unix)]
    #[test]
    fn removal_refuses_a_symlink_target() {
        use std::os::unix::fs::symlink;

        let root = tempfile::tempdir().unwrap();
        let outside = tempfile::tempdir().unwrap();
        std::fs::write(outside.path().join("keep"), "private").unwrap();
        symlink(outside.path(), root.path().join("huggingface")).unwrap();

        let error = remove_exact_managed_dir(root.path(), "huggingface").unwrap_err();
        assert!(error.contains("refusing to remove unexpected cache path"));
        assert_eq!(std::fs::read_to_string(outside.path().join("keep")).unwrap(), "private");
    }
}
