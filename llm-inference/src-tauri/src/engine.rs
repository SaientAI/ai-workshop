use anyhow::{Context, Result};
use futures::StreamExt;
use reqwest::Client;
#[cfg(unix)]
use std::os::unix::process::CommandExt;   // pre_exec (Unix only)
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex, atomic::{AtomicBool, Ordering}};
use std::time::Duration;

use crate::gguf::{GgufFile, ModelSummary};
use crate::resolve::NoConsole;

const DESKTOP_BIND_HOST: &str = "127.0.0.1";

// Known ports for user-started tinyq4 instances.
pub const PROBE_PORTS: &[u16] = &[18081, 18082, 33115, 18080];

// Disk-persistent PID file — survives crashes so the next launch can reap any leftover.
// Cross-platform temp paths (Windows has no /tmp).
fn pid_file() -> PathBuf { std::env::temp_dir().join("saient-server.pid") }
fn engine_log() -> PathBuf { std::env::temp_dir().join("saient-tinyq4.log") }
const FIRST_TOKEN_TIMEOUT_SECS: u64 = 300;
const STREAM_IDLE_TIMEOUT_SECS: u64 = 180;

// Track PIDs we spawned so we can kill stale ones before spawning again.
// Covers only the current process lifetime; PID_FILE covers cross-session leaks.
static SPAWNED_PIDS: std::sync::OnceLock<Mutex<Vec<u32>>> = std::sync::OnceLock::new();

fn spawned_pids() -> &'static Mutex<Vec<u32>> {
    SPAWNED_PIDS.get_or_init(|| Mutex::new(Vec::new()))
}

/// Terminate a process by PID — `kill` on Unix, `taskkill /F /T` on Windows.
fn kill_pid(pid: u32) -> std::io::Result<std::process::ExitStatus> {
    #[cfg(unix)]
    { Command::new("kill").arg(pid.to_string()).status() }
    #[cfg(windows)]
    { Command::new("taskkill").args(["/F", "/T", "/PID", &pid.to_string()]).no_console().status() }
}

/// Kill every server we spawned — both in this session and any leftover from a
/// previous session (tracked via PID_FILE).  Safe to call multiple times.
pub fn kill_our_stale_servers() {
    // In-session PIDs (current process lifetime)
    let mut pids = spawned_pids().lock().unwrap();
    for pid in pids.drain(..) {
        let _ = kill_pid(pid);
    }
    drop(pids);

    // Cross-session: read the PID file left by a previous app run
    if let Ok(s) = std::fs::read_to_string(pid_file()) {
        if let Ok(pid) = s.trim().parse::<u32>() {
            let _ = kill_pid(pid);
        }
        let _ = std::fs::remove_file(pid_file());
    }
}

// ── Public types ───────────────────────────────────────────────────────────────

pub struct Engine {
    _process: Option<Child>,  // Some when we spawned it; None when attached to external
    pub port: u16,
    pub client: Client,
    pub summary: ModelSummary,
}

impl Drop for Engine {
    fn drop(&mut self) {
        if let Some(child) = self._process.as_mut() {
            let pid = child.id();
            let _ = child.kill();
            let _ = child.wait();
            unregister_spawned_pid(pid);
        }
    }
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct DetectedServer {
    pub pid: u32,
    pub port: u16,
    pub model_id: String,
    pub model_path: Option<String>,
    pub source: String,
}

#[derive(Clone)]
pub struct SamplingParams {
    pub max_tokens: usize,
    pub temperature: f64,
    pub top_p: f64,
    pub top_k: usize,
    pub repeat_penalty: f32,
    pub seed: u64,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            max_tokens: 512,
            temperature: 0.7,
            top_p: 0.95,
            top_k: 40,
            repeat_penalty: 1.1,
            seed: 42,
        }
    }
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct GenerateResult {
    pub tokens_generated: usize,
    pub total_tokens: usize,
    pub elapsed_ms: u64,
    pub tokens_per_sec: f64,
}

// ── Engine ─────────────────────────────────────────────────────────────────────

impl Engine {
    /// Load a model, auto-attaching to an already-running server if one has it loaded.
    pub async fn load(
        model_path: &Path,
        _gpu_layers: i32,   // tinyq4 currently manages GPU residency internally
        _ctx_size: u32,
    ) -> Result<Self> {
        let client = Client::builder()
            .timeout(Duration::from_secs(1800))
            .build()?;

        let gguf_meta = GgufFile::open(model_path)
            .with_context(|| format!("Cannot read model: {:?}", model_path))?;
        let summary = gguf_meta.summary();
        drop(gguf_meta);

        let model_stem = model_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("")
            .to_lowercase();

        // Kill any servers WE previously spawned before probing or starting fresh
        kill_our_stale_servers();

        // Locally-started tinyq4 servers can be discovered by command line. This avoids
        // loading the same huge GGUF twice and lets us attach after a UI restart.
        for srv in detect_tinyq4_servers() {
            if !detected_server_matches(&srv, model_path, &model_stem) {
                continue;
            }
            if health_ok(&client, srv.port).await {
                return Ok(Self {
                    _process: None,
                    port: srv.port,
                    client: client.clone(),
                    summary,
                });
            }
        }

        // Probe known tinyq4 ports for an already-loaded user-started instance.
        for &port in PROBE_PORTS {
            if let Ok(eng) = try_attach(&client, port, &model_stem, summary.clone()).await {
                return Ok(eng);
            }
        }

        // Nothing found — start our own tinyq4 runtime. Keep this stack
        // self-contained; do not fall back to llama-server.
        let server_bin = find_tinyq4()?;
        let port = find_free_port()?;

        // tinyq4 CLI: tinyq4 <model_path> --server <port>
        let args = vec![
            model_path.to_str().unwrap().to_string(),
            "--server".to_string(), port.to_string(),
        ];

        let log = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(engine_log())
            .ok();
        let stdout = log
            .as_ref()
            .and_then(|file| file.try_clone().ok())
            .map(Stdio::from)
            .unwrap_or_else(Stdio::null);
        let stderr = log
            .map(Stdio::from)
            .unwrap_or_else(Stdio::null);

        // Put the bundled CUDA runtime (libcudart, shipped next to the binary) on the
        // loader path so the CUDA engine runs without a system CUDA toolkit install.
        let (lib_key, lib_val) = server_bin.parent()
            .map(engine_lib_env)
            .unwrap_or_else(|| ("LD_LIBRARY_PATH".into(),
                std::env::var("LD_LIBRARY_PATH").unwrap_or_default()));

        let mut cmd = Command::new(&server_bin);
        cmd.args(&args)
            .env(&lib_key, &lib_val)
            .no_console()
            .stdout(stdout)
            .stderr(stderr);
        harden_desktop_server_command(&mut cmd);
        // On Unix, have the kernel SIGTERM the child if the app dies, so a crash can't
        // orphan a VRAM-holding server. On Windows we rely on kill_our_stale_servers()
        // (called at startup and on window-close) plus the PID_FILE for cleanup.
        #[cfg(unix)]
        unsafe {
            cmd.pre_exec(|| {
                libc::prctl(
                    libc::PR_SET_PDEATHSIG,
                    libc::SIGTERM as libc::c_ulong,
                    0, 0, 0,
                );
                Ok(())
            });
        }
        let mut process = cmd.spawn()
            .with_context(|| format!("Failed to start {:?}", server_bin))?;

        let pid = process.id();
        spawned_pids().lock().unwrap().push(pid);
        let _ = std::fs::write(pid_file(), pid.to_string());

        if let Err(e) = wait_for_ready(&client, port, &mut process).await {
            let _ = process.kill();
            let _ = process.wait();
            return Err(e);
        }
        let _ = server_bin; // (kept for the spawn above; readiness errors are self-describing)

        Ok(Self { _process: Some(process), port, client, summary })
    }

    /// Attach to a server that is already running at `port`.
    /// Still reads GGUF metadata for the summary; does NOT kill the server on drop.
    pub async fn attach(port: u16, model_path: &Path) -> Result<Self> {
        let client = Client::builder()
            .timeout(Duration::from_secs(1800))
            .build()?;

        let gguf_meta = GgufFile::open(model_path)
            .with_context(|| format!("Cannot read model: {:?}", model_path))?;
        let summary = gguf_meta.summary();
        drop(gguf_meta);

        let url = format!("http://127.0.0.1:{}/health", port);
        client.get(&url)
            .timeout(Duration::from_millis(2000))
            .send()
            .await
            .context("Server not responding")?
            .error_for_status()
            .context("Server health check failed")?;

        Ok(Self { _process: None, port, client, summary })
    }
}

pub fn detect_tinyq4_servers() -> Vec<DetectedServer> {
    detect_tinyq4_servers_impl()
}

pub fn detect_local_model_servers() -> Vec<DetectedServer> {
    detect_tinyq4_servers_impl()
}

#[cfg(target_os = "linux")]
fn detect_tinyq4_servers_impl() -> Vec<DetectedServer> {
    detect_local_model_servers_impl()
        .into_iter()
        .filter(|server| server.source.starts_with("tinyq4 "))
        .collect()
}

#[cfg(target_os = "linux")]
fn detect_local_model_servers_impl() -> Vec<DetectedServer> {
    let mut out = Vec::new();
    let Ok(entries) = std::fs::read_dir("/proc") else {
        return out;
    };

    for entry in entries.flatten() {
        let Some(pid) = entry.file_name().to_str().and_then(|s| s.parse::<u32>().ok()) else {
            continue;
        };
        let Ok(raw) = std::fs::read(format!("/proc/{}/cmdline", pid)) else {
            continue;
        };
        if raw.is_empty() {
            continue;
        }
        let args: Vec<String> = raw
            .split(|b| *b == 0)
            .filter(|part| !part.is_empty())
            .map(|part| String::from_utf8_lossy(part).into_owned())
            .collect();
        if args.is_empty() {
            continue;
        }

        if !looks_like_tinyq4(&args[0]) {
            continue;
        }

        let port = args.windows(2)
            .find_map(|w| (w[0] == "--server").then(|| w[1].parse::<u16>().ok()).flatten());
        let Some(port) = port else { continue; };

        let model_path = args.iter()
            .skip(1)
            .find(|arg| !arg.starts_with('-') && arg.to_lowercase().ends_with(".gguf"))
            .cloned();
        let model_id = model_path
            .as_deref()
            .and_then(|p| Path::new(p).file_stem())
            .and_then(|s| s.to_str())
            .unwrap_or("tinyq4")
            .to_string();

        out.push(DetectedServer {
            pid,
            port,
            model_id,
            model_path,
            source: format!("tinyq4 pid {}", pid),
        });
    }

    out.sort_by_key(|s| s.port);
    out.dedup_by_key(|s| s.port);
    out
}

#[cfg(not(target_os = "linux"))]
fn detect_tinyq4_servers_impl() -> Vec<DetectedServer> {
    Vec::new()
}

#[cfg(not(target_os = "linux"))]
fn detect_local_model_servers_impl() -> Vec<DetectedServer> {
    Vec::new()
}

pub fn stop_tinyq4_server(port: u16) -> Result<bool> {
    stop_local_model_server(port)
}

pub fn stop_local_model_server(port: u16) -> Result<bool> {
    let Some(server) = detect_local_model_servers()
        .into_iter()
        .find(|server| server.port == port)
    else {
        return Ok(false);
    };

    kill_pid(server.pid)
        .with_context(|| format!("failed to stop tinyq4 server on port {}", port))?;

    if let Ok(s) = std::fs::read_to_string(pid_file()) {
        if s.trim().parse::<u32>().ok() == Some(server.pid) {
            let _ = std::fs::remove_file(pid_file());
        }
    }

    if let Ok(mut pids) = spawned_pids().lock() {
        pids.retain(|pid| *pid != server.pid);
    }

    Ok(true)
}

fn unregister_spawned_pid(pid: u32) {
    if let Ok(s) = std::fs::read_to_string(pid_file()) {
        if s.trim().parse::<u32>().ok() == Some(pid) {
            let _ = std::fs::remove_file(pid_file());
        }
    }

    if let Ok(mut pids) = spawned_pids().lock() {
        pids.retain(|known| *known != pid);
    }
}

fn looks_like_tinyq4(cmd: &str) -> bool {
    Path::new(cmd)
        .file_name()
        .and_then(|s| s.to_str())
        .map(|s| s == "tinyq4")
        .unwrap_or(false)
}

fn detected_server_matches(srv: &DetectedServer, model_path: &Path, model_stem: &str) -> bool {
    if let Some(path) = &srv.model_path {
        let path = Path::new(path);
        if same_file(path, model_path) {
            return true;
        }
        let detected_stem = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("")
            .to_lowercase();
        if !detected_stem.is_empty()
            && (detected_stem.contains(model_stem) || model_stem.contains(&detected_stem))
        {
            return true;
        }
    }

    let running_id = srv.model_id.to_lowercase();
    !running_id.is_empty()
        && (running_id.contains(model_stem) || model_stem.contains(&running_id))
}

fn same_file(a: &Path, b: &Path) -> bool {
    match (a.canonicalize(), b.canonicalize()) {
        (Ok(a), Ok(b)) => a == b,
        _ => a == b,
    }
}

async fn health_ok(client: &Client, port: u16) -> bool {
    let url = format!("http://127.0.0.1:{}/health", port);
    let Ok(resp) = client
        .get(&url)
        .timeout(Duration::from_millis(800))
        .send()
        .await
    else {
        return false;
    };
    if !resp.status().is_success() {
        return false;
    }
    resp.text()
        .await
        .map(|text| text.trim() == "ok")
        .unwrap_or(false)
}

/// Find the PID listening on `port` via `ss`, returns None if not found.
// Try to attach to `port` if it's serving the model whose stem is `model_stem`.
async fn try_attach(
    client: &Client,
    port: u16,
    model_stem: &str,
    summary: ModelSummary,
) -> Result<Engine> {
    let health_url = format!("http://127.0.0.1:{}/health", port);
    let resp = client
        .get(&health_url)
        .timeout(Duration::from_millis(300))
        .send()
        .await?;
    if !resp.status().is_success() {
        anyhow::bail!("not healthy");
    }

    // Ask the server what model it has loaded
    let models_url = format!("http://127.0.0.1:{}/v1/models", port);
    if let Ok(mr) = client
        .get(&models_url)
        .timeout(Duration::from_millis(300))
        .send()
        .await
    {
        if mr.status().is_success() {
            if let Ok(v) = mr.json::<serde_json::Value>().await {
                let running_id = v["data"]
                    .as_array()
                    .and_then(|a| a.first())
                    .and_then(|m| m["id"].as_str())
                    .unwrap_or("")
                    .to_lowercase();
                // Fuzzy match: filename stem is a substring of the model ID or vice versa
                if !running_id.is_empty()
                    && (running_id.contains(model_stem) || model_stem.contains(&running_id))
                {
                    return Ok(Engine {
                        _process: None,
                        port,
                        client: client.clone(),
                        summary,
                    });
                }
            }
        }
    }
    anyhow::bail!("no matching model on port {}", port)
}

// ── Streaming generation — called without holding the Engine lock ──────────────

pub async fn stream_generate(
    client: &Client,
    port: u16,
    messages: &[serde_json::Value],
    params: &SamplingParams,
    mut on_token: impl FnMut(&str, bool),       // (piece, is_reasoning)
    mut on_prefill: impl FnMut(usize, usize),   // (done, total) during prefill
    stop_flag: Arc<AtomicBool>,
) -> Result<GenerateResult> {
    let body = serde_json::json!({
        "messages": messages,
        "max_tokens": params.max_tokens,
        "temperature": params.temperature,
        "top_p": params.top_p,
        "top_k": params.top_k,
        "repeat_penalty": params.repeat_penalty,
        "seed": params.seed,
        "stream": true,
        "cache_prompt": true,
    });

    let t_start = std::time::Instant::now();
    let mut tokens_generated = 0usize;

    let resp = client
        .post(format!("http://127.0.0.1:{}/v1/chat/completions", port))
        .json(&body)
        .send()
        .await
        .context("Request to tinyq4 request failed")?;

    if !resp.status().is_success() {
        let status = resp.status();
        let text = resp.text().await.unwrap_or_default();
        anyhow::bail!("tinyq4 returned {}: {}", status, text);
    }

    let mut byte_stream = resp.bytes_stream();
    let mut buf = String::new();

    let mut saw_data = false;
    let mut last_token_at = t_start;

    // CPU inference is far slower than GPU — prefill of a large prompt can take
    // many minutes before the first token — so give it much longer before we
    // assume the model/quant is broken. Tokens and prefill progress reset the clock.
    let have_gpu = gpu_available();
    let first_token_to = if have_gpu { FIRST_TOKEN_TIMEOUT_SECS } else { 1800 };
    let idle_to        = if have_gpu { STREAM_IDLE_TIMEOUT_SECS } else { 600 };
    let no_token_msg = format!(
        "tinyq4 accepted the request but produced no tokens after {first_token_to}s.{} The model may still be warming up, or this architecture/quantization isn't supported by tinyq4.",
        if have_gpu { "" } else { " (CPU inference is slow — a smaller / more-quantized model will respond faster.)" }
    );

    'outer: loop {
        if stop_flag.load(Ordering::Relaxed) { break; }
        let timeout_secs = if saw_data { idle_to } else { first_token_to };
        if last_token_at.elapsed() > Duration::from_secs(timeout_secs) {
            if !saw_data {
                anyhow::bail!("{no_token_msg}");
            }
            anyhow::bail!("tinyq4 stopped sending tokens for {idle_to} seconds");
        }
        let chunk = match tokio::time::timeout(Duration::from_secs(timeout_secs), byte_stream.next()).await {
            Ok(Some(chunk)) => chunk,
            Ok(None) => break,
            Err(_) if !saw_data => {
                anyhow::bail!("{no_token_msg}");
            }
            Err(_) => {
                anyhow::bail!("tinyq4 stopped sending tokens for {idle_to} seconds");
            }
        };
        buf.push_str(&String::from_utf8_lossy(&chunk?));

        while let Some(nl) = buf.find('\n') {
            let line = buf[..nl].trim().to_string();
            buf.drain(..=nl);

            if let Some(data) = line.strip_prefix("data: ") {
                if data == "[DONE]" { break 'outer; }
                if let Ok(v) = serde_json::from_str::<serde_json::Value>(data) {
                    let delta = &v["choices"][0]["delta"];
                    if let Some(pp) = delta.get("prefill_progress") {
                        let done  = pp["done"].as_u64().unwrap_or(0) as usize;
                        let total = pp["total"].as_u64().unwrap_or(0) as usize;
                        last_token_at = std::time::Instant::now(); // reset timeout
                        on_prefill(done, total);
                    } else if let Some(piece) = delta["reasoning_content"].as_str() {
                        if !piece.is_empty() {
                            saw_data = true;
                            last_token_at = std::time::Instant::now();
                            on_token(piece, true);
                            tokens_generated += 1;
                        }
                    } else if let Some(piece) = delta["content"].as_str() {
                        if !piece.is_empty() {
                            saw_data = true;
                            last_token_at = std::time::Instant::now();
                            on_token(piece, false);
                            tokens_generated += 1;
                        }
                    }
                }
            }
        }
    }

    if tokens_generated == 0 {
        anyhow::bail!(
            "tinyq4 accepted the request but the stream ended without text. Try a shorter prompt, lower Max tokens, or restart the server."
        );
    }

    let elapsed = t_start.elapsed();
    Ok(GenerateResult {
        tokens_generated,
        total_tokens: tokens_generated,
        elapsed_ms: elapsed.as_millis() as u64,
        tokens_per_sec: if elapsed.as_secs_f64() > 0.0 {
            tokens_generated as f64 / elapsed.as_secs_f64()
        } else { 0.0 },
    })
}

// ── Discovery ──────────────────────────────────────────────────────────────────

/// The Tauri resource directory, captured at startup (see main.rs setup hook). The
/// bundled engine lives in `<resource_dir>/engine/`.
static RESOURCE_DIR: std::sync::OnceLock<PathBuf> = std::sync::OnceLock::new();
pub fn set_resource_dir(p: PathBuf) { let _ = RESOURCE_DIR.set(p); }

fn cuda_bin_name() -> &'static str { if cfg!(windows) { "tinyq4-cuda.exe" } else { "tinyq4-cuda" } }
fn cpu_bin_name()  -> &'static str { if cfg!(windows) { "tinyq4-cpu.exe" }  else { "tinyq4-cpu" } }

/// Directory holding the bundled engine binaries + CUDA runtime lib (libcudart).
/// Override with SAIENT_ENGINE_DIR; otherwise `<resource_dir>/engine`, else next to the exe.
pub fn engine_dir() -> Option<PathBuf> {
    // Only accept a directory that actually holds an engine binary — not just any
    // dir that happens to exist (the bundled layout varies per platform/installer).
    let has_engine = |d: &Path| d.join(cuda_bin_name()).exists() || d.join(cpu_bin_name()).exists();

    if let Ok(d) = std::env::var("SAIENT_ENGINE_DIR") {
        let p = PathBuf::from(d);
        if has_engine(&p) { return Some(p); }
    }
    if let Some(rd) = RESOURCE_DIR.get() {
        // Tauri preserves the resource's relative path, so `resources/engine/*`
        // installs to <resource_dir>/resources/engine — check that and the flat form.
        for cand in [rd.join("engine"), rd.join("resources").join("engine")] {
            if has_engine(&cand) { return Some(cand); }
        }
    }
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            for cand in [
                dir.join("engine"),
                dir.join("resources").join("engine"),
                dir.to_path_buf(),
            ] {
                if has_engine(&cand) { return Some(cand); }
            }
        }
    }
    None
}

/// A usable NVIDIA GPU is present (driver installed). We don't need the CUDA toolkit —
/// only the driver — because the engine bundles its own libcudart.
pub fn gpu_available() -> bool {
    std::process::Command::new("nvidia-smi").arg("-L").no_console().output()
        .map(|o| o.status.success() && !o.stdout.is_empty())
        .unwrap_or(false)
}

/// Environment to make the bundled libcudart loadable without a system CUDA install:
/// prepend the engine dir to LD_LIBRARY_PATH (Linux) / PATH (Windows).
fn engine_lib_env(lib_dir: &Path) -> (String, String) {
    let key = if cfg!(windows) { "PATH" } else { "LD_LIBRARY_PATH" };
    let sep = if cfg!(windows) { ";" } else { ":" };
    let dir = lib_dir.to_string_lossy().into_owned();
    let val = match std::env::var(key) {
        Ok(cur) if !cur.is_empty() => format!("{dir}{sep}{cur}"),
        _ => dir,
    };
    (key.to_string(), val)
}

pub fn find_tinyq4() -> Result<PathBuf> {
    // 1. Explicit override.
    if let Ok(p) = std::env::var("TINYQ4_PATH") {
        let path = PathBuf::from(&p);
        if path.exists() { return Ok(path); }
    }

    // 2. Bundled engine (shipped with the app). Prefer the CUDA build when a GPU/driver
    //    is present, else the CPU build. libcudart ships alongside; engine_lib_env() puts
    //    it on the loader path at spawn time, so no system CUDA install is needed.
    if let Some(dir) = engine_dir() {
        if gpu_available() {
            let cuda = dir.join(cuda_bin_name());
            if cuda.exists() { return Ok(cuda); }
        }
        let cpu = dir.join(cpu_bin_name());
        if cpu.exists() { return Ok(cpu); }
    }

    // 3. The setup wizard's managed venv (where `run_setup` pip-installs tinyq4).
    let venv_tinyq4 = crate::setup::venv_bin("tinyq4");
    if venv_tinyq4.exists() { return Ok(venv_tinyq4); }

    // 3. pip-installed console script — `pip install tinyq4` drops a `tinyq4`
    //    executable into a bin dir that is normally on $PATH.
    if let Some(p) = which_in_path("tinyq4") {
        return Ok(p);
    }

    // 3. Common pip bin dirs that may not be on this process's $PATH (the app is
    //    often launched from a desktop entry with a minimal environment).
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
    let pip_dirs = [
        format!("{home}/.local/bin/tinyq4"),
        "/usr/local/bin/tinyq4".to_string(),
        "/usr/bin/tinyq4".to_string(),
    ];
    for p in &pip_dirs {
        if Path::new(p).exists() { return Ok(PathBuf::from(p)); }
    }

    // 4. Local source build — dev only, never baked into a release binary.
    #[cfg(debug_assertions)]
    {
        let runtime = crate::paths::llm_runtime_dir();
        for sub in ["tinyq4/target/release/tinyq4",
                    "tinyq4/target/debug/tinyq4"] {
            let p = runtime.join(sub);
            if p.exists() { return Ok(p); }
        }
    }

    // 5. Bundled next to our own binary.
    if let Ok(exe) = std::env::current_exe() {
        let bundled = exe.parent().unwrap_or(Path::new(".")).join("tinyq4");
        if bundled.exists() { return Ok(bundled); }
    }

    anyhow::bail!(
        "Saient's inference engine could not be found. It normally ships bundled with the app; \
         reinstall Saient, or set the TINYQ4_PATH env var to a tinyq4 binary."
    )
}

/// Find an executable by name on `$PATH`, returning the first match.
fn which_in_path(name: &str) -> Option<PathBuf> {
    let path_var = std::env::var_os("PATH")?;
    for dir in std::env::split_paths(&path_var) {
        let candidate = dir.join(name);
        if candidate.is_file() {
            // On Unix, confirm it's executable.
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                if let Ok(meta) = std::fs::metadata(&candidate) {
                    if meta.permissions().mode() & 0o111 != 0 {
                        return Some(candidate);
                    }
                }
            }
            #[cfg(not(unix))]
            { return Some(candidate); }
        }
    }
    None
}

// ── Helpers ────────────────────────────────────────────────────────────────────

fn find_free_port() -> Result<u16> {
    let listener = std::net::TcpListener::bind((DESKTOP_BIND_HOST, 0))?;
    Ok(listener.local_addr()?.port())
}

/// Standalone tinyq4 supports an explicit wildcard bind. The desktop parent
/// must override, not inherit, that environment variable.
fn harden_desktop_server_command(command: &mut Command) {
    command.env("TINYQ4_BIND", DESKTOP_BIND_HOST);
}

async fn wait_for_ready(client: &Client, port: u16, process: &mut Child) -> Result<()> {
    let url = format!("http://127.0.0.1:{}/health", port);
    for _ in 0..600 {   // up to 5 minutes; larger GGUFs can spend minutes uploading to CUDA
        // Fail FAST if tinyq4 already died (e.g. a cudaMalloc OOM panic). Otherwise we'd
        // pointlessly poll a dead process for 5 minutes and report a useless timeout.
        if let Ok(Some(status)) = process.try_wait() {
            let why = last_tinyq4_error()
                .unwrap_or_else(|| format!("tinyq4 exited ({status}) during model load"));
            anyhow::bail!("{why}");
        }
        tokio::time::sleep(Duration::from_millis(500)).await;
        if let Ok(r) = client.get(&url).send().await {
            if r.status().is_success() { return Ok(()); }
        }
    }
    let tail = last_tinyq4_error().map(|s| format!(" ({s})")).unwrap_or_default();
    anyhow::bail!("Model didn't become ready within 5 minutes{tail}")
}

/// Extract the most useful error line from the tinyq4 log so the UI can show the real
/// reason a load failed (CUDA OOM, panic, …) instead of a generic "did not become ready".
fn last_tinyq4_error() -> Option<String> {
    let log = std::fs::read_to_string(engine_log()).ok()?;
    let recent: Vec<&str> = log.lines().rev().take(40).collect();
    for line in &recent {
        let l = line.to_lowercase();
        if l.contains("cudamalloc") || l.contains("out of memory") || l.contains(" oom") {
            return Some(format!(
                "GPU out of memory — another model is still using VRAM. Stop it (or your other tinyq4/Saient server) and try again."
            ));
        }
        if l.contains("panicked") {
            return Some(format!("tinyq4 crashed: {}", line.trim()));
        }
    }
    recent.first().map(|s| s.trim().to_string()).filter(|s| !s.is_empty())
}

/// Free VRAM in MiB via nvidia-smi (first GPU). `None` if nvidia-smi is unavailable.
pub fn gpu_free_mib() -> Option<u64> {
    let out = Command::new("nvidia-smi")
        .args(["--query-gpu=memory.free", "--format=csv,noheader,nounits"])
        .no_console()
        .output()
        .ok()?;
    if !out.status.success() { return None; }
    String::from_utf8_lossy(&out.stdout)
        .lines()
        .next()?
        .trim()
        .parse()
        .ok()
}

// ── Handle ─────────────────────────────────────────────────────────────────────

pub type EngineHandle = Arc<tokio::sync::Mutex<Option<Engine>>>;

pub fn new_handle() -> EngineHandle {
    Arc::new(tokio::sync::Mutex::new(None))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn desktop_child_overrides_a_hostile_wildcard_bind_environment() {
        let mut command = Command::new("tinyq4-test-only");
        harden_desktop_server_command(&mut command);
        let bind = command
            .get_envs()
            .find(|(key, _)| *key == "TINYQ4_BIND")
            .and_then(|(_, value)| value)
            .and_then(|value| value.to_str());
        assert_eq!(bind, Some("127.0.0.1"));
    }
}
