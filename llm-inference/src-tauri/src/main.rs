// Copyright 2024 — Licensed under the Apache License, Version 2.0
// See LICENSE file in the project root for full license text.

#![cfg_attr(
  all(not(debug_assertions), target_os = "windows"),
  windows_subsystem = "windows"
)]

mod engine;
mod gguf;
mod imggen;
mod lora;
mod merge;
mod pty;
mod resolve;
mod setup;
mod tts;
mod tools;
mod memory;
mod planner;

use engine::{Engine, EngineHandle, GenerateResult, SamplingParams, stream_generate};
use gguf::{GgufFile, ModelSummary};
use memory::store::{Fact, Memory, MemoryStore, ToolCallRecord};
use planner::{Plan, PlanStep, PlanSummary, StepStatus, Verifier, VerifyResult};
use tools::{
    fs_tool::{FileEntry, FsTool, ReadResult, SearchResult, TreeEntry, WriteResult},
    patch::{DiffResult, HistoryEntry, PatchEngine, PatchResult},
    sandbox::{ExecRequest, ExecResult, Sandbox, SandboxConfig},
};

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicBool, Ordering};
use tauri::{command, Emitter, State};
use tauri::WebviewWindow;

// ── App state ─────────────────────────────────────────────────────────────────

struct AppState {
    // Inference — single model
    engine: EngineHandle,
    // Inference — dual agent
    drafter: EngineHandle,
    critic: EngineHandle,
    stop_flag: Arc<AtomicBool>,
    models_dir: Arc<Mutex<PathBuf>>,
    // Agent
    fs: Arc<Mutex<FsTool>>,
    sandbox: Arc<Sandbox>,
    patch: Arc<Mutex<PatchEngine>>,
    memory: Arc<Mutex<Memory>>,
    current_plan: Arc<Mutex<Option<Plan>>>,
    sandbox_root: PathBuf,
    /// When false, all file writes and arbitrary command execution are blocked.
    write_mode: Arc<AtomicBool>,
    /// Append-only audit log of every destructive agent action.
    audit_log: Arc<Mutex<AuditLog>>,
}

// ── Audit log ─────────────────────────────────────────────────────────────────

struct AuditLog {
    pub(crate) path: PathBuf,
}

impl AuditLog {
    fn open(path: PathBuf) -> Self {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).ok();
        }
        AuditLog { path }
    }

    fn record(&self, action: &str, detail: serde_json::Value) {
        let entry = serde_json::json!({
            "ts": now_ms(),
            "action": action,
            "detail": detail,
        });
        if let Ok(mut f) = std::fs::OpenOptions::new()
            .create(true).append(true).open(&self.path)
        {
            use std::io::Write;
            let _ = writeln!(f, "{}", entry);
        }
    }
}

/// Commands allowed even when write mode is off.
/// Only truly read-only OS tools — interpreters (python, node, cargo, git)
/// are intentionally excluded because they can exec arbitrary code.
const SAFE_COMMANDS: &[&str] = &[
    "ls", "cat", "head", "tail", "grep", "find", "wc", "pwd", "echo",
    "file", "stat", "diff", "which", "env", "printenv", "whoami", "hostname",
    "du", "df", "uname", "date", "id",
];

fn is_safe_command(cmd: &str) -> bool {
    SAFE_COMMANDS.iter().any(|s| cmd.trim() == *s || cmd.trim().starts_with(&format!("{} ", s)))
}

fn make_state() -> AppState {
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
    let root = PathBuf::from(&home).join("agent-workspace");
    std::fs::create_dir_all(&root).ok();

    // Managed models directory — created on first launch. If this machine already
    // has GGUF files under a common model folder, use that as the visible default.
    let managed_models_dir = PathBuf::from(&home).join("llm-runtime").join("models");
    std::fs::create_dir_all(&managed_models_dir).ok();
    let models_dir = choose_default_models_dir(&home, &managed_models_dir);

    let fs = FsTool::new(&root).expect("sandbox root");
    let memory_path = root.join(".agent/memory.json");
    let memory = Memory::load(memory_path).expect("memory");

    let sandbox_config = SandboxConfig {
        root: root.clone(),
        max_timeout_secs: 300,
        max_output_bytes: 10 * 1024 * 1024,
        blocked_commands: vec![
            // Disk destruction
            "rm -rf /".into(), "rm -rf ~".into(),
            "mkfs".into(), "fdisk".into(), "dd if=/dev/".into(), "shred".into(),
            "wipefs".into(), "sgdisk".into(),
            // Privilege escalation
            "sudo".into(), "su -".into(), "pkexec".into(), "doas".into(),
            // System control
            "shutdown".into(), "reboot".into(), "halt".into(), "poweroff".into(),
            "systemctl poweroff".into(), "systemctl reboot".into(),
            // Fork bombs / resource exhaustion
            ":(){ :|:& };:".into(), "fork bomb".into(),
            // User/credential management
            "useradd".into(), "userdel".into(), "usermod".into(),
            "passwd ".into(), "/etc/shadow".into(),
            // Firewall / network changes
            "iptables".into(), "ufw ".into(), "nft ".into(),
            // Persistence
            "crontab -e".into(), "systemctl enable".into(),
            // Pipe from internet to shell
            "curl | bash".into(), "curl | sh".into(),
            "wget | bash".into(), "wget | sh".into(),
            "curl|bash".into(), "curl|sh".into(),
        ],
        allowed_commands: None,
        base_env: {
            let mut e = HashMap::new();
            e.insert("PATH".into(), std::env::var("PATH").unwrap_or_default());
            e.insert("HOME".into(), std::env::var("HOME").unwrap_or_default());
            e
        },
    };

    let audit_path = PathBuf::from(&home)
        .join(".local/share/ai-workshop/audit.jsonl");

    AppState {
        engine: engine::new_handle(),
        drafter: engine::new_handle(),
        critic: engine::new_handle(),
        stop_flag: Arc::new(AtomicBool::new(false)),
        models_dir: Arc::new(Mutex::new(models_dir)),
        fs: Arc::new(Mutex::new(fs)),
        sandbox: Arc::new(Sandbox::new(sandbox_config)),
        patch: Arc::new(Mutex::new(PatchEngine::new(root.clone()))),
        memory: Arc::new(Mutex::new(memory)),
        current_plan: Arc::new(Mutex::new(None)),
        sandbox_root: root,
        write_mode: Arc::new(AtomicBool::new(false)),
        audit_log: Arc::new(Mutex::new(AuditLog::open(audit_path))),
    }
}

// ── Inference: Load model ─────────────────────────────────────────────────────

#[command]
async fn load_model(
    window: WebviewWindow,
    state: State<'_, AppState>,
    model_path: String,
    gpu_layers: i32,       // 999 = full GPU, 0 = CPU only, N = partial
    ctx_size: u32,
) -> Result<ModelSummary, String> {
    macro_rules! phase {
        ($step:expr, $detail:expr) => {
            window.emit("load-phase", serde_json::json!({"step": $step, "detail": $detail})).ok();
        };
    }

    phase!("stopping", "Stopping previous server…");

    // Kill any existing server we own
    *state.engine.lock().await = None;

    // Also kill any externally-started tinyq4 servers so we always spawn fresh binary
    for srv in engine::detect_tinyq4_servers() {
        let _ = engine::stop_tinyq4_server(srv.port);
    }

    // Give CUDA time to free VRAM before starting new one
    phase!("freeing_vram", "Freeing VRAM…");
    tokio::time::sleep(tokio::time::Duration::from_secs(5)).await;

    phase!("launching", "Launching tinyq4…");
    let result = Engine::load(Path::new(&model_path), gpu_layers, ctx_size)
        .await
        .map_err(|e| e.to_string())?;

    let summary = result.summary.clone();
    *state.engine.lock().await = Some(result);
    phase!("ready", "Model ready");
    window.emit("model-loaded", &summary).ok();
    Ok(summary)
}

#[command]
async fn unload_model(state: State<'_, AppState>) -> Result<(), String> {
    *state.engine.lock().await = None;
    Ok(())
}

#[command]
async fn current_model_port(state: State<'_, AppState>) -> Result<Option<u16>, String> {
    Ok(state.engine.lock().await.as_ref().map(|engine| engine.port))
}

#[command]
async fn stop_model_server(
    state: State<'_, AppState>,
    port: Option<u16>,
) -> Result<(), String> {
    let target_port = {
        let mut engine = state.engine.lock().await;
        let active_port = engine.as_ref().map(|engine| engine.port);
        let target_port = port.or(active_port);

        if port.is_none() || port == active_port {
            *engine = None;
        }

        target_port
    };

    if let Some(port) = target_port {
        engine::stop_local_model_server(port)
            .map_err(|e| e.to_string())?;
    }

    Ok(())
}

/// Attach to an already-running OpenAI-compatible server without spawning anything.
#[command]
async fn attach_model(
    window: WebviewWindow,
    state: State<'_, AppState>,
    model_path: String,
    port: u16,
) -> Result<ModelSummary, String> {
    window.emit("model-loading", format!("Attaching to port {}…", port)).ok();
    *state.engine.lock().await = None;

    let result = Engine::attach(port, Path::new(&model_path))
        .await
        .map_err(|e| e.to_string())?;

    let summary = result.summary.clone();
    *state.engine.lock().await = Some(result);
    window.emit("model-loaded", &summary).ok();
    Ok(summary)

}

/// Return the list of probe ports that are currently responding with /health.
#[command]
async fn scan_running_servers() -> Vec<serde_json::Value> {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_millis(300))
        .build()
        .unwrap_or_default();

    let probe_ports: &[u16] = engine::PROBE_PORTS;
    let mut found = Vec::new();
    let mut seen_ports = std::collections::HashSet::new();

    for srv in engine::detect_local_model_servers() {
        let url = format!("http://127.0.0.1:{}/health", srv.port);
        let healthy = match client
            .get(&url)
            .send()
            .await
        {
            Ok(r) if r.status().is_success() => true,
            _ => false,
        };
        if healthy && seen_ports.insert(srv.port) {
            found.push(serde_json::json!({
                "pid": srv.pid,
                "port": srv.port,
                "model_id": srv.model_id,
                "model_path": srv.model_path,
                "source": srv.source,
            }));
        }
    }

    for &port in probe_ports {
        if seen_ports.contains(&port) {
            continue;
        }

        let model_id = async {
            let r = client.get(format!("http://127.0.0.1:{}/v1/models", port)).send().await.ok()?;
            if !r.status().is_success() {
                return None;
            }
            let v: serde_json::Value = r.json().await.ok()?;
            v["data"].as_array()?.first()?.get("id")?.as_str().map(String::from)
        }.await;

        let health_ok = async {
            let r = client.get(format!("http://127.0.0.1:{}/health", port)).send().await.ok()?;
            if !r.status().is_success() {
                return None;
            }
            r.text().await.ok().map(|text| text.trim() == "ok")
        }.await.unwrap_or(false);

        if model_id.is_some() || health_ok {
            let model_id = model_id.unwrap_or_else(|| format!("port:{}", port));
            seen_ports.insert(port);
            found.push(serde_json::json!({ "port": port, "model_id": model_id }));
        }
    }
    found
}

// ── Dual-agent: load drafter / critic ────────────────────────────────────────

#[command]
async fn load_drafter(
    window: WebviewWindow,
    state: State<'_, AppState>,
    model_path: String,
    gpu_layers: i32,
    ctx_size: u32,
) -> Result<ModelSummary, String> {
    window.emit("drafter-loading", "Starting drafter…").ok();
    *state.drafter.lock().await = None;
    let eng = Engine::load(Path::new(&model_path), gpu_layers, ctx_size)
        .await.map_err(|e| e.to_string())?;
    let summary = eng.summary.clone();
    *state.drafter.lock().await = Some(eng);
    window.emit("drafter-loaded", &summary).ok();
    Ok(summary)
}

#[command]
async fn load_critic(
    window: WebviewWindow,
    state: State<'_, AppState>,
    model_path: String,
    gpu_layers: i32,
    ctx_size: u32,
) -> Result<ModelSummary, String> {
    window.emit("critic-loading", "Starting critic…").ok();
    *state.critic.lock().await = None;
    let eng = Engine::load(Path::new(&model_path), gpu_layers, ctx_size)
        .await.map_err(|e| e.to_string())?;
    let summary = eng.summary.clone();
    *state.critic.lock().await = Some(eng);
    window.emit("critic-loaded", &summary).ok();
    Ok(summary)
}

#[command]
async fn dual_agent_status(state: State<'_, AppState>) -> Result<serde_json::Value, String> {
    let drafter = state.drafter.lock().await;
    let critic  = state.critic.lock().await;
    Ok(serde_json::json!({
        "drafter": drafter.as_ref().map(|e| serde_json::to_value(&e.summary).unwrap()),
        "critic":  critic.as_ref().map(|e| serde_json::to_value(&e.summary).unwrap()),
    }))
}

#[command]
async fn dual_generate(
    window: WebviewWindow,
    state: State<'_, AppState>,
    req: GenerateRequest,
) -> Result<serde_json::Value, String> {
    state.stop_flag.store(false, Ordering::Relaxed);

    let (d_port, d_client) = {
        let g = state.drafter.lock().await;
        let e = g.as_ref().ok_or("No drafter loaded — pick a model for slot A")?;
        (e.port, e.client.clone())
    };
    let (c_port, c_client) = {
        let g = state.critic.lock().await;
        let e = g.as_ref().ok_or("No critic loaded — pick a model for slot B")?;
        (e.port, e.client.clone())
    };

    let messages: Vec<serde_json::Value> = req.messages.iter()
        .map(|m| serde_json::json!({"role": m.role, "content": m.content}))
        .collect();
    let params = SamplingParams {
        max_tokens: req.max_tokens, temperature: req.temperature,
        top_p: req.top_p, top_k: req.top_k,
        repeat_penalty: req.repeat_penalty, seed: req.seed,
    };
    let stop = state.stop_flag.clone();

    // ── Phase 1: Drafter ──────────────────────────────────────────────────────
    window.emit("dual-start", ()).ok();
    let drafter_buf: Arc<Mutex<String>> = Arc::new(Mutex::new(String::new()));
    let db = drafter_buf.clone();
    let win = window.clone();
    let dr = stream_generate(&d_client, d_port, &messages, &params,
        move |piece, is_r| {
            db.lock().unwrap().push_str(piece);
            win.emit(if is_r { "drafter-reasoning" } else { "drafter-token" }, piece).ok();
        }, |_, _| {}, stop.clone(),
    ).await.map_err(|e| e.to_string())?;

    let drafter_output = drafter_buf.lock().unwrap().clone();
    window.emit("drafter-done", &dr).ok();

    if stop.load(Ordering::Relaxed) {
        window.emit("dual-done", serde_json::json!({"drafter": dr, "critic": null})).ok();
        return Ok(serde_json::json!({"drafter": dr, "critic": null}));
    }

    // ── Phase 2: Critic ───────────────────────────────────────────────────────
    // Critic sees the full conversation + drafter answer, then a review prompt
    let mut critic_msgs = messages.clone();
    critic_msgs.push(serde_json::json!({"role": "assistant", "content": drafter_output}));
    critic_msgs.push(serde_json::json!({"role": "user",
        "content": "Review your previous response. If it's accurate and complete, confirm it briefly. If there are errors or gaps, provide a corrected version. Be concise and direct."}));

    window.emit("critic-start", ()).ok();
    let win = window.clone();
    let cr = stream_generate(&c_client, c_port, &critic_msgs, &params,
        move |piece, is_r| {
            win.emit(if is_r { "critic-reasoning" } else { "critic-token" }, piece).ok();
        }, |_, _| {}, stop,
    ).await.map_err(|e| e.to_string())?;

    let combined = serde_json::json!({
        "drafter": dr, "critic": cr,
        "tokens_generated": dr.tokens_generated + cr.tokens_generated,
        "total_tokens": dr.total_tokens + cr.total_tokens,
        "elapsed_ms": dr.elapsed_ms + cr.elapsed_ms,
        "tokens_per_sec": if (dr.elapsed_ms + cr.elapsed_ms) > 0 {
            (dr.tokens_generated + cr.tokens_generated) as f64 /
            ((dr.elapsed_ms + cr.elapsed_ms) as f64 / 1000.0)
        } else { 0.0 },
    });
    window.emit("dual-done", &combined).ok();
    Ok(combined)
}

// ── Inference: inspect ────────────────────────────────────────────────────────

#[command]
fn inspect_gguf(path: String) -> Result<ModelSummary, String> {
    GgufFile::open(Path::new(&path))
        .map(|g| g.summary())
        .map_err(|e| e.to_string())
}

// ── Inference: Generate (chat) ────────────────────────────────────────────────

#[derive(Deserialize)]
pub struct GenerateRequest {
    pub messages: Vec<ChatMessage>,
    pub max_tokens: usize,
    pub temperature: f64,
    pub top_p: f64,
    pub top_k: usize,
    pub repeat_penalty: f32,
    pub seed: u64,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

#[command]
async fn generate(
    window: WebviewWindow,
    state: State<'_, AppState>,
    req: GenerateRequest,
) -> Result<GenerateResult, String> {
    state.stop_flag.store(false, Ordering::Relaxed);

    // Grab port + client without holding the lock across the streaming await
    let (port, client) = {
        let guard = state.engine.lock().await;
        let eng = guard.as_ref().ok_or("No model loaded — pick a GGUF file first")?;
        (eng.port, eng.client.clone())
    };

    let messages: Vec<serde_json::Value> = req.messages.iter()
        .map(|m| serde_json::json!({"role": m.role, "content": m.content}))
        .collect();

    let params = SamplingParams {
        max_tokens: req.max_tokens,
        temperature: req.temperature,
        top_p: req.top_p,
        top_k: req.top_k,
        repeat_penalty: req.repeat_penalty,
        seed: req.seed,
    };

    let stop_flag = state.stop_flag.clone();
    let win = window.clone();
    window.emit("stream-start", ()).ok();

    let win2 = window.clone();
    let result = stream_generate(
        &client, port, &messages, &params,
        move |piece, is_reasoning| {
            if is_reasoning {
                win.emit("stream-reasoning", piece).ok();
            } else {
                win.emit("stream-token", piece).ok();
            }
        },
        move |done, total| {
            win2.emit("prefill-progress", serde_json::json!({"done": done, "total": total})).ok();
        },
        stop_flag,
    ).await.map_err(|e| e.to_string())?;

    window.emit("stream-done", &result).ok();
    Ok(result)
}

#[command]
fn stop_generate(state: State<'_, AppState>) {
    state.stop_flag.store(true, Ordering::Relaxed);
}

// ── Inference: GPU stats / session ────────────────────────────────────────────

#[command]
fn get_gpu_stats() -> serde_json::Value {
    let out = std::process::Command::new("nvidia-smi")
        .args([
            "--query-gpu=name,memory.total,memory.used,memory.free,temperature.gpu,utilization.gpu",
            "--format=csv,noheader,nounits",
        ])
        .output();
    match out {
        Ok(o) if o.status.success() => {
            let s = String::from_utf8_lossy(&o.stdout);
            let p: Vec<&str> = s.trim().split(", ").collect();
            if p.len() >= 6 {
                return serde_json::json!({
                    "available": true,
                    "name": p[0],
                    "mem_total": p[1].parse::<u64>().unwrap_or(0),
                    "mem_used":  p[2].parse::<u64>().unwrap_or(0),
                    "mem_free":  p[3].parse::<u64>().unwrap_or(0),
                    "temp":      p[4].parse::<u64>().unwrap_or(0),
                    "gpu_util":  p[5].parse::<u64>().unwrap_or(0),
                });
            }
        }
        _ => {}
    }
    serde_json::json!({ "available": false })
}

/// Ask the LLM to evaluate whether the agent's goal has been fully achieved,
/// using episodic memory as evidence. Returns {"complete": bool, "reason": "..."}.
#[command]
async fn check_goal_completion(
    state: State<'_, AppState>,
    goal: String,
) -> Result<serde_json::Value, String> {
    let mem_ctx = state.memory.lock().map_err(|e| e.to_string())?.context_for_prompt(10);

    let prompt = format!(
        "You are evaluating whether an AI agent has completed its assigned goal.\n\n\
         GOAL: {goal}\n\n\
         AGENT WORK LOG (from memory):\n{mem_ctx}\n\n\
         Based on the work log, has the goal been fully achieved?\n\
         Reply with ONLY one line of valid JSON — nothing before or after it:\n\
         {{\"complete\": true, \"reason\": \"brief explanation\"}} \
         or {{\"complete\": false, \"reason\": \"what still needs to be done\"}}"
    );

    let (port, client) = {
        let guard = state.engine.lock().await;
        let eng = guard.as_ref().ok_or("No model loaded")?;
        (eng.port, eng.client.clone())
    };

    let messages = vec![serde_json::json!({"role": "user", "content": prompt})];
    let params = SamplingParams {
        max_tokens: 200,
        temperature: 0.1,
        top_p: 0.9,
        top_k: 40,
        repeat_penalty: 1.0,
        seed: 42,
    };

    let buf: Arc<Mutex<String>> = Arc::new(Mutex::new(String::new()));
    let b = buf.clone();
    let stop_flag = state.stop_flag.clone();

    stream_generate(
        &client, port, &messages, &params,
        move |piece, _| { b.lock().unwrap().push_str(piece); },
        |_, _| {},
        stop_flag,
    ).await.map_err(|e| e.to_string())?;

    let raw = buf.lock().unwrap().clone();
    let json_str = extract_first_json_object(&raw);
    serde_json::from_str::<serde_json::Value>(&json_str)
        .map_err(|e| format!("Completion check parse failed: {e}. Model said: {}", &raw[..raw.len().min(200)]))
}

fn extract_first_json_object(text: &str) -> String {
    if let Some(start) = text.find('{') {
        if let Some(end) = text.rfind('}') {
            if end >= start {
                return text[start..=end].to_string();
            }
        }
    }
    text.trim().to_string()
}

/// Delete user data from disk based on which categories are requested.
/// Returns a summary of what was cleared.
#[command]
fn clear_user_data(
    state: State<'_, AppState>,
    clear_memory: bool,
    clear_audit: bool,
    clear_logs: bool,
) -> Result<Vec<String>, String> {
    let mut cleared = Vec::new();
    if clear_memory {
        state.memory.lock().map_err(|e| e.to_string())?.clear_all();
        cleared.push("agent_memory".into());
    }
    if clear_audit {
        let path = state.audit_log.lock().map_err(|e| e.to_string())?.path.clone();
        std::fs::remove_file(&path).ok();
        cleared.push("audit_log".into());
    }
    if clear_logs {
        std::fs::remove_file("/tmp/ai-workshop-tinyq4.log").ok();
        std::fs::remove_file("/tmp/llm-inference-tauri-server.pid").ok();
        cleared.push("temp_logs".into());
    }
    Ok(cleared)
}

#[command]
fn save_session(json: String, path: String) -> Result<(), String> {
    let p = std::path::Path::new(&path);
    let fname = p.file_name().unwrap_or_default().to_string_lossy();
    let mut tmp = p.to_path_buf();
    tmp.set_file_name(format!("{}.tmp", fname));
    std::fs::write(&tmp, &json).map_err(|e| e.to_string())?;
    std::fs::rename(&tmp, p).map_err(|e| e.to_string())
}

/// Write a base64-encoded binary file (used for saving PNG/WAV from the frontend).
#[command]
fn write_binary_b64(path: String, b64: String) -> Result<(), String> {
    use base64::Engine as _;
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(&b64)
        .map_err(|e| format!("base64 decode: {e}"))?;
    std::fs::write(&path, bytes).map_err(|e| e.to_string())
}

#[command]
fn load_session(path: String) -> Result<String, String> {
    std::fs::read_to_string(&path).map_err(|e| e.to_string())
}

// ── Agent: Filesystem ─────────────────────────────────────────────────────────

#[command]
fn fs_read(state: State<'_, AppState>, path: String) -> Result<ReadResult, String> {
    state.fs.lock().unwrap().read_file(&path).map_err(|e| e.to_string())
}
#[command]
fn get_agent_write_mode(state: State<'_, AppState>) -> bool {
    state.write_mode.load(Ordering::Relaxed)
}
#[command]
fn set_agent_write_mode(state: State<'_, AppState>, enabled: bool) {
    state.write_mode.store(enabled, Ordering::Relaxed);
}

#[command]
fn fs_write(state: State<'_, AppState>, path: String, content: String) -> Result<WriteResult, String> {
    if !state.write_mode.load(Ordering::Relaxed) {
        return Err("Agent write mode is OFF — enable it in the agent panel to write files.".into());
    }
    let result = state.fs.lock().unwrap().write_file(&path, &content).map_err(|e| e.to_string())?;
    state.audit_log.lock().unwrap().record("fs_write", serde_json::json!({
        "path": path, "bytes": content.len()
    }));
    Ok(result)
}
#[command]
fn fs_append(state: State<'_, AppState>, path: String, content: String) -> Result<WriteResult, String> {
    if !state.write_mode.load(Ordering::Relaxed) {
        return Err("Agent write mode is OFF — enable it to write files.".into());
    }
    state.fs.lock().unwrap().append_file(&path, &content).map_err(|e| e.to_string())
}
#[command]
fn fs_delete(state: State<'_, AppState>, path: String) -> Result<String, String> {
    if !state.write_mode.load(Ordering::Relaxed) {
        return Err("Agent write mode is OFF — enable it to delete files.".into());
    }
    let result = state.fs.lock().unwrap().delete_file(&path).map_err(|e| e.to_string())?;
    state.audit_log.lock().unwrap().record("fs_delete", serde_json::json!({ "path": path }));
    Ok(result)
}
#[command]
fn fs_list(state: State<'_, AppState>, path: String) -> Result<Vec<FileEntry>, String> {
    state.fs.lock().unwrap().list_dir(&path).map_err(|e| e.to_string())
}
#[command]
fn fs_tree(state: State<'_, AppState>, path: String, max_depth: usize) -> Result<Vec<TreeEntry>, String> {
    state.fs.lock().unwrap().tree(&path, max_depth).map_err(|e| e.to_string())
}
#[command]
fn fs_search(state: State<'_, AppState>, path: String, pattern: String, context: usize) -> Result<Vec<SearchResult>, String> {
    state.fs.lock().unwrap().search(&path, &pattern, context).map_err(|e| e.to_string())
}
#[command]
fn fs_move(state: State<'_, AppState>, from: String, to: String) -> Result<String, String> {
    state.fs.lock().unwrap().move_file(&from, &to).map_err(|e| e.to_string())
}
#[command]
fn fs_copy(state: State<'_, AppState>, from: String, to: String) -> Result<String, String> {
    state.fs.lock().unwrap().copy_file(&from, &to).map_err(|e| e.to_string())
}
#[command]
fn fs_mkdir(state: State<'_, AppState>, path: String) -> Result<String, String> {
    state.fs.lock().unwrap().create_dir(&path).map_err(|e| e.to_string())
}
#[command]
fn fs_exists(state: State<'_, AppState>, path: String) -> bool {
    state.fs.lock().unwrap().file_exists(&path)
}
#[command]
fn get_sandbox_root(state: State<'_, AppState>) -> String {
    state.sandbox_root.to_string_lossy().into_owned()
}
#[command]
fn set_sandbox_root(state: State<'_, AppState>, path: String) -> Result<(), String> {
    let new_root = PathBuf::from(&path);
    std::fs::create_dir_all(&new_root).map_err(|e| e.to_string())?;
    // FsTool
    let new_fs = FsTool::new(&new_root).map_err(|e| e.to_string())?;
    *state.fs.lock().unwrap() = new_fs;
    // PatchEngine
    state.patch.lock().unwrap().set_root(new_root.clone());
    // Sandbox
    state.sandbox.set_root(new_root.clone());
    Ok(())
}

// ── Agent: Sandbox ────────────────────────────────────────────────────────────

#[command]
async fn exec_command(
    window: WebviewWindow,
    state: State<'_, AppState>,
    req: ExecRequest,
) -> Result<ExecResult, String> {
    let write_mode = state.write_mode.load(Ordering::Relaxed);
    if !write_mode && !is_safe_command(&req.command) {
        return Err(format!(
            "Agent write mode is OFF. '{}' is not in the safe read-only list. \
             Enable write mode to run arbitrary commands.",
            req.command
        ));
    }
    let id = format!("exec-{}", now_ms());
    let sandbox = state.sandbox.clone();
    let win_out = window.clone();
    let win_err = window.clone();
    let id2 = id.clone();
    window.emit("exec-start", &id).ok();
    let result = sandbox.exec(
        &id, &req,
        move |line| { win_out.emit("exec-stdout", serde_json::json!({"id": id2, "line": line})).ok(); },
        move |line| { win_err.emit("exec-stderr", serde_json::json!({"id": "", "line": line})).ok(); },
    ).await.map_err(|e| e.to_string())?;
    state.audit_log.lock().unwrap().record("exec_command", serde_json::json!({
        "command": req.command, "args": req.args, "cwd": req.cwd,
        "exit_code": result.exit_code, "ts": now_ms(),
    }));
    window.emit("exec-done", &result).ok();
    Ok(result)
}
#[command]
fn kill_process(state: State<'_, AppState>, id: String) -> bool {
    state.sandbox.kill_process(&id)
}
#[command]
fn list_processes(state: State<'_, AppState>) -> Vec<String> {
    state.sandbox.list_processes()
}

// ── Agent: Patch / Diff ───────────────────────────────────────────────────────

#[command]
fn diff_proposed(state: State<'_, AppState>, path: String, new_content: String) -> Result<DiffResult, String> {
    state.patch.lock().unwrap().diff_proposed(&path, &new_content).map_err(|e| e.to_string())
}
#[command]
fn apply_patch(state: State<'_, AppState>, path: String, content: String, description: String) -> Result<PatchResult, String> {
    if !state.write_mode.load(Ordering::Relaxed) {
        return Err("Agent write mode is OFF — enable it to apply patches.".into());
    }
    let result = state.patch.lock().unwrap().apply(&path, &content, &description).map_err(|e| e.to_string())?;
    state.audit_log.lock().unwrap().record("apply_patch", serde_json::json!({
        "path": path, "description": description
    }));
    Ok(result)
}
#[command]
fn apply_unified_diff(state: State<'_, AppState>, path: String, diff: String) -> Result<PatchResult, String> {
    if !state.write_mode.load(Ordering::Relaxed) {
        return Err("Agent write mode is OFF — enable it to apply diffs.".into());
    }
    state.patch.lock().unwrap().apply_unified_diff(&path, &diff).map_err(|e| e.to_string())
}
#[command]
fn undo_patch(state: State<'_, AppState>, path: String) -> Result<PatchResult, String> {
    state.patch.lock().unwrap().undo(&path).map_err(|e| e.to_string())
}
#[command]
fn patch_history(state: State<'_, AppState>, path: String) -> Vec<HistoryEntry> {
    state.patch.lock().unwrap().history(&path)
}
#[command]
fn diff_files(state: State<'_, AppState>, path_a: String, path_b: String) -> Result<DiffResult, String> {
    state.patch.lock().unwrap().diff_files(&path_a, &path_b).map_err(|e| e.to_string())
}

// ── Agent: Memory ─────────────────────────────────────────────────────────────

#[command]
fn mem_start_task(state: State<'_, AppState>, goal: String) -> Result<String, String> {
    state.memory.lock().unwrap().start_task(&goal).map_err(|e| e.to_string())
}
#[command]
fn mem_finish_task(state: State<'_, AppState>, outcome: String, summary: String) -> Result<(), String> {
    state.memory.lock().unwrap().finish_task(&outcome, &summary).map_err(|e| e.to_string())
}
#[command]
fn mem_remember(state: State<'_, AppState>, key: String, value: String, category: String, source: String, confidence: f32) -> Result<String, String> {
    state.memory.lock().unwrap().remember(&key, &value, &category, &source, confidence).map_err(|e| e.to_string())
}
#[command]
fn mem_recall(state: State<'_, AppState>, query: String) -> Vec<Fact> {
    state.memory.lock().unwrap().recall(&query)
}
#[command]
fn mem_forget(state: State<'_, AppState>, id: String) -> Result<bool, String> {
    state.memory.lock().unwrap().forget(&id).map_err(|e| e.to_string())
}
#[command]
fn mem_context(state: State<'_, AppState>, max_facts: usize) -> String {
    state.memory.lock().unwrap().context_for_prompt(max_facts)
}
#[command]
fn mem_store(state: State<'_, AppState>) -> MemoryStore {
    state.memory.lock().unwrap().full_store().clone()
}

// ── Agent: Planner ────────────────────────────────────────────────────────────

#[command]
fn plan_parse(goal: String, json: String) -> Result<PlanSummary, String> {
    planner::parse_plan_response(&goal, &json)
        .map(|p| p.summary())
        .map_err(|e| e.to_string())
}
#[command]
fn plan_get(state: State<'_, AppState>) -> Option<PlanSummary> {
    state.current_plan.lock().unwrap().as_ref().map(|p| p.summary())
}
#[command]
fn plan_prompt_template(goal: String, memory_context: String) -> String {
    planner::plan_prompt(&goal, &memory_context, &AGENT_TOOLS)
}

const AGENT_TOOLS: &[&str] = &[
    "fs_read", "fs_write", "fs_list", "fs_tree", "fs_search",
    "fs_mkdir", "fs_delete", "fs_move", "fs_copy",
    "exec", "diff_proposed", "apply_patch", "apply_unified_diff",
    "mem_remember", "mem_recall",
];

/// Execute a pre-parsed plan JSON manually (paste workflow).
#[command]
async fn plan_execute(
    window: WebviewWindow,
    state: State<'_, AppState>,
    plan_json: String,
    goal: String,
) -> Result<PlanSummary, String> {
    let plan = planner::parse_plan_response(&goal, &plan_json)
        .map_err(|e| e.to_string())?;
    run_plan(
        window,
        state.fs.clone(), state.sandbox.clone(),
        state.patch.clone(), state.memory.clone(),
        state.current_plan.clone(),
        state.write_mode.clone(),
        plan,
    ).await
}

/// Fully autonomous: LLM generates the plan from `goal`, then executes it.
#[command]
async fn agent_run(
    window: WebviewWindow,
    state: State<'_, AppState>,
    goal: String,
) -> Result<PlanSummary, String> {
    // Build the planning prompt
    let mem_ctx = state.memory.lock().unwrap().context_for_prompt(10);
    let plan_prompt_text = planner::plan_prompt(&goal, &mem_ctx, AGENT_TOOLS);

    window.emit("agent-planning", &goal).ok();

    // Get port + client without holding lock across the stream
    let (port, client) = {
        let guard = state.engine.lock().await;
        let eng = guard.as_ref().ok_or("No model loaded — load a GGUF model first")?;
        (eng.port, eng.client.clone())
    };

    let messages = vec![serde_json::json!({"role": "user", "content": plan_prompt_text})];

    let params = SamplingParams {
        max_tokens: 2048,
        temperature: 0.15,
        top_p: 0.9,
        top_k: 40,
        repeat_penalty: 1.05,
        seed: 42,
    };

    let stop_flag = state.stop_flag.clone();
    let json_buf: Arc<Mutex<String>> = Arc::new(Mutex::new(String::new()));
    let jb = json_buf.clone();
    let win_tok = window.clone();

    stream_generate(
        &client, port, &messages, &params,
        move |piece, _is_reasoning| {
            jb.lock().unwrap().push_str(piece);
            win_tok.emit("agent-plan-token", piece).ok();
        },
        |_, _| {},
        stop_flag,
    ).await.map_err(|e| e.to_string())?;

    let plan_json = json_buf.lock().unwrap().clone();
    window.emit("agent-plan-ready", &plan_json).ok();

    let plan = planner::parse_plan_response(&goal, &plan_json).map_err(|e| {
        let preview = &plan_json[..plan_json.len().min(300)];
        format!("LLM plan parse failed: {}.\nRaw output start: {}", e, preview)
    })?;

    run_plan(
        window,
        state.fs.clone(), state.sandbox.clone(),
        state.patch.clone(), state.memory.clone(),
        state.current_plan.clone(),
        state.write_mode.clone(),
        plan,
    ).await
}

// ── Shared plan execution engine ──────────────────────────────────────────────

async fn run_plan(
    window: WebviewWindow,
    fs: Arc<Mutex<FsTool>>,
    sandbox: Arc<Sandbox>,
    patch: Arc<Mutex<PatchEngine>>,
    memory: Arc<Mutex<Memory>>,
    current_plan_slot: Arc<Mutex<Option<Plan>>>,
    write_mode: Arc<AtomicBool>,
    mut plan: Plan,
) -> Result<PlanSummary, String> {
    plan.status = StepStatus::Running;
    *current_plan_slot.lock().unwrap() = Some(plan.clone());
    memory.lock().unwrap().start_task(&plan.goal).ok();
    window.emit("plan-start", plan.summary()).ok();
    let total = plan.steps.len();

    loop {
        let ready: Vec<usize> = {
            let p = current_plan_slot.lock().unwrap();
            p.as_ref().unwrap().ready_steps().iter().map(|s| s.index).collect()
        };
        if ready.is_empty() { break; }

        for idx in ready {
            let step = {
                let mut p = current_plan_slot.lock().unwrap();
                let p = p.as_mut().unwrap();
                p.steps[idx].status = StepStatus::Running;
                p.steps[idx].started_at = Some(now_ms());
                p.steps[idx].clone()
            };

            window.emit("plan-step-start", serde_json::json!({
                "step_id": step.id, "index": step.index,
                "description": step.description,
                "tool": step.tool_call.tool, "total": total,
            })).ok();

            let result = execute_step(&window, &fs, &sandbox, &patch, &memory, &write_mode, &step).await;

            let (output, success, error) = match result {
                Ok(v)  => (v, true, None),
                Err(e) => (serde_json::Value::Null, false, Some(e)),
            };

            let verify = if success {
                let step_ref = current_plan_slot.lock().unwrap()
                    .as_ref().unwrap().steps[idx].clone();
                Verifier::verify(&step_ref, &output)
            } else {
                VerifyResult { passed: false, reason: error.clone().unwrap_or_default(), suggestions: vec![] }
            };

            window.emit("plan-step-verify", serde_json::json!({
                "step_id": step.id, "passed": verify.passed, "reason": verify.reason,
            })).ok();

            let final_status = if verify.passed {
                StepStatus::Done
            } else if step.retry_count < step.max_retries {
                StepStatus::Retrying
            } else {
                StepStatus::Failed
            };

            {
                let mut p = current_plan_slot.lock().unwrap();
                let s = &mut p.as_mut().unwrap().steps[idx];
                s.status = final_status.clone();
                s.output = Some(output.clone());
                s.error = error.clone();
                s.finished_at = Some(now_ms());
                if final_status == StepStatus::Retrying { s.retry_count += 1; }
            }

            memory.lock().unwrap().record_tool_call(ToolCallRecord {
                step: idx,
                tool: step.tool_call.tool.clone(),
                input: serde_json::to_value(&step.tool_call.params).unwrap_or_default(),
                output: output.clone(),
                success: verify.passed,
                duration_ms: 0,
                ts: now_ms(),
            }).ok();

            window.emit("plan-step-done",
                current_plan_slot.lock().unwrap().as_ref().unwrap().summary()).ok();

            if final_status == StepStatus::Failed {
                window.emit("plan-step-failed", serde_json::json!({
                    "step_id": step.id, "error": error, "suggestions": verify.suggestions,
                })).ok();
            }
        }

        let done = current_plan_slot.lock().unwrap()
            .as_ref().map(|p| p.is_complete()).unwrap_or(true);
        if done { break; }
    }

    let summary = {
        let mut p = current_plan_slot.lock().unwrap();
        let plan = p.as_mut().unwrap();
        plan.status = if plan.has_failed() { StepStatus::Failed } else { StepStatus::Done };
        plan.summary()
    };

    memory.lock().unwrap().finish_task(
        &summary.status,
        &format!("{}/{} steps completed", summary.done_steps, summary.total_steps),
    ).ok();

    window.emit("plan-done", &summary).ok();
    Ok(summary)
}

// ── Step executor — dispatches tool name → Rust function ──────────────────────

async fn execute_step(
    window: &WebviewWindow,
    fs: &Arc<Mutex<FsTool>>,
    sandbox: &Arc<Sandbox>,
    patch: &Arc<Mutex<PatchEngine>>,
    memory: &Arc<Mutex<Memory>>,
    write_mode: &Arc<AtomicBool>,
    step: &PlanStep,
) -> Result<serde_json::Value, String> {
    let p = &step.tool_call.params;
    let write_ok = write_mode.load(Ordering::Relaxed);

    fn sp(p: &HashMap<String, serde_json::Value>, key: &str) -> String {
        p.get(key).and_then(|v| v.as_str()).unwrap_or("").to_string()
    }
    fn up(p: &HashMap<String, serde_json::Value>, key: &str, d: usize) -> usize {
        p.get(key).and_then(|v| v.as_u64()).unwrap_or(d as u64) as usize
    }

    match step.tool_call.tool.as_str() {
        "fs_read"   => Ok(serde_json::to_value(fs.lock().unwrap().read_file(&sp(p,"path")).map_err(|e|e.to_string())?).unwrap()),
        "fs_list"   => Ok(serde_json::to_value(fs.lock().unwrap().list_dir(&sp(p,"path")).map_err(|e|e.to_string())?).unwrap()),
        "fs_search" => Ok(serde_json::to_value(fs.lock().unwrap().search(&sp(p,"path"),&sp(p,"pattern"),up(p,"context",2)).map_err(|e|e.to_string())?).unwrap()),
        "fs_write"  => {
            if !write_ok { return Err("Agent write mode is OFF — enable it to write files.".into()); }
            Ok(serde_json::to_value(fs.lock().unwrap().write_file(&sp(p,"path"),&sp(p,"content")).map_err(|e|e.to_string())?).unwrap())
        }
        "fs_mkdir"  => {
            if !write_ok { return Err("Agent write mode is OFF — enable it to create directories.".into()); }
            Ok(serde_json::Value::String(fs.lock().unwrap().create_dir(&sp(p,"path")).map_err(|e|e.to_string())?))
        }
        "fs_delete" => {
            if !write_ok { return Err("Agent write mode is OFF — enable it to delete files.".into()); }
            Ok(serde_json::Value::String(fs.lock().unwrap().delete_file(&sp(p,"path")).map_err(|e|e.to_string())?))
        }
        "fs_move"   => {
            if !write_ok { return Err("Agent write mode is OFF.".into()); }
            Ok(serde_json::Value::String(fs.lock().unwrap().move_file(&sp(p,"from"),&sp(p,"to")).map_err(|e|e.to_string())?))
        }
        "fs_copy"   => {
            if !write_ok { return Err("Agent write mode is OFF.".into()); }
            Ok(serde_json::Value::String(fs.lock().unwrap().copy_file(&sp(p,"from"),&sp(p,"to")).map_err(|e|e.to_string())?))
        }
        "exec" => {
            let req = ExecRequest {
                command: sp(p, "command"),
                args: p.get("args").and_then(|v| v.as_array())
                    .map(|a| a.iter().filter_map(|v| v.as_str().map(String::from)).collect())
                    .unwrap_or_default(),
                cwd: p.get("cwd").and_then(|v| v.as_str()).map(String::from),
                env: HashMap::new(),
                timeout_secs: p.get("timeout_secs").and_then(|v| v.as_u64()).unwrap_or(60),
                stdin_data: p.get("stdin_data").and_then(|v| v.as_str()).map(String::from),
                capture_output: true,
            };
            let id = format!("plan-{}-{}", step.id, now_ms());
            let win = window.clone();
            let win2 = window.clone();
            let r = sandbox.exec(&id, &req,
                move |line| { win.emit("exec-stdout", &line).ok(); },
                move |line| { win2.emit("exec-stderr", &line).ok(); },
            ).await.map_err(|e| e.to_string())?;
            Ok(serde_json::to_value(r).unwrap())
        }
        "diff_proposed" => Ok(serde_json::to_value(
            patch.lock().unwrap().diff_proposed(&sp(p,"path"),&sp(p,"new_content")).map_err(|e|e.to_string())?
        ).unwrap()),
        "apply_patch" => {
            if !write_ok { return Err("Agent write mode is OFF — enable it to apply patches.".into()); }
            Ok(serde_json::to_value(
                patch.lock().unwrap().apply(&sp(p,"path"),&sp(p,"content"),&sp(p,"description")).map_err(|e|e.to_string())?
            ).unwrap())
        }
        "apply_unified_diff" => {
            if !write_ok { return Err("Agent write mode is OFF — enable it to apply diffs.".into()); }
            Ok(serde_json::to_value(
                patch.lock().unwrap().apply_unified_diff(&sp(p,"path"),&sp(p,"diff")).map_err(|e|e.to_string())?
            ).unwrap())
        }
        "mem_remember" => {
            let key = sp(p, "key");
            let value = sp(p, "value");
            let cat = sp(p, "category");
            let src = sp(p, "source");
            let conf = p.get("confidence").and_then(|v| v.as_f64()).unwrap_or(0.8) as f32;
            let id = memory.lock().unwrap()
                .remember(&key, &value,
                    if cat.is_empty() { "learned" } else { &cat },
                    if src.is_empty() { "agent_plan" } else { &src },
                    conf)
                .map_err(|e| e.to_string())?;
            Ok(serde_json::Value::String(id))
        }
        "mem_recall" => {
            let query = sp(p, "query");
            let facts = memory.lock().unwrap().recall(&query);
            Ok(serde_json::to_value(facts).unwrap_or_default())
        }
        "noop" => Ok(serde_json::Value::String("noop".into())),
        other  => Err(format!("Unknown tool: {}", other)),
    }
}

// ── Models directory ─────────────────────────────────────────────────────────

#[derive(Serialize, Clone)]
struct ModelEntry {
    name: String,
    gguf_path: String,
    tokenizer_path: Option<String>,
    size_gb: f32,
    dir: String,
}

fn is_gguf_path(path: &Path) -> bool {
    path.extension()
        .and_then(|x| x.to_str())
        .map(|x| x.eq_ignore_ascii_case("gguf"))
        .unwrap_or(false)
}

fn dir_has_gguf(dir: &Path) -> bool {
    if !dir.exists() {
        return false;
    }
    walkdir::WalkDir::new(dir)
        .max_depth(6)
        .follow_links(true)
        .into_iter()
        .filter_map(|e| e.ok())
        .any(|e| is_gguf_path(e.path()))
}

fn candidate_model_dirs(home: &str) -> Vec<PathBuf> {
    let home = PathBuf::from(home);
    vec![
        home.join("models"),
        home.join("projects").join("models"),
        home.join(".lmstudio").join("models"),
        PathBuf::from("/data/models"),
    ]
}

fn choose_default_models_dir(home: &str, managed_dir: &Path) -> PathBuf {
    if dir_has_gguf(managed_dir) {
        return managed_dir.to_path_buf();
    }
    candidate_model_dirs(home)
        .into_iter()
        .find(|dir| dir_has_gguf(dir))
        .unwrap_or_else(|| managed_dir.to_path_buf())
}

fn push_model_root(roots: &mut Vec<PathBuf>, seen: &mut HashSet<PathBuf>, path: PathBuf) {
    if !path.exists() {
        return;
    }
    let key = path.canonicalize().unwrap_or_else(|_| path.clone());
    if seen.insert(key) {
        roots.push(path);
    }
}

fn model_scan_roots(primary: &Path) -> Vec<PathBuf> {
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
    let mut roots = Vec::new();
    let mut seen = HashSet::new();
    push_model_root(&mut roots, &mut seen, primary.to_path_buf());
    push_model_root(
        &mut roots,
        &mut seen,
        PathBuf::from(&home).join("llm-runtime").join("models"),
    );
    for dir in candidate_model_dirs(&home) {
        push_model_root(&mut roots, &mut seen, dir);
    }
    roots
}

#[command]
fn scan_models_dir(state: State<'_, AppState>) -> Vec<ModelEntry> {
    use walkdir::WalkDir;
    let dir = state.models_dir.lock().unwrap().clone();
    let mut entries = Vec::new();
    let mut seen_files = HashSet::new();
    for root in model_scan_roots(&dir) {
        for e in WalkDir::new(&root)
            .max_depth(6)
            .follow_links(true)
            .into_iter()
            .filter_map(|e| e.ok())
        {
            let path = e.path().to_path_buf();
            if !is_gguf_path(&path) {
                continue;
            }
            let file_key = path.canonicalize().unwrap_or_else(|_| path.clone());
            if !seen_files.insert(file_key) {
                continue;
            }
            let size = path.metadata().map(|m| m.len()).unwrap_or(0);
            let size_gb = size as f32 / (1024.0 * 1024.0 * 1024.0);
            let parent = path.parent().unwrap_or(&root).to_path_buf();
            let tok = parent.join("tokenizer.json");
            let tokenizer_path = if tok.exists() { Some(tok.to_string_lossy().into_owned()) } else { None };
            let name = if parent != root {
                parent.file_name().map(|n| n.to_string_lossy().into_owned())
                    .unwrap_or_else(|| path.file_stem().unwrap_or_default().to_string_lossy().into_owned())
            } else {
                path.file_stem().unwrap_or_default().to_string_lossy().into_owned()
            };
            entries.push(ModelEntry {
                name,
                gguf_path: path.to_string_lossy().into_owned(),
                tokenizer_path,
                size_gb,
                dir: parent.to_string_lossy().into_owned(),
            });
        }
    }
    entries.sort_by(|a, b| a.name.to_lowercase().cmp(&b.name.to_lowercase()));
    entries
}

#[command]
fn get_models_dir(state: State<'_, AppState>) -> String {
    state.models_dir.lock().unwrap().to_string_lossy().into_owned()
}

#[command]
fn set_models_dir(state: State<'_, AppState>, path: String) -> Result<(), String> {
    let p = PathBuf::from(&path);
    std::fs::create_dir_all(&p).map_err(|e| e.to_string())?;
    *state.models_dir.lock().unwrap() = p;
    Ok(())
}

#[command]
fn open_models_dir(state: State<'_, AppState>) {
    let dir = state.models_dir.lock().unwrap().clone();
    #[cfg(target_os = "linux")]
    let _ = std::process::Command::new("xdg-open").arg(&dir).spawn();
    #[cfg(target_os = "macos")]
    let _ = std::process::Command::new("open").arg(&dir).spawn();
    #[cfg(target_os = "windows")]
    let _ = std::process::Command::new("explorer").arg(&dir).spawn();
}

#[command]
fn check_dependencies(state: State<'_, AppState>) -> resolve::DepReport {
    let models_dir = state.models_dir.lock().unwrap().clone();
    resolve::check_dependencies(&models_dir)
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

// ── Entry point ───────────────────────────────────────────────────────────────

fn main() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_fs::init())
        .plugin(tauri_plugin_shell::init())
        .setup(|_app| {
            // Kill any server left over from a previous session (crash or force-quit).
            engine::kill_our_stale_servers();
            Ok(())
        })
        .on_window_event(|_window, event| {
            // Kill our server the moment the last window closes — don't hold VRAM.
            if matches!(event, tauri::WindowEvent::Destroyed) {
                engine::kill_our_stale_servers();
            }
        })
        .manage(make_state())
        .manage(lora::new_lora_handle())
        .manage(merge::new_merge_handle())
        .manage(imggen::new_daemon_handle())
        .manage(pty::new_handle())
        .invoke_handler(tauri::generate_handler![
            // Inference — single
            load_model, unload_model, current_model_port, stop_model_server,
            attach_model, scan_running_servers,
            inspect_gguf, generate, stop_generate,
            get_gpu_stats, save_session, load_session, write_binary_b64,
            clear_user_data,
            // Inference — dual agent
            load_drafter, load_critic, dual_agent_status, dual_generate,
            // Models directory / startup
            scan_models_dir, get_models_dir, set_models_dir, open_models_dir,
            check_dependencies,
            // Agent write mode
            get_agent_write_mode, set_agent_write_mode,
            // Agent — Filesystem
            fs_read, fs_write, fs_append, fs_delete, fs_list, fs_tree,
            fs_search, fs_move, fs_copy, fs_mkdir, fs_exists,
            get_sandbox_root, set_sandbox_root,
            // Agent — Sandbox
            exec_command, kill_process, list_processes,
            // Agent — Patch
            diff_proposed, apply_patch, apply_unified_diff,
            undo_patch, patch_history, diff_files,
            // Agent — Memory
            mem_start_task, mem_finish_task, mem_remember,
            mem_recall, mem_forget, mem_context, mem_store,
            // Agent — Planner
            plan_parse, plan_get, plan_prompt_template,
            plan_execute, agent_run, check_goal_completion,
            // PTY terminal
            pty::pty_spawn, pty::pty_write, pty::pty_resize, pty::pty_kill,
            // Setup wizard
            setup::detect_system, setup::run_setup, setup::skip_setup, setup::reset_setup,
            // Image Gen
            imggen::imggen_scan_models, imggen::imggen_scan_checkpoints,
            imggen::imggen_scan_loras, imggen::imggen_generate,
            imggen::imggen_load, imggen::imggen_unload, imggen::imggen_loaded_model,
            // TTS
            tts::tts_voices, tts::tts_generate,
            // LoRA trainer
            lora::lora_start_training, lora::lora_stop_training,
            lora::lora_clean_dataset,
            // Checkpoint merger
            merge::merge_start, merge::merge_cancel,
        ])
        .run(tauri::generate_context!())
        .expect("error while running llm-runtime");
}
