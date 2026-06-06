//! vision.rs — local image understanding (Moondream2) via a warm Python daemon.
//! Mirrors the imggen daemon: spawn scripts/vision.py once, then stream JSON requests.

use serde::Serialize;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::{Arc, Mutex};
use tauri::State;

use crate::resolve;

#[derive(Serialize)]
pub struct VisionResult {
    pub answer:  String,
    pub elapsed: f64,
    pub device:  String,
}

pub struct VisionDaemon {
    _child: Child,
    stdin:  BufWriter<ChildStdin>,
    stdout: BufReader<ChildStdout>,
    pub device: String,
}

impl Drop for VisionDaemon {
    fn drop(&mut self) { let _ = self._child.kill(); }
}

pub type VisionHandle = Arc<Mutex<Option<VisionDaemon>>>;
pub fn new_vision_handle() -> VisionHandle { Arc::new(Mutex::new(None)) }

// ── Commands ────────────────────────────────────────────────────────────────────

/// Describe/answer about an image (base64 PNG/JPEG). Loads the model on first use.
#[tauri::command]
pub async fn vision_describe(
    daemon: State<'_, VisionHandle>,
    image_b64: String,
    question: String,
) -> Result<VisionResult, String> {
    let arc = daemon.inner().clone();
    tokio::task::spawn_blocking(move || do_describe(arc, image_b64, question))
        .await
        .map_err(|e| format!("task join: {e}"))?
}

/// Free the vision model's VRAM.
#[tauri::command]
pub async fn vision_unload(daemon: State<'_, VisionHandle>) -> Result<(), String> {
    *daemon.lock().map_err(|e| e.to_string())? = None;
    Ok(())
}

#[tauri::command]
pub fn vision_loaded(daemon: State<'_, VisionHandle>) -> Result<bool, String> {
    Ok(daemon.lock().map_err(|e| e.to_string())?.is_some())
}

// ── Internal ────────────────────────────────────────────────────────────────────

fn ensure_loaded(guard: &mut Option<VisionDaemon>) -> Result<(), String> {
    if guard.is_some() { return Ok(()); }

    let python = resolve::find_python().map_err(|e: anyhow::Error| e.to_string())?;
    let script = resolve::find_script("vision.py").map_err(|e: anyhow::Error| e.to_string())?;

    let mut child = Command::new(python)
        .arg(script)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()
        .map_err(|e| format!("Failed to spawn Python: {e}"))?;

    let mut stdin  = BufWriter::new(child.stdin.take().ok_or("no stdin")?);
    let mut stdout = BufReader::new(child.stdout.take().ok_or("no stdout")?);

    let cfg = serde_json::json!({ "model": "moondream2", "device": "auto" });
    writeln!(stdin, "{cfg}").map_err(|e| format!("stdin write: {e}"))?;
    stdin.flush().map_err(|e| format!("stdin flush: {e}"))?;

    let device = loop {
        let mut line = String::new();
        let n = stdout.read_line(&mut line).map_err(|e| format!("read ready: {e}"))?;
        if n == 0 {
            return Err("The vision model couldn't start. Run Full setup to install the vision tools (Python + transformers).".into());
        }
        let t = line.trim();
        if t.is_empty() { continue; }
        let v: serde_json::Value = serde_json::from_str(t).map_err(|e| format!("bad JSON from vision daemon: {e}"))?;
        if let Some(err) = v["error"].as_str() { return Err(err.to_string()); }
        if v["ready"].as_bool() == Some(true) {
            break v["device"].as_str().unwrap_or("unknown").to_string();
        }
    };

    *guard = Some(VisionDaemon { _child: child, stdin, stdout, device });
    Ok(())
}

fn do_describe(arc: VisionHandle, image_b64: String, question: String) -> Result<VisionResult, String> {
    let mut guard = arc.lock().map_err(|e| e.to_string())?;
    ensure_loaded(&mut guard)?;
    let daemon = guard.as_mut().ok_or("vision daemon not loaded")?;
    let device = daemon.device.clone();

    let req = serde_json::json!({ "image_b64": image_b64, "question": question });
    writeln!(daemon.stdin, "{req}").map_err(|e| format!("stdin write: {e}"))?;
    daemon.stdin.flush().map_err(|e| format!("stdin flush: {e}"))?;

    loop {
        let mut line = String::new();
        let n = daemon.stdout.read_line(&mut line).map_err(|e| format!("read: {e}"))?;
        if n == 0 { return Err("vision daemon exited".into()); }
        let t = line.trim();
        if t.is_empty() { continue; }
        let v: serde_json::Value = serde_json::from_str(t).map_err(|e| format!("bad JSON: {e}"))?;
        if let Some(err) = v["error"].as_str() { return Err(err.to_string()); }
        if let Some(ans) = v["answer"].as_str() {
            return Ok(VisionResult {
                answer: ans.to_string(),
                elapsed: v["elapsed"].as_f64().unwrap_or(0.0),
                device,
            });
        }
    }
}
