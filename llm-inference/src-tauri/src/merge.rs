/// merge.rs — checkpoint blending commands
///
/// Spawns `merge_checkpoints.py` in a background thread and forwards
/// its JSON stdout lines to the frontend as `merge-event` Tauri events.

use serde::Deserialize;
use std::io::BufRead;
use std::process::{Command, Stdio};
use std::sync::{Arc, Mutex};
use tauri::{Emitter, WebviewWindow};
use crate::resolve;
use crate::resolve::NoConsole;

#[derive(Clone)]
pub struct MergeHandle(Arc<Mutex<Option<u32>>>);

impl std::ops::Deref for MergeHandle {
    type Target = Arc<Mutex<Option<u32>>>;
    fn deref(&self) -> &Self::Target { &self.0 }
}

pub fn new_merge_handle() -> MergeHandle {
    MergeHandle(Arc::new(Mutex::new(None)))
}

#[derive(Deserialize)]
pub struct MergePayload {
    pub model_a:     String,
    pub model_b:     String,
    pub model_c:     Option<String>,
    pub method:      String,   // "weighted_sum" | "add_difference"
    pub weight:      f64,
    pub output_path: String,
}

// ── Commands ──────────────────────────────────────────────────────────────────

#[tauri::command]
pub async fn merge_start(
    payload: MergePayload,
    window: WebviewWindow,
    handle: tauri::State<'_, MergeHandle>,
) -> Result<(), String> {
    cancel_impl(handle.inner());

    let cfg = serde_json::json!({
        "model_a":     payload.model_a,
        "model_b":     payload.model_b,
        "model_c":     payload.model_c,
        "method":      payload.method,
        "weight":      payload.weight,
        "output_path": payload.output_path,
    });

    let tmp = std::env::temp_dir().join(format!(
        "merge_cfg_{}.json",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis()
    ));
    std::fs::write(&tmp, cfg.to_string()).map_err(|e| e.to_string())?;

    let handle_arc = handle.inner().clone();
    let tmp_clone  = tmp.clone();

    tokio::task::spawn_blocking(move || {
        let emit_err = |msg: String| {
            let _ = window.emit("merge-event",
                serde_json::json!({"type": "error", "msg": msg}));
        };

        let python = match resolve::find_python() {
            Ok(p) => p, Err(e) => { emit_err(e.to_string()); return; }
        };
        let script = match resolve::find_script("merge_checkpoints.py") {
            Ok(p) => p, Err(e) => { emit_err(e.to_string()); return; }
        };
        let mut child = match Command::new(python)
            .arg(script)
            .arg(&tmp_clone)
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .no_console()
            .spawn()
        {
            Ok(c)  => c,
            Err(e) => { emit_err(format!("Failed to start merge script: {e}")); return; }
        };

        *handle_arc.lock().unwrap() = Some(child.id());

        if let Some(stdout) = child.stdout.take() {
            for line in std::io::BufReader::new(stdout).lines() {
                let Ok(line) = line else { break };
                if let Ok(v) = serde_json::from_str::<serde_json::Value>(&line) {
                    let _ = window.emit("merge-event", v);
                }
            }
        }

        let _ = child.wait();
        *handle_arc.lock().unwrap() = None;
        let _ = std::fs::remove_file(&tmp_clone);
    });

    Ok(())
}

#[tauri::command]
pub async fn merge_cancel(
    handle: tauri::State<'_, MergeHandle>,
) -> Result<(), String> {
    cancel_impl(handle.inner());
    Ok(())
}

fn cancel_impl(handle: &MergeHandle) {
    if let Some(pid) = handle.lock().unwrap().take() {
        let _ = Command::new("kill")
            .args(["-TERM", &pid.to_string()])
            .status();
    }
}
