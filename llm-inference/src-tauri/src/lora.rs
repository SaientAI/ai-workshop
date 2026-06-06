/// lora.rs — LoRA trainer commands
///
/// Spawns `train_lora_sdxl.py` in a background thread and forwards
/// its JSON stdout lines to the frontend as `lora-event` Tauri events.

use serde::Deserialize;
use std::io::BufRead;
use std::process::{Command, Stdio};
use std::sync::{Arc, Mutex};
use tauri::{Emitter, WebviewWindow};
use crate::resolve;
use crate::resolve::NoConsole;

/// PID of the currently running training process (None = not training).
#[derive(Clone)]
pub struct LoraHandle(Arc<Mutex<Option<u32>>>);

impl std::ops::Deref for LoraHandle {
    type Target = Arc<Mutex<Option<u32>>>;
    fn deref(&self) -> &Self::Target { &self.0 }
}

pub fn new_lora_handle() -> LoraHandle {
    LoraHandle(Arc::new(Mutex::new(None)))
}

#[derive(Deserialize)]
pub struct LoraTrainPayload {
    pub model_path:  String,
    pub dataset_dir: String,
    pub output_name: String,
    pub output_dir:  Option<String>,
    pub rank:        Option<u32>,
    pub alpha:       Option<f32>,
    pub lr:          Option<f64>,
    pub epochs:      Option<u32>,
    pub batch_size:  Option<u32>,
    pub resolution:  Option<u32>,
}

// ── Commands ──────────────────────────────────────────────────────────────────

#[tauri::command]
pub async fn lora_start_training(
    payload: LoraTrainPayload,
    window: WebviewWindow,
    handle: tauri::State<'_, LoraHandle>,
) -> Result<(), String> {
    stop_impl(handle.inner());

    let output_dir = payload.output_dir.clone()
        .unwrap_or_else(|| resolve::default_lora_dir().to_string_lossy().into_owned());
    let output_path = std::path::PathBuf::from(&output_dir)
        .join(format!("{}.safetensors", payload.output_name.trim()));

    let cfg = serde_json::json!({
        "model_path":  payload.model_path,
        "dataset_dir": payload.dataset_dir,
        "output_path": output_path,
        "rank":        payload.rank.unwrap_or(16),
        "alpha":       payload.alpha.unwrap_or(16.0),
        "lr":          payload.lr.unwrap_or(1e-4_f64),
        "epochs":      payload.epochs.unwrap_or(10),
        "batch_size":  payload.batch_size.unwrap_or(1),
        "resolution":  payload.resolution.unwrap_or(1024),
    });

    let tmp = std::env::temp_dir().join(format!(
        "lora_cfg_{}.json",
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
            let _ = window.emit("lora-event",
                serde_json::json!({"type": "error", "msg": msg}));
        };

        let python = match resolve::find_python() {
            Ok(p) => p,
            Err(e) => { emit_err(e.to_string()); return; }
        };
        let script = match resolve::find_script("train_lora_sdxl.py") {
            Ok(p) => p,
            Err(e) => { emit_err(e.to_string()); return; }
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
            Err(e) => { emit_err(format!("Failed to start training script: {e}")); return; }
        };

        *handle_arc.lock().unwrap() = Some(child.id());

        if let Some(stdout) = child.stdout.take() {
            for line in std::io::BufReader::new(stdout).lines() {
                let Ok(line) = line else { break };
                if let Ok(v) = serde_json::from_str::<serde_json::Value>(&line) {
                    let _ = window.emit("lora-event", v);
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
pub async fn lora_stop_training(
    handle: tauri::State<'_, LoraHandle>,
) -> Result<(), String> {
    stop_impl(handle.inner());
    Ok(())
}

// ── Dataset cleaner ───────────────────────────────────────────────────────────

#[tauri::command]
pub async fn lora_clean_dataset(
    dataset_dir: String,
    window: WebviewWindow,
) -> Result<(), String> {
    if !std::path::Path::new(&dataset_dir).is_dir() {
        return Err(format!("Not a directory: {dataset_dir}"));
    }

    tokio::task::spawn_blocking(move || {
        let emit_err = |msg: String| {
            let _ = window.emit("lora-event",
                serde_json::json!({"type": "error", "msg": msg}));
        };

        let python = match resolve::find_python() {
            Ok(p) => p,
            Err(e) => { emit_err(e.to_string()); return; }
        };
        let script = match resolve::find_script("clean_lora_dataset.py") {
            Ok(p) => p,
            Err(e) => { emit_err(e.to_string()); return; }
        };
        let mut child = match Command::new(python)
            .arg(script)
            .arg(&dataset_dir)
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .no_console()
            .spawn()
        {
            Ok(c)  => c,
            Err(e) => { emit_err(format!("Failed to start cleaner: {e}")); return; }
        };

        if let Some(stdout) = child.stdout.take() {
            for line in std::io::BufReader::new(stdout).lines() {
                let Ok(line) = line else { break };
                if let Ok(v) = serde_json::from_str::<serde_json::Value>(&line) {
                    let _ = window.emit("lora-event", v);
                }
            }
        }
        let _ = child.wait();
    });

    Ok(())
}

fn stop_impl(handle: &LoraHandle) {
    if let Some(pid) = handle.lock().unwrap().take() {
        let _ = Command::new("kill")
            .args(["-TERM", &pid.to_string()])
            .status();
    }
}
