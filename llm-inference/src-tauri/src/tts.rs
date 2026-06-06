use serde::{Deserialize, Serialize};
use std::io::Write;
use std::process::{Command, Stdio};
use tauri::{Emitter, WebviewWindow};
use crate::resolve;
use crate::resolve::NoConsole;

#[derive(Deserialize)]
pub struct TtsPayload {
    pub text:  String,
    pub voice: Option<String>,
    pub speed: Option<f32>,
    pub lang:  Option<String>,
}

#[derive(Serialize)]
pub struct TtsResult {
    pub base64_wav:  String,
    pub duration:    f64,
    pub sample_rate: u32,
}

#[tauri::command]
pub fn tts_voices() -> Vec<serde_json::Value> {
    serde_json::from_str(VOICES_JSON).unwrap_or_default()
}

#[tauri::command]
pub async fn tts_generate(payload: TtsPayload, window: WebviewWindow) -> Result<TtsResult, String> {
    // Run the blocking Python call on the thread-pool so the UI stays responsive
    tokio::task::spawn_blocking(move || run_tts(payload, window))
        .await
        .map_err(|e| format!("task join: {e}"))?
}

fn run_tts(payload: TtsPayload, window: WebviewWindow) -> Result<TtsResult, String> {
    let input = serde_json::json!({
        "text":  payload.text,
        "voice": payload.voice.unwrap_or_else(|| "af_heart".into()),
        "speed": payload.speed.unwrap_or(1.0_f32),
        "lang":  payload.lang.unwrap_or_else(|| "a".into()),
    });

    let python = resolve::find_python().map_err(|e: anyhow::Error| e.to_string())?;
    let script = resolve::find_script("tts_kokoro.py").map_err(|e: anyhow::Error| e.to_string())?;
    let mut child = Command::new(python)
        .arg(script)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .no_console()
        .spawn()
        .map_err(|e| format!("Failed to spawn Python: {}", e))?;

    if let Some(mut stdin) = child.stdin.take() {
        stdin.write_all(input.to_string().as_bytes())
            .map_err(|e| format!("stdin write: {}", e))?;
    }

    // Stream progress from stderr
    if let Some(stderr) = child.stderr.take() {
        let win = window.clone();
        std::thread::spawn(move || {
            use std::io::{BufRead, BufReader};
            for line in BufReader::new(stderr).lines().map_while(Result::ok) {
                if let Ok(v) = serde_json::from_str::<serde_json::Value>(&line) {
                    if let Some(p) = v["progress"].as_u64() {
                        let _ = win.emit("tts_progress", p);
                    }
                }
            }
        });
    }

    let output = child.wait_with_output().map_err(|e| format!("wait: {}", e))?;
    let stdout  = String::from_utf8_lossy(&output.stdout);
    let v: serde_json::Value = serde_json::from_str(&stdout)
        .map_err(|_| format!("Bad JSON from Python: {}", stdout))?;

    if let Some(err) = v["error"].as_str() {
        return Err(err.to_string());
    }

    Ok(TtsResult {
        base64_wav:  v["base64_wav"].as_str().unwrap_or("").to_string(),
        duration:    v["duration"].as_f64().unwrap_or(0.0),
        sample_rate: v["sample_rate"].as_u64().unwrap_or(24000) as u32,
    })
}

// Available Kokoro voices
const VOICES_JSON: &str = r#"[
  {"id":"af_heart",  "label":"Heart (F · American)",   "lang":"a"},
  {"id":"af_bella",  "label":"Bella (F · American)",   "lang":"a"},
  {"id":"af_sarah",  "label":"Sarah (F · American)",   "lang":"a"},
  {"id":"af_nicole", "label":"Nicole (F · American)",  "lang":"a"},
  {"id":"am_adam",   "label":"Adam (M · American)",    "lang":"a"},
  {"id":"am_michael","label":"Michael (M · American)", "lang":"a"},
  {"id":"bf_emma",   "label":"Emma (F · British)",     "lang":"b"},
  {"id":"bf_isabella","label":"Isabella (F · British)","lang":"b"},
  {"id":"bm_george", "label":"George (M · British)",   "lang":"b"},
  {"id":"bm_lewis",  "label":"Lewis (M · British)",    "lang":"b"}
]"#;
