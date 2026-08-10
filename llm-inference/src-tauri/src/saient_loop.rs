//! Saient's heartbeat, tied to the app's lifetime.
//!
//! She used to live only in a terminal somebody had to remember to leave open,
//! while the title-bar button claimed to switch her on and off and governed
//! nothing but a chat prompt. A control that lies about something with real
//! consequences is worse than no control, so the button now writes the flag this
//! loop reads every tick.
//!
//! Two independent things end her, because one was not enough: `stop()` on app
//! exit, and PR_SET_PDEATHSIG inside the loop itself, which fires even if the
//! app crashes rather than exits.
//!
//! This comment previously claimed `kill_on_drop` — that is tokio's Command, not
//! `std`'s, and the child was in fact orphaned and still ticking after the app
//! was killed. A mind that outlives the thing that switched it on is the ghost
//! process this machine has had before.

use std::fs::{self, OpenOptions};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::{Mutex, OnceLock};

use crate::paths;
use crate::resolve::NoConsole;

pub const RUNTIME_DIR_ENV: &str = "SAIENT_RUNTIME_DIR";
pub const STATE_DIR_ENV: &str = "SAIENT_STATE_DIR";

/// Captured from Tauri at startup. Release bundles preserve resource paths, but
/// the platform-specific resource root itself is only known at runtime.
static RESOURCE_DIR: OnceLock<PathBuf> = OnceLock::new();

pub fn set_resource_dir(path: PathBuf) {
    let _ = RESOURCE_DIR.set(path);
}

pub struct LoopHandle(pub Mutex<Option<Child>>);

impl Default for LoopHandle {
    fn default() -> Self {
        LoopHandle(Mutex::new(None))
    }
}

/// The file the app writes and the loop reads. Deliberately a file rather than a
/// signal: it survives either side restarting, and the loop can be started or
/// stopped independently without the two disagreeing about her state.
pub fn enabled_flag() -> PathBuf {
    paths::data_dir().join("saient_enabled")
}

fn is_runtime_dir(path: &Path) -> bool {
    path.join("run_saient.py").is_file() && path.join("orchestrator.py").is_file()
}

/// The runtime shipped inside this desktop repository/application.
///
/// The environment override exists for development, but there is deliberately
/// no machine-specific fallback to the old research checkout: if the bundled
/// runtime is absent, startup reports that fact instead of silently borrowing
/// code from somewhere outside Saient Desktop.
pub fn runtime_dir() -> Option<PathBuf> {
    if let Some(path) = std::env::var_os(RUNTIME_DIR_ENV).map(PathBuf::from) {
        if is_runtime_dir(&path) {
            return Some(path);
        }
    }

    if let Some(root) = RESOURCE_DIR.get() {
        for candidate in [root.join("saient"), root.join("resources").join("saient")] {
            if is_runtime_dir(&candidate) {
                return Some(candidate);
            }
        }
    }

    let source_resource = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("resources")
        .join("saient");
    is_runtime_dir(&source_resource).then_some(source_resource)
}

pub fn state_dir() -> PathBuf {
    paths::data_dir().join("saient")
}

/// Write the effective loop gate. Read on her next beat, not at her next restart.
pub fn set_enabled(enabled: bool) -> Result<(), String> {
    let flag = enabled_flag();
    if let Some(parent) = flag.parent() {
        fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    fs::write(&flag, if enabled { "on" } else { "off" }).map_err(|e| e.to_string())
}

pub fn is_enabled() -> bool {
    fs::read_to_string(enabled_flag())
        .map(|s| s.trim() == "on")
        .unwrap_or(false)
}

/// Start her alongside the app.
///
/// Writes are on so she can build something observable; commands stay off. She
/// runs unattended, and the distance between "edits files in one confined
/// directory" and "has a shell" is the whole of the risk.
pub fn start(handle: &LoopHandle, workspace: PathBuf, interval: f64) -> Result<(), String> {
    let mut guard = handle.0.lock().map_err(|e| e.to_string())?;
    if let Some(child) = guard.as_mut() {
        if matches!(child.try_wait(), Ok(None)) {
            return Ok(());
        }
        *guard = None;
    }

    let runtime = runtime_dir().ok_or_else(|| {
        "Saient's bundled runtime is missing (run_saient.py/orchestrator.py)".to_string()
    })?;
    let script = runtime.join("run_saient.py");
    fs::create_dir_all(&workspace).map_err(|e| e.to_string())?;
    let state = state_dir();
    fs::create_dir_all(&state).map_err(|e| e.to_string())?;
    let log_path = state.join("heartbeat.log");
    let log = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&log_path)
        .map_err(|e| format!("could not open {}: {e}", log_path.display()))?;
    let log_err = log
        .try_clone()
        .map_err(|e| format!("could not clone Saient heartbeat log: {e}"))?;

    let python = crate::resolve::find_python()
        .map_err(|e| format!("could not find Python for Saient: {e}"))?;
    let mut command = Command::new(python);
    command
        .arg(&script)
        .arg("--workspace")
        .arg(&workspace)
        .arg("--interval")
        .arg(interval.to_string())
        .arg("--enabled-file")
        .arg(enabled_flag())
        .env(STATE_DIR_ENV, &state)
        .env("PYTHONDONTWRITEBYTECODE", "1")
        .current_dir(&runtime)
        // Nobody consumed the old pipes, so a long-running heartbeat could
        // eventually fill them and freeze. Append to an inspectable local log.
        .stdout(Stdio::from(log))
        .stderr(Stdio::from(log_err))
        .no_console();
    paths::apply_child_env(&mut command);

    let child = command
        .spawn()
        .map_err(|e| format!("could not start Saient: {e}"))?;

    *guard = Some(child);
    Ok(())
}

/// Move the heartbeat to a newly opened project. The executor owns its
/// workspace for its entire lifetime, so changing only the desktop tool roots
/// would leave autonomous edits landing in the previous project.
pub fn restart(handle: &LoopHandle, workspace: PathBuf, interval: f64) -> Result<(), String> {
    stop(handle);
    start(handle, workspace, interval)
}

pub fn stop(handle: &LoopHandle) {
    if let Ok(mut guard) = handle.0.lock() {
        if let Some(mut child) = guard.take() {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

pub fn running(handle: &LoopHandle) -> bool {
    handle
        .0
        .lock()
        .map(|mut g| match g.as_mut() {
            Some(child) => matches!(child.try_wait(), Ok(None)),
            None => false,
        })
        .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bundled_runtime_is_present_in_source_builds() {
        let runtime = runtime_dir().expect("bundled Saient runtime");
        assert!(runtime.join("run_saient.py").is_file());
        assert!(runtime.join("orchestrator.py").is_file());
    }

    #[test]
    fn mutable_state_lives_outside_the_bundled_runtime() {
        let runtime = runtime_dir().expect("bundled Saient runtime");
        assert_ne!(state_dir(), runtime.join("data"));
        assert!(state_dir().starts_with(paths::data_dir()));
    }
}
