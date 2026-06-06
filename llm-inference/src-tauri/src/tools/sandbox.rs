/// sandbox.rs — Command execution sandbox
/// Runs shell commands in a controlled environment:
/// - Scoped working directory
/// - Configurable timeout with SIGTERM + SIGKILL
/// - Allowlist/denylist for commands
/// - Environment variable control
/// - Streaming stdout/stderr via Tauri events
/// - Exit code, duration, output capture

use anyhow::{bail, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::Command;
use tokio::time::timeout;

/// Collapse `..` / `.` without requiring the path to exist on disk.
/// Never pops past a root or prefix component — excess `..` are ignored so the
/// result always stays at (or below) the filesystem root.
pub(crate) fn normalize_path(p: &Path) -> PathBuf {
    let mut out: Vec<std::path::Component> = Vec::new();
    for c in p.components() {
        match c {
            std::path::Component::ParentDir => {
                // Only pop Normal components — never pop RootDir/Prefix.
                if matches!(out.last(), Some(std::path::Component::Normal(_))) {
                    out.pop();
                }
            }
            std::path::Component::CurDir => {}
            other => out.push(other),
        }
    }
    out.iter().collect()
}

// ── Types ─────────────────────────────────────────────────────────────────────

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ExecRequest {
    pub command: String,
    pub args: Vec<String>,
    pub cwd: Option<String>,       // relative to sandbox root
    pub env: HashMap<String, String>,
    pub timeout_secs: u64,
    pub stdin_data: Option<String>,
    pub capture_output: bool,      // false = stream events only
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ExecResult {
    pub command: String,
    pub exit_code: Option<i32>,
    pub stdout: String,
    pub stderr: String,
    pub duration_ms: u64,
    pub timed_out: bool,
    pub killed: bool,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ProcessInfo {
    pub id: String,
    pub command: String,
    pub started_at: String,
    pub running: bool,
}

// ── Sandbox config ────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct SandboxConfig {
    pub root: PathBuf,
    pub max_timeout_secs: u64,
    pub max_output_bytes: usize,
    /// Commands that are always blocked regardless of request
    pub blocked_commands: Vec<String>,
    /// If Some, only these commands are allowed
    pub allowed_commands: Option<Vec<String>>,
    /// Environment vars always passed through
    pub base_env: HashMap<String, String>,
}

impl Default for SandboxConfig {
    fn default() -> Self {
        let mut base_env = HashMap::new();
        base_env.insert("PATH".into(), std::env::var("PATH").unwrap_or_default());
        base_env.insert("HOME".into(), std::env::var("HOME").unwrap_or_default());
        base_env.insert("LANG".into(), "en_US.UTF-8".into());
        base_env.insert("TERM".into(), "xterm-256color".into());

        Self {
            root: PathBuf::from("."),
            max_timeout_secs: 300,
            max_output_bytes: 10 * 1024 * 1024, // 10 MB
            blocked_commands: vec![
                "rm -rf /".into(),
                ":(){ :|:& };:".into(), // fork bomb
                "dd if=/dev/".into(),
            ],
            allowed_commands: None, // None = allow all not blocked
            base_env,
        }
    }
}

// ── Running process registry ──────────────────────────────────────────────────

#[derive(Default)]
pub struct ProcessRegistry {
    processes: HashMap<String, Arc<Mutex<bool>>>, // id → kill_flag
}

impl ProcessRegistry {
    pub fn register(&mut self, id: String) -> Arc<Mutex<bool>> {
        let flag = Arc::new(Mutex::new(false));
        self.processes.insert(id, flag.clone());
        flag
    }

    pub fn kill(&mut self, id: &str) -> bool {
        if let Some(flag) = self.processes.get(id) {
            *flag.lock().unwrap() = true;
            true
        } else {
            false
        }
    }

    pub fn remove(&mut self, id: &str) {
        self.processes.remove(id);
    }

    pub fn list(&self) -> Vec<String> {
        self.processes.keys().cloned().collect()
    }
}

// ── Sandbox executor ──────────────────────────────────────────────────────────

pub struct Sandbox {
    pub config: SandboxConfig,
    pub registry: Arc<Mutex<ProcessRegistry>>,
    /// Interior-mutable root so set_sandbox_root can update it without Mutex<Sandbox>.
    pub root: RwLock<PathBuf>,
}

impl Sandbox {
    pub fn new(config: SandboxConfig) -> Self {
        let root = config.root.clone();
        Self {
            config,
            registry: Arc::new(Mutex::new(ProcessRegistry::default())),
            root: RwLock::new(root),
        }
    }

    pub fn set_root(&self, new_root: PathBuf) {
        *self.root.write().unwrap() = new_root;
    }

    pub fn validate(&self, req: &ExecRequest) -> Result<()> {
        // Check blocked patterns
        let full_cmd = format!("{} {}", req.command, req.args.join(" "));
        for blocked in &self.config.blocked_commands {
            if full_cmd.contains(blocked.as_str()) {
                bail!("Command blocked by sandbox policy: {}", blocked);
            }
        }

        // Check allowlist
        if let Some(allowed) = &self.config.allowed_commands {
            if !allowed.iter().any(|a| req.command == *a || req.command.ends_with(&format!("/{}", a))) {
                bail!("Command '{}' not in allowlist", req.command);
            }
        }

        // Timeout cap
        if req.timeout_secs > self.config.max_timeout_secs {
            bail!("Requested timeout {}s exceeds maximum {}s",
                req.timeout_secs, self.config.max_timeout_secs);
        }

        Ok(())
    }

    pub fn resolve_cwd(&self, req: &ExecRequest) -> PathBuf {
        let root = self.root.read().unwrap().clone();
        match &req.cwd {
            Some(rel) => {
                // Normalize to collapse any `..` before the starts_with check
                let p = normalize_path(&root.join(rel.trim_start_matches('/')));
                if p.starts_with(&root) { p } else { root }
            }
            None => root,
        }
    }

    /// Execute a command, streaming output via `on_stdout` / `on_stderr` callbacks.
    pub async fn exec<F, G>(
        &self,
        id: &str,
        req: &ExecRequest,
        on_stdout: F,
        on_stderr: G,
    ) -> Result<ExecResult>
    where
        F: Fn(String) + Send + 'static,
        G: Fn(String) + Send + 'static,
    {
        self.validate(req)?;

        let cwd = self.resolve_cwd(req);
        let kill_flag = self.registry.lock().unwrap().register(id.to_string());
        let kill_flag_clone = kill_flag.clone();

        // Build environment
        let mut env = self.config.base_env.clone();
        env.extend(req.env.clone());

        let timeout_dur = Duration::from_secs(req.timeout_secs.max(1));
        let started = Instant::now();

        // Build command
        let mut cmd = Command::new(&req.command);
        cmd.args(&req.args)
            .current_dir(&cwd)
            .env_clear()
            .envs(&env)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true);

        let stdin_bytes = req.stdin_data.as_deref().map(|s| s.as_bytes().to_vec());
        if stdin_bytes.is_some() {
            cmd.stdin(Stdio::piped());
        } else {
            cmd.stdin(Stdio::null());
        }

        // Don't pop a console window for each spawned command on Windows.
        #[cfg(windows)]
        cmd.creation_flags(0x0800_0000);

        let mut child = cmd.spawn()?;

        // Write stdin and drop handle to signal EOF before reading output
        if let Some(data) = stdin_bytes {
            if let Some(mut stdin) = child.stdin.take() {
                use tokio::io::AsyncWriteExt;
                let _ = stdin.write_all(&data).await;
            }
        }

        let max_bytes = self.config.max_output_bytes;

        // Shared capture buffers — tasks write here, we read after child exits
        let stdout_capture = Arc::new(Mutex::new(String::new()));
        let stderr_capture = Arc::new(Mutex::new(String::new()));

        // Stream stdout
        let stdout_task = if let Some(stdout) = child.stdout.take() {
            let mut reader = BufReader::new(stdout).lines();
            let cap = stdout_capture.clone();
            Some(tokio::spawn(async move {
                while let Ok(Some(line)) = reader.next_line().await {
                    on_stdout(line.clone());
                    let mut b = cap.lock().unwrap();
                    if b.len() < max_bytes { b.push_str(&line); b.push('\n'); }
                }
            }))
        } else { None };

        // Stream stderr
        let stderr_task = if let Some(stderr) = child.stderr.take() {
            let mut reader = BufReader::new(stderr).lines();
            let cap = stderr_capture.clone();
            Some(tokio::spawn(async move {
                while let Ok(Some(line)) = reader.next_line().await {
                    on_stderr(line.clone());
                    let mut b = cap.lock().unwrap();
                    if b.len() < max_bytes { b.push_str(&line); b.push('\n'); }
                }
            }))
        } else { None };

        // Wait with timeout, polling kill flag
        let mut timed_out = false;
        let mut killed = false;
        let exit_status;

        loop {
            // Check kill flag
            if *kill_flag_clone.lock().unwrap() {
                let _ = child.kill().await;
                killed = true;
                exit_status = None;
                break;
            }

            // Try to wait with short poll
            match timeout(Duration::from_millis(100), child.wait()).await {
                Ok(Ok(status)) => {
                    exit_status = Some(status.code().unwrap_or(-1));
                    break;
                }
                Ok(Err(e)) => {
                    return Err(e.into());
                }
                Err(_) => {
                    // Still running — check overall timeout
                    if started.elapsed() >= timeout_dur {
                        let _ = child.kill().await;
                        timed_out = true;
                        exit_status = None;
                        break;
                    }
                }
            }
        }

        self.registry.lock().unwrap().remove(id);

        // Wait for reader tasks to drain before reading captured output
        if let Some(t) = stdout_task { let _ = t.await; }
        if let Some(t) = stderr_task { let _ = t.await; }

        let stdout = stdout_capture.lock().unwrap().clone();
        let stderr = stderr_capture.lock().unwrap().clone();

        Ok(ExecResult {
            command: format!("{} {}", req.command, req.args.join(" ")),
            exit_code: exit_status,
            stdout,
            stderr,
            duration_ms: started.elapsed().as_millis() as u64,
            timed_out,
            killed,
        })
    }

    pub fn kill_process(&self, id: &str) -> bool {
        self.registry.lock().unwrap().kill(id)
    }

    pub fn list_processes(&self) -> Vec<String> {
        self.registry.lock().unwrap().list()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_path_collapses_dotdot() {
        let p = Path::new("/sandbox/root/sub/../file.txt");
        assert_eq!(normalize_path(p), PathBuf::from("/sandbox/root/file.txt"));
    }

    #[test]
    fn normalize_path_collapses_dot() {
        let p = Path::new("/sandbox/./root/./file.txt");
        assert_eq!(normalize_path(p), PathBuf::from("/sandbox/root/file.txt"));
    }

    #[test]
    fn normalize_path_cannot_escape_with_excess_dotdot() {
        // Excess `..` beyond the filesystem root are clamped — the result stays absolute.
        let p = Path::new("/sandbox/../../etc/passwd");
        let result = normalize_path(p);
        // The fixed implementation never pops past root, so result is /etc/passwd
        // (still absolute). resolve_cwd then rejects it via starts_with(&sandbox_root).
        assert!(result.is_absolute(), "result must be absolute, got {:?}", result);
        assert!(!result.starts_with("/sandbox"), "should not be inside /sandbox");
    }

    #[test]
    fn validate_blocks_listed_pattern() {
        let cfg = SandboxConfig {
            blocked_commands: vec!["rm -rf /".into()],
            ..SandboxConfig::default()
        };
        let sb = Sandbox::new(cfg);
        let req = ExecRequest {
            command: "rm".into(),
            args: vec!["-rf".into(), "/".into()],
            cwd: None, env: HashMap::new(),
            timeout_secs: 10, stdin_data: None, capture_output: true,
        };
        assert!(sb.validate(&req).is_err());
    }

    #[test]
    fn validate_allows_safe_command() {
        let sb = Sandbox::new(SandboxConfig::default());
        let req = ExecRequest {
            command: "ls".into(),
            args: vec!["-la".into()],
            cwd: None, env: HashMap::new(),
            timeout_secs: 5, stdin_data: None, capture_output: true,
        };
        assert!(sb.validate(&req).is_ok());
    }

    #[test]
    fn validate_rejects_excessive_timeout() {
        let cfg = SandboxConfig { max_timeout_secs: 10, ..SandboxConfig::default() };
        let sb = Sandbox::new(cfg);
        let req = ExecRequest {
            command: "sleep".into(), args: vec!["9999".into()],
            cwd: None, env: HashMap::new(),
            timeout_secs: 9999, stdin_data: None, capture_output: true,
        };
        assert!(sb.validate(&req).is_err());
    }

    #[test]
    fn resolve_cwd_stays_inside_root() {
        let root = std::env::temp_dir().join("sb_test_resolve");
        std::fs::create_dir_all(&root).unwrap();
        let cfg = SandboxConfig { root: root.clone(), ..SandboxConfig::default() };
        let sb = Sandbox::new(cfg);

        // dotdot escape attempt
        let req = ExecRequest {
            command: "ls".into(), args: vec![],
            cwd: Some("../../etc".into()),
            env: HashMap::new(), timeout_secs: 5, stdin_data: None, capture_output: true,
        };
        let cwd = sb.resolve_cwd(&req);
        assert!(cwd.starts_with(&root), "cwd {:?} escaped sandbox {:?}", cwd, root);
        std::fs::remove_dir_all(&root).ok();
    }
}
