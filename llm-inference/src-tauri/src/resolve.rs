//! resolve.rs — runtime path discovery for Python, scripts, and tinyq4.
//!
//! All hardcoded /home/tiny paths live here as fallbacks only.
//! Callers should prefer env-var overrides and PATH lookups.

use std::path::{Path, PathBuf};

/// Suppress the console window Windows pops for each child process. No-op on Unix.
/// Apply to EVERY `std::process::Command` before spawn/output/status so the app
/// never flashes stray cmd windows (nvidia-smi, python, the engine server, …).
pub trait NoConsole {
    fn no_console(&mut self) -> &mut Self;
}
impl NoConsole for std::process::Command {
    fn no_console(&mut self) -> &mut Self {
        #[cfg(windows)]
        {
            use std::os::windows::process::CommandExt;
            const CREATE_NO_WINDOW: u32 = 0x0800_0000;
            self.creation_flags(CREATE_NO_WINDOW);
        }
        self
    }
}

/// Find the Python interpreter to use for helper scripts.
///
/// Resolution order:
///   1. `PYTHON_PATH` env var
///   2. `~/.venvs/ltx/bin/python` (legacy location)
///   3. `python3` on PATH
///   4. `/usr/bin/python3`
pub fn find_python() -> anyhow::Result<PathBuf> {
    // 1. Explicit override
    if let Ok(p) = std::env::var("PYTHON_PATH") {
        let path = PathBuf::from(&p);
        if path.exists() { return Ok(path); }
        anyhow::bail!("PYTHON_PATH set to '{}' but file not found", p);
    }

    // 2. The setup wizard's managed venv (has the CUDA-matched torch + creative stack)
    let managed = crate::setup::venv_python();
    if managed.exists() { return Ok(managed); }

    // 3. Legacy venv relative to $HOME
    if let Some(home) = home_dir() {
        let venv = home.join(".venvs/ltx/bin/python");
        if venv.exists() { return Ok(venv); }
    }

    // 3. python3 on PATH
    if let Ok(out) = std::process::Command::new("which").arg("python3").output() {
        if out.status.success() {
            let p = PathBuf::from(String::from_utf8_lossy(&out.stdout).trim());
            if p.exists() { return Ok(p); }
        }
    }

    // 4. Absolute fallback
    let fallback = PathBuf::from("/usr/bin/python3");
    if fallback.exists() { return Ok(fallback); }

    anyhow::bail!(
        "Python not found. Set PYTHON_PATH env var or install python3. \
         Alternatively create ~/.venvs/ltx with the required packages."
    )
}

/// Find a bundled Python script by filename (e.g. "generate_sdxl.py").
///
/// Resolution order:
///   1. `SCRIPTS_DIR` env var / name
///   2. Next to the current executable
///   3. `scripts/` directory two levels up from the executable (dev layout)
pub fn find_script(name: &str) -> anyhow::Result<PathBuf> {
    // 1. Explicit scripts dir override
    if let Ok(dir) = std::env::var("SCRIPTS_DIR") {
        let p = PathBuf::from(&dir).join(name);
        if p.exists() { return Ok(p); }
        anyhow::bail!("SCRIPTS_DIR='{}' but '{}' not found there", dir, name);
    }

    // 2. Bundled next to our executable
    if let Ok(exe) = std::env::current_exe() {
        let exe_dir = exe.parent().unwrap_or(Path::new("."));
        let candidate = exe_dir.join(name);
        if candidate.exists() { return Ok(candidate); }

        // 3. scripts/ relative to exe (dev: target/release/  → ../../scripts/)
        for up in 1..=4 {
            let mut anc = exe_dir.to_path_buf();
            for _ in 0..up { anc = anc.parent().unwrap_or(Path::new(".")).to_path_buf(); }
            let candidate = anc.join("scripts").join(name);
            if candidate.exists() { return Ok(candidate); }
        }
    }

    anyhow::bail!(
        "Script '{}' not found. Set SCRIPTS_DIR env var to the directory containing helper scripts.",
        name
    )
}

/// Return candidate directories for diffusers/SDXL model scanning.
pub fn model_scan_dirs() -> Vec<PathBuf> {
    let mut dirs = Vec::new();
    if let Some(home) = home_dir() {
        dirs.push(home.join("models"));
        dirs.push(home.join("projects/models"));
        dirs.push(home.join("llm-runtime/models"));
    }
    dirs
}

/// Return candidate directories for .safetensors checkpoint scanning.
pub fn checkpoint_scan_dirs() -> Vec<PathBuf> {
    let mut dirs = Vec::new();
    if let Some(home) = home_dir() {
        dirs.push(home.join("projects/models/checkpoints"));
        dirs.push(home.join("models/checkpoints"));
        dirs.push(home.join("models"));
    }
    dirs
}

/// Return candidate directories for LoRA .safetensors scanning.
pub fn lora_scan_dirs() -> Vec<PathBuf> {
    let mut dirs = Vec::new();
    if let Some(home) = home_dir() {
        dirs.push(home.join("projects/models/lora"));
        dirs.push(home.join("models/lora"));
    }
    dirs
}

/// Return default LoRA output directory.
pub fn default_lora_dir() -> PathBuf {
    home_dir()
        .map(|h| h.join("models/lora"))
        .unwrap_or_else(|| std::env::temp_dir().join("lora"))
}

fn home_dir() -> Option<PathBuf> {
    std::env::var("HOME").ok().map(PathBuf::from)
}

// ── Dependency checker ────────────────────────────────────────────────────────

use serde::Serialize;

#[derive(Serialize, Clone)]
pub struct DepStatus {
    pub ok: bool,
    pub detail: String,
}

#[derive(Serialize, Clone)]
pub struct DepReport {
    pub tinyq4: DepStatus,
    pub python: DepStatus,
    pub scripts: DepStatus,
    pub gpu: DepStatus,
    pub models_dir: DepStatus,
}

pub fn check_dependencies(models_dir: &Path) -> DepReport {
    DepReport {
        tinyq4: check_tinyq4(),
        python: check_python(),
        scripts: check_scripts(),
        gpu: check_gpu(),
        models_dir: check_models_dir(models_dir),
    }
}

fn check_tinyq4() -> DepStatus {
    match crate::engine::find_tinyq4() {
        Ok(p) => DepStatus { ok: true, detail: p.to_string_lossy().into_owned() },
        Err(e) => DepStatus { ok: false, detail: e.to_string() },
    }
}

fn check_python() -> DepStatus {
    match find_python() {
        Ok(p) => DepStatus { ok: true, detail: p.to_string_lossy().into_owned() },
        Err(e) => DepStatus { ok: false, detail: e.to_string() },
    }
}

fn check_scripts() -> DepStatus {
    let names = ["generate_sdxl.py", "tts_kokoro.py", "train_lora_sdxl.py",
                 "clean_lora_dataset.py", "merge_checkpoints.py"];
    let mut missing = Vec::new();
    for name in &names {
        if find_script(name).is_err() { missing.push(*name); }
    }
    if missing.is_empty() {
        DepStatus { ok: true, detail: "all scripts found".into() }
    } else {
        DepStatus {
            ok: false,
            detail: format!("missing: {}. Set SCRIPTS_DIR env var.", missing.join(", ")),
        }
    }
}

fn check_gpu() -> DepStatus {
    match std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=name,memory.total", "--format=csv,noheader,nounits"])
        .no_console()
        .output()
    {
        Ok(out) if out.status.success() => {
            let info = String::from_utf8_lossy(&out.stdout).trim().to_string();
            DepStatus { ok: true, detail: info }
        }
        _ => DepStatus { ok: false, detail: "nvidia-smi not found or no GPU".into() },
    }
}

fn check_models_dir(dir: &Path) -> DepStatus {
    if !dir.exists() {
        return DepStatus {
            ok: false,
            detail: format!("{} does not exist", dir.display()),
        };
    }
    let count = std::fs::read_dir(dir)
        .map(|e| e.count())
        .unwrap_or(0);
    DepStatus {
        ok: true,
        detail: format!("{} ({} entries)", dir.display(), count),
    }
}
