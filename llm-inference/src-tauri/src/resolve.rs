//! resolve.rs — runtime path discovery for Python, scripts, and tinyq4.
//!
//! Callers should prefer env-var overrides and project-local paths.

use std::path::{Path, PathBuf};
use std::sync::OnceLock;

static RESOURCE_DIR: OnceLock<PathBuf> = OnceLock::new();

pub fn set_resource_dir(path: PathBuf) {
    let _ = RESOURCE_DIR.set(path);
}

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
///   2. project-local managed venv
///   3. Windows Python launchers
///   4. `python3` on PATH
///   5. `/usr/bin/python3`
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

    // 3. Windows has no `which` command. Probe the same launchers accepted by
    // setup so an explicitly managed system environment can still be used.
    #[cfg(target_os = "windows")]
    for candidate in ["python", "python3", "py"] {
        if let Ok(out) = std::process::Command::new(candidate)
            .arg("--version")
            .no_console()
            .output()
        {
            if out.status.success() { return Ok(PathBuf::from(candidate)); }
        }
    }

    // 4. python3 on PATH
    if let Ok(out) = std::process::Command::new("which").arg("python3").output() {
        if out.status.success() {
            let p = PathBuf::from(String::from_utf8_lossy(&out.stdout).trim());
            if p.exists() { return Ok(p); }
        }
    }

    // 5. Absolute fallback
    let fallback = PathBuf::from("/usr/bin/python3");
    if fallback.exists() { return Ok(fallback); }

    anyhow::bail!(
        "Python not found. Set PYTHON_PATH env var or install python3. \
         Alternatively run Saient setup to create the managed venv."
    )
}

/// Resolve Python and verify the modules needed by Image Gen before starting
/// its long-lived daemon. This turns an opaque Python traceback into a direct
/// recovery route when setup was skipped or only partially completed.
pub fn find_image_python() -> anyhow::Result<PathBuf> {
    let python = find_python()?;
    let code = r#"import importlib.util, sys
required = ("torch", "diffusers", "transformers", "PIL")
missing = [name for name in required if importlib.util.find_spec(name) is None]
print(", ".join(missing))
sys.exit(1 if missing else 0)
"#;
    let output = std::process::Command::new(&python)
        .args(["-c", code])
        .no_console()
        .output()
        .map_err(|e| anyhow::anyhow!("Could not check the Image Gen Python environment: {e}"))?;
    if output.status.success() { return Ok(python); }

    let missing = String::from_utf8_lossy(&output.stdout).trim().to_string();
    let detail = if missing.is_empty() {
        String::from_utf8_lossy(&output.stderr).trim().to_string()
    } else {
        format!("missing: {missing}")
    };
    let detail = if detail.is_empty() {
        String::new()
    } else {
        format!(" ({detail})")
    };
    anyhow::bail!(
        "Image Gen is not set up for this Python environment{detail}. Open Settings → Setup and run Full setup."
    )
}

pub fn image_runtime_ready() -> bool {
    find_image_python().is_ok()
}

/// Find a bundled Python script by filename (e.g. "generate_sdxl.py").
///
/// Resolution order:
///   1. `SCRIPTS_DIR` env var / name
///   2. Tauri's packaged resource directory
///   3. Next to the current executable
///   4. `scripts/` directory above the executable (dev layout)
pub fn find_script(name: &str) -> anyhow::Result<PathBuf> {
    // 1. Explicit scripts dir override
    if let Ok(dir) = std::env::var("SCRIPTS_DIR") {
        let p = PathBuf::from(&dir).join(name);
        if p.exists() { return Ok(p); }
        anyhow::bail!("SCRIPTS_DIR='{}' but '{}' not found there", dir, name);
    }

    // 2. Packaged resources. Tauri preserves the configured resources/ prefix.
    if let Some(root) = RESOURCE_DIR.get() {
        for candidate in [root.join("scripts").join(name),
                          root.join("resources").join("scripts").join(name)] {
            if candidate.is_file() { return Ok(candidate); }
        }
    }

    // 3. Bundled next to our executable
    if let Ok(exe) = std::env::current_exe() {
        let exe_dir = exe.parent().unwrap_or(Path::new("."));
        let candidate = exe_dir.join(name);
        if candidate.exists() { return Ok(candidate); }

        // 4. scripts/ relative to exe (dev: target/release/ → ancestors/scripts/)
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

/// Return candidate directories for diffusers/video model scanning.
///
/// Only Saient-owned category folders are scanned — the app never reaches into
/// arbitrary locations elsewhere on the machine. Image and video each have their
/// own folder; the callers filter results by pipeline class.
pub fn model_scan_dirs() -> Vec<PathBuf> {
    vec![
        crate::paths::image_models_dir(),
        crate::paths::video_models_dir(),
    ]
}

/// Saient's managed folder where in-app downloads land (and are scanned first).
pub fn checkpoints_download_dir() -> PathBuf {
    let d = crate::paths::checkpoints_dir();
    std::fs::create_dir_all(&d).ok();
    d
}
pub fn loras_download_dir() -> PathBuf {
    let d = crate::paths::loras_dir();
    std::fs::create_dir_all(&d).ok();
    d
}
pub fn models_download_dir() -> PathBuf {
    let d = crate::paths::models_dir();
    std::fs::create_dir_all(&d).ok();
    d
}

/// Return candidate directories for .safetensors checkpoint scanning.
pub fn checkpoint_scan_dirs() -> Vec<PathBuf> {
    vec![
        crate::paths::checkpoints_dir(),
        crate::paths::image_models_dir(),
    ]
}

/// Return candidate directories for LoRA .safetensors scanning.
pub fn lora_scan_dirs() -> Vec<PathBuf> {
    vec![
        crate::paths::loras_dir(),
        crate::paths::image_models_dir().join("lora"),
    ]
}

/// Return default LoRA output directory.
pub fn default_lora_dir() -> PathBuf {
    crate::paths::loras_dir()
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
