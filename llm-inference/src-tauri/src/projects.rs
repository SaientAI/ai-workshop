//! Projects — one folder per piece of work.
//!
//! Before this everything shared a single `agent-workspace`, so a snake game, a
//! fibonacci script and a stray test log all sat in one heap. That is untidy on
//! its own, but it also quietly broke checkpoints: a snapshot of "the workspace"
//! captured every unrelated file too, so restoring one task's state could drag
//! back another's.
//!
//! A project is just a directory with a name. Its checkpoints live inside it
//! under `.saient/`, so a project is self-contained — copy it, move it, or
//! delete it, and its history travels with it.

use anyhow::{bail, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// Per-project Saient data (checkpoints and friends). Hidden so it stays out of
/// the user's way, and skipped when snapshotting so checkpoints never contain
/// previous checkpoints.
pub const PROJECT_DATA_DIR: &str = ".saient";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectInfo {
    pub name: String,
    pub path: String,
    /// Seconds since the epoch, newest activity first in listings.
    pub modified: u64,
    /// Files at the top level, so the picker can show "empty" vs "has work in it".
    pub entry_count: usize,
    /// How much of Saient runs behind the agent here: off | guided | companion |
    /// autonomous. Stored per project because it changes what the agent may do,
    /// and someone may well want one project autonomous and another not.
    pub agi_level: String,
}

/// Per-project settings, kept beside its checkpoints.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct ProjectSettings {
    #[serde(default)]
    agi_level: String,
}

fn settings_file(project_path: &Path) -> PathBuf {
    project_path.join(PROJECT_DATA_DIR).join("project.json")
}

fn read_level(project_path: &Path) -> String {
    std::fs::read_to_string(settings_file(project_path))
        .ok()
        .and_then(|raw| serde_json::from_str::<ProjectSettings>(&raw).ok())
        .map(|s| s.agi_level)
        .filter(|l| !l.is_empty())
        // Anything missing or unrecognised means off. A project should never end
        // up acting on its own because a settings file failed to parse.
        .unwrap_or_else(|| "off".to_string())
}

pub fn set_level(name: &str, level: &str) -> Result<ProjectInfo> {
    const VALID: &[&str] = &["off", "guided", "companion", "autonomous"];
    if !VALID.contains(&level) {
        bail!("Unknown AGI level: {level}");
    }
    let path = path_for(name)?;
    std::fs::create_dir_all(path.join(PROJECT_DATA_DIR))?;
    let json = serde_json::to_string_pretty(&ProjectSettings { agi_level: level.to_string() })?;
    std::fs::write(settings_file(&path), json)?;
    Ok(describe(name, &path))
}

pub fn projects_dir() -> PathBuf {
    crate::paths::data_dir().join("projects")
}

fn active_pref_file() -> PathBuf {
    crate::paths::config_dir().join("active_project.txt")
}

/// Validate a project name.
///
/// Names become directory names, so anything that could climb out of the
/// projects directory is refused rather than sanitised — silently renaming
/// someone's project is worse than telling them the name won't do.
pub fn validate_name(name: &str) -> Result<String> {
    let trimmed = name.trim();
    if trimmed.is_empty() {
        bail!("Project name cannot be empty");
    }
    if trimmed.len() > 64 {
        bail!("Project name is too long (max 64 characters)");
    }
    if trimmed.starts_with('.') {
        bail!("Project name cannot start with a dot");
    }
    if trimmed.contains('/') || trimmed.contains('\\') || trimmed.contains("..") {
        bail!("Project name cannot contain slashes or '..'");
    }
    // Reserved on Windows; refuse everywhere so projects stay portable.
    const RESERVED: &[&str] = &[
        "con", "prn", "aux", "nul", "com1", "com2", "com3", "com4",
        "lpt1", "lpt2", "lpt3", "lpt4",
    ];
    if RESERVED.contains(&trimmed.to_ascii_lowercase().as_str()) {
        bail!("'{trimmed}' is a reserved name on some systems");
    }
    if trimmed.chars().any(|c| matches!(c, '<' | '>' | ':' | '"' | '|' | '?' | '*') || c.is_control()) {
        bail!("Project name contains characters that are not allowed in a folder name");
    }
    Ok(trimmed.to_string())
}

pub fn list() -> Result<Vec<ProjectInfo>> {
    let root = projects_dir();
    if !root.is_dir() {
        return Ok(Vec::new());
    }
    let mut out = Vec::new();
    for entry in std::fs::read_dir(&root)? {
        let entry = entry?;
        if !entry.file_type()?.is_dir() {
            continue;
        }
        let name = entry.file_name().to_string_lossy().into_owned();
        if name.starts_with('.') {
            continue;
        }
        out.push(describe(&name, &entry.path()));
    }
    out.sort_by(|a, b| b.modified.cmp(&a.modified));
    Ok(out)
}

fn describe(name: &str, path: &Path) -> ProjectInfo {
    let modified = std::fs::metadata(path)
        .and_then(|m| m.modified())
        .ok()
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|d| d.as_secs())
        .unwrap_or(0);

    // Top-level entries only, ignoring our own data directory.
    let entry_count = std::fs::read_dir(path)
        .map(|rd| {
            rd.filter_map(|e| e.ok())
                .filter(|e| e.file_name().to_string_lossy() != PROJECT_DATA_DIR)
                .count()
        })
        .unwrap_or(0);

    ProjectInfo {
        name: name.to_string(),
        path: path.to_string_lossy().into_owned(),
        modified,
        entry_count,
        agi_level: read_level(path),
    }
}

/// Create a project directory. Fails if one of that name already exists, rather
/// than opening it — "New project" that silently opens an old one loses work.
pub fn create(name: &str) -> Result<ProjectInfo> {
    let name = validate_name(name)?;
    let path = projects_dir().join(&name);
    if path.exists() {
        bail!("A project called '{name}' already exists");
    }
    std::fs::create_dir_all(path.join(PROJECT_DATA_DIR))?;
    Ok(describe(&name, &path))
}

/// The directory for a project that must already exist.
pub fn path_for(name: &str) -> Result<PathBuf> {
    let name = validate_name(name)?;
    let path = projects_dir().join(&name);
    if !path.is_dir() {
        bail!("No project called '{name}'");
    }
    Ok(path)
}

pub fn set_active(name: &str) -> Result<ProjectInfo> {
    let path = path_for(name)?;
    let pref = active_pref_file();
    if let Some(parent) = pref.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&pref, name)?;
    Ok(describe(name, &path))
}

/// Leave managed-project mode without deleting any project or its history.
pub fn clear_active() -> Result<()> {
    let pref = active_pref_file();
    if let Some(parent) = pref.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(pref, "")?;
    Ok(())
}

/// The remembered project, or None on first run or if it has been deleted.
pub fn active() -> Option<ProjectInfo> {
    let name = std::fs::read_to_string(active_pref_file()).ok()?;
    let name = name.trim();
    let path = path_for(name).ok()?;
    Some(describe(name, &path))
}

/// Where a project keeps its checkpoints.
pub fn checkpoint_dir(project_path: &Path) -> PathBuf {
    project_path.join(PROJECT_DATA_DIR).join("checkpoints")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_names_that_could_climb_out() {
        for bad in ["../escape", "a/b", "a\\b", "..", "../../etc"] {
            assert!(validate_name(bad).is_err(), "{bad} should be refused");
        }
    }

    #[test]
    fn rejects_empty_hidden_and_overlong_names() {
        assert!(validate_name("").is_err());
        assert!(validate_name("   ").is_err());
        assert!(validate_name(".hidden").is_err());
        assert!(validate_name(&"x".repeat(65)).is_err());
    }

    #[test]
    fn rejects_characters_no_filesystem_will_take() {
        for bad in ["a:b", "a?b", "a*b", "a|b", "a\"b", "a<b", "a>b"] {
            assert!(validate_name(bad).is_err(), "{bad} should be refused");
        }
    }

    #[test]
    fn rejects_windows_reserved_names_everywhere() {
        // Refused on Linux too, so a project stays portable.
        for bad in ["CON", "con", "nul", "COM1"] {
            assert!(validate_name(bad).is_err(), "{bad} should be refused");
        }
    }

    #[test]
    fn accepts_ordinary_names_and_trims() {
        assert_eq!(validate_name("snake game").unwrap(), "snake game");
        assert_eq!(validate_name("  my-project  ").unwrap(), "my-project");
        assert_eq!(validate_name("fib_v2").unwrap(), "fib_v2");
    }

    #[test]
    fn checkpoints_live_inside_the_project() {
        let p = Path::new("/tmp/projects/demo");
        assert_eq!(
            checkpoint_dir(p),
            Path::new("/tmp/projects/demo/.saient/checkpoints")
        );
    }
}
