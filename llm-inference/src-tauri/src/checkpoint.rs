//! Checkpoints — the saved state of a working session.
//!
//! Saving the conversation alone gives Saient memories but no shoes: it can
//! recall what was said and nothing about where it was. A checkpoint therefore
//! captures the *working* state too — the active goal, which step was in flight,
//! the terminal's working directory, what remained outstanding, and the contents
//! of the workspace itself.
//!
//! Storage is content-addressed, git-style: file bodies live once in `objects/`
//! keyed by their SHA-256, and a checkpoint is a manifest of path → hash. Two
//! checkpoints taken either side of a one-line edit therefore cost one extra
//! object, not two full copies. That matters because the Flight Recorder wants
//! many checkpoints, and it is what makes "restore" and "branch from here"
//! cheap enough to be worth offering.
//!
//! Restore is the only destructive operation here, so it takes a safety
//! checkpoint of the current state first — restoring is itself undoable.

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

/// Directories never worth snapshotting: regenerable, and large enough to make
/// checkpointing feel expensive if included.
const IGNORED_DIRS: &[&str] = &[
    ".git", "node_modules", "target", "__pycache__", ".venv", "venv",
    "dist", "build", ".next", ".cache", ".mypy_cache", ".pytest_cache",
];

/// Skip individual files above this size. A checkpoint is for source and notes,
/// not for model weights that happen to be sitting in the workspace.
const MAX_FILE_BYTES: u64 = 2 * 1024 * 1024;

/// Refuse to snapshot a workspace larger than this in total, rather than
/// silently taking minutes and filling the disk.
const MAX_TOTAL_BYTES: u64 = 256 * 1024 * 1024;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum CheckpointKind {
    /// The user asked for it.
    Manual,
    /// Auto-save after each completed turn.
    AutoTurn,
    /// Auto-save after a task reaches a terminal state.
    AutoTask,
    /// Taken automatically just before a restore, so the restore can be undone.
    PreRestore,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointMeta {
    pub id: String,
    pub name: String,
    pub created_at: u64,
    pub kind: CheckpointKind,
    /// The checkpoint this one followed, so history can be walked and branched.
    pub parent: Option<String>,

    // ── The working state, not just the words ──
    pub goal: String,
    pub turn_state: String,
    pub terminal_cwd: String,
    /// Which step was in flight, 1-based for display.
    pub step_index: Option<usize>,
    pub step_total: Option<usize>,
    /// Steps not yet done — what Saient still owes the user.
    pub outstanding: Vec<String>,

    pub file_count: usize,
    pub total_bytes: u64,
}

/// The full record. `conversation` and `plan` are opaque JSON: the frontend owns
/// those shapes, and this module has no business knowing them.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Checkpoint {
    pub meta: CheckpointMeta,
    /// Workspace-relative path → content hash in `objects/`.
    pub files: BTreeMap<String, String>,
    pub conversation: serde_json::Value,
    pub plan: serde_json::Value,
    pub terminal: Vec<String>,
}

/// What a restore actually did, so the result can be reported rather than assumed.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RestoreReport {
    pub restored: Vec<String>,
    pub unchanged: Vec<String>,
    /// Files present now but absent from the checkpoint. Left alone rather than
    /// deleted — a restore that silently removes new work is a data-loss bug.
    pub left_in_place: Vec<String>,
    /// The safety checkpoint taken before restoring.
    pub undo_checkpoint: String,
}

pub struct CheckpointStore {
    root: PathBuf,
}

impl CheckpointStore {
    pub fn new(root: PathBuf) -> Self {
        Self { root }
    }

    /// Default location under the app's data dir.
    pub fn default_store() -> Self {
        Self::new(crate::paths::data_dir().join("checkpoints"))
    }

    fn objects_dir(&self) -> PathBuf {
        self.root.join("objects")
    }
    fn meta_dir(&self) -> PathBuf {
        self.root.join("checkpoints")
    }

    fn ensure_dirs(&self) -> Result<()> {
        std::fs::create_dir_all(self.objects_dir())?;
        std::fs::create_dir_all(self.meta_dir())?;
        Ok(())
    }

    /// Snapshot `workspace` plus the supplied session state.
    pub fn create(
        &self,
        workspace: &Path,
        name: &str,
        kind: CheckpointKind,
        parent: Option<String>,
        session: SessionState,
    ) -> Result<CheckpointMeta> {
        self.ensure_dirs()?;

        let (files, total_bytes) = self.snapshot_files(workspace)?;

        let id = new_id();
        let meta = CheckpointMeta {
            id: id.clone(),
            name: if name.trim().is_empty() {
                default_name(&session.goal, kind_label(&kind))
            } else {
                name.trim().to_string()
            },
            created_at: now_secs(),
            kind,
            parent,
            goal: session.goal,
            turn_state: session.turn_state,
            terminal_cwd: session.terminal_cwd,
            step_index: session.step_index,
            step_total: session.step_total,
            outstanding: session.outstanding,
            file_count: files.len(),
            total_bytes,
        };

        let checkpoint = Checkpoint {
            meta: meta.clone(),
            files,
            conversation: session.conversation,
            plan: session.plan,
            terminal: session.terminal,
        };

        let path = self.meta_dir().join(format!("{id}.json"));
        let json = serde_json::to_string_pretty(&checkpoint)?;
        write_atomic(&path, json.as_bytes())
            .with_context(|| format!("writing checkpoint {id}"))?;

        Ok(meta)
    }

    /// Walk the workspace, storing each file's body once by hash.
    fn snapshot_files(&self, workspace: &Path) -> Result<(BTreeMap<String, String>, u64)> {
        if !workspace.is_dir() {
            bail!("workspace does not exist: {}", workspace.display());
        }
        let mut files = BTreeMap::new();
        let mut total: u64 = 0;

        for entry in walkdir::WalkDir::new(workspace)
            .follow_links(false)
            .into_iter()
            .filter_entry(|e| !is_ignored(e.path()))
        {
            let entry = entry?;
            if !entry.file_type().is_file() {
                continue;
            }
            let meta = entry.metadata()?;
            if meta.len() > MAX_FILE_BYTES {
                continue;
            }
            total += meta.len();
            if total > MAX_TOTAL_BYTES {
                bail!(
                    "workspace exceeds the {} MiB checkpoint limit — exclude large \
                     directories or check the workspace root is right",
                    MAX_TOTAL_BYTES / 1024 / 1024
                );
            }

            let rel = entry
                .path()
                .strip_prefix(workspace)?
                .to_string_lossy()
                .replace('\\', "/");
            let bytes = std::fs::read(entry.path())?;
            let hash = self.put_object(&bytes)?;
            files.insert(rel, hash);
        }
        Ok((files, total))
    }

    /// Store bytes under their hash. Already-present objects are left alone,
    /// which is what makes repeated checkpoints cheap.
    fn put_object(&self, bytes: &[u8]) -> Result<String> {
        let hash = hex(&Sha256::digest(bytes));
        let path = self.objects_dir().join(&hash);
        if !path.exists() {
            write_atomic(&path, bytes)?;
        }
        Ok(hash)
    }

    pub fn load(&self, id: &str) -> Result<Checkpoint> {
        let path = self.meta_dir().join(format!("{}.json", safe_id(id)?));
        let raw = std::fs::read_to_string(&path)
            .with_context(|| format!("no such checkpoint: {id}"))?;
        Ok(serde_json::from_str(&raw)?)
    }

    /// Newest first.
    pub fn list(&self) -> Result<Vec<CheckpointMeta>> {
        let dir = self.meta_dir();
        if !dir.is_dir() {
            return Ok(Vec::new());
        }
        let mut out = Vec::new();
        for entry in std::fs::read_dir(&dir)? {
            let path = entry?.path();
            if path.extension().and_then(|e| e.to_str()) != Some("json") {
                continue;
            }
            // A single corrupt file must not hide every other checkpoint.
            if let Ok(raw) = std::fs::read_to_string(&path) {
                if let Ok(cp) = serde_json::from_str::<Checkpoint>(&raw) {
                    out.push(cp.meta);
                }
            }
        }
        out.sort_by(|a, b| b.created_at.cmp(&a.created_at));
        Ok(out)
    }

    pub fn delete(&self, id: &str) -> Result<()> {
        let path = self.meta_dir().join(format!("{}.json", safe_id(id)?));
        std::fs::remove_file(path)?;
        // Objects are left behind deliberately: they may be shared with other
        // checkpoints. Reclaiming them needs a sweep across all manifests.
        Ok(())
    }

    /// Write a checkpoint's files back into `workspace`.
    ///
    /// Takes a safety checkpoint first, so this is reversible. Files present now
    /// but absent from the checkpoint are reported and left alone — deleting
    /// them would turn a restore into silent data loss.
    pub fn restore(
        &self,
        id: &str,
        workspace: &Path,
        session: SessionState,
    ) -> Result<RestoreReport> {
        let cp = self.load(id)?;

        let undo = self.create(
            workspace,
            &format!("Before restoring “{}”", cp.meta.name),
            CheckpointKind::PreRestore,
            Some(cp.meta.id.clone()),
            session,
        )?;

        let mut report = RestoreReport {
            restored: Vec::new(),
            unchanged: Vec::new(),
            left_in_place: Vec::new(),
            undo_checkpoint: undo.id,
        };

        for (rel, hash) in &cp.files {
            let target = resolve_inside(workspace, rel)?;
            let bytes = std::fs::read(self.objects_dir().join(hash))
                .with_context(|| format!("missing object {hash} for {rel}"))?;

            if let Ok(current) = std::fs::read(&target) {
                if current == bytes {
                    report.unchanged.push(rel.clone());
                    continue;
                }
            }
            if let Some(parent) = target.parent() {
                std::fs::create_dir_all(parent)?;
            }
            write_atomic(&target, &bytes)?;
            report.restored.push(rel.clone());
        }

        // Anything in the workspace the checkpoint never knew about.
        let (present, _) = self.snapshot_files(workspace)?;
        for rel in present.keys() {
            if !cp.files.contains_key(rel) {
                report.left_in_place.push(rel.clone());
            }
        }

        Ok(report)
    }

    /// Human-readable session export.
    pub fn export_markdown(&self, id: &str) -> Result<String> {
        let cp = self.load(id)?;
        let m = &cp.meta;
        let mut s = String::new();
        s.push_str(&format!("# {}\n\n", m.name));
        s.push_str(&format!("- **Saved:** {}\n", fmt_time(m.created_at)));
        s.push_str(&format!("- **Goal:** {}\n", if m.goal.is_empty() { "—" } else { &m.goal }));
        s.push_str(&format!("- **State:** {}\n", m.turn_state));
        if let (Some(i), Some(t)) = (m.step_index, m.step_total) {
            s.push_str(&format!("- **Step:** {i} of {t}\n"));
        }
        s.push_str(&format!("- **Working directory:** {}\n", m.terminal_cwd));
        s.push_str(&format!("- **Files captured:** {} ({} KiB)\n", m.file_count, m.total_bytes / 1024));

        if !m.outstanding.is_empty() {
            s.push_str("\n## Outstanding\n\n");
            for item in &m.outstanding {
                s.push_str(&format!("- [ ] {item}\n"));
            }
        }

        if let Some(msgs) = cp.conversation.as_array() {
            s.push_str("\n## Conversation\n\n");
            for msg in msgs {
                let role = msg.get("role").and_then(|v| v.as_str()).unwrap_or("?");
                let content = msg.get("content").and_then(|v| v.as_str()).unwrap_or("");
                s.push_str(&format!("### {role}\n\n{content}\n\n"));
            }
        }

        if !cp.terminal.is_empty() {
            s.push_str("\n## Terminal\n\n```\n");
            for line in &cp.terminal {
                s.push_str(line);
                s.push('\n');
            }
            s.push_str("```\n");
        }

        if !cp.files.is_empty() {
            s.push_str("\n## Files\n\n");
            for path in cp.files.keys() {
                s.push_str(&format!("- `{path}`\n"));
            }
        }
        Ok(s)
    }

    pub fn export_json(&self, id: &str) -> Result<String> {
        Ok(serde_json::to_string_pretty(&self.load(id)?)?)
    }
}

/// Everything the frontend hands over at save time.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct SessionState {
    pub goal: String,
    pub turn_state: String,
    pub terminal_cwd: String,
    pub step_index: Option<usize>,
    pub step_total: Option<usize>,
    pub outstanding: Vec<String>,
    pub conversation: serde_json::Value,
    pub plan: serde_json::Value,
    pub terminal: Vec<String>,
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn is_ignored(path: &Path) -> bool {
    path.file_name()
        .and_then(|n| n.to_str())
        .map(|n| IGNORED_DIRS.contains(&n))
        .unwrap_or(false)
}

/// Join `rel` under `root`, refusing anything that escapes it.
///
/// A checkpoint file is data, and data from disk can carry `../` — restoring it
/// blindly would write outside the workspace.
fn resolve_inside(root: &Path, rel: &str) -> Result<PathBuf> {
    // Refuse an absolute path outright rather than reinterpreting it. Splitting
    // "/etc/passwd" on '/' yields an empty leading component, and skipping that
    // would quietly turn it into <workspace>/etc/passwd — inside the workspace,
    // so not an escape, but not what the manifest said either. A manifest that
    // disagrees with what gets written is exactly what this function exists to
    // prevent.
    if rel.starts_with('/') || rel.starts_with('\\') || Path::new(rel).is_absolute() {
        bail!("checkpoint path must be workspace-relative: {rel}");
    }

    let mut out = root.to_path_buf();
    for part in rel.split('/') {
        match part {
            "" | "." => continue,
            ".." => bail!("checkpoint path escapes the workspace: {rel}"),
            p => {
                if p.contains('\\') || Path::new(p).is_absolute() {
                    bail!("unsafe checkpoint path: {rel}");
                }
                out.push(p);
            }
        }
    }
    if !out.starts_with(root) {
        bail!("checkpoint path escapes the workspace: {rel}");
    }
    Ok(out)
}

/// Ids are used as filenames, so they must not carry path separators.
fn safe_id(id: &str) -> Result<String> {
    if id.is_empty() || !id.chars().all(|c| c.is_ascii_alphanumeric() || c == '-') {
        bail!("invalid checkpoint id: {id}");
    }
    Ok(id.to_string())
}

fn write_atomic(path: &Path, bytes: &[u8]) -> Result<()> {
    let tmp = path.with_extension("tmp-write");
    std::fs::write(&tmp, bytes)?;
    std::fs::rename(&tmp, path)?;
    Ok(())
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn new_id() -> String {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    format!("cp-{nanos:x}")
}

fn now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn kind_label(kind: &CheckpointKind) -> &'static str {
    match kind {
        CheckpointKind::Manual => "Manual save",
        CheckpointKind::AutoTurn => "Auto-save",
        CheckpointKind::AutoTask => "Task complete",
        CheckpointKind::PreRestore => "Before restore",
    }
}

/// A name someone can recognise later, from the goal when there is one.
fn default_name(goal: &str, fallback: &str) -> String {
    let g = goal.trim();
    if g.is_empty() {
        return fallback.to_string();
    }
    let short: String = g.chars().take(60).collect();
    if g.chars().count() > 60 { format!("{short}…") } else { short }
}

fn fmt_time(secs: u64) -> String {
    // Local formatting without pulling in a date crate: the exact wall-clock
    // rendering belongs to the frontend, which has one.
    format!("{secs} (unix seconds)")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn store() -> (CheckpointStore, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        (CheckpointStore::new(dir.path().join("store")), dir)
    }

    fn workspace(dir: &Path, files: &[(&str, &str)]) -> PathBuf {
        let ws = dir.join("ws");
        for (rel, body) in files {
            let p = ws.join(rel);
            std::fs::create_dir_all(p.parent().unwrap()).unwrap();
            std::fs::write(p, body).unwrap();
        }
        ws
    }

    fn session(goal: &str) -> SessionState {
        SessionState {
            goal: goal.into(),
            turn_state: "COMPLETED".into(),
            terminal_cwd: "/workspace".into(),
            step_index: Some(2),
            step_total: Some(3),
            outstanding: vec!["run the tests".into()],
            conversation: serde_json::json!([{"role": "user", "content": "hi"}]),
            plan: serde_json::json!({"steps": []}),
            terminal: vec!["$ cargo test".into()],
        }
    }

    #[test]
    fn captures_working_state_not_just_conversation() {
        let (s, dir) = store();
        let ws = workspace(dir.path(), &[("a.txt", "one")]);
        let meta = s.create(&ws, "", CheckpointKind::Manual, None, session("build it")).unwrap();

        // The "shoes": where it was, not only what was said.
        assert_eq!(meta.goal, "build it");
        assert_eq!(meta.terminal_cwd, "/workspace");
        assert_eq!(meta.step_index, Some(2));
        assert_eq!(meta.step_total, Some(3));
        assert_eq!(meta.outstanding, vec!["run the tests".to_string()]);
        assert_eq!(meta.file_count, 1);

        let cp = s.load(&meta.id).unwrap();
        assert_eq!(cp.terminal, vec!["$ cargo test".to_string()]);
        assert_eq!(cp.conversation[0]["content"], "hi");
    }

    #[test]
    fn round_trips_file_contents() {
        let (s, dir) = store();
        let ws = workspace(dir.path(), &[("src/a.rs", "fn main() {}"), ("notes.md", "# hi")]);
        let meta = s.create(&ws, "cp", CheckpointKind::Manual, None, session("g")).unwrap();

        std::fs::write(ws.join("src/a.rs"), "BROKEN").unwrap();
        let report = s.restore(&meta.id, &ws, session("g")).unwrap();

        assert_eq!(std::fs::read_to_string(ws.join("src/a.rs")).unwrap(), "fn main() {}");
        assert_eq!(report.restored, vec!["src/a.rs".to_string()]);
        assert_eq!(report.unchanged, vec!["notes.md".to_string()]);
    }

    #[test]
    fn identical_content_is_stored_once() {
        let (s, dir) = store();
        let ws = workspace(dir.path(), &[("a.txt", "same"), ("b.txt", "same")]);
        s.create(&ws, "cp", CheckpointKind::Manual, None, session("g")).unwrap();
        let objects = std::fs::read_dir(s.objects_dir()).unwrap().count();
        assert_eq!(objects, 1, "duplicate bodies should share one object");
    }

    #[test]
    fn repeat_checkpoints_only_add_what_changed() {
        let (s, dir) = store();
        let ws = workspace(dir.path(), &[("a.txt", "one"), ("b.txt", "two")]);
        s.create(&ws, "first", CheckpointKind::Manual, None, session("g")).unwrap();
        let before = std::fs::read_dir(s.objects_dir()).unwrap().count();

        std::fs::write(ws.join("a.txt"), "one changed").unwrap();
        s.create(&ws, "second", CheckpointKind::Manual, None, session("g")).unwrap();
        let after = std::fs::read_dir(s.objects_dir()).unwrap().count();

        assert_eq!(after, before + 1, "only the changed body should be new");
    }

    #[test]
    fn restore_is_itself_undoable() {
        let (s, dir) = store();
        let ws = workspace(dir.path(), &[("a.txt", "original")]);
        let first = s.create(&ws, "first", CheckpointKind::Manual, None, session("g")).unwrap();

        std::fs::write(ws.join("a.txt"), "work in progress").unwrap();
        let report = s.restore(&first.id, &ws, session("g")).unwrap();
        assert_eq!(std::fs::read_to_string(ws.join("a.txt")).unwrap(), "original");

        // The in-progress work is recoverable from the safety checkpoint.
        s.restore(&report.undo_checkpoint, &ws, session("g")).unwrap();
        assert_eq!(std::fs::read_to_string(ws.join("a.txt")).unwrap(), "work in progress");
    }

    #[test]
    fn restore_never_deletes_newer_files() {
        let (s, dir) = store();
        let ws = workspace(dir.path(), &[("a.txt", "one")]);
        let cp = s.create(&ws, "cp", CheckpointKind::Manual, None, session("g")).unwrap();

        std::fs::write(ws.join("new.txt"), "written later").unwrap();
        let report = s.restore(&cp.id, &ws, session("g")).unwrap();

        assert!(ws.join("new.txt").exists(), "restore must not delete unknown files");
        assert!(report.left_in_place.contains(&"new.txt".to_string()));
    }

    #[test]
    fn regenerable_directories_are_skipped() {
        let (s, dir) = store();
        let ws = workspace(
            dir.path(),
            &[("src/a.rs", "x"), ("node_modules/dep/i.js", "y"), ("target/debug/bin", "z")],
        );
        let meta = s.create(&ws, "cp", CheckpointKind::Manual, None, session("g")).unwrap();
        let cp = s.load(&meta.id).unwrap();
        assert_eq!(cp.files.keys().collect::<Vec<_>>(), vec!["src/a.rs"]);
    }

    #[test]
    fn a_traversing_path_cannot_escape_the_workspace() {
        let root = Path::new("/tmp/ws");
        assert!(resolve_inside(root, "../../etc/passwd").is_err());
        assert!(resolve_inside(root, "a/../../b").is_err());
        assert!(resolve_inside(root, "/etc/passwd").is_err());
        assert!(resolve_inside(root, "src/ok.rs").is_ok());
    }

    #[test]
    fn ids_used_as_filenames_are_validated() {
        assert!(safe_id("../../etc/passwd").is_err());
        assert!(safe_id("cp-1a2b").is_ok());
        assert!(safe_id("").is_err());
    }

    #[test]
    fn listing_is_newest_first_and_survives_a_corrupt_file() {
        let (s, dir) = store();
        let ws = workspace(dir.path(), &[("a.txt", "x")]);
        let a = s.create(&ws, "older", CheckpointKind::Manual, None, session("g")).unwrap();
        std::thread::sleep(std::time::Duration::from_millis(1100));
        let b = s.create(&ws, "newer", CheckpointKind::Manual, None, session("g")).unwrap();

        std::fs::write(s.meta_dir().join("cp-garbage.json"), "{{{ not json").unwrap();

        let list = s.list().unwrap();
        assert_eq!(list.len(), 2, "one corrupt file must not hide the rest");
        assert_eq!(list[0].id, b.id);
        assert_eq!(list[1].id, a.id);
    }

    #[test]
    fn name_defaults_to_the_goal_when_none_given() {
        let (s, dir) = store();
        let ws = workspace(dir.path(), &[("a.txt", "x")]);
        let m = s.create(&ws, "", CheckpointKind::Manual, None, session("refactor the scheduler")).unwrap();
        assert_eq!(m.name, "refactor the scheduler");

        let m2 = s.create(&ws, "", CheckpointKind::AutoTurn, None, SessionState::default()).unwrap();
        assert_eq!(m2.name, "Auto-save", "no goal falls back to the kind");
    }

    #[test]
    fn markdown_export_includes_the_working_state() {
        let (s, dir) = store();
        let ws = workspace(dir.path(), &[("a.txt", "x")]);
        let m = s.create(&ws, "Before Vulkan changes", CheckpointKind::Manual, None, session("ship it")).unwrap();
        let md = s.export_markdown(&m.id).unwrap();

        assert!(md.contains("# Before Vulkan changes"));
        assert!(md.contains("ship it"));
        assert!(md.contains("Step:** 2 of 3"));
        assert!(md.contains("- [ ] run the tests"));
        assert!(md.contains("$ cargo test"));
        assert!(md.contains("`a.txt`"));
    }

    #[test]
    fn json_export_round_trips() {
        let (s, dir) = store();
        let ws = workspace(dir.path(), &[("a.txt", "x")]);
        let m = s.create(&ws, "cp", CheckpointKind::Manual, None, session("g")).unwrap();
        let parsed: Checkpoint = serde_json::from_str(&s.export_json(&m.id).unwrap()).unwrap();
        assert_eq!(parsed.meta.id, m.id);
    }

    #[test]
    fn parent_links_allow_walking_history() {
        let (s, dir) = store();
        let ws = workspace(dir.path(), &[("a.txt", "x")]);
        let first = s.create(&ws, "one", CheckpointKind::Manual, None, session("g")).unwrap();
        let second = s
            .create(&ws, "two", CheckpointKind::Manual, Some(first.id.clone()), session("g"))
            .unwrap();
        assert_eq!(second.parent.as_deref(), Some(first.id.as_str()));
    }
}
