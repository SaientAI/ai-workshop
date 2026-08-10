/// patch.rs — Patch and diff engine
/// Generate unified diffs between file versions.
/// Apply patches with preview and rollback support.
/// Maintains a change history per file for undo.

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use similar::{ChangeTag, TextDiff};
use std::collections::HashMap;
use std::path::PathBuf;

// ── Types ─────────────────────────────────────────────────────────────────────

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct DiffResult {
    pub path: String,
    pub unified: String,        // standard unified diff output
    pub hunks: Vec<DiffHunk>,
    pub additions: usize,
    pub deletions: usize,
    pub unchanged: usize,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct DiffHunk {
    pub old_start: usize,
    pub new_start: usize,
    pub lines: Vec<DiffLine>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct DiffLine {
    pub tag: String,    // "add" | "del" | "eq"
    pub content: String,
    pub old_lineno: Option<usize>,
    pub new_lineno: Option<usize>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct PatchResult {
    pub path: String,
    pub applied: bool,
    pub conflicts: Vec<String>,
    pub backup_key: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct HistoryEntry {
    pub timestamp: String,
    pub description: String,
    pub old_content: String,
    pub new_content: String,
    pub diff_lines: usize,
}

// ── PatchEngine ───────────────────────────────────────────────────────────────

use super::sandbox::normalize_path;

pub struct PatchEngine {
    /// path → stack of history entries
    history: HashMap<String, Vec<HistoryEntry>>,
    pub root: PathBuf,
}

impl PatchEngine {
    pub fn new(root: PathBuf) -> Self {
        Self { history: HashMap::new(), root }
    }

    pub fn set_root(&mut self, root: PathBuf) {
        self.root = root;
    }

    /// Generate a unified diff between old and new text.
    pub fn diff_text(&self, path: &str, old: &str, new: &str) -> DiffResult {
        let diff = TextDiff::from_lines(old, new);
        let mut unified = String::new();
        let mut hunks: Vec<DiffHunk> = vec![];
        let mut additions = 0usize;
        let mut deletions = 0usize;
        let mut unchanged = 0usize;

        // Build unified header
        unified.push_str(&format!("--- a/{}\n+++ b/{}\n", path, path));

        for group in diff.grouped_ops(3) {
            let mut hunk_lines = vec![];
            let first = &group[0];
            let _last = &group[group.len() - 1];
            let old_start = first.old_range().start + 1;
            let new_start = first.new_range().start + 1;
            let old_len: usize = group.iter().map(|op| op.old_range().len()).sum();
            let new_len: usize = group.iter().map(|op| op.new_range().len()).sum();

            unified.push_str(&format!("@@ -{},{} +{},{} @@\n", old_start, old_len, new_start, new_len));

            let mut old_lineno = old_start;
            let mut new_lineno = new_start;

            for op in &group {
                for change in diff.iter_changes(op) {
                    let (tag_str, prefix) = match change.tag() {
                        ChangeTag::Insert => ("add", "+"),
                        ChangeTag::Delete => ("del", "-"),
                        ChangeTag::Equal  => ("eq",  " "),
                    };
                    let content = change.value().trim_end_matches('\n').to_string();
                    unified.push_str(&format!("{}{}\n", prefix, content));

                    let (old_ln, new_ln) = match change.tag() {
                        ChangeTag::Delete => { let o = Some(old_lineno); old_lineno += 1; deletions += 1; (o, None) }
                        ChangeTag::Insert => { let n = Some(new_lineno); new_lineno += 1; additions += 1; (None, n) }
                        ChangeTag::Equal  => {
                            unchanged += 1;
                            let o = Some(old_lineno); old_lineno += 1;
                            let n = Some(new_lineno); new_lineno += 1;
                            (o, n)
                        }
                    };
                    hunk_lines.push(DiffLine { tag: tag_str.into(), content, old_lineno: old_ln, new_lineno: new_ln });
                }
            }
            hunks.push(DiffHunk { old_start, new_start, lines: hunk_lines });
        }

        DiffResult { path: path.into(), unified, hunks, additions, deletions, unchanged }
    }

    /// Diff two files on disk.
    pub fn diff_files(&self, path_a: &str, path_b: &str) -> Result<DiffResult> {
        let a = std::fs::read_to_string(self.root.join(path_a))
            .with_context(|| format!("read {}", path_a))?;
        let b = std::fs::read_to_string(self.root.join(path_b))
            .with_context(|| format!("read {}", path_b))?;
        Ok(self.diff_text(path_a, &a, &b))
    }

    /// Diff current file content against proposed new content.
    pub fn diff_proposed(&self, path: &str, new_content: &str) -> Result<DiffResult> {
        let p = self.root.join(path);
        let old = if p.exists() {
            std::fs::read_to_string(&p).unwrap_or_default()
        } else {
            String::new()
        };
        Ok(self.diff_text(path, &old, new_content))
    }

    /// Apply new content to a file, saving history for undo.
    pub fn apply(&mut self, path: &str, new_content: &str, description: &str) -> Result<PatchResult> {
        let p = normalize_path(&self.root.join(path));
        if !p.starts_with(&self.root) {
            bail!("Path '{}' escapes workspace root", path);
        }
        if let Some(parent) = p.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let old_content = if p.exists() {
            std::fs::read_to_string(&p).unwrap_or_default()
        } else {
            String::new()
        };

        let diff = self.diff_text(path, &old_content, new_content);
        let diff_lines = diff.additions + diff.deletions;

        // Save history
        let entry = HistoryEntry {
            timestamp: now_str(),
            description: description.into(),
            old_content,
            new_content: new_content.into(),
            diff_lines,
        };
        self.history.entry(path.into()).or_default().push(entry);

        // Write
        std::fs::write(&p, new_content)?;

        Ok(PatchResult {
            path: path.into(),
            applied: true,
            conflicts: vec![],
            backup_key: Some(format!("{}:{}", path, self.history[path].len() - 1)),
        })
    }

    /// Apply a unified diff string to a file.
    pub fn apply_unified_diff(&mut self, path: &str, unified_diff: &str) -> Result<PatchResult> {
        let p = self.root.join(path);
        let old = if p.exists() {
            std::fs::read_to_string(&p).unwrap_or_default()
        } else {
            String::new()
        };

        // Parse and apply the unified diff manually
        let new_content = apply_unified_patch(&old, unified_diff)?;
        self.apply(path, &new_content, &format!("apply unified diff ({} chars)", unified_diff.len()))
    }

    /// Undo the last change to a file.
    pub fn undo(&mut self, path: &str) -> Result<PatchResult> {
        let stack = self.history.get_mut(path)
            .ok_or_else(|| anyhow::anyhow!("No history for {}", path))?;

        let entry = stack.pop()
            .ok_or_else(|| anyhow::anyhow!("Nothing to undo for {}", path))?;

        let p = self.root.join(path);
        std::fs::write(&p, &entry.old_content)?;

        Ok(PatchResult {
            path: path.into(),
            applied: true,
            conflicts: vec![],
            backup_key: None,
        })
    }

    /// Get history for a file.
    pub fn history(&self, path: &str) -> Vec<HistoryEntry> {
        self.history.get(path).cloned().unwrap_or_default()
    }

    /// Show a preview diff without applying.
    pub fn preview(&self, path: &str, new_content: &str) -> Result<DiffResult> {
        self.diff_proposed(path, new_content)
    }
}

// ── Unified diff application ──────────────────────────────────────────────────

fn apply_unified_patch(original: &str, patch: &str) -> Result<String> {
    let orig_lines: Vec<&str> = original.lines().collect();
    let mut result_lines: Vec<String> = vec![];
    let mut orig_pos = 0usize; // 0-indexed

    let patch_lines: Vec<&str> = patch.lines().collect();
    let mut i = 0usize;

    while i < patch_lines.len() {
        let line = patch_lines[i];

        // Skip header lines
        if line.starts_with("---") || line.starts_with("+++") {
            i += 1;
            continue;
        }

        // Hunk header: @@ -old_start,old_len +new_start,new_len @@
        if line.starts_with("@@") {
            let hunk = parse_hunk_header(line)
                .with_context(|| format!("bad hunk header: {}", line))?;

            // Copy unchanged lines up to the hunk's old start
            let old_start = hunk.0.saturating_sub(1); // convert to 0-indexed
            while orig_pos < old_start && orig_pos < orig_lines.len() {
                result_lines.push(orig_lines[orig_pos].to_string());
                orig_pos += 1;
            }

            i += 1;
            // Process hunk body
            while i < patch_lines.len() && !patch_lines[i].starts_with("@@")
                && !patch_lines[i].starts_with("---") && !patch_lines[i].starts_with("+++") {
                let pl = patch_lines[i];
                if let Some(content) = pl.strip_prefix('+') {
                    result_lines.push(content.to_string());
                } else if let Some(_) = pl.strip_prefix('-') {
                    // Skip deleted line, advance orig
                    orig_pos += 1;
                } else if let Some(content) = pl.strip_prefix(' ') {
                    result_lines.push(content.to_string());
                    orig_pos += 1;
                }
                i += 1;
            }
            continue;
        }
        i += 1;
    }

    // Copy remaining original lines
    while orig_pos < orig_lines.len() {
        result_lines.push(orig_lines[orig_pos].to_string());
        orig_pos += 1;
    }

    Ok(result_lines.join("\n") + if original.ends_with('\n') { "\n" } else { "" })
}

/// Returns (old_start, old_len, new_start, new_len)
fn parse_hunk_header(line: &str) -> Result<(usize, usize, usize, usize)> {
    // @@ -a,b +c,d @@
    let inner = line.trim_start_matches('@').trim_end_matches('@').trim();
    let parts: Vec<&str> = inner.split_whitespace().collect();
    if parts.len() < 2 { bail!("malformed hunk"); }

    fn parse_range(s: &str) -> (usize, usize) {
        let s = s.trim_start_matches(['-', '+']);
        if let Some((a, b)) = s.split_once(',') {
            (a.parse().unwrap_or(1), b.parse().unwrap_or(1))
        } else {
            (s.parse().unwrap_or(1), 1)
        }
    }

    let (os, ol) = parse_range(parts[0]);
    let (ns, nl) = parse_range(parts[1]);
    Ok((os, ol, ns, nl))
}

fn now_str() -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    format!("{}", secs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU32, Ordering};

    static COUNTER: AtomicU32 = AtomicU32::new(0);

    fn tmp_engine() -> PatchEngine {
        let id = COUNTER.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!("ai_ws_patch_{}_{}", std::process::id(), id));
        std::fs::create_dir_all(&dir).unwrap();
        PatchEngine::new(dir)
    }

    #[test]
    fn diff_counts_additions_and_deletions() {
        let engine = tmp_engine();
        let old = "line1\nline2\nline3\n";
        let new = "line1\nLINE2_CHANGED\nline3\nline4_new\n";
        let result = engine.diff_text("test.txt", old, new);
        // one deletion (line2) + one addition (LINE2_CHANGED) + one new addition (line4_new)
        assert_eq!(result.deletions, 1, "deletions: {}", result.deletions);
        assert_eq!(result.additions, 2, "additions: {}", result.additions);
        assert_eq!(result.unchanged, 2, "unchanged: {}", result.unchanged);
        std::fs::remove_dir_all(&engine.root).ok();
    }

    #[test]
    fn apply_and_undo_round_trip() {
        let mut engine = tmp_engine();
        let root = engine.root.clone();
        std::fs::write(root.join("file.txt"), "original\n").unwrap();
        engine.apply("file.txt", "modified\n", "test change").unwrap();
        assert_eq!(std::fs::read_to_string(root.join("file.txt")).unwrap(), "modified\n");
        engine.undo("file.txt").unwrap();
        assert_eq!(std::fs::read_to_string(root.join("file.txt")).unwrap(), "original\n");
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn unified_diff_apply_add_lines() {
        let mut engine = tmp_engine();
        let root = engine.root.clone();
        std::fs::write(root.join("f.txt"), "a\nb\n").unwrap();
        let patch = "--- a/f.txt\n+++ b/f.txt\n@@ -2,1 +2,2 @@\n b\n+c\n";
        engine.apply_unified_diff("f.txt", patch).unwrap();
        let content = std::fs::read_to_string(root.join("f.txt")).unwrap();
        assert!(content.contains("c"), "added line missing: {}", content);
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn cannot_escape_workspace_via_dotdot() {
        let mut engine = tmp_engine();
        let result = engine.apply("../../evil.txt", "bad", "escape");
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("escapes"));
        std::fs::remove_dir_all(&engine.root).ok();
    }

    #[test]
    fn diff_unified_header_format() {
        let engine = tmp_engine();
        let result = engine.diff_text("myfile.rs", "old\n", "new\n");
        assert!(result.unified.contains("--- a/myfile.rs"), "bad header: {}", result.unified);
        assert!(result.unified.contains("+++ b/myfile.rs"), "bad header: {}", result.unified);
        std::fs::remove_dir_all(&engine.root).ok();
    }
}
