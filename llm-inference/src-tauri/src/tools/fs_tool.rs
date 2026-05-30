/// fs_tool.rs — Filesystem tool layer
/// Safe scoped read/write/search/watch over a working directory.
/// All paths are canonicalized and checked against the sandbox root.

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};
use walkdir::WalkDir;

// ── Types ─────────────────────────────────────────────────────────────────────

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct FileEntry {
    pub path: String,
    pub name: String,
    pub is_dir: bool,
    pub size: u64,
    pub modified: String,
    pub extension: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ReadResult {
    pub path: String,
    pub content: String,
    pub size: u64,
    pub hash: String,
    pub lines: usize,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct WriteResult {
    pub path: String,
    pub bytes_written: usize,
    pub hash: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct SearchResult {
    pub path: String,
    pub line_number: usize,
    pub line: String,
    pub context_before: Vec<String>,
    pub context_after: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct TreeEntry {
    pub path: String,
    pub name: String,
    pub is_dir: bool,
    pub depth: usize,
    pub children: Vec<TreeEntry>,
}

// ── FsTool ────────────────────────────────────────────────────────────────────

pub struct FsTool {
    pub root: PathBuf,
}

impl FsTool {
    pub fn new(root: &Path) -> Result<Self> {
        std::fs::create_dir_all(root).context("create sandbox root")?;
        Ok(Self { root: root.canonicalize().context("canonicalize root")? })
    }

    /// Resolve and verify a path stays within the sandbox root.
    fn safe_path(&self, rel: &str) -> Result<PathBuf> {
        // Strip leading slashes so joining works correctly
        let rel = rel.trim_start_matches('/');
        let joined = self.root.join(rel);

        // Resolve without requiring existence (for new files)
        let resolved = if joined.exists() {
            joined.canonicalize()?
        } else {
            // Resolve parent, then re-attach filename
            let parent = joined.parent().unwrap_or(&self.root);
            let parent_canon = if parent.exists() {
                parent.canonicalize()?
            } else {
                std::fs::create_dir_all(parent)?;
                parent.canonicalize()?
            };
            parent_canon.join(joined.file_name().unwrap_or_default())
        };

        if !resolved.starts_with(&self.root) {
            bail!("Path escape attempt: {:?} is outside sandbox", resolved);
        }
        Ok(resolved)
    }

    pub fn read_file(&self, path: &str) -> Result<ReadResult> {
        let p = self.safe_path(path)?;
        let content = std::fs::read_to_string(&p)
            .with_context(|| format!("read {:?}", p))?;
        let hash = sha256_str(&content);
        let lines = content.lines().count();
        let size = content.len() as u64;
        Ok(ReadResult {
            path: self.rel_str(&p),
            content,
            size,
            hash,
            lines,
        })
    }

    pub fn write_file(&self, path: &str, content: &str) -> Result<WriteResult> {
        let p = self.safe_path(path)?;
        if let Some(parent) = p.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(&p, content)
            .with_context(|| format!("write {:?}", p))?;
        Ok(WriteResult {
            path: self.rel_str(&p),
            bytes_written: content.len(),
            hash: sha256_str(content),
        })
    }

    pub fn append_file(&self, path: &str, content: &str) -> Result<WriteResult> {
        let p = self.safe_path(path)?;
        use std::io::Write;
        let mut f = std::fs::OpenOptions::new().append(true).create(true).open(&p)?;
        f.write_all(content.as_bytes())?;
        let full = std::fs::read_to_string(&p)?;
        Ok(WriteResult {
            path: self.rel_str(&p),
            bytes_written: content.len(),
            hash: sha256_str(&full),
        })
    }

    pub fn delete_file(&self, path: &str) -> Result<String> {
        let p = self.safe_path(path)?;
        if p.is_dir() {
            std::fs::remove_dir_all(&p)?;
        } else {
            std::fs::remove_file(&p)?;
        }
        Ok(format!("deleted {}", self.rel_str(&p)))
    }

    pub fn create_dir(&self, path: &str) -> Result<String> {
        let p = self.safe_path(path)?;
        std::fs::create_dir_all(&p)?;
        Ok(self.rel_str(&p))
    }

    pub fn list_dir(&self, path: &str) -> Result<Vec<FileEntry>> {
        let p = self.safe_path(path)?;
        let mut entries = vec![];
        for entry in std::fs::read_dir(&p).with_context(|| format!("list {:?}", p))? {
            let e = entry?;
            let meta = e.metadata()?;
            let modified = meta.modified()
                .ok()
                .and_then(|t| {
                    let secs = t.duration_since(std::time::UNIX_EPOCH).ok()?.as_secs();
                    Some(format_unix_ts(secs))
                })
                .unwrap_or_default();
            let name = e.file_name().to_string_lossy().into_owned();
            let ext = Path::new(&name).extension()
                .map(|e| e.to_string_lossy().into_owned())
                .unwrap_or_default();
            entries.push(FileEntry {
                path: self.rel_str(&e.path()),
                name,
                is_dir: meta.is_dir(),
                size: if meta.is_file() { meta.len() } else { 0 },
                modified,
                extension: ext,
            });
        }
        entries.sort_by(|a, b| b.is_dir.cmp(&a.is_dir).then(a.name.cmp(&b.name)));
        Ok(entries)
    }

    pub fn tree(&self, path: &str, max_depth: usize) -> Result<Vec<TreeEntry>> {
        let p = self.safe_path(path)?;
        Ok(build_tree(&p, &self.root, 0, max_depth))
    }

    pub fn search(&self, path: &str, pattern: &str, context_lines: usize) -> Result<Vec<SearchResult>> {
        let p = self.safe_path(path)?;
        let re = regex::Regex::new(pattern).context("invalid regex")?;
        let mut results = vec![];

        let walker = WalkDir::new(&p).follow_links(false).into_iter()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_type().is_file());

        for entry in walker {
            let content = match std::fs::read_to_string(entry.path()) {
                Ok(c) => c,
                Err(_) => continue, // skip binary files
            };
            let lines: Vec<&str> = content.lines().collect();
            for (i, line) in lines.iter().enumerate() {
                if re.is_match(line) {
                    let before: Vec<String> = lines[i.saturating_sub(context_lines)..i]
                        .iter().map(|s| s.to_string()).collect();
                    let after: Vec<String> = lines[i+1..((i+1+context_lines).min(lines.len()))]
                        .iter().map(|s| s.to_string()).collect();
                    results.push(SearchResult {
                        path: self.rel_str(entry.path()),
                        line_number: i + 1,
                        line: line.to_string(),
                        context_before: before,
                        context_after: after,
                    });
                }
            }
        }
        Ok(results)
    }

    pub fn move_file(&self, from: &str, to: &str) -> Result<String> {
        let src = self.safe_path(from)?;
        let dst = self.safe_path(to)?;
        if let Some(parent) = dst.parent() { std::fs::create_dir_all(parent)?; }
        std::fs::rename(&src, &dst)?;
        Ok(format!("{} → {}", self.rel_str(&src), self.rel_str(&dst)))
    }

    pub fn copy_file(&self, from: &str, to: &str) -> Result<String> {
        let src = self.safe_path(from)?;
        let dst = self.safe_path(to)?;
        if let Some(parent) = dst.parent() { std::fs::create_dir_all(parent)?; }
        std::fs::copy(&src, &dst)?;
        Ok(format!("copied {} → {}", self.rel_str(&src), self.rel_str(&dst)))
    }

    pub fn file_exists(&self, path: &str) -> bool {
        self.safe_path(path).map(|p| p.exists()).unwrap_or(false)
    }

    pub fn hash_file(&self, path: &str) -> Result<String> {
        let p = self.safe_path(path)?;
        let bytes = std::fs::read(&p)?;
        let mut h = Sha256::new();
        h.update(&bytes);
        Ok(format!("{:x}", h.finalize()))
    }

    fn rel_str(&self, p: &Path) -> String {
        p.strip_prefix(&self.root)
            .map(|r| r.to_string_lossy().into_owned())
            .unwrap_or_else(|_| p.to_string_lossy().into_owned())
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn sha256_str(s: &str) -> String {
    let mut h = Sha256::new();
    h.update(s.as_bytes());
    format!("{:x}", h.finalize())
}

fn format_unix_ts(secs: u64) -> String {
    // Simple ISO-like format without chrono dependency here
    let days = secs / 86400;
    let epoch_days = 719_468u64;
    let z = days + epoch_days;
    let era = z / 146097;
    let doe = z - era * 146097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    let h = (secs % 86400) / 3600;
    let min = (secs % 3600) / 60;
    format!("{:04}-{:02}-{:02} {:02}:{:02}", y, m, d, h, min)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU32, Ordering};

    static COUNTER: AtomicU32 = AtomicU32::new(0);

    fn tmp_root() -> PathBuf {
        let id = COUNTER.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!("ai_ws_fs_{}_{}",std::process::id(), id));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn write_and_read_round_trip() {
        let root = tmp_root();
        let fs = FsTool::new(&root).unwrap();
        fs.write_file("hello.txt", "hello world").unwrap();
        let result = fs.read_file("hello.txt").unwrap();
        assert_eq!(result.content, "hello world");
        assert_eq!(result.lines, 1);
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn cannot_escape_sandbox_with_dotdot() {
        let root = tmp_root();
        let fs = FsTool::new(&root).unwrap();
        let err = fs.read_file("../../etc/passwd");
        assert!(err.is_err(), "dotdot traversal should be rejected");
        let msg = err.unwrap_err().to_string();
        assert!(msg.contains("escape") || msg.contains("outside"), "got: {}", msg);
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn cannot_escape_sandbox_with_absolute_path() {
        let root = tmp_root();
        let fs = FsTool::new(&root).unwrap();
        // safe_path strips leading '/', so /etc/passwd becomes root/etc/passwd
        // which is inside the sandbox — verify it doesn't leave the root
        let result = fs.safe_path("/etc/passwd");
        if let Ok(p) = result {
            assert!(p.starts_with(&root), "resolved {:?} must be inside {:?}", p, root);
        }
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn delete_removes_file() {
        let root = tmp_root();
        let fs = FsTool::new(&root).unwrap();
        fs.write_file("to_delete.txt", "bye").unwrap();
        assert!(root.join("to_delete.txt").exists());
        fs.delete_file("to_delete.txt").unwrap();
        assert!(!root.join("to_delete.txt").exists());
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn subdirectory_created_on_write() {
        let root = tmp_root();
        let fs = FsTool::new(&root).unwrap();
        fs.write_file("subdir/nested.txt", "deep").unwrap();
        assert!(root.join("subdir/nested.txt").exists());
        std::fs::remove_dir_all(&root).ok();
    }
}

fn build_tree(dir: &Path, root: &Path, depth: usize, max_depth: usize) -> Vec<TreeEntry> {
    if depth >= max_depth { return vec![]; }
    let mut entries = vec![];
    let rd = match std::fs::read_dir(dir) { Ok(r) => r, Err(_) => return entries };
    let mut items: Vec<_> = rd.filter_map(|e| e.ok()).collect();
    items.sort_by(|a, b| {
        let ad = a.metadata().map(|m| m.is_dir()).unwrap_or(false);
        let bd = b.metadata().map(|m| m.is_dir()).unwrap_or(false);
        bd.cmp(&ad).then(a.file_name().cmp(&b.file_name()))
    });
    for entry in items {
        let meta = match entry.metadata() { Ok(m) => m, Err(_) => continue };
        let name = entry.file_name().to_string_lossy().into_owned();
        if name.starts_with('.') { continue; } // skip hidden
        let is_dir = meta.is_dir();
        let children = if is_dir {
            build_tree(&entry.path(), root, depth + 1, max_depth)
        } else {
            vec![]
        };
        let rel = entry.path().strip_prefix(root)
            .map(|p| p.to_string_lossy().into_owned())
            .unwrap_or_else(|_| name.clone());
        entries.push(TreeEntry { path: rel, name, is_dir, depth, children });
    }
    entries
}
