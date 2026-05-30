/// memory/store.rs — Long-horizon agent memory
///
/// Three layers:
///   1. Working memory  — current task context, volatile
///   2. Episodic memory — past task records, append-only log
///   3. Semantic memory — key facts the agent has learned, searchable
///
/// All persisted to a single JSON file on disk.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;

// ── Types ─────────────────────────────────────────────────────────────────────

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct WorkingMemory {
    pub task_id: String,
    pub task_goal: String,
    pub task_status: String,     // "pending" | "running" | "done" | "failed"
    pub current_step: usize,
    pub total_steps: usize,
    pub scratch: HashMap<String, serde_json::Value>,  // arbitrary KV scratchpad
    pub tool_calls: Vec<ToolCallRecord>,
    pub started_at: u64,
    pub updated_at: u64,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ToolCallRecord {
    pub step: usize,
    pub tool: String,
    pub input: serde_json::Value,
    pub output: serde_json::Value,
    pub success: bool,
    pub duration_ms: u64,
    pub ts: u64,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct EpisodicEntry {
    pub task_id: String,
    pub goal: String,
    pub outcome: String,
    pub steps_taken: usize,
    pub tool_calls: usize,
    pub files_modified: Vec<String>,
    pub commands_run: Vec<String>,
    pub started_at: u64,
    pub ended_at: u64,
    pub summary: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Fact {
    pub id: String,
    pub key: String,
    pub value: String,
    pub category: String,    // "file", "code", "infra", "user", "learned"
    pub confidence: f32,     // 0.0 – 1.0
    pub source: String,      // which task/tool learned this
    pub created_at: u64,
    pub accessed_at: u64,
    pub access_count: u32,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct MemoryStore {
    pub working: Option<WorkingMemory>,
    pub episodes: Vec<EpisodicEntry>,
    pub facts: Vec<Fact>,
    pub version: u32,
}

// ── Memory engine ─────────────────────────────────────────────────────────────

pub struct Memory {
    store: MemoryStore,
    pub(crate) path: PathBuf,
}

impl Memory {
    pub fn load(path: PathBuf) -> Result<Self> {
        let store = if path.exists() {
            let raw = std::fs::read_to_string(&path)
                .with_context(|| format!("read memory {:?}", path))?;
            serde_json::from_str(&raw).unwrap_or_default()
        } else {
            MemoryStore::default()
        };
        Ok(Self { store, path })
    }

    /// Wipe all stored data in-memory and delete the file on disk.
    pub fn clear_all(&mut self) {
        self.store = MemoryStore::default();
        std::fs::remove_file(&self.path).ok();
        // Also remove any stale .tmp left from an interrupted write
        let mut tmp = self.path.clone();
        let fname = self.path.file_name().unwrap_or_default().to_string_lossy().into_owned();
        tmp.set_file_name(format!("{}.tmp", fname));
        std::fs::remove_file(&tmp).ok();
    }

    fn save(&self) -> Result<()> {
        if let Some(parent) = self.path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let json = serde_json::to_string_pretty(&self.store)?;
        // Atomic write: write to .tmp then rename to avoid partial-write corruption
        let fname = self.path.file_name().unwrap_or_default().to_string_lossy();
        let mut tmp = self.path.clone();
        tmp.set_file_name(format!("{}.tmp", fname));
        std::fs::write(&tmp, &json)?;
        std::fs::rename(&tmp, &self.path)?;
        Ok(())
    }

    // ── Working memory ──────────────────────────────────────────────────────

    pub fn start_task(&mut self, goal: &str) -> Result<String> {
        let id = uuid_v4();
        self.store.working = Some(WorkingMemory {
            task_id: id.clone(),
            task_goal: goal.into(),
            task_status: "running".into(),
            current_step: 0,
            total_steps: 0,
            scratch: HashMap::new(),
            tool_calls: vec![],
            started_at: now_secs(),
            updated_at: now_secs(),
        });
        self.save()?;
        Ok(id)
    }

    pub fn update_step(&mut self, step: usize, total: usize) -> Result<()> {
        if let Some(wm) = &mut self.store.working {
            wm.current_step = step;
            wm.total_steps = total;
            wm.updated_at = now_secs();
        }
        self.save()
    }

    pub fn record_tool_call(&mut self, call: ToolCallRecord) -> Result<()> {
        if let Some(wm) = &mut self.store.working {
            wm.tool_calls.push(call);
            wm.updated_at = now_secs();
        }
        self.save()
    }

    pub fn set_scratch(&mut self, key: &str, value: serde_json::Value) -> Result<()> {
        if let Some(wm) = &mut self.store.working {
            wm.scratch.insert(key.into(), value);
            wm.updated_at = now_secs();
        }
        self.save()
    }

    pub fn get_scratch(&self, key: &str) -> Option<&serde_json::Value> {
        self.store.working.as_ref()?.scratch.get(key)
    }

    pub fn finish_task(&mut self, outcome: &str, summary: &str) -> Result<()> {
        if let Some(wm) = self.store.working.take() {
            let files: Vec<String> = wm.tool_calls.iter()
                .filter(|t| t.tool.contains("fs") || t.tool.contains("file") || t.tool.contains("patch"))
                .filter_map(|t| t.input.get("path").and_then(|v| v.as_str()).map(|s| s.to_string()))
                .collect();
            let cmds: Vec<String> = wm.tool_calls.iter()
                .filter(|t| t.tool == "exec")
                .filter_map(|t| t.input.get("command").and_then(|v| v.as_str()).map(|s| s.to_string()))
                .collect();

            self.store.episodes.push(EpisodicEntry {
                task_id: wm.task_id,
                goal: wm.task_goal,
                outcome: outcome.into(),
                steps_taken: wm.current_step,
                tool_calls: wm.tool_calls.len(),
                files_modified: files,
                commands_run: cmds,
                started_at: wm.started_at,
                ended_at: now_secs(),
                summary: summary.into(),
            });
        }
        self.save()
    }

    pub fn working(&self) -> Option<&WorkingMemory> {
        self.store.working.as_ref()
    }

    // ── Semantic memory ──────────────────────────────────────────────────────

    pub fn remember(&mut self, key: &str, value: &str, category: &str, source: &str, confidence: f32) -> Result<String> {
        // Update existing fact if key matches
        if let Some(existing) = self.store.facts.iter_mut().find(|f| f.key == key) {
            existing.value = value.into();
            existing.confidence = confidence;
            existing.source = source.into();
            existing.accessed_at = now_secs();
            existing.access_count += 1;
            let id = existing.id.clone();
            self.save()?;
            return Ok(id);
        }

        let id = uuid_v4();
        self.store.facts.push(Fact {
            id: id.clone(),
            key: key.into(),
            value: value.into(),
            category: category.into(),
            confidence,
            source: source.into(),
            created_at: now_secs(),
            accessed_at: now_secs(),
            access_count: 1,
        });
        self.save()?;
        Ok(id)
    }

    pub fn recall(&mut self, query: &str) -> Vec<Fact> {
        let q = query.to_lowercase();
        let mut results: Vec<Fact> = self.store.facts.iter()
            .filter(|f| f.key.to_lowercase().contains(&q) || f.value.to_lowercase().contains(&q))
            .cloned()
            .collect();

        // Update access stats
        let now = now_secs();
        for f in self.store.facts.iter_mut() {
            if f.key.to_lowercase().contains(&q) || f.value.to_lowercase().contains(&q) {
                f.accessed_at = now;
                f.access_count += 1;
            }
        }
        let _ = self.save();

        results.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap()
            .then(b.access_count.cmp(&a.access_count)));
        results
    }

    pub fn forget(&mut self, id: &str) -> Result<bool> {
        let before = self.store.facts.len();
        self.store.facts.retain(|f| f.id != id);
        let removed = self.store.facts.len() < before;
        self.save()?;
        Ok(removed)
    }

    pub fn all_facts(&self) -> &[Fact] {
        &self.store.facts
    }

    // ── Episodic ──────────────────────────────────────────────────────────────

    pub fn episodes(&self) -> &[EpisodicEntry] {
        &self.store.episodes
    }

    pub fn recent_episodes(&self, n: usize) -> Vec<&EpisodicEntry> {
        self.store.episodes.iter().rev().take(n).collect()
    }

    /// Build a context string summarising memory for injection into a prompt.
    pub fn context_for_prompt(&self, max_facts: usize) -> String {
        let mut out = String::new();

        if let Some(wm) = &self.store.working {
            out.push_str(&format!("## Current task\nGoal: {}\nStep: {}/{}\nStatus: {}\n\n",
                wm.task_goal, wm.current_step, wm.total_steps, wm.task_status));
        }

        if !self.store.facts.is_empty() {
            out.push_str("## Known facts\n");
            for f in self.store.facts.iter().take(max_facts) {
                out.push_str(&format!("- [{}] {}: {} (confidence: {:.0}%)\n",
                    f.category, f.key, f.value, f.confidence * 100.0));
            }
            out.push('\n');
        }

        if !self.store.episodes.is_empty() {
            out.push_str("## Recent task history\n");
            for ep in self.store.episodes.iter().rev().take(3) {
                out.push_str(&format!("- {}: {} → {}\n", ep.task_id, ep.goal, ep.outcome));
            }
        }

        out
    }

    pub fn full_store(&self) -> &MemoryStore {
        &self.store
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn uuid_v4() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let t = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_nanos();
    format!("{:016x}-{:04x}", t, (t >> 48) as u16 ^ 0xABCD)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp_memory() -> Memory {
        use std::sync::atomic::{AtomicU32, Ordering};
        static C: AtomicU32 = AtomicU32::new(0);
        let n = C.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir()
            .join(format!("ai_ws_mem_{}_{}.json", std::process::id(), n));
        Memory::load(path).unwrap()
    }

    #[test]
    fn remember_and_recall_by_key() {
        let mut m = tmp_memory();
        m.remember("db_host", "localhost:5432", "infra", "test", 0.9).unwrap();
        let results = m.recall("db_host");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].value, "localhost:5432");
    }

    #[test]
    fn recall_returns_empty_for_unknown_query() {
        let mut m = tmp_memory();
        let results = m.recall("xyzzy_nonexistent_key");
        assert!(results.is_empty());
    }

    #[test]
    fn remember_updates_existing_key() {
        let mut m = tmp_memory();
        m.remember("api_url", "v1", "code", "t1", 0.5).unwrap();
        m.remember("api_url", "v2", "code", "t2", 0.9).unwrap();
        let results = m.recall("api_url");
        assert_eq!(results.len(), 1, "should not duplicate facts with same key");
        assert_eq!(results[0].value, "v2");
        assert!((results[0].confidence - 0.9).abs() < 1e-5);
    }

    #[test]
    fn forget_removes_fact() {
        let mut m = tmp_memory();
        let id = m.remember("temp_key", "temp_val", "learned", "test", 0.7).unwrap();
        assert!(!m.recall("temp_key").is_empty());
        let removed = m.forget(&id).unwrap();
        assert!(removed);
        assert!(m.recall("temp_key").is_empty());
    }

    #[test]
    fn forget_unknown_id_returns_false() {
        let mut m = tmp_memory();
        let removed = m.forget("nonexistent-id-xyz").unwrap();
        assert!(!removed);
    }

    #[test]
    fn recall_sorts_by_confidence_desc() {
        let mut m = tmp_memory();
        m.remember("topic", "low", "learned", "t", 0.2).unwrap();
        m.remember("on_topic", "high", "learned", "t", 0.95).unwrap();
        m.remember("topic_mid", "mid", "learned", "t", 0.6).unwrap();
        let results = m.recall("topic");
        assert!(results[0].confidence >= results[1].confidence);
        assert!(results[1].confidence >= results[2].confidence);
    }

    #[test]
    fn atomic_save_produces_valid_json() {
        let mut m = tmp_memory();
        m.remember("save_test", "value", "test", "test", 1.0).unwrap();
        // Verify the file exists and is valid JSON after save
        let raw = std::fs::read_to_string(&m.path).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&raw).unwrap();
        assert!(parsed["facts"].is_array());
        std::fs::remove_file(&m.path).ok();
    }
}
