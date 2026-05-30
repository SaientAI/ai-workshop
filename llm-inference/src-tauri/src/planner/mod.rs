/// planner/mod.rs — Planner + Verifier loop
///
/// Flow:
///   1. Planner receives a goal → emits a Plan (ordered list of Steps)
///   2. Executor runs each Step using the tool layer
///   3. Verifier checks step output against success criteria
///   4. On failure: retry with backoff, or emit a re-plan request
///   5. All state tracked in Memory
///
/// The LLM is called for planning and verification.
/// Tool execution is pure Rust (no LLM needed for fs/exec/patch).

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ── Plan types ────────────────────────────────────────────────────────────────

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum StepStatus {
    Pending,
    Running,
    Done,
    Failed,
    Skipped,
    Retrying,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ToolCall {
    pub tool: String,
    pub params: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct VerificationCriteria {
    /// Check that output contains this string
    pub output_contains: Option<String>,
    /// Check that output does NOT contain this
    pub output_excludes: Option<String>,
    /// Check exit code equals this
    pub exit_code: Option<i32>,
    /// Check that a file exists at this path
    pub file_exists: Option<String>,
    /// Check that a file contains this string
    pub file_contains: Option<(String, String)>,
    /// Custom regex match on output
    pub output_matches_regex: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct PlanStep {
    pub id: String,
    pub index: usize,
    pub description: String,
    pub tool_call: ToolCall,
    pub verification: Option<VerificationCriteria>,
    pub max_retries: usize,
    pub retry_count: usize,
    pub depends_on: Vec<String>,   // step IDs this must wait for
    pub status: StepStatus,
    pub output: Option<serde_json::Value>,
    pub error: Option<String>,
    pub started_at: Option<u64>,
    pub finished_at: Option<u64>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Plan {
    pub id: String,
    pub goal: String,
    pub steps: Vec<PlanStep>,
    pub status: StepStatus,
    pub created_at: u64,
    pub context: String,           // extra context injected into each LLM call
}

impl Plan {
    pub fn new(goal: &str) -> Self {
        Self {
            id: gen_id(),
            goal: goal.into(),
            steps: vec![],
            status: StepStatus::Pending,
            created_at: now_secs(),
            context: String::new(),
        }
    }

    pub fn add_step(
        &mut self,
        description: &str,
        tool: &str,
        params: HashMap<String, serde_json::Value>,
        verification: Option<VerificationCriteria>,
        depends_on: Vec<String>,
    ) -> String {
        let id = gen_id();
        self.steps.push(PlanStep {
            id: id.clone(),
            index: self.steps.len(),
            description: description.into(),
            tool_call: ToolCall { tool: tool.into(), params },
            verification,
            max_retries: 2,
            retry_count: 0,
            depends_on,
            status: StepStatus::Pending,
            output: None,
            error: None,
            started_at: None,
            finished_at: None,
        });
        id
    }

    pub fn ready_steps(&self) -> Vec<&PlanStep> {
        let done_ids: Vec<&str> = self.steps.iter()
            .filter(|s| s.status == StepStatus::Done)
            .map(|s| s.id.as_str())
            .collect();

        self.steps.iter()
            .filter(|s| {
                matches!(s.status, StepStatus::Pending | StepStatus::Retrying) &&
                s.depends_on.iter().all(|dep| done_ids.contains(&dep.as_str()))
            })
            .collect()
    }

    pub fn is_complete(&self) -> bool {
        self.steps.iter().all(|s| matches!(s.status, StepStatus::Done | StepStatus::Skipped | StepStatus::Failed))
    }

    pub fn has_failed(&self) -> bool {
        self.steps.iter().any(|s| s.status == StepStatus::Failed)
    }

    pub fn progress(&self) -> (usize, usize) {
        let done = self.steps.iter().filter(|s| matches!(s.status, StepStatus::Done | StepStatus::Skipped)).count();
        (done, self.steps.len())
    }

    pub fn summary(&self) -> PlanSummary {
        let (done, total) = self.progress();
        PlanSummary {
            id: self.id.clone(),
            goal: self.goal.clone(),
            total_steps: total,
            done_steps: done,
            failed_steps: self.steps.iter().filter(|s| s.status == StepStatus::Failed).count(),
            pending_steps: self.steps.iter().filter(|s| s.status == StepStatus::Pending).count(),
            status: format!("{:?}", self.status),
            steps: self.steps.iter().map(|s| StepSummary {
                id: s.id.clone(),
                index: s.index,
                description: s.description.clone(),
                tool: s.tool_call.tool.clone(),
                status: format!("{:?}", s.status),
                error: s.error.clone(),
                retry_count: s.retry_count,
            }).collect(),
        }
    }
}

// ── Summary types (for frontend) ──────────────────────────────────────────────

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct PlanSummary {
    pub id: String,
    pub goal: String,
    pub total_steps: usize,
    pub done_steps: usize,
    pub failed_steps: usize,
    pub pending_steps: usize,
    pub status: String,
    pub steps: Vec<StepSummary>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct StepSummary {
    pub id: String,
    pub index: usize,
    pub description: String,
    pub tool: String,
    pub status: String,
    pub error: Option<String>,
    pub retry_count: usize,
}

// ── Verifier ──────────────────────────────────────────────────────────────────

pub struct Verifier;

impl Verifier {
    /// Check a step's output against its verification criteria.
    pub fn verify(step: &PlanStep, output: &serde_json::Value) -> VerifyResult {
        let criteria = match &step.verification {
            None => return VerifyResult { passed: true, reason: "no criteria".into(), suggestions: vec![] },
            Some(c) => c,
        };

        let output_str = match output {
            serde_json::Value::String(s) => s.clone(),
            other => other.to_string(),
        };

        // output_contains
        if let Some(required) = &criteria.output_contains {
            if !output_str.contains(required.as_str()) {
                return VerifyResult {
                    passed: false,
                    reason: format!("output missing required string: '{}'", required),
                    suggestions: vec!["Check command succeeded".into(), "Verify the correct tool was used".into()],
                };
            }
        }

        // output_excludes
        if let Some(banned) = &criteria.output_excludes {
            if output_str.contains(banned.as_str()) {
                return VerifyResult {
                    passed: false,
                    reason: format!("output contains forbidden string: '{}'", banned),
                    suggestions: vec!["Error text detected in output".into()],
                };
            }
        }

        // exit_code
        if let Some(expected_code) = criteria.exit_code {
            if let Some(actual) = output.get("exit_code").and_then(|v| v.as_i64()) {
                if actual != expected_code as i64 {
                    return VerifyResult {
                        passed: false,
                        reason: format!("exit code {} ≠ expected {}", actual, expected_code),
                        suggestions: vec!["Check command syntax".into(), "Verify dependencies are installed".into()],
                    };
                }
            }
        }

        // output_matches_regex
        if let Some(pattern) = &criteria.output_matches_regex {
            match regex::Regex::new(pattern) {
                Ok(re) => {
                    if !re.is_match(&output_str) {
                        return VerifyResult {
                            passed: false,
                            reason: format!("output did not match regex: {}", pattern),
                            suggestions: vec!["Output format was unexpected".into()],
                        };
                    }
                }
                Err(e) => {
                    return VerifyResult {
                        passed: false,
                        reason: format!("invalid regex in verification criteria: {}", e),
                        suggestions: vec![],
                    };
                }
            }
        }

        // file_exists
        if let Some(path) = &criteria.file_exists {
            if !std::path::Path::new(path).exists() {
                return VerifyResult {
                    passed: false,
                    reason: format!("expected file does not exist: '{}'", path),
                    suggestions: vec!["Check that the write step completed successfully".into()],
                };
            }
        }

        // file_contains
        if let Some((path, needle)) = &criteria.file_contains {
            match std::fs::read_to_string(path) {
                Ok(contents) => {
                    if !contents.contains(needle.as_str()) {
                        return VerifyResult {
                            passed: false,
                            reason: format!("file '{}' does not contain: '{}'", path, needle),
                            suggestions: vec!["Check that the correct content was written".into()],
                        };
                    }
                }
                Err(e) => {
                    return VerifyResult {
                        passed: false,
                        reason: format!("could not read file '{}': {}", path, e),
                        suggestions: vec!["Check that the file exists and is readable".into()],
                    };
                }
            }
        }

        VerifyResult { passed: true, reason: "all criteria met".into(), suggestions: vec![] }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct VerifyResult {
    pub passed: bool,
    pub reason: String,
    pub suggestions: Vec<String>,
}

// ── Planner (template-based, wired to LLM in main.rs) ────────────────────────

/// A structured prompt template the planner uses to ask the LLM for a plan.
pub fn plan_prompt(goal: &str, memory_context: &str, available_tools: &[&str]) -> String {
    format!(
        r#"You are an autonomous agent planner. Given a goal, produce a JSON execution plan.

## Goal
{goal}

## Available tools
{tools}

## Memory context
{memory}

## Instructions
Return ONLY valid JSON matching this schema:
{{
  "goal": "...",
  "steps": [
    {{
      "description": "human readable description",
      "tool": "tool_name",
      "params": {{ ... }},
      "depends_on": [],
      "verification": {{
        "output_contains": "optional string that must appear in output",
        "output_excludes": "optional string that must NOT appear",
        "exit_code": 0,
        "file_exists": "optional/path/to/file",
        "output_matches_regex": "optional_regex"
      }}
    }}
  ]
}}

Rules:
- steps execute in order unless depends_on creates a dependency
- verification fields are all optional, use only what makes sense
- tool params must exactly match the tool's expected parameters
- prefer small, atomic steps over large ones
- if a step might fail, add a verification so it can be retried"#,
        goal = goal,
        tools = available_tools.join(", "),
        memory = if memory_context.is_empty() { "none" } else { memory_context },
    )
}

/// Parse a JSON plan from LLM output.
pub fn parse_plan_response(goal: &str, json: &str) -> Result<Plan> {
    // Strip markdown fences if present
    let clean = json
        .trim()
        .trim_start_matches("```json")
        .trim_start_matches("```")
        .trim_end_matches("```")
        .trim();

    let val: serde_json::Value = serde_json::from_str(clean)
        .with_context(|| format!("LLM returned invalid JSON: {}", &clean[..clean.len().min(200)]))?;

    let mut plan = Plan::new(goal);

    if let Some(steps) = val["steps"].as_array() {
        for step in steps {
            let tool = step["tool"].as_str().unwrap_or("noop").to_string();
            let description = step["description"].as_str().unwrap_or("").to_string();
            let depends_on: Vec<String> = step["depends_on"]
                .as_array()
                .map(|a| a.iter().filter_map(|v| v.as_str().map(String::from)).collect())
                .unwrap_or_default();

            let params: HashMap<String, serde_json::Value> = step["params"]
                .as_object()
                .map(|o| o.iter().map(|(k, v)| (k.clone(), v.clone())).collect())
                .unwrap_or_default();

            let verification = step.get("verification").and_then(|v| {
                serde_json::from_value::<VerificationCriteria>(v.clone()).ok()
            });

            plan.add_step(&description, &tool, params, verification, depends_on);
        }
    }

    Ok(plan)
}

// ── Helpers ───────────────────────────────────────────────────────────────────
fn now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn gen_id() -> String {
    let t = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    format!("{:x}", t ^ (t >> 32))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_valid_plan() {
        let json = r#"{"steps":[
            {"tool":"fs_write","description":"write hello","params":{"path":"hello.txt","content":"hi"}},
            {"tool":"exec_command","description":"run it","params":{"command":"cat","args":["hello.txt"]},"depends_on":[]}
        ]}"#;
        let plan = parse_plan_response("test goal", json).unwrap();
        assert_eq!(plan.goal, "test goal");
        assert_eq!(plan.steps.len(), 2);
        assert_eq!(plan.steps[0].tool_call.tool, "fs_write");
        assert_eq!(plan.steps[1].tool_call.tool, "exec_command");
    }

    #[test]
    fn parse_rejects_invalid_json() {
        let result = parse_plan_response("goal", "this is not json {{{{");
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("invalid") || msg.contains("JSON"), "got: {}", msg);
    }

    #[test]
    fn parse_markdown_fenced_json() {
        let json = "```json\n{\"steps\":[]}\n```";
        let plan = parse_plan_response("goal", json).unwrap();
        assert_eq!(plan.steps.len(), 0);
    }

    #[test]
    fn plan_ready_steps_respects_depends_on() {
        let mut plan = Plan::new("test");
        let id_a = plan.add_step("step A", "noop", HashMap::new(), None, vec![]);
        let id_b = plan.add_step("step B", "noop", HashMap::new(), None, vec![id_a.clone()]);
        // B depends on A; A is Pending → B should not be ready
        let ready = plan.ready_steps();
        assert_eq!(ready.len(), 1);
        assert_eq!(ready[0].id, id_a);

        // Mark A done
        plan.steps[0].status = StepStatus::Done;
        let ready = plan.ready_steps();
        assert_eq!(ready.len(), 1);
        assert_eq!(ready[0].id, id_b);
    }

    #[test]
    fn plan_is_complete_when_all_done() {
        let mut plan = Plan::new("goal");
        plan.add_step("s1", "noop", HashMap::new(), None, vec![]);
        plan.add_step("s2", "noop", HashMap::new(), None, vec![]);
        assert!(!plan.is_complete());
        for s in &mut plan.steps { s.status = StepStatus::Done; }
        assert!(plan.is_complete());
    }
}
