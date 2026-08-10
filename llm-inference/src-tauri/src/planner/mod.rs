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

use anyhow::{Context, Result};
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

    /// Steps that actually completed, over the total.
    ///
    /// Skipped deliberately does not count as done. A step is only skipped when a
    /// prerequisite failed, so counting it here would report a plan that fell over
    /// after step one as fully finished.
    pub fn progress(&self) -> (usize, usize) {
        let done = self.steps.iter().filter(|s| s.status == StepStatus::Done).count();
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
            skipped_steps: self.steps.iter().filter(|s| s.status == StepStatus::Skipped).count(),
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
    /// Steps never attempted because something they depend on failed.
    pub skipped_steps: usize,
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
///
/// Layout matters as much as wording here. Everything fixed — role, tool list,
/// schema, rules — comes first, and the two parts that change per run (memory,
/// then the goal) come last. The engine's KV cache reuses the longest shared
/// token prefix between consecutive prompts, so holding the boilerplate constant
/// means only the goal is actually prefilled on a second planning call. With the
/// goal near the top, as it used to be, the shared prefix ended after about a
/// dozen tokens and the whole template was re-processed every time.
///
/// Putting the goal last also helps a small model follow it, since it lands
/// nearest the generation point.
pub fn plan_prompt(goal: &str, memory_context: &str, available_tools: &[&str]) -> String {
    format!(
        r#"You are an autonomous agent planner. Given a goal, produce a JSON execution plan.

## Available tools
Use these exact tool names, and these exact parameter names. `?` marks optional.
{tools}

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

A correct single step looks exactly like this:
{{
  "description": "Create the reports folder",
  "tool": "fs_mkdir",
  "params": {{ "path": "reports" }},
  "depends_on": [],
  "verification": {{ "file_exists": "reports" }}
}}

Rules:
- steps execute in order unless depends_on creates a dependency
- verification fields are all optional, use only what makes sense
- every entry in "params" must be a "name": value pair using the names listed above
- paths are relative to the workspace root; do not invent absolute paths
- prefer small, atomic steps over large ones
- if a step might fail, add a verification so it can be retried
- emit the smallest plan that achieves the goal; a one-step goal gets one step
- every string must be double-quoted, and emit no text outside the JSON object

## Memory context
{memory}

## Goal
{goal}"#,
        goal = goal,
        tools = available_tools
            .iter()
            .map(|t| format!("  {t}"))
            .collect::<Vec<_>>()
            .join("\n"),
        memory = if memory_context.is_empty() { "none" } else { memory_context },
    )
}

/// Best-effort repair of the JSON defects a small model actually emits.
///
/// A 7B asked for a 2000-token strict-JSON document gets it *nearly* right and
/// then drops a quote somewhere in the middle. Before this, one bad character
/// threw away the whole plan — and the ~11s spent generating it. Repair is free;
/// re-asking is not.
pub fn repair_json(raw: &str) -> String {
    // Take the outermost object. Models like to wrap it in prose or fences, and
    // both are harmless once we slice to the braces.
    let body = match (raw.find('{'), raw.rfind('}')) {
        (Some(s), Some(e)) if e > s => &raw[s..=e],
        _ => raw,
    };

    let mut out = String::with_capacity(body.len() + 32);
    for line in body.lines() {
        out.push_str(&repair_line(line));
        out.push('\n');
    }

    // Trailing comma before a closing brace/bracket — the other habitual defect.
    match regex::Regex::new(r",(\s*[}\]])") {
        Ok(re) => re.replace_all(&out, "$1").into_owned(),
        Err(_) => out,
    }
}

/// Fix the two per-line defects small models produce: a bare key, and a value
/// that lost its opening quote.
fn repair_line(line: &str) -> String {
    repair_value(&quote_bare_key(line))
}

/// Re-quote a key that was never quoted: `  verification: {}` → `  "verification": {}`
///
/// Observed from Qwen2.5-Coder-14B, which is otherwise more reliable than the 7B
/// but drops key quotes where the 7B drops value quotes.
fn quote_bare_key(line: &str) -> String {
    let indent_len = line.len() - line.trim_start().len();
    let (indent, rest) = line.split_at(indent_len);

    // Already quoted, or not the start of a key.
    if rest.starts_with('"') || rest.is_empty() {
        return line.to_string();
    }
    let Some(colon) = rest.find(':') else {
        return line.to_string();
    };
    let key = &rest[..colon];
    // Only a plain identifier is safe to assume was meant as a key. Anything with
    // a space, quote or brace is something else entirely — leave it alone.
    if key.is_empty()
        || !key.chars().all(|c| c.is_alphanumeric() || c == '_' || c == '-')
    {
        return line.to_string();
    }
    format!("{indent}\"{key}\"{}", &rest[colon..])
}

/// Re-quote a value that lost its opening quote: `"k": text",` → `"k": "text",`
fn repair_value(line: &str) -> String {
    let Some(idx) = line.find("\":") else {
        return line.to_string();
    };
    let after_colon = &line[idx + 2..];
    let ws = after_colon.len() - after_colon.trim_start().len();
    let split = idx + 2 + ws;
    let (head, tail) = line.split_at(split);

    let trimmed = tail.trim_end();
    let (value, trailing) = match trimmed.strip_suffix(',') {
        Some(v) => (v, ","),
        None => (trimmed, ""),
    };

    // Leave anything that already parses as a JSON value alone.
    if value.is_empty()
        || value.starts_with('"')
        || value.starts_with('{')
        || value.starts_with('[')
        || value == "true"
        || value == "false"
        || value == "null"
        || value.starts_with(|c: char| c.is_ascii_digit() || c == '-')
    {
        return line.to_string();
    }

    // The signature we repair: closes with a quote but never opened one.
    if value.ends_with('"') {
        format!("{head}\"{value}{trailing}")
    } else {
        line.to_string()
    }
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

    // Try it as written; fall back to a repaired copy rather than losing the plan.
    let val: serde_json::Value = match serde_json::from_str(clean) {
        Ok(v) => v,
        Err(strict_err) => {
            let repaired = repair_json(clean);
            serde_json::from_str(&repaired).with_context(|| {
                format!(
                    "LLM returned invalid JSON ({}), and repair did not fix it: {}",
                    strict_err,
                    &repaired[..repaired.len().min(200)]
                )
            })?
        }
    };

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

    /// Verbatim output from Qwen2.5-Coder-7B for "create a folder called
    /// test-folder": the description value lost its opening quote. Before the
    /// repair pass this threw the whole plan away after ~11s of generation.
    #[test]
    fn repairs_the_real_missing_quote_defect() {
        let json = r#"```json
{
  "goal": "create a folder called test-folder in my workspace",
  "steps": [
    {
      "description": "Check if the folder already exists.",
      "tool": "fs_list",
      "params": { "path": "/workspace" },
      "depends_on": []
    },
    {
      "description": If the folder does not exist, create it.",
      "tool": "fs_mkdir",
      "params": { "path": "/workspace/test-folder" },
      "depends_on": []
    }
  ]
}
```"#;
        let plan = parse_plan_response("goal", json).expect("repair should rescue this");
        assert_eq!(plan.steps.len(), 2);
        assert_eq!(plan.steps[1].tool_call.tool, "fs_mkdir");
        assert_eq!(
            plan.steps[1].description,
            "If the folder does not exist, create it."
        );
    }

    /// Verbatim from Qwen2.5-Coder-14B for "list the files in my workspace":
    /// every key quoted except the last one.
    #[test]
    fn repairs_the_real_unquoted_key_defect() {
        let json = r#"```json
{
  "goal": "list the files",
  "steps": [
    {
      "description": "List the files",
      "tool": "fs_list",
      "params": {
        "path": "."
      },
      "depends_on": [],
      verification: {}
    }
  ]
}
```"#;
        let plan = parse_plan_response("goal", json).expect("repair should rescue this");
        assert_eq!(plan.steps.len(), 1);
        assert_eq!(plan.steps[0].tool_call.tool, "fs_list");
    }

    #[test]
    fn key_repair_leaves_urls_and_prose_alone() {
        // A colon inside a quoted value must not be mistaken for a bare key.
        let json = r#"{"steps":[{"tool":"fs_write","description":"see https://x/y",
                     "params":{"path":"a.txt","content":"note: hello"}}]}"#;
        let plan = parse_plan_response("goal", json).unwrap();
        assert_eq!(plan.steps[0].description, "see https://x/y");
        assert_eq!(
            plan.steps[0].tool_call.params["content"].as_str().unwrap(),
            "note: hello"
        );
    }

    #[test]
    fn repairs_trailing_commas() {
        let json = r#"{"steps":[{"tool":"fs_mkdir","description":"d","params":{"path":"x"},},]}"#;
        let plan = parse_plan_response("goal", json).unwrap();
        assert_eq!(plan.steps.len(), 1);
    }

    #[test]
    fn repair_strips_prose_around_the_object() {
        let json = "Sure! Here is the plan you asked for:\n\
                    {\"steps\":[{\"tool\":\"fs_list\",\"description\":\"d\",\"params\":{}}]}\n\
                    Let me know if you want changes.";
        let plan = parse_plan_response("goal", json).unwrap();
        assert_eq!(plan.steps.len(), 1);
        assert_eq!(plan.steps[0].tool_call.tool, "fs_list");
    }

    #[test]
    fn repair_leaves_valid_values_untouched() {
        // Numbers, bools, null, objects and arrays must survive a repair pass.
        let json = r#"{"steps":[{"tool":"exec","description":"d","params":{},
                      "depends_on":[],"verification":{"exit_code":0,
                      "output_contains":"ok","output_excludes":null}}]}"#;
        let plan = parse_plan_response("goal", json).unwrap();
        assert_eq!(plan.steps.len(), 1);
        let v = plan.steps[0].verification.as_ref().unwrap();
        assert_eq!(v.exit_code, Some(0));
        assert_eq!(v.output_contains.as_deref(), Some("ok"));
    }

    #[test]
    fn repair_does_not_rescue_genuine_garbage() {
        assert!(parse_plan_response("goal", "this is not json at all").is_err());
    }

    /// The cache reuses the longest shared token prefix, so everything that varies
    /// per run has to sit at the end or the saving disappears.
    #[test]
    fn plan_prompt_keeps_the_variable_parts_last() {
        let tools = ["fs_mkdir(path) — create a directory"];
        let p = plan_prompt("MY_GOAL_MARKER", "MY_MEMORY_MARKER", &tools);

        let goal_at = p.find("MY_GOAL_MARKER").expect("goal present");
        let mem_at = p.find("MY_MEMORY_MARKER").expect("memory present");
        let rules_at = p.find("Rules:").expect("rules present");
        let tools_at = p.find("fs_mkdir(path)").expect("tool signature present");

        assert!(tools_at < rules_at, "tools must precede rules");
        assert!(rules_at < mem_at, "fixed rules must precede variable memory");
        assert!(mem_at < goal_at, "memory must precede the goal");
        assert!(
            goal_at > p.len() * 3 / 4,
            "the goal should sit near the very end so the prefix stays cacheable"
        );

        // Two different goals must share everything up to the memory section.
        let q = plan_prompt("OTHER_GOAL", "MY_MEMORY_MARKER", &tools);
        let shared = p
            .chars()
            .zip(q.chars())
            .take_while(|(a, b)| a == b)
            .count();
        assert!(
            shared > mem_at,
            "prompts for different goals should share the whole template ({shared} chars shared, \
             memory starts at {mem_at})"
        );
    }

    /// Print the rendered prompt so it can be checked against a real model.
    /// `cargo test --bins render_plan_prompt -- --nocapture --ignored`
    #[test]
    #[ignore]
    fn render_plan_prompt() {
        let tools = crate::AGENT_TOOLS;
        println!("<<<PROMPT_START>>>");
        println!(
            "{}",
            plan_prompt("create a folder called test-folder in my workspace", "", tools)
        );
        println!("<<<PROMPT_END>>>");
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
