---
claim_id: 18a33f4c0fbb7ba401ad07a4
entity: cogops
status: inferred
confidence: 0.75
sources:
  - cogops.rs:26
  - cogops.rs:43
  - cogops.rs:63
  - cogops.rs:72
  - cogops.rs:84
  - cogops.rs:91
  - cogops.rs:101
  - cogops.rs:112
  - cogops.rs:124
  - cogops.rs:136
last_checked: 2026-04-04T19:51:14Z
derived_from:
  - cogops_compiler
  - sast
---

# Module: cogops

**Language:** rs
**Lines of code:** 1839

## Types
- `pub enum EpistemicIntent`
- `pub enum EpistemicFlow`
- `pub enum RiskLevel`
- `pub struct CodeEntity`
- `pub struct BeliefArtifact`
- `pub struct RoutingDecision`
- `pub struct Contradiction`
- `pub struct ChangeSet`
- `pub struct ReviewFinding`
- `pub struct CogOpsEngine`

## Functions
- `fn as_str(&self) -> &'static str`
- `fn as_str(&self) -> &'static str`
- `fn as_str(&self) -> &'static str`
- `pub fn classify_intent(query: &str) -> EpistemicIntent`
- `pub fn assess_risk(query: &str) -> RiskLevel` — Assess risk level from query content.
- `pub fn extract_entities(content: &str, file_path: &str) -> Vec<CodeEntity>`
- `fn extract_python_entities(content: &str, file_path: &str) -> Vec<CodeEntity>`
- `fn extract_rust_entities(content: &str, file_path: &str) -> Vec<CodeEntity>`
- `fn extract_js_entities(content: &str, file_path: &str) -> Vec<CodeEntity>`
- `fn get_next_docstring(lines: &[&str], after: usize) -> String`
- `fn get_rust_doc(lines: &[&str], before: usize) -> String`
- `fn ensure_vault(vault_path: &Path)`
- `fn generate_claim_id() -> String`
- `fn write_belief_artifact(vault_path: &Path, artifact: &BeliefArtifact) -> Result<String, String>`
- `fn read_all_beliefs(vault_path: &Path) -> Vec<BeliefArtifact>`
- `fn parse_belief_frontmatter(content: &str) -> Option<BeliefArtifact>`
- `fn collect_markdown_files(dir: &Path, out: &mut Vec<PathBuf>)`
- `fn normalize_rel_path(path: &str) -> String`
- `fn source_pointer_path(source: &str) -> String`
- `fn belief_matches_changed_file(belief: &BeliefArtifact, changed_file: &str) -> bool`
- `fn mark_belief_stale_content(content: &str) -> Option<String>`
- `fn mark_beliefs_stale(vault_path: &Path, changed_files: &[String]) -> Vec<String>`
- `fn build_belief_artifact(root: &Path, fpath: &Path, content: &str) -> Option<(BeliefArtifact, usize)>`
- `fn compile_source_paths(`
- `pub fn detect_contradictions(beliefs: &[BeliefArtifact]) -> Vec<Contradiction>`
- `pub fn compute_blast_radius(beliefs: &[BeliefArtifact], changed_files: &[&str]) -> (Vec<String>, Vec<String>, &'static str)`
- `pub fn parse_diff(diff_text: &str, commit_msg: &str) -> ChangeSet`
- `pub fn review_diff(diff_text: &str) -> Vec<ReviewFinding>` — Review a diff for common issues (hardcoded secrets, TODOs, unsafe patterns).
- `pub fn select_flow(`
- `pub fn new(vault_path: String, miss_threshold: u32, freshness_hours: f64, min_confidence: f64) -> Self`
- `pub fn classify_intent(&self, query: &str) -> String` — Classify intent from a query string.
- `pub fn route(&mut self, query: &str, is_event: bool, event_type: &str) -> PyResult<PyObject>` — Route a query through the epistemic routing matrix.
- `pub fn extract_entities(&self, content: &str, file_path: &str) -> PyResult<PyObject>` — Extract entities from source code.
- `pub fn compile_beliefs(&mut self, directory: &str, max_files: usize) -> PyResult<PyObject>` — Compile a directory of source files into belief artifacts.
- `pub fn verify_beliefs(&self) -> PyResult<PyObject>` — Run full verification pass on all beliefs.
- `pub fn blast_radius(&self, changed_files: Vec<String>) -> PyResult<PyObject>` — Compute blast radius for changed files.
- `pub fn process_change(&self, diff_text: &str, commit_msg: &str, pr_title: &str) -> PyResult<PyObject>` — Parse and analyze a diff.
- `pub fn write_belief(&self, entity: &str, title: &str, body: &str,` — Write a belief artifact to the vault.
- `pub fn vault_status(&self) -> PyResult<PyObject>` — Get vault status and coverage index.
- `pub fn create_skill(&self, entity_key: &str, failing_queries: Vec<String>) -> PyResult<PyObject>` — Create a new skill from a gap report.
- `pub fn list_skills(&self) -> PyResult<PyObject>` — List all skills.
- `pub fn coverage_gaps(&self, directory: &str) -> PyResult<PyObject>` — Find source files with no corresponding belief in the vault.
- `pub fn refresh_beliefs(&self, changed_files: Vec<String>) -> PyResult<PyObject>` — Mark beliefs as stale after file changes.
- `pub fn benchmark_skill(&self, skill_id: &str) -> PyResult<PyObject>` — Benchmark a skill by running its test cases in a subprocess.
- `pub fn promote_skill(&self, skill_id: &str) -> PyResult<PyObject>` — Promote or prune a skill based on fitness score.
- `pub fn execute_flow(&mut self, query: &str, diff_text: &str, is_event: bool, event_type: &str) -> PyResult<PyObject>` — Execute a full canonical epistemic flow end-to-end.
- `fn extract_entity_key(query: &str) -> String`
- `fn chrono_iso() -> String`
- `fn days_to_ymd(mut days: u64) -> (u64, u64, u64)`
- `fn is_leap(y: u64) -> bool`
- `fn extract_fm_value(content: &str, key: &str) -> Option<String>` — Extract a value from YAML-like frontmatter.
- `fn collect_source_files(dir: &Path, skip: &HashSet<&str>, exts: &HashSet<&str>, out: &mut Vec<PathBuf>, max: usize)`
