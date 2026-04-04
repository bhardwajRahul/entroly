---
claim_id: 18a33f4c11631a840354a684
entity: sast
status: inferred
confidence: 0.75
sources:
  - sast.rs:30
  - sast.rs:40
  - sast.rs:51
  - sast.rs:64
  - sast.rs:86
  - sast.rs:103
  - sast.rs:120
  - sast.rs:1887
  - sast.rs:1912
  - sast.rs:1937
last_checked: 2026-04-04T19:51:14Z
derived_from:
  - cogops_compiler
  - sast
---

# Module: sast

**Language:** rs
**Lines of code:** 2881

## Types
- `pub enum Severity`
- `pub struct SastRule` — A single SAST rule.
- `pub struct TaintSource` — A taint source — a variable or expression that receives user-controlled data.
- `pub struct SastFinding` — A single SAST finding.
- `pub struct SastReport` — The full result of scanning a fragment.
- `pub struct CommentTracker` — Detect if a line is inside a comment block.

## Functions
- `pub fn cvss_weight(self) -> f64` — CVSS base score contribution [0.0, 4.0] (used in aggregate scoring)
- `pub(crate) fn label(self) -> &'static str`
- `fn detect_lang(source: &str) -> Option<&'static str>`
- `fn rule_applies(rule: &SastRule, lang: Option<&str>) -> bool`
- `fn is_non_code_file(source: &str) -> bool` — Detect non-code files where structural security rules are meaningless. Markdown relative links (`../guide.md`), HTML hrefs, CSS paths, etc. are NOT security vulnerabilities — scanning them produces fa
- `fn collect_taint_sources(lines: &[&str]) -> HashMap<usize, Vec<String>>` — Collect variable names that hold tainted (user-controlled) values.
- `fn extract_assignment_lhs(line: &str) -> Option<String>` — Extract the left-hand side variable name from a simple assignment. Works for: `var_name = ...`, `var_name: Type = ...`
- `fn propagate_taint(lines: &[&str], direct_sources: &HashMap<usize, Vec<String>>) -> HashSet<String>` — Given source lines and the set of taint sources, propagate taint through assignments. Returns a set of tainted variable names as of each line, plus which line they were last updated.  Algorithm: singl
- `fn line_is_tainted(line_lower: &str, tainted_vars: &HashSet<String>, direct_sources: &HashMap<usize, Vec<String>>, line_idx: usize) -> bool` — Check if a line refers to any tainted variable.
- `fn new() -> Self`
- `fn update_and_check(&mut self, line: &str) -> bool`
- `fn confidence_for_context(source: &str, line: &str, rule: &SastRule) -> f64` — Confidence modifier based on context.
- `fn compute_risk_score(findings: &[SastFinding]) -> f64`
- `pub fn scan_content(content: &str, source: &str) -> SastReport` — Scan `content` from `source` file and return a full SastReport.  This is the primary entry point. Call once per `ingest()`.
- `fn scan(code: &str, file: &str) -> SastReport`
- `fn test_hardcoded_password_critical()`
- `fn test_hardcoded_secret_redacted_in_line_content()`
- `fn test_openai_key_redacted_not_leaked()`
- `fn test_non_secret_finding_preserves_line_content()`
- `fn test_openai_key_flagged()`
- `fn test_sql_injection_taint_aware()`
- `fn test_yaml_load_without_loader()`
- `fn test_yaml_safe_load_not_flagged()`
- `fn test_md5_password_critical()`
- `fn test_os_system_flagged()`
- `fn test_debug_true_flagged()`
- `fn test_test_file_lower_confidence()`
- `fn test_pickle_loads_critical()`
- `fn test_risk_score_increases_with_severity()`
- `fn test_nosec_suppresses_finding()`
- `fn test_rust_unsafe_without_safety_comment()`
- `fn dangerous()`
- `fn test_rust_unsafe_with_safety_comment_not_flagged()`
- `fn read_aligned(ptr: *const u8) -> u8`
- `fn test_xss_innerhtml_flagged()`
- `fn test_jwt_algorithms_none()`
- `fn test_empty_file_zero_risk()`
- `fn test_taint_propagation()`
- `fn test_path_traversal_not_flagged_in_markdown()`
- `fn test_path_traversal_still_flagged_in_python()`
- `fn test_hardcoded_secret_still_flagged_in_markdown()`
- `fn test_cpp_buffer_overflow_strcpy()`
- `fn test_cpp_gets()`
- `fn test_cpp_system_injection()`
- `fn test_cpp_format_string()`
- `fn test_c_inherits_cpp_rules()`
- `fn test_swift_userdefaults_password()`
- `fn test_swift_nscoding_deserialization()`
- `fn test_cs_binaryformatter()`
- `fn test_cs_process_start()`
- `fn test_cs_sql_injection()`
- `fn test_php_eval()`
- `fn test_php_sql_injection()`
- `fn test_php_xss()`
- `fn test_php_unserialize()`
- `fn test_vue_v_html()`
- `fn test_angular_bypass_security()`
- `fn test_angular_innerhtml_binding()`
- `fn test_svelte_html_tag()`
- `fn test_html_javascript_uri()`
- `fn test_css_expression()`
- `fn test_css_javascript_url()`
- `fn test_localstorage_token()`
- `fn test_postmessage_star_origin()`
- `fn test_target_blank_without_noopener()`
- `fn test_target_blank_with_noopener_suppressed()`
- `fn test_vue_file_detects_js_rules()`
- `fn test_html_onerror_xss()`
