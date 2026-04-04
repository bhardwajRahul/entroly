---
claim_id: 18a33f4c10584a380249d638
entity: health
status: inferred
confidence: 0.75
sources:
  - health.rs:44
  - health.rs:56
  - health.rs:69
  - health.rs:79
  - health.rs:87
  - health.rs:98
  - health.rs:108
  - health.rs:119
  - health.rs:130
  - health.rs:139
last_checked: 2026-04-04T19:51:14Z
derived_from:
  - cogops_compiler
  - sast
---

# Module: health

**Language:** rs
**Lines of code:** 850

## Types
- `pub struct ClonePair` — A pair of fragments that are near-duplicates (code clones).
- `pub enum CloneType` — Clone types (after Koschke et al., adapted for SimHash detection).
- `pub struct DeadSymbol` — A symbol that appears to be defined but never referenced across all known fragments.
- `pub struct GodFile` — A "god file" — a fragment that too many others depend on.
- `pub struct ArchViolation` — An architectural layer violation — a lower-layer importing a higher-layer module.
- `pub struct NamingIssue` — Naming consistency issue: files that break established naming conventions.
- `pub struct HealthReport` — Full codebase health report.

## Functions
- `pub(crate) fn label(self) -> &'static str`
- `fn from_hamming(dist: u32) -> Option<CloneType>`
- `pub(crate) fn label(self) -> &'static str`
- `fn detect_clones(fragments: &[&ContextFragment]) -> Vec<ClonePair>`
- `fn find_dead_symbols(`
- `fn is_generic_symbol(sym: &str) -> bool` — Skip symbols that are so common they're meaningless as dead-code indicators.
- `fn find_god_files(`
- `fn classify_layer(source: &str) -> Option<(usize, &'static str)>`
- `fn find_arch_violations(fragments: &[&ContextFragment]) -> Vec<ArchViolation>`
- `fn find_layer_in_import(import_line: &str) -> Option<(usize, &'static str)>`
- `fn find_naming_issues(fragments: &[&ContextFragment]) -> Vec<NamingIssue>`
- `fn compute_code_health(`
- `fn health_grade(score: f64) -> &'static str`
- `pub fn analyze_health(`
- `fn basename(path: &str) -> &str`
- `fn to_snake_case(s: &str) -> String`
- `fn to_pascal_case(s: &str) -> String`
- `fn make_frag(id: &str, content: &str, source: &str) -> ContextFragment`
- `fn test_exact_clone_detected()`
- `fn test_same_file_not_cloned()`
- `fn test_god_file_detection()`
- `fn test_code_health_perfect_score()`
- `fn test_code_health_decreases_with_clones()`
- `fn test_arch_violation_domain_imports_api()`
- `fn test_health_grade_mapping()`
- `fn test_snake_case_conversion()`
- `fn test_empty_codebase_no_panic()`
- `fn test_naming_issue_python_camel_case()`
