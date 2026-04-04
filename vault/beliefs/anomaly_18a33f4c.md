---
claim_id: 18a33f4c0f3459800125e580
entity: anomaly
status: inferred
confidence: 0.75
sources:
  - anomaly.rs:44
  - anomaly.rs:55
  - anomaly.rs:65
  - anomaly.rs:85
  - anomaly.rs:93
  - anomaly.rs:106
  - anomaly.rs:119
  - anomaly.rs:264
  - anomaly.rs:275
  - anomaly.rs:282
last_checked: 2026-04-04T19:51:14Z
derived_from:
  - cogops_compiler
  - sast
---

# Module: anomaly

**Language:** rs
**Lines of code:** 378

## Types
- `pub enum AnomalyType`
- `pub struct EntropyAnomaly` — A single entropy anomaly detected in the codebase.
- `pub struct AnomalyReport` — Full anomaly scan report.

## Functions
- `pub fn label(&self) -> &'static str`
- `fn median(sorted: &[f64]) -> f64` — Compute the median of a sorted slice.
- `fn directory_of(source: &str) -> String` — Extract directory from a source path. "src/handlers/auth.rs" → "src/handlers" "auth.rs" → ""
- `pub fn scan_anomalies(fragments: &[&ContextFragment]) -> AnomalyReport` — Scan all fragments for entropy anomalies using robust Z-scores.  Groups fragments by directory, computes MAD-based Z-scores within each group, and flags fragments with |z| > 2.5.
- `fn basename(path: &str) -> &str`
- `fn make_frag(id: &str, source: &str, entropy: f64, content: &str) -> ContextFragment`
- `fn test_spike_detection()`
- `fn test_drop_detection()`
- `fn test_small_group_skipped()`
- `fn test_uniform_group_no_anomalies()`
- `fn test_cross_directory_isolation()`
