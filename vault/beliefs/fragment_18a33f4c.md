---
claim_id: 18a33f4c102f57cc0220e3cc
entity: fragment
status: inferred
confidence: 0.75
sources:
  - fragment.rs:16
  - fragment.rs:67
  - fragment.rs:104
  - fragment.rs:148
  - fragment.rs:161
  - fragment.rs:179
  - fragment.rs:192
  - fragment.rs:212
last_checked: 2026-04-04T19:51:14Z
derived_from:
  - cogops_compiler
  - sast
---

# Module: fragment

**Language:** rs
**Lines of code:** 236

## Types
- `pub struct ContextFragment` — A single piece of context (code snippet, file content, tool result, etc.)

## Functions
- `pub fn new(fragment_id: String, content: String, token_count: u32, source: String) -> Self`
- `pub fn compute_relevance(` — Compute composite relevance score for a fragment.  Direct port of ebbiforge-core ContextScorer::score() but with entropy replacing emotion as the fourth dimension.  `feedback_multiplier` comes from Fe
- `pub fn softcap(x: f64, cap: f64) -> f64` — Logit softcap: `c · tanh(x / c)`.  Gemini-style bounded scoring. When `cap ≤ 0`, falls back to `min(x, 1)`.
- `pub fn apply_ebbinghaus_decay(` — Apply Ebbinghaus forgetting curve decay to all fragments.  recency(t) = exp(-λ · Δt) where λ = ln(2) / half_life  Same math as ebbiforge-core HippocampusEngine.
- `fn test_ebbinghaus_half_life()`
- `fn test_relevance_scoring()`
- `fn test_softcap_properties()`
