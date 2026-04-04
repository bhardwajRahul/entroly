---
claim_id: 18a33f4c1183decc03756acc
entity: semantic_dedup
status: inferred
confidence: 0.75
sources:
  - semantic_dedup.rs:43
  - semantic_dedup.rs:50
  - semantic_dedup.rs:72
  - semantic_dedup.rs:99
  - semantic_dedup.rs:133
  - semantic_dedup.rs:142
  - semantic_dedup.rs:168
  - semantic_dedup.rs:173
  - semantic_dedup.rs:189
  - semantic_dedup.rs:203
last_checked: 2026-04-04T19:51:14Z
derived_from:
  - cogops_compiler
  - sast
---

# Module: semantic_dedup

**Language:** rs
**Lines of code:** 263

## Types
- `pub struct DeduplicationResult` — Convenience: deduplicate and return count stats.

## Functions
- `fn content_overlap(a: &str, b: &str) -> f64` — Compute content overlap between two fragments using: 50% n-gram overlap (word trigrams) 50% identifier overlap  Returns [0, 1] where 1.0 = identical information.
- `fn trigram_jaccard(a: &str, b: &str) -> f64` — Symmetric trigram Jaccard similarity.
- `fn identifier_jaccard(a: &str, b: &str) -> f64` — Symmetric identifier Jaccard similarity.
- `pub fn semantic_deduplicate(` — Returns the subset of indices to keep — fragments that each contribute at least `threshold` marginal information.  # Arguments * `fragments` - all fragments in the engine * `sorted_indices` - indices 
- `pub fn semantic_deduplicate_with_stats(`
- `fn make_frag(id: &str, content: &str, tokens: u32) -> ContextFragment`
- `fn test_redundant_fragments_deduplicated()`
- `fn test_unique_fragments_kept()`
- `fn test_three_way_dedup()`
- `fn test_dedup_stats()`
- `fn test_empty_input()`
- `fn test_threshold_sensitivity()`
