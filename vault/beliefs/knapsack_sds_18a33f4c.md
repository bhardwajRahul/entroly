---
claim_id: 18a33f4c1098b280028a3e80
entity: knapsack_sds
status: inferred
confidence: 0.75
sources:
  - knapsack_sds.rs:38
  - knapsack_sds.rs:50
  - knapsack_sds.rs:56
  - knapsack_sds.rs:63
  - knapsack_sds.rs:71
  - knapsack_sds.rs:82
  - knapsack_sds.rs:92
  - knapsack_sds.rs:113
  - knapsack_sds.rs:135
  - knapsack_sds.rs:173
last_checked: 2026-04-04T19:51:14Z
derived_from:
  - cogops_compiler
  - sast
---

# Module: knapsack_sds

**Language:** rs
**Lines of code:** 636

## Types
- `pub enum Resolution` — Resolution level for a selected fragment.
- `pub struct InfoFactors` — Configurable information retention factors for each resolution level. These control the value/cost trade-off in multi-resolution knapsack. Tunable via tuning_config.json → autotune daemon.
- `pub struct Candidate` — A candidate item for the SDS+MRK optimizer. Each fragment generates 1-3 candidates (one per resolution).
- `pub struct SdsResult` — Result of the IOS selection.

## Functions
- `fn default() -> Self`
- `fn info_factor(&self, factors: &InfoFactors) -> f64` — Information retention factor for this resolution level.
- `pub fn as_str(&self) -> &'static str`
- `fn diversity_factor(candidate_hash: u64, selected_hashes: &[u64]) -> f64` —  Similarity is estimated from SimHash Hamming distance: sim(a, b) = 1 - hamming(a, b) / 64  When the selected set is empty, diversity = 1.0 (no penalty).  Returns a value in [0, 1] where: 1.0 = comple
- `fn compute_pairwise_diversity(hashes: &[u64]) -> f64` — Compute average pairwise diversity from SimHash fingerprints.  diversity = mean over all pairs of (hamming_distance / 64). Returns 1.0 when ≤ 1 hash (trivially diverse).
- `pub fn ios_select(` — 2. Separate pinned fragments (always full resolution, always included) 3. Greedy loop (greedy-by-density with diversity penalty): - compute marginal_value = base_value × diversity_factor(hash) - selec
- `fn empty_feedback() -> HashMap<String, f64>`
- `fn default_factors() -> InfoFactors`
- `fn make_frag(id: &str, content: &str, tokens: u32, source: &str) -> ContextFragment`
- `fn test_empty_fragments()`
- `fn test_single_fragment_selected()`
- `fn test_pinned_always_included()`
- `fn test_diversity_penalizes_duplicates()`
- `fn test_multi_resolution_fits_more()`
- `fn test_budget_respected()`
- `fn test_feedback_multiplier_affects_selection()`
- `fn test_reference_resolution_very_cheap()`
- `fn test_diversity_score_range()`
- `fn test_resolution_preference_by_budget()`
- `fn test_fast_path_selects_all_when_budget_generous()`
