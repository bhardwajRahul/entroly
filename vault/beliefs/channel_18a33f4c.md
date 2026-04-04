---
claim_id: 18a33f4c0f86b45001784050
entity: channel
status: inferred
confidence: 0.75
sources:
  - channel.rs:41
  - channel.rs:59
  - channel.rs:76
  - channel.rs:100
  - channel.rs:183
  - channel.rs:203
  - channel.rs:319
  - channel.rs:359
  - channel.rs:394
  - channel.rs:412
last_checked: 2026-04-04T19:51:14Z
derived_from:
  - cogops_compiler
  - sast
---

# Module: channel

**Language:** rs
**Lines of code:** 1330

## Types
- `pub struct ContradictionReport` — Result of contradiction scan.

## Functions
- `fn trigram_hashes(text: &str) -> Vec<u64>` — Extract character 3-gram hashes from text. Uses FNV-1a-style mixing: fast, zero-alloc per gram, good distribution.
- `fn build_trigram_set(frags: &[ContextFragment], indices: &[usize]) -> HashSet<u64>` — Build the set of trigram hashes from selected fragments' content. This is the "information already in the channel."
- `fn marginal_gain(` — Marginal information gain of adding candidate x to selection S.  ΔI(x | S) = entropy(x) × (1 − overlap(x, S)) × dep_bonus  Where overlap = fraction of x's trigrams already in S's trigram set. Submodul
- `pub fn channel_trailing_pass(` — Channel-Aware Trailing Pass: fill the KKT token gap using marginal information gain instead of greedy density.  After IOS/knapsack selects fragments, a token gap Δ = budget − used remains. This functi
- `pub fn attention_weight(position: usize, total: usize) -> f64` — LLM attention weight at position p in context of length L.  α(p, L) = 0.4·exp(−p/(0.15·L)) + 0.4·exp(−(L−p−1)/(0.15·L)) + 0.2  U-shaped: peaks at start (primacy) and end (recency), valley in the middl
- `pub fn semantic_interleave(` — Attention-aware semantic interleaving of selected fragments.  Algorithm: 1. Topological sort for causal ordering (defs before refs) 2. Within each level, sort by importance descending 3. Place level-0
- `pub fn information_reward(` — Attention-weighted utilization reward.  R = η × (1 + sufficiency_bonus)  η = Σᵢ(util_i × entropy_i × α_i) / Σᵢ(entropy_i × α_i)  This is "information received by decoder" / "information sent by encode
- `pub fn modulated_reward(success: bool, sufficiency: f64) -> f64` —  Success: R ∈ [0.5, 1.0] — higher when sufficiency is high Failure: R ∈ [−1.5, −0.5] — penalizes more when sufficiency is low  Better credit assignment than flat ±1: "failed with bad context" is a str
- `pub fn contradiction_guard(` —  Two fragments contradict if: 1. Their source paths are structurally similar (SimHash of source ≥ threshold) 2. Their content is divergent (content SimHash distance > threshold)  The fragment with low
- `pub fn bookend_calibrate(` —  Within each causal level, reorders fragments so that the most important ones occupy the attention-peak positions (start and end of that level's span within the full sequence).  `ordered_indices`: out
- `fn frag(id: &str, content: &str, tokens: u32, source: &str) -> ContextFragment`
- `fn test_trailing_pass_fills_gap()`
- `fn test_trailing_pass_zero_gap_returns_empty()`
- `fn test_trailing_pass_respects_budget()`
- `fn test_marginal_gain_is_submodular()`
- `fn test_attention_u_shaped()`
- `fn test_attention_single_fragment()`
- `fn test_interleave_preserves_all_indices()`
- `fn test_interleave_respects_causal_order()`
- `fn test_interleave_two_fragments()`
- `fn test_information_reward_high_util()`
- `fn test_information_reward_zero_util()`
- `fn test_modulated_reward_bounds()`
- `fn test_modulated_reward_credit_assignment()`
- `fn test_trailing_pass_performance_1000_frags()`
- `fn test_trailing_pass_empty_fragment_list()`
- `fn test_trailing_pass_all_already_selected()`
- `fn test_trailing_pass_all_candidates_oversized()`
- `fn test_trailing_pass_single_token_gap()`
- `fn test_trailing_pass_zero_entropy_fragments()`
- `fn test_trailing_pass_no_duplicate_indices()`
- `fn test_trailing_pass_incremental_submodularity()`
- `fn test_trailing_pass_with_very_short_content()`
- `fn test_interleave_empty()`
- `fn test_interleave_single()`
- `fn test_interleave_cyclic_dependencies()`
- `fn test_interleave_deep_dependency_chain()`
- `fn test_interleave_star_dependency()`
- `fn test_interleave_mismatched_relevances()`
- `fn test_attention_large_context_no_nan_inf()`
- `fn test_attention_zero_total()`
- `fn test_attention_symmetry()`
- `fn test_information_reward_empty_inputs()`
- `fn test_information_reward_single_fragment()`
- `fn test_information_reward_no_nan_with_zero_entropy()`
- `fn test_modulated_reward_extreme_sufficiency()`
- `fn test_modulated_reward_is_always_finite()`
- `fn test_full_pipeline_trailing_then_interleave()`
- `fn test_contradiction_guard_no_contradictions()`
- `fn test_contradiction_guard_same_source_different_content()`
- `fn test_contradiction_guard_single_fragment()`
- `fn test_contradiction_guard_empty()`
- `fn test_contradiction_guard_different_sources_no_eviction()`
- `fn test_bookend_preserves_all_indices()`
- `fn test_bookend_small_input()`
- `fn test_bookend_empty()`
