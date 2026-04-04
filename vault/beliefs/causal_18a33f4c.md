---
claim_id: 18a33f4c0f7143a40162cfa4
entity: causal
status: inferred
confidence: 0.75
sources:
  - causal.rs:98
  - causal.rs:112
  - causal.rs:140
  - causal.rs:154
  - causal.rs:175
  - causal.rs:195
  - causal.rs:201
  - causal.rs:223
  - causal.rs:342
  - causal.rs:364
last_checked: 2026-04-04T19:51:14Z
derived_from:
  - cogops_compiler
  - sast
---

# Module: causal

**Language:** rs
**Lines of code:** 1065

## Types
- `pub struct CausalTrace` — A single optimization+feedback event.
- `pub struct InterventionEstimate` — Per-fragment interventional vs observational statistics.  Tracks outcomes separately for when the fragment was: - Naturally selected by the policy (observational) - Randomly included via exploration (
- `pub struct TemporalLinkEntry` — Temporal causal link: source at T−1 → target at T.  Measures how much selecting `source` at T−1 improves outcomes when `target` is selected at T (transfer entropy approximation).
- `pub struct CausalStats` — Statistics for observability.
- `pub struct CausalContextGraph` — Causal Context Graph — learns fragment causal effects via natural experiments.  Uses the exploration mechanism as an instrumental variable to separate true causal effects from selection bias, discover

## Functions
- `fn default() -> Self`
- `pub fn new() -> Self`
- `pub fn record_trace(` — Record an optimization+feedback trace and update all causal estimates.  # Arguments - `turn`: current engine turn - `query_hash`: SimHash of the query string - `selected_ids`: all fragments selected (
- `pub fn gravity_bonuses(&self, candidate_ids: &[&str]) -> HashMap<String, f64>` — Compute information gravity bonuses for candidate fragments.  Returns fragment_id → gravity_bonus ∈ [0, 1). Based on conformal metric distortion: high causal mass fragments appear "closer" to all quer
- `pub fn temporal_bonuses(` — Compute temporal chain bonuses for candidates given previous selection.  For each candidate B, returns the confidence-weighted mean temporal effect across all A→B links where A was in `prev_selected_i
- `pub fn decay_tick(&mut self, current_turn: u32)` — Per-turn maintenance: evict stale temporal links, recompute fields.
- `pub fn is_empty(&self) -> bool`
- `pub fn tracked_fragments(&self) -> usize` — Number of fragments with causal estimates.
- `pub fn interventional_fragments(&self) -> usize` — Number of fragments with interventional data (from exploration).
- `pub fn temporal_link_count(&self) -> usize` — Number of active temporal links.
- `pub fn gravity_source_count(&self) -> usize` — Number of fragments with positive causal mass (gravity sources).
- `pub fn mean_causal_mass(&self) -> f64` — Mean causal mass across all tracked fragments.
- `pub fn top_causal_fragments(&self, n: usize) -> Vec<(String, f64, f64, f64)>` — Top fragments ranked by causal effect (for observability).  Returns: (fragment_id, causal_effect, confounding_bias, confidence).
- `pub fn top_temporal_chains(&self, n: usize) -> Vec<(String, String, f64, f64)>` — Top temporal chains ranked by effect strength (for observability).  Returns: (source_id, target_id, temporal_effect, confidence).
- `pub fn stats(&self) -> CausalStats` — Full statistics for dashboard/CLI observability.
- `fn recompute_causal_effects(&mut self)` — Recompute causal_effect and confounding_bias for all fragments.  Causal effect = E[Y|do(include f)] − base_rate Uses interventional data (exploration = natural experiment)  Confounding bias = E[Y|obse
- `fn recompute_temporal_effects(&mut self)` — Recompute temporal effects for all links.  TE(A→B) ≈ E[Y | B@T, A@T−1] − E[Y | B@T, ¬A@T−1]  Uses per-fragment total stats as the marginal denominator: marginal = (total_Y_B − conditional_Y_AB) / (tot
- `fn recompute_gravity_field(&mut self)` — Recompute the information gravity field from causal estimates.  causal_mass(f) = max(0, causal_effect(f)) × confidence(f)  Only positive causal effects generate gravity — fragments that causally HURT 
- `fn evict_temporal_links(&mut self, current_turn: u32)` — Evict the stalest temporal links to stay under capacity.
- `fn make_ids(names: &[&str]) -> Vec<String>`
- `fn test_new_graph_is_empty()`
- `fn test_single_trace_updates_estimates()`
- `fn test_interventional_vs_observational_separation()`
- `fn test_confounding_bias_detection()`
- `fn test_temporal_links_created()`
- `fn test_temporal_effect_computation()`
- `fn test_gravity_bonuses_positive_effect()`
- `fn test_gravity_zero_for_negative_effect()`
- `fn test_temporal_bonuses_computation()`
- `fn test_decay_tick_removes_stale_links()`
- `fn test_base_rate_tracking()`
- `fn test_circular_buffer_capacity()`
- `fn test_confidence_increases_with_trials()`
- `fn test_top_causal_fragments()`
- `fn test_stats_consistency()`
- `fn test_empty_selected_ignored()`
- `fn test_nan_outcome_ignored()`
- `fn test_self_links_excluded()`
- `fn test_serde_roundtrip()`
- `fn test_temporal_eviction_under_capacity()`
