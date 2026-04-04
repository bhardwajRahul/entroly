---
claim_id: 18a33f4c1144b09403363c94
entity: resonance
status: inferred
confidence: 0.75
sources:
  - resonance.rs:61
  - resonance.rs:64
  - resonance.rs:73
  - resonance.rs:78
  - resonance.rs:91
  - resonance.rs:100
  - resonance.rs:102
  - resonance.rs:105
  - resonance.rs:108
  - resonance.rs:124
last_checked: 2026-04-04T19:51:14Z
derived_from:
  - cogops_compiler
  - sast
---

# Module: resonance

**Language:** rs
**Lines of code:** 712

## Types
- `pub struct PairKey` — A symmetric pair key — order-independent. Always stores (min, max) to ensure R[a][b] == R[b][a].
- `pub struct PairKeyMapVisitor`
- `pub struct ResonanceEntry` — A single resonance entry tracking pairwise interaction strength.
- `pub struct ResonanceMatrix` — The Resonance Matrix — learns which fragment pairs produce synergistic LLM outputs through outcome tracking.
- `pub struct ConsolidationResult` — Fragment Consolidation Result
- `pub struct CoverageEstimate` — Coverage estimation result.

## Functions
- `fn new(a: &str, b: &str) -> Self`
- `fn to_key_string(&self) -> String` — Convert to a string key for JSON serialization.
- `fn from_key_string(s: &str) -> Option<Self>` — Parse from a string key.
- `pub fn serialize<S>(map: &HashMap<PairKey, super::ResonanceEntry>, serializer: S) -> Result<S::Ok, S::Error>`
- `pub fn deserialize<'de, D>(deserializer: D) -> Result<HashMap<PairKey, super::ResonanceEntry>, D::Error>`
- `fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result`
- `fn visit_map<M: MapAccess<'de>>(self, mut access: M) -> Result<Self::Value, M::Error>`
- `fn default() -> Self`
- `pub fn new() -> Self`
- `pub fn record_outcome(&mut self, fragment_ids: &[String], reward: f64, current_turn: u32)` — Record outcome for a set of co-selected fragments.  For N selected fragments, updates C(N,2) = N*(N-1)/2 pairs. With typical N ≈ 10-20, this is 45-190 updates — negligible.
- `pub fn resonance_bonus(&self, candidate_id: &str, selected_ids: &[&str]) -> f64` — Compute the resonance bonus for a candidate fragment given the already-selected set.  Returns the mean pairwise resonance strength, weighted by confidence: bonus = Σ_{s ∈ selected} w(s) · R[candidate]
- `pub fn batch_resonance_bonuses(` — Compute resonance bonuses for all candidate fragments at once.  Returns a map from fragment_id → resonance_bonus. Candidates not in the resonance matrix get 0.0 (cold start).
- `pub fn decay_tick(&mut self)` — Apply per-turn decay to all resonance entries. Removes entries that have decayed below threshold (|strength| < 0.001).
- `pub fn len(&self) -> usize` — Total number of tracked pairs.
- `pub fn is_empty(&self) -> bool` — Whether the matrix is empty (no learned resonances yet).
- `pub fn top_pairs(&self, top_k: usize) -> Vec<(String, String, f64, u32)>` — Get the strongest resonance pairs (for diagnostics). Returns up to `top_k` pairs sorted by |strength| descending.
- `pub fn mean_strength(&self) -> f64` — Mean absolute resonance strength (health metric). Low (< 0.01) = cold start, no patterns learned yet. High (> 0.5) = strong interaction patterns detected.
- `pub fn find_consolidation_groups(` —  By consolidating based on *outcome* (which version led to better LLM outputs), we're doing Maxwell's Demon: reducing entropy by keeping the thermodynamically "useful" variant.  # Complexity: O(N²) pa
- `pub fn estimate_coverage(` —  - Zero overlap: degenerate case, falls back to union size estimate.  # Confidence Estimation  The coefficient of variation of the Chapman estimator is approximately: CV = sqrt((N₁ + 1)(N₂ + 1)(N₁ − m
- `fn test_resonance_symmetric()`
- `fn test_resonance_positive_reinforcement()`
- `fn test_resonance_negative_suppression()`
- `fn test_resonance_cold_start_zero()`
- `fn test_resonance_decay()`
- `fn test_resonance_multiple_partners()`
- `fn test_consolidation_near_duplicates()`
- `fn test_consolidation_pinned_exempt()`
- `fn test_coverage_high_overlap()`
- `fn test_coverage_low_overlap()`
- `fn test_coverage_zero_selected()`
- `fn test_coverage_monotone_in_overlap()`
- `fn test_chapman_estimator_accuracy()`
