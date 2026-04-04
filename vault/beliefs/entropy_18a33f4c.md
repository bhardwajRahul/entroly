---
claim_id: 18a33f4c101e0378020f8f78
entity: entropy
status: inferred
confidence: 0.75
sources:
  - entropy.rs:26
  - entropy.rs:54
  - entropy.rs:80
  - entropy.rs:122
  - entropy.rs:171
  - entropy.rs:187
  - entropy.rs:216
  - entropy.rs:240
  - entropy.rs:254
  - entropy.rs:278
last_checked: 2026-04-04T19:51:14Z
derived_from:
  - cogops_compiler
  - sast
---

# Module: entropy

**Language:** rs
**Lines of code:** 580


## Functions
- `pub fn shannon_entropy(text: &str) -> f64` — Character-level Shannon entropy in bits per character.  Uses a 256-element byte histogram for O(n) computation with virtually zero allocation overhead.
- `pub fn normalized_entropy(text: &str) -> f64` — Normalize Shannon entropy to [0, 1]. Max entropy for source code is empirically ~6.0 bits/char.
- `pub fn renyi_entropy_2(text: &str) -> f64` —  Computational advantage: requires only Σ p² (no per-symbol log), making it ~30% faster than Shannon on large fragments.  Used as a secondary signal in the IOS knapsack: fragments with high Shannon bu
- `pub fn renyi_entropy_alpha(scores: &[f64], alpha: f64) -> f64` — The input `scores` do NOT need to be normalized — this function normalizes them to a probability distribution internally.  Used by EGSC's admission gate: given per-fragment entropy scores s₁, ..., sₖ,
- `pub fn renyi_max(n: usize) -> f64` — Maximum possible Rényi entropy for n elements: H₂_max = log₂(n).  When all pᵢ = 1/n (uniform distribution), H₂ = log₂(n). Used to normalize EGSC admission threshold to [0, 1] scale.
- `pub fn entropy_divergence(text: &str) -> f64` — This measures "entropy inflation" — when Shannon entropy is high but collision entropy is low, the fragment has many unique-but-rare symbols (e.g., binary-encoded data, UUID strings, minified code).  
- `pub(crate) fn bits_per_byte(text: &str) -> f64` — Typical application code: BPB ≈ 0.55–0.65 Config / boilerplate:     BPB ≈ 0.30–0.45 Minified / compressed:    BPB ≈ 0.85–0.95  Used in the autotune composite score to reward configs that select high-i
- `pub(crate) fn bpb_quality(text: &str, redundancy: f64) -> f64` — BPB-weighted quality score: 60% density + 40% uniqueness.
- `pub fn boilerplate_ratio(text: &str) -> f64` — Boilerplate pattern matcher. Returns the fraction of non-empty lines matching common boilerplate.  Hardcoded patterns for speed (no regex dependency): - import/from imports - pass/... - dunder methods
- `fn is_boilerplate(trimmed: &str) -> bool` — Fast boilerplate check without regex.
- `pub fn cross_fragment_redundancy(` — argument patterns. The "standard" measure. - 4-grams (n=4) catch near-verbatim duplication: almost identical code blocks. Discriminative for long files where n=3 is too permissive.  **Adaptive weights
- `fn ngram_redundancy(` — Compute single-scale n-gram overlap ratio against a set of other fragments. Parallelises over others when len > 10 (Rayon).
- `pub fn information_score(` — Compute the final information density score.  Combines: 40% Shannon entropy (normalized) 30% Boilerplate penalty (1 - ratio) 30% Uniqueness (1 - adaptive multi-scale redundancy)
- `fn test_entropy_identical_chars()`
- `fn test_entropy_increases_with_diversity()`
- `fn test_boilerplate_detection()`
- `fn test_redundancy_identical()`
- `fn test_multiscale_short_fragment_uses_bigrams()`
- `fn test_multiscale_long_fragment_discriminates()`
- `fn test_bpb_empty()`
- `fn test_bpb_uniform_char()`
- `fn test_bpb_range()`
- `fn test_bpb_boilerplate_lower()`
- `fn test_bpb_quality_combines_density_and_uniqueness()`
- `fn test_renyi_entropy_empty()`
- `fn test_renyi_entropy_uniform()`
- `fn test_renyi_leq_shannon()`
- `fn test_entropy_divergence_nonnegative()`
- `fn test_entropy_divergence_low_for_code()`
- `fn test_noise_penalty_in_information_score()`
