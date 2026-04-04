---
claim_id: 18a33f4c1082edd8027479d8
entity: knapsack
status: inferred
confidence: 0.75
sources:
  - knapsack.rs:44
  - knapsack.rs:52
  - knapsack.rs:64
  - knapsack.rs:91
  - knapsack.rs:107
  - knapsack.rs:123
  - knapsack.rs:242
  - knapsack.rs:311
  - knapsack.rs:412
  - knapsack.rs:474
last_checked: 2026-04-04T19:51:14Z
derived_from:
  - cogops_compiler
  - sast
---

# Module: knapsack

**Language:** rs
**Lines of code:** 627

## Types
- `pub struct ScoringWeights` — Weights for the four-dimensional relevance scoring.
- `pub struct KnapsackResult` — Result of a knapsack optimization run.

## Functions
- `fn default() -> Self`
- `fn sigmoid(x: f64) -> f64` — Numerically stable sigmoid σ(x). Clamped to [-500, 500] — no NaN, no Inf, no overflow.
- `fn linear_score(frag: &ContextFragment, w: &ScoringWeights, fm: f64) -> f64` — Raw linear score for a fragment, scaled by the per-fragment RL feedback multiplier.  This is the **pre-softcap** score — the same landscape used in the REINFORCE backward pass. Feedback multipliers sh
- `pub fn knapsack_optimize(` — Select the most valuable subset of fragments within the token budget.  `temperature` controls the forward-pass mode: - `temperature < 0.05` → exact 0/1 DP (optimal, used at weight convergence) - `temp
- `pub fn compute_lambda_star(` — fragments with high σ((sᵢ − λ*·tokensᵢ)/τ) are the ones the sigmoid model "expected" to be included given the actual budget consumed by IOS.  # Arguments - `scored`: (fragment_idx, linear_score) pairs
- `fn soft_bisection_select(` — Previous "additive threshold" version (p*ᵢ = σ((sᵢ − th*) / τ)) is only exact when all tokens_i are equal — a bias-inducing simplification.  Dual feasibility: find λ* ≥ 0 such that Σ p*ᵢ·tokensᵢ = B. 
- `fn knapsack_dp(` — Exact 0/1 knapsack via DP with budget quantization.  Quantize budget into Q=1000 bins to bound the DP table at N×1000. Precision loss: < 0.1% of optimal value.  Small fragments (token_count < granular
- `fn knapsack_greedy(` — Greedy approximation for very large sets (N > 2000) under hard τ. Sort by relevance/token density. Provable 0.5 optimality (Dantzig, 1957).
- `fn pinned_relevance(` — Compute total relevance for pinned fragments only.
- `fn no_feedback() -> HashMap<String, f64>`
- `fn test_knapsack_selects_optimal()`
- `fn test_soft_bisection_selects_optimal()`
- `fn test_small_fragments_not_penalized()`
- `fn test_feedback_affects_selection()`
- `fn test_soft_bisection_respects_budget()`
- `fn test_temperature_transition()`
