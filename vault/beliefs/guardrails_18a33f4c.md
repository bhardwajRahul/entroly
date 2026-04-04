---
claim_id: 18a33f4c103fac4402313844
entity: guardrails
status: inferred
confidence: 0.75
sources:
  - guardrails.rs:24
  - guardrails.rs:36
  - guardrails.rs:142
  - guardrails.rs:207
  - guardrails.rs:224
  - guardrails.rs:237
  - guardrails.rs:274
  - guardrails.rs:292
  - guardrails.rs:324
  - guardrails.rs:345
last_checked: 2026-04-04T19:51:14Z
derived_from:
  - cogops_compiler
  - sast
---

# Module: guardrails

**Language:** rs
**Lines of code:** 570

## Types
- `pub enum Criticality` — Criticality level — overrides entropy and relevance scoring.
- `pub enum TaskType` — Adaptive budget allocation based on task type.  Different tasks need different context volumes: - Bug tracing: LARGE budget (need call chains, logs, history) - Refactoring: MEDIUM budget (need interfa
- `pub struct FeedbackTracker` — Feedback loop: record which fragments influenced a successful output.  Extended with per-fragment Welford variance tracking for RAVEN-UCB adaptive exploration (arXiv:2506.02933, June 2025).

## Functions
- `pub fn file_criticality(path: &str) -> Criticality` — Check if a file path matches critical file patterns.
- `pub fn has_safety_signal(content: &str) -> bool` — Check content for safety signals that must never be stripped.
- `pub(crate) fn criticality_boost(criticality: Criticality) -> f64` — Compute the criticality boost for a fragment. Returns a multiplier [1.0, 10.0] for the relevance score.
- `pub fn classify(query: &str) -> TaskType` — Classify task type from natural language query.
- `pub fn budget_multiplier(&self) -> f64` — Get the recommended budget multiplier for this task type.
- `pub fn compute_ordering_priority(` — Context ordering strategy.  LLMs are order-sensitive. Fragment ordering affects reasoning quality. We order by: pinned first → critical → high relevance → imports → rest
- `pub fn new() -> Self`
- `pub fn record_success(&mut self, fragment_ids: &[String])` — Record that these fragments contributed to a successful output.
- `pub fn record_failure(&mut self, fragment_ids: &[String])` — Record that these fragments were present during a failure.
- `fn record_reward_signal(&mut self, fragment_ids: &[String], reward: f64)` — Welford online variance update — O(1) per fragment.
- `pub fn variance(&self, fragment_id: &str) -> f64` — Sample variance (Welford's M₂/(n-1)). Returns 1.0 for unseen fragments.
- `pub fn visit_count(&self, fragment_id: &str) -> u32` — Visit count for a fragment.
- `pub fn welford_mean(&self, fragment_id: &str) -> f64` — Welford mean reward for a fragment.
- `pub fn adaptive_exploration_rate(&self, alpha_0: f64) -> f64` — Current annealed exploration coefficient α_t for RAVEN-UCB.
- `pub fn ucb_score(&self, fragment_id: &str, alpha_0: f64) -> f64` — RAVEN-UCB score: UCB_i = μ_i + α_t · √(σ²_i / (n_i + 1)) where α_t = α₀ / ln(t + e) self-anneals exploration.  Reference: arXiv:2506.02933, June 2025.
- `pub fn total_observations(&self) -> u64` — Total optimization calls (global clock).
- `pub fn learned_value(&self, fragment_id: &str) -> f64` — Compute a learned value adjustment for a fragment.  Returns a multiplier: - > 1.0 = fragment has been historically useful - < 1.0 = fragment has been historically unhelpful - = 1.0 = no data
- `fn test_critical_files()`
- `fn test_safety_signals()`
- `fn test_task_classification()`
- `fn test_feedback_tracker()`
- `fn test_ucb_variance_tracking()`
- `fn test_raven_ucb_convergence()`
