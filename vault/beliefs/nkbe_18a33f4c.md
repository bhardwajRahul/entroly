---
claim_id: 18a33f4c10e8f2a402da7ea4
entity: nkbe
status: inferred
confidence: 0.75
sources:
  - nkbe.rs:32
  - nkbe.rs:42
  - nkbe.rs:53
  - nkbe.rs:71
  - nkbe.rs:94
  - nkbe.rs:109
  - nkbe.rs:136
  - nkbe.rs:266
  - nkbe.rs:290
  - nkbe.rs:305
last_checked: 2026-04-04T19:51:14Z
derived_from:
  - cogops_compiler
  - sast
---

# Module: nkbe

**Language:** rs
**Lines of code:** 507

## Types
- `pub struct AgentBudgetState` — Per-agent state for budget allocation.
- `pub struct NkbeFragment` — Fragment descriptor for NKBE allocation.
- `pub struct NkbeAllocator` — NKBE Allocator — multi-agent token budget allocation.  Implements two-phase KKT bisection with Nash Bargaining refinement and REINFORCE gradient for RL weight learning.

## Functions
- `pub fn new(`
- `pub fn register_agent(` — Register an agent for budget allocation.
- `pub fn add_fragment(` — Add a fragment to an agent (for utility estimation).
- `pub fn allocate<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>>` — Run NKBE allocation. Returns per-agent budgets as a Python dict.  Two-phase KKT bisection: 1. Bisect for global λ* such that Σ demand(λ*) = B 2. Each agent gets budget proportional to their utility at
- `pub fn reinforce(&mut self, outcomes_json: &str) -> bool` — REINFORCE gradient: update agent weights based on outcomes.  Δwᵢ = η · (Rᵢ − R̄) · wᵢ  Agents producing better outcomes get more budget next time.
- `pub fn stats<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>>` — Get allocator statistics.
- `fn agent_utility(&self, agent_idx: usize, lambda: f64) -> f64` — Agent utility at price λ: U_i(λ) = Σⱼ σ((sⱼ - λ·cⱼ)/τ) · sⱼ · wᵢ
- `fn agent_demand(&self, agent_idx: usize, lambda: f64) -> f64` — Agent demand at price λ: D_i(λ) = Σⱼ σ((sⱼ - λ·cⱼ)/τ) · cⱼ
- `fn total_demand(&self, lambda: f64) -> f64` — Total demand across all agents at price λ.
- `fn compute_dual(&self, lambda: f64) -> f64` — Dual objective: D(λ) = τ·Σ log(1+exp((sᵢ−λ·cᵢ)/τ)) + λ·B
- `pub fn reinforce_gradient(` — REINFORCE gradient computation for 4D scoring weights.  ∂E[R]/∂wₖ = Σᵢ (aᵢ − p*ᵢ) · R · σ'(zᵢ/τ) · featureᵢₖ  Returns gradient vector [Δw_recency, Δw_frequency, Δw_semantic, Δw_entropy].
- `fn sigmoid(x: f64) -> f64` — Numerically stable sigmoid.
- `fn softplus(x: f64) -> f64` — Numerically stable softplus: log(1 + exp(x)).
- `fn test_sigmoid_bounds()`
- `fn test_softplus()`
- `fn test_single_agent_gets_full_budget()`
- `fn test_demand_decreases_with_lambda()`
- `fn test_reinforce_gradient_basic()`
- `fn test_dual_gap_nonnegative()`
- `fn test_two_agents_split()`
