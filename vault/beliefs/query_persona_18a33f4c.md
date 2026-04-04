---
claim_id: 18a33f4c112f12fc03209efc
entity: query_persona
status: inferred
confidence: 0.75
sources:
  - query_persona.rs:73
  - query_persona.rs:79
  - query_persona.rs:86
  - query_persona.rs:120
  - query_persona.rs:133
  - query_persona.rs:142
  - query_persona.rs:154
  - query_persona.rs:191
  - query_persona.rs:199
  - query_persona.rs:229
last_checked: 2026-04-04T19:51:14Z
derived_from:
  - cogops_compiler
  - sast
---

# Module: query_persona

**Language:** rs
**Lines of code:** 842

## Types
- `pub struct RbfKernel`
- `pub struct ParticleCloud`
- `pub struct QueryArchetype` — A query archetype = a probability measure over query feature vectors + a learned PRISM weight set for context selection.
- `pub struct PitmanYorProcess`
- `pub struct QueryPersonaManifold` — Discovers query archetypes from TF-IDF feature vectors and maintains per-archetype PRISM weight sets for specialized context selection.
- `pub struct ManifoldStats`
- `pub struct ArchetypeInfo`

## Functions
- `pub fn new(sigma: f32) -> Self`
- `pub fn silverman(particles: &[Vec<f32>]) -> Self` — Adaptive bandwidth via Silverman's rule: σ = 1.06 × median_std × n^(-1/(d+4))
- `pub fn eval(&self, x: &[f32], y: &[f32]) -> f32` — k(x, y) = exp(-||x-y||² / (2σ²))
- `pub fn new(dim: usize, max_particles: usize) -> Self`
- `pub fn add_particle(&mut self, x: &[f32])` — Add a particle. Pruning uses kernel herding: removes the most redundant particle (highest total affinity to neighbors).
- `pub fn affinity(&self, t: &[f32], kernel: &RbfKernel) -> f32` — Kernel mean embedding affinity: A(t) = Σᵢ wᵢ k(t, xᵢ)
- `pub fn mmd_squared(&self, other: &ParticleCloud, kernel: &RbfKernel) -> f32` — MMD²(μ_p, μ_q) = E[k(x,x')] + E[k(y,y')] - 2E[k(x,y)]
- `pub fn effective_support(&self, kernel: &RbfKernel) -> f32` — Effective support size (kernel-weighted entropy).
- `pub fn merge(&mut self, other: &ParticleCloud)` — Merge another cloud into this one.
- `pub fn new(id: String, default_weights: [f64; 4], tick: u64) -> Self`
- `pub fn affinity(&self, query_features: &[f32]) -> f32` — Kernel mean embedding affinity.
- `pub fn record_result(&mut self, gradient: &[f64; 4], success: bool)` — Record a result and update per-archetype PRISM weights.
- `pub fn update_kernel(&mut self)` — Update kernel bandwidth from particles (Silverman's rule).
- `pub fn new(alpha: f32, discount: f32) -> Self`
- `pub fn new_archetype_prob(&self, n_existing: usize) -> f32` — P(new archetype | K existing, n observations).
- `pub fn existing_weight(&self, n_k: u64) -> f32` — Weight for existing archetype with n_k observations.
- `pub fn update_weights(&self, archetypes: &mut [QueryArchetype])`
- `pub fn build_query_features(` — Build a QUERY_DIM-dimensional feature vector from TF-IDF key terms and query metadata. This is the embedding that PSM operates on.  Layout: [0..11]  = top-12 TF-IDF scores (padded with 0.0 if fewer) [
- `pub fn new(`
- `pub fn population(&self) -> usize`
- `pub fn assign(&mut self, query_features: &[f32]) -> (String, [f64; 4], bool)` — Assign a query to an archetype. Returns (archetype_id, weights, is_new).  Algorithm: 1. Compute kernel mean embedding affinity for each archetype 2. Weight by Pitman-Yor stick weights 3. Deterministic
- `pub fn record_result(` — Record optimization result for an archetype.
- `pub fn get_weights(&self, archetype_id: &str) -> [f64; 4]` — Get weights for a specific archetype. Returns default if not found.
- `pub fn lifecycle_tick(&mut self)` — Lifecycle tick: Ebbinghaus decay + death + MMD fusion.
- `pub fn stats(&self) -> ManifoldStats` — Return statistics about the manifold.
- `fn birth_archetype(&mut self, query_features: &[f32]) -> String`
- `fn check_fusion(&mut self)` — Fusion via MMD² — proper metric on probability measures.
- `fn make_features(primary: usize, vagueness: f32) -> Vec<f32>`
- `fn default_weights() -> [f64; 4]`
- `fn test_rbf_kernel_self()`
- `fn test_rbf_kernel_distance()`
- `fn test_birth_on_empty()`
- `fn test_repeated_assignment_converges()`
- `fn test_different_queries_can_birth()`
- `fn test_death_by_decay()`
- `fn test_fusion_identical_archetypes()`
- `fn test_per_archetype_weight_learning()`
- `fn test_mmd_identical_clouds()`
- `fn test_build_query_features()`
- `fn test_manifold_stats()`
- `fn test_max_archetypes_cap()`
- `fn test_pitman_yor_probability()`
