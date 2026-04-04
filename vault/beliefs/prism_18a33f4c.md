---
claim_id: 18a33f4c10fe32e002efbee0
entity: prism
status: inferred
confidence: 0.75
sources:
  - prism.rs:33
  - prism.rs:39
  - prism.rs:43
  - prism.rs:50
  - prism.rs:55
  - prism.rs:60
  - prism.rs:74
  - prism.rs:157
  - prism.rs:168
  - prism.rs:203
last_checked: 2026-04-04T19:51:14Z
derived_from:
  - cogops_compiler
  - sast
---

# Module: prism

**Language:** rs
**Lines of code:** 761

## Types
- `pub struct SymMatrixN` — An NxN symmetric matrix stored as a flat array for tracking gradient covariance.  Uses `N * N` flat layout (row-major) since const generic expressions like `[f64; N * N]` require nightly features. The
- `pub struct PrismOptimizerN` — certain fragment pairs produce dramatically better LLM outputs than the sum of their individual contributions. PRISM learns to weight this signal relative to the four individual-fragment dimensions.  
- `pub struct ResonanceDiagnostics` — Diagnostics for the resonance dimension in 5D PRISM.

## Functions
- `pub fn new() -> Self`
- `pub fn identity() -> Self`
- `pub fn get(&self, i: usize, j: usize) -> f64`
- `pub fn set(&mut self, i: usize, j: usize, val: f64)`
- `pub fn update_ema(&mut self, g: &[f64], beta: f64)` — Update running covariance: $C = \beta C + (1-\beta) g g^T$
- `pub fn jacobi_eigendecomposition(&self) -> (SymMatrixN<N>, Vec<f64>)` — Computes Eigenvalue Decomposition $C = Q \Lambda Q^T$ using the Cyclic Jacobi method. Returns (Q, Eigenvalues), where Q columns are eigenvectors.  Complexity: O(N² × max_iters) — negligible for N ≤ 5.
- `pub fn from_array(data: [[f64; 4]; 4]) -> Self` — Construct from a fixed 4x4 array (backward compat with old checkpoint format).
- `pub fn to_array(&self) -> [[f64; 4]; 4]` — Export as a fixed 4x4 array (backward compat).
- `pub fn new(learning_rate: f64) -> Self`
- `pub fn compute_update(&mut self, g: &[f64]) -> Vec<f64>` — Computes $P g = Q \Lambda^{-1/2} Q^T g$ and returns the update $\Delta w = \alpha P g$.  For the 5D case, the resonance dimension (index 4) typically shows higher gradient variance than dimensions 0–3
- `pub fn condition_number(&self) -> f64` — Return the spectral condition number κ = sqrt(λ_max / λ_min).  κ encodes weight-space uncertainty: κ ≈ 1  → isotropic, well-conditioned (all dimensions equally informative) κ >> 1 → ill-conditioned (s
- `pub fn eigenvalues(&self) -> Vec<f64>` — Return current eigenvalues of the gradient covariance matrix.
- `pub fn spectral_energy(&self) -> Vec<f64>` — Spectral energy per dimension: fraction of total eigenvalue mass per eigenvector.  Returns N values that sum to 1.0. Useful for diagnostics: - If dimension 4 (resonance) carries <5% spectral energy, i
- `pub fn compute_update_4d(&mut self, g: &[f64; 4]) -> [f64; 4]` — Fixed-size 4D update (backward compat with existing call sites). Delegates to the generic implementation.
- `pub fn condition_number_4d(&self) -> f64` — Condition number (4D). Same as generic, but named for clarity.
- `pub fn from_4d(opt4: &PrismOptimizer) -> Self` —  ┌─────────────┬────┐ │   C₄ₓ₄      │ 0  │ ├─────────────┼────┤ │   0ᵀ         │ ε  │ └─────────────┴────┘  This means the resonance dimension starts with the same prior variance as the initial 4D dim
- `pub fn resonance_diagnostics(&self) -> ResonanceDiagnostics` — Extract the resonance-specific spectral diagnostics.  Returns a struct with: - resonance_eigenvalue: the eigenvalue most aligned with dim 4 - resonance_energy_fraction: what % of spectral energy is in
- `fn test_jacobi_eigendecomposition_diagonal_4d()`
- `fn test_anisotropic_shaping_dampens_noise_4d()`
- `fn test_4d_backward_compat_api()`
- `fn test_jacobi_eigendecomposition_5d()`
- `fn test_5d_resonance_damping()`
- `fn test_5d_from_4d_preserves_covariance()`
- `fn test_5d_cross_correlation_emergence()`
- `fn test_5d_condition_number_increases_with_resonance()`
- `fn test_spectral_energy_sums_to_one()`
- `fn test_sym_matrix4_array_roundtrip()`
- `fn test_5d_non_diagonal_eigendecomposition()`
