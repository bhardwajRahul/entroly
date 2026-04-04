---
claim_id: 18a33f4c10b7118002a89d80
entity: lib
status: inferred
confidence: 0.75
sources:
  - lib.rs:72
  - lib.rs:216
  - lib.rs:223
  - lib.rs:249
  - lib.rs:350
  - lib.rs:445
  - lib.rs:619
  - lib.rs:1491
  - lib.rs:1559
  - lib.rs:1776
last_checked: 2026-04-04T19:51:14Z
derived_from:
  - cogops_compiler
  - sast
---

# Module: lib

**Language:** rs
**Lines of code:** 3741

## Types
- `pub struct EntrolyEngine` — The core engine that orchestrates all subsystems.  Modeled after ebbiforge-core HippocampusEngine: ingest → SimHash → dedup check → entropy score → store optimize → Ebbinghaus decay → knapsack DP → ra
- `pub struct OptimizationSnapshot` — Snapshot of the last optimization for explainability.
- `pub struct FragmentScore` — Per-fragment scoring breakdown.
- `pub struct IndexSnapshot`
- `pub struct IndexSnapshot`
- `pub struct EngineState` — Borrowed state for serialization (avoids cloning dedup/dep/feedback).
- `pub struct OwnedEngineState` — Owned state for deserialization.
- `pub struct PyDedupIndex` — Python-friendly DedupIndex — wraps the Rust DedupIndex struct.

## Functions
- `pub fn new(`
- `pub fn advance_turn(&mut self)` — Advance the turn counter and apply Ebbinghaus decay.
- `pub fn ingest(&mut self, content: String, source: String, token_count: u32, is_pinned: bool) -> PyResult<PyObject>` — Ingest a new context fragment.  Pipeline: tokens → SimHash → dedup → entropy → criticality → depgraph → store
- `pub fn optimize(&mut self, token_budget: u32, query: String) -> PyResult<PyObject>` — Select the optimal context subset within the token budget.  Two-pass optimization: Pass 1: Initial knapsack selection Pass 2: Boost unselected dependencies of selected fragments, re-run  Wires in: fee
- `pub fn recall(&self, query: String, top_k: usize) -> PyResult<PyObject>` — Semantic recall of relevant fragments.  Uses ebbiforge-ported LSH multi-probe index for sub-linear candidate selection, then scores per ContextScorer (similarity+recency+entropy +frequency+feedback). 
- `pub fn stats(&mut self) -> PyResult<PyObject>` — Get session statistics.
- `pub fn get_turn(&self) -> u32` — Get the current turn number.
- `pub fn fragment_count(&self) -> usize` — Get fragment count.
- `pub fn query_manifold_stats(&self) -> PyResult<PyObject>` — Get detailed query manifold stats (per-archetype weights, health, particles).
- `pub fn set_query_personas_enabled(&mut self, enabled: bool)` — Enable or disable query persona manifold.
- `pub fn set_weights(&mut self, w_recency: f64, w_frequency: f64, w_semantic: f64, w_entropy: f64)` — Hot-reload scoring weights mid-session (autotune live update). Normalizes to sum=1.0 and clamps to [0.05, 0.80].
- `pub fn persist_index(&self, path: &str) -> PyResult<()>` — Persist the full fragment index to disk as compressed JSON. Called automatically after each ingest batch so sessions resume without re-indexing the whole codebase.
- `pub fn load_index(&mut self, path: &str) -> PyResult<usize>` — Load a previously persisted index from disk. Returns the number of fragments restored.
- `fn default_gradient_temperature() -> f64`
- `pub fn record_success(&mut self, fragment_ids: Vec<String>)` — Record that the selected fragments led to a successful output. This feeds the reinforcement learning loop.  Channel coding path: R = modulated_reward(suff), A = R − μ. The advantage A is what drives t
- `pub fn record_failure(&mut self, fragment_ids: Vec<String>)` — Record that the selected fragments led to a failed output.  Channel coding path: R = modulated_reward(suff), A = R − μ. Low sufficiency → stronger penalty → faster weight correction.
- `pub fn record_reward(&mut self, fragment_ids: Vec<String>, reward: f64)` —  This is the preferred path when ΔPerplexity is available: R = PPL(response | no context) − PPL(response | context)  Positive R means our context helped (perplexity dropped). Negative R means our cont
- `pub fn set_channel_coding_enabled(&mut self, enabled: bool)` — Enable or disable channel coding framework.
- `pub fn cache_clear(&mut self)` — Clear the EGSC cache (useful after major project changes).
- `pub fn cache_len(&self) -> usize` — Get current cache entry count.
- `pub fn cache_is_empty(&self) -> bool` — Check if cache is empty.
- `pub fn cache_hit_rate(&mut self) -> f64` — Get the current cache hit rate (from shift detector EMA).
- `pub fn set_model(&mut self, model_name: &str)` — Set the cost model from a model name (auto-detects pricing).  Covers 20+ models: OpenAI (gpt-4o, gpt-4, o1, o3), Anthropic (claude-3.5), Google (gemini), DeepSeek, Meta (llama), Mistral, and more. Unk
- `pub fn set_cache_cost_per_token(&mut self, cost: f64)` — Set cost-per-token directly (power users only). Most developers should use `set_model()` instead. Default is already $0.000015 (GPT-4o output) — no config needed.
- `pub fn classify_task(&self, query: &str) -> PyResult<PyObject>` — Classify a task query and return the recommended budget multiplier.
- `pub fn dep_graph_stats(&self) -> PyResult<PyObject>` — Get dependency graph stats.
- `pub fn hierarchical_compress(&self, token_budget: u32, query: String) -> PyResult<PyObject>` — Hierarchical Context Compression — 3-level codebase compression.  Level 1: Skeleton map of entire codebase (~5% budget) Level 2: Expanded skeletons for dep-graph connected cluster (~25%) Level 3: Full
- `pub fn explain_selection(&self) -> PyResult<PyObject>` — Explain why each fragment was included or excluded in the last optimization.  Returns per-fragment scoring breakdowns with all dimensions visible. Call after optimize() to understand selection decisio
- `pub fn export_state(&self) -> PyResult<String>` — Export full engine state as JSON string for checkpoint/restore.  Serializes: fragments, dedup index, dep graph, feedback tracker, turn counter, stats — everything needed for perfect resume.
- `pub fn import_state(&mut self, json_str: &str) -> PyResult<()>` — Import engine state from JSON string (checkpoint restore).  Replaces all engine state with the deserialized data.
- `pub fn set_exploration_rate(&mut self, rate: f64)` — Set the exploration rate (0.0 = pure exploitation, 1.0 = always explore).
- `pub fn scan_fragment(&self, fragment_id: &str) -> PyResult<String>` — Scan a specific fragment for security vulnerabilities.  Returns a JSON-encoded SastReport with: - findings: list of {rule_id, cwe, severity, line_number, description, fix, confidence} - risk_score: CV
- `pub fn security_report(&self) -> PyResult<String>` — Scan all ingested fragments and return an aggregated security report.  Returns JSON with per-fragment findings + session-level statistics: - total_findings, critical_total, max_risk_score - most_vulne
- `pub fn analyze_health(&self) -> PyResult<String>` — Analyze codebase health across all ingested fragments.  Returns a JSON-encoded HealthReport with: - code_health_score [0–100] and health_grade (A/B/C/D/F) - clone_pairs: near-duplicate file pairs (Typ
- `pub fn entropy_anomalies(&self) -> PyResult<String>` — Scan for entropy anomalies — code that is statistically surprising relative to its neighbors. Uses robust Z-scores (MAD) grouped by directory. High-entropy spikes = copy-paste errors, unusual patterns
- `pub fn score_utilization(&self, response: &str) -> PyResult<String>` — Score how much of the injected context the LLM actually used. Call after receiving the LLM response. Measures trigram overlap and identifier reuse to determine context efficiency.  Returns JSON with p
- `pub fn semantic_dedup_report(&self) -> PyResult<String>` — Run semantic deduplication on all fragments and report which ones are informationally redundant. Returns removal candidates and estimated token savings.
- `pub fn export_fragments(&self) -> PyResult<PyObject>` — Export fragments for checkpoint (returns list of dicts).
- `fn rebuild_lsh_index(&mut self)` — Rebuild the LSH index and slot list from the current fragment map.  Called after batch eviction in advance_turn(). O(N) but infrequent.
- `fn sigmoid(x: f64) -> f64` — Numerically stable sigmoid: σ(x) = 1 / (1 + e^{-x}). Clamps input to [-500, 500] to prevent overflow.
- `fn apply_prism_rl_update(&mut self, fragment_ids: &[String], reward: f64)` —  The σ'(score/τ) = p(1-p)/τ term focuses gradient on decision-boundary fragments.  Temperature τ anneals via τ *= 0.995 per update but resets on regime changes (gradient norm spike > 3× EMA), allowing
- `fn compute_sufficiency(&self, frags: &[ContextFragment], selected_indices: &[usize]) -> f64` — Compute context sufficiency: fraction of referenced symbols that have definitions in the selected context.
- `fn default_max_fragments() -> usize`
- `fn default_w_recency() -> f64`
- `fn default_w_frequency() -> f64`
- `fn default_w_semantic() -> f64`
- `fn default_w_entropy() -> f64`
- `fn default_gradient_temperature() -> f64`
- `fn default_rng_state() -> u64`
- `fn py_shannon_entropy(text: &str) -> f64` — Compute Shannon entropy of a text string (bits per character).
- `fn py_normalized_entropy(text: &str) -> f64` — Compute normalized Shannon entropy [0, 1].
- `fn py_boilerplate_ratio(text: &str) -> f64` — Compute boilerplate ratio [0, 1].
- `fn py_renyi_entropy_2(text: &str) -> f64` — Compute Rényi entropy of order 2 (collision entropy). More sensitive to concentrated probability mass than Shannon entropy.
- `fn py_entropy_divergence(text: &str) -> f64` — Shannon–Rényi divergence: H₁(X) - H₂(X). High divergence indicates noise or encoded data, not useful context.
- `fn py_simhash(text: &str) -> u64` — Compute 64-bit SimHash fingerprint.
- `fn py_hamming_distance(a: u64, b: u64) -> u32` — Compute Hamming distance between two fingerprints.
- `fn py_information_score(text: &str, other_fragments: Vec<String>) -> f64` — Compute information density score [0, 1].
- `fn py_scan_content(content: &str, source: &str) -> String` — Scan content for security vulnerabilities — returns JSON-encoded SastReport.
- `fn py_analyze_query(`
- `fn py_refine_heuristic(query: &str, fragment_summaries: Vec<String>) -> String`
- `fn py_analyze_health_info() -> String` — Health analysis is available via engine.analyze_health() (a #[pymethod]). This standalone function is intentionally minimal.
- `fn py_prune_conversation(` — Prune a conversation to fit within a token budget.  Uses multi-choice knapsack via KKT dual bisection with causal DAG coherence enforcement.  Returns JSON-encoded PruneResult.  `blocks` is a list of d
- `fn py_progressive_thresholds(` — Progressive compression: assign resolutions based on context utilization. Returns JSON: [{index, resolution}]
- `fn py_compress_block(` — Compress a single block at a given resolution level. Returns the compressed text.
- `fn py_classify_block(role: &str, content: &str, tool_name: Option<String>) -> String` — Classify a conversation block by role and content. Returns: "user", "assistant", "tool_call", "tool_result", "thinking", "system"
- `fn py_cross_fragment_redundancy(text: &str, others: Vec<String>) -> f64` — Cross-fragment redundancy: how much of `text` is already covered by `others` [0,1].
- `fn py_apply_ebbinghaus_decay(` — Apply Ebbinghaus decay in-place to a list of ContextFragments. Mutates recency_score based on turns elapsed since turn_last_accessed.
- `fn py_knapsack_optimize(` — Convenience knapsack optimizer for standalone use (default weights, empty feedback). Returns (selected_fragments, stats_dict).
- `fn rebuild_selected_list(` — Reconstruct the `selected` PyList from cached fragment IDs + live fragment data.  On cache hit, the stored JSON contains `selected_ids: [{id, variant}, ...]`. We look up each fragment by ID in `self.f
- `fn new(hamming_threshold: u32) -> Self`
- `fn insert(&mut self, fragment_id: &str, text: &str) -> Option<String>` — Insert a fragment. Returns the ID of the duplicate if one was found, else None.
- `fn remove(&mut self, fragment_id: &str)` — Remove a fragment by ID.
- `fn size(&self) -> usize` — Current number of indexed fragments.
- `fn entroly_core(m: &Bound<'_, PyModule>) -> PyResult<()>`
- `fn test_simhash_distance_uses_64_not_32()`
- `fn test_task_budget_multiplier()`
- `fn test_export_import_roundtrip()`
- `fn test_sufficiency_full()`
- `fn test_exploration_rate_bounds()`
- `fn test_recall_returns_correct_fragment_not_random()` — Recall must return the EXACT fragment that matches the query content — not a random or irrelevant one. This is the most fundamental correctness guarantee: if you store "fn connect_database()" and quer
- `fn test_recall_ranking_is_monotone_descending()` — Recall ranking must be MONOTONE: fragment[0].relevance >= fragment[1].relevance >= ... If this fails, the LLM sees irrelevant context before relevant context.
- `fn test_context_scorer_similarity_monotone()` — ContextScorer must be MONOTONE in similarity: higher similarity → higher score (everything else equal). If this fails, the scorer would rank distant fragments above close ones.
- `fn test_lsh_never_drops_exact_match_after_scale()` — LshIndex must not silently drop exact-match entries even after 1000 inserts. If this fails, some fragments will NEVER be recalled regardless of query.
