---
claim_id: 18a33f4c106fc9d8026155d8
entity: hierarchical
status: inferred
confidence: 0.75
sources:
  - hierarchical.rs:29
  - hierarchical.rs:54
  - hierarchical.rs:82
  - hierarchical.rs:175
  - hierarchical.rs:224
  - hierarchical.rs:263
  - hierarchical.rs:326
  - hierarchical.rs:372
  - hierarchical.rs:392
  - hierarchical.rs:416
last_checked: 2026-04-04T19:51:14Z
derived_from:
  - cogops_compiler
  - sast
---

# Module: hierarchical

**Language:** rs
**Lines of code:** 749

## Types
- `pub struct HccResult` — Result of hierarchical compression.

## Functions
- `pub fn compress_level1(fragments: &[ContextFragment]) -> (String, u32)` — Generate a one-line summary for each fragment from its skeleton.  Format: "file:auth.py → AuthService, login(), verify_token()" If no skeleton, falls back to source name only.  Target: ~3-8 tokens per
- `fn extract_oneliner_from_skeleton(skeleton: &str) -> String` — Extract a compact one-liner from a skeleton string. Pulls out function/class/struct/trait names.
- `pub fn identify_cluster(` — Identify the fragment cluster connected to query-relevant fragments via the dependency graph.  This is our "symbol-reachability slice" — an approximation of backward+forward program slicing using the 
- `pub fn compress_level2(` — Generate expanded skeleton text for the cluster.  Cluster fragments get their full skeleton. This gives the LLM structural understanding of the query neighborhood.
- `pub(crate) fn compute_pagerank(` — Compute PageRank scores for all fragments in the dep graph.  Hub files (imported/called by many others) get higher scores. Used to prioritize L2 expansion toward structurally central code.  Classic po
- `pub fn allocate_budget(` — Allocate token budget across the three levels.  Strategy: - L1 gets min(budget * 0.05, 500) — always fits - L2 gets min(budget * 0.25, remaining * 0.35) — scales with budget - L3 gets the rest — knaps
- `pub fn submodular_marginal_gain(` — Uses a facility-location style objective: marginal_gain(f | S) = relevance(f) × diversity_penalty(f, S)  where diversity_penalty decreases as more fragments from the same "cluster" (connected componen
- `fn extract_module(source: &str) -> String` — Extract the module/directory path from a source string. "file:src/auth/login.py" → "src/auth"
- `pub fn hierarchical_compress(` — Main entry point: compress the entire fragment set hierarchically.  Arguments: - fragments: all indexed fragments - dep_graph: the dependency graph - query_relevant_ids: fragment IDs ranked by query r
- `fn make_frag(id: &str, source: &str, tokens: u32, skeleton: Option<&str>) -> ContextFragment`
- `fn test_level1_generates_oneliners()`
- `fn test_extract_oneliner_caps_at_6()`
- `fn test_identify_cluster_follows_deps()`
- `fn test_budget_allocation()`
- `fn test_submodular_diversity()`
- `fn test_pagerank_hub_gets_highest()`
- `fn test_hierarchical_compress_end_to_end()`
- `fn test_empty_fragments()`
