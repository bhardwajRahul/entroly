---
claim_id: 18a33f4c10cb1d8802bca988
entity: lsh
status: inferred
confidence: 0.75
sources:
  - lsh.rs:35
  - lsh.rs:45
  - lsh.rs:73
  - lsh.rs:83
  - lsh.rs:89
  - lsh.rs:100
  - lsh.rs:123
  - lsh.rs:128
  - lsh.rs:136
  - lsh.rs:144
last_checked: 2026-04-04T19:51:14Z
derived_from:
  - cogops_compiler
  - sast
---

# Module: lsh

**Language:** rs
**Lines of code:** 313

## Types
- `pub struct LshTable` — A single LSH table mapping hash keys → fragment ID lists.
- `pub struct LshIndex` — Multi-Probe LSH Index for sub-linear SimHash similarity search.  Public API: - `insert(fp, idx)` — called on every `ingest()` - `remove(fp, idx)` — called on eviction / dedup replacement - `query(fp) 
- `pub struct ContextScorer` — Context-weighted composite scorer for fragment recall.  Combines similarity (Hamming-based), recency (Ebbinghaus decay), and entropy (information density) into a single relevance score.  Ported from e

## Functions
- `fn new(table_idx: usize) -> Self` — Create a table with deterministically-chosen bit positions. Uses golden ratio hashing so each table gets a different bit subset.
- `fn hash_key(&self, fp: u64) -> u16` — Extract the hash key by sampling BITS_PER_KEY bits from a 64-bit fingerprint.
- `fn insert(&mut self, fp: u64, idx: usize)`
- `fn remove(&mut self, fp: u64, idx: usize)`
- `fn query_multiprobe(&self, fp: u64) -> Vec<usize>` — Multi-probe: exact bucket + MULTI_PROBE_DEPTH single-bit-flip neighbors.
- `pub fn new() -> Self`
- `pub fn insert(&mut self, fp: u64, idx: usize)` — Insert a fragment by its SimHash fingerprint and storage index.
- `pub(crate) fn remove(&mut self, fp: u64, idx: usize)` — Remove a fragment entry (called on eviction).
- `pub fn query(&self, fp: u64) -> Vec<usize>` — Query for candidate fragment indices similar to the given fingerprint. Returns a deduplicated set — caller computes exact Hamming distance.
- `pub fn clear(&mut self)` — Clear all tables (e.g., after bulk eviction requiring index rebuild).
- `pub(crate) fn approx_size(&self) -> usize` — Number of unique entries in the first table (approximate size).
- `fn default() -> Self`
- `fn default() -> Self`
- `pub fn score(` — Compute a composite relevance score for a candidate fragment.  - `hamming`: Hamming distance between query fingerprint and fragment (0–64). - `recency_score`:   Pre-computed Ebbinghaus decay [0, 1]. -
- `fn test_insert_and_query_finds_exact_match()`
- `fn test_similar_fingerprints_found()`
- `fn test_remove_not_returned()`
- `fn test_no_duplicates_in_results()`
- `fn test_scale_1k_query_returns_small_set()`
- `fn test_context_scorer_ordering()`
