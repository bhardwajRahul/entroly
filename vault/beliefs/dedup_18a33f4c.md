---
claim_id: 18a33f4c0fec70d801ddfcd8
entity: dedup
status: inferred
confidence: 0.75
sources:
  - dedup.rs:23
  - dedup.rs:34
  - dedup.rs:46
  - dedup.rs:92
  - dedup.rs:105
  - dedup.rs:124
  - dedup.rs:137
  - dedup.rs:142
  - dedup.rs:153
  - dedup.rs:196
last_checked: 2026-04-04T19:51:14Z
derived_from:
  - cogops_compiler
  - sast
---

# Module: dedup

**Language:** rs
**Lines of code:** 255

## Types
- `pub struct DedupIndex` — LSH-bucketed deduplication index.  Split 64-bit fingerprints into 4 bands of 16 bits. Two fingerprints sharing any band → candidate pair. Verify with full Hamming distance.  Same approach as ebbiforge

## Functions
- `fn hash_token(token: &str) -> u64` — Hash a token to a 64-bit integer using MD5.
- `fn normalized_content_hash(text: &str) -> u64` — Hash normalized content so duplicate verification can distinguish truly repeated fragments from large files that merely share a stable SimHash despite minor semantic edits.
- `pub fn simhash(text: &str) -> u64` — Compute the 64-bit SimHash fingerprint of a text.  Algorithm: 1. Extract word trigrams as features 2. Hash each feature → 64-bit 3. For each bit: feature hash bit=1 → +1, bit=0 → -1 4. Final bit = 1 i
- `pub fn hamming_distance(a: u64, b: u64) -> u32` — Hamming distance between two 64-bit fingerprints.
- `pub fn new(hamming_threshold: u32) -> Self`
- `pub fn hamming_threshold(&self) -> u32`
- `fn extract_bands(&self, fp: u64) -> Vec<u64>` — Extract band hashes from a fingerprint.
- `pub fn insert(&mut self, fragment_id: &str, text: &str) -> Option<String>` — Insert a fragment. Returns Some(duplicate_id) if near-dup found.
- `pub fn remove(&mut self, fragment_id: &str)` — Remove a fragment from the index.
- `pub fn size(&self) -> usize`
- `pub fn get_fingerprint(&self, fragment_id: &str) -> Option<u64>` — Get the stored fingerprint for a fragment.
- `pub fn export_fingerprints(&self) -> Vec<(String, u64)>` — Export all fingerprints for checkpointing.
- `fn test_identical_texts()`
- `fn test_dedup_catches_exact()`
- `fn test_dedup_allows_different()`
