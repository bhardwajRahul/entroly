---
claim_id: 18a33f4c1111762003030220
entity: query
status: inferred
confidence: 0.75
sources:
  - query.rs:58
  - query.rs:74
  - query.rs:81
  - query.rs:86
  - query.rs:99
  - query.rs:118
  - query.rs:170
  - query.rs:223
  - query.rs:250
  - query.rs:311
last_checked: 2026-04-04T19:51:14Z
derived_from:
  - cogops_compiler
  - sast
---

# Module: query

**Language:** rs
**Lines of code:** 422

## Types
- `pub struct QueryAnalysis`

## Functions
- `fn tokenize(text: &str) -> Vec<String>` — Tokenize text into lowercase words, removing stop words and punctuation.
- `fn is_stop_word(word: &str) -> bool`
- `fn tf(tokens: &[String]) -> HashMap<String, f64>` — Compute TF (term frequency) for a token list.
- `fn idf(corpus: &[Vec<String>]) -> HashMap<String, f64>` — Compute IDF (inverse document frequency) given a corpus of token lists. Uses log(N / (1 + df)) + 1 (smooth IDF).
- `pub fn extract_key_terms(` — Extract top-N key terms from a query using TF-IDF over the fragment corpus.  If `fragment_summaries` is empty, falls back to TF-only ranking.  Returns sorted (descending score) vector of term strings.
- `pub fn compute_vagueness(query: &str) -> (f64, String)` — Score the vagueness of a query.  Algorithm: vagueness = generic_verb_ratio × 0.5 + short_penalty × 0.3 − specificity_bonus × 0.2  - `generic_verb_ratio`: fraction of tokens that are generic verbs ("fi
- `pub fn analyze_query(query: &str, fragment_summaries: &[String]) -> QueryAnalysis` — Full query analysis: vagueness + key term extraction.
- `pub fn refine_heuristic(query: &str, fragment_summaries: &[String]) -> String` — Refine a vague query by grounding it in vocabulary from the fragment corpus.  Algorithm: 1. Extract key terms from query 2. Find which fragments have highest vocabulary overlap with query terms 3. Inj
- `pub fn py_analyze_query(` — Analyze a query for vagueness and extract key terms.  Returns a tuple: (vagueness_score: float, key_terms: list[str], needs_refinement: bool, reason: str)
- `pub fn py_refine_heuristic(query: &str, fragment_summaries: Vec<String>) -> String` — Heuristic query refinement — grounded in fragment vocabulary, no LLM needed.  Returns the refined query string.
- `fn test_vague_query_high_score()`
- `fn test_specific_query_low_score()`
- `fn test_key_term_extraction_removes_stopwords()`
- `fn test_key_terms_prefers_specific_tokens()`
- `fn test_refinement_adds_context_terms()`
- `fn test_refinement_noop_on_empty_fragments()`
- `fn test_analyze_query_full()`
- `fn test_analyze_query_vague_needs_refinement()`
- `fn test_stopwords_filtered()`
- `fn test_idf_penalizes_common_terms()`
