---
claim_id: 18a33f4c13a8c974059a5574
entity: query_refiner
status: inferred
confidence: 0.75
sources:
  - query_refiner.py:27
  - query_refiner.py:34
  - query_refiner.py:43
  - query_refiner.py:54
  - query_refiner.py:63
  - query_refiner.py:83
  - query_refiner.py:113
last_checked: 2026-04-04T19:51:14Z
derived_from:
  - cogops_compiler
  - sast
---

# Module: query_refiner

**Language:** py
**Lines of code:** 172

## Types
- `class QueryRefiner:` —  Refines vague developer queries into precise context-selection prompts.  Rust handles all compute. Python only does LLM I/O (optional).  Usage: refiner = QueryRefiner() result = refiner.refine("fix t

## Functions
- `def py_analyze_query(query: str, summaries: list) -> tuple:  # type: ignore[misc]` — Pure-Python fallback for query analysis.
- `def py_refine_heuristic(query: str, summaries: list) -> str:  # type: ignore[misc]` — Pure-Python fallback — returns query unchanged.
- `def __init__(self, llm_fn: Optional[Callable[[str], str]] = None)` —  Args: llm_fn: Optional async/sync function that takes a query string and returns a refined query string using an LLM. If None, only the Rust heuristic path is used.
- `def analyze(self, query: str, fragment_summaries: List[str] | None = None) -> dict` —  Analyze a query for vagueness and key terms.  Returns: dict with: - vagueness_score (float, 0–1) - key_terms (list[str]) - needs_refinement (bool) - reason (str)
- `def refine(self, query: str, fragment_summaries: List[str] | None = None) -> str` —  Refine a query using: 1. Rust heuristic (always runs, zero latency) 2. LLM refinement (only if query needs_refinement AND llm_fn is set)  Returns the refined query string (original if no refinement a
- `def make_openai_refine_fn(api_key: str, model: str = "gpt-4o-mini") -> Callable[[str], str]` —  Return a function that calls OpenAI to refine a query.  Requires: openai package (`pip install openai`). The function is synchronous and makes one API call per invocation.  Usage: refiner = QueryRefi
