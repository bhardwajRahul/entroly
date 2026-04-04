---
claim_id: 18a33f4c134b88cc053d14cc
entity: prefetch
status: inferred
confidence: 0.75
sources:
  - prefetch.py:45
  - prefetch.py:91
  - prefetch.py:103
  - prefetch.py:133
  - prefetch.py:181
  - prefetch.py:200
  - prefetch.py:209
  - prefetch.py:289
last_checked: 2026-04-04T19:51:14Z
derived_from:
  - cogops_compiler
  - sast
---

# Module: prefetch

**Language:** py
**Lines of code:** 297

## Types
- `class PrefetchResult:` — A predicted context fragment that might be needed next.
- `class PrefetchEngine:` —  Predictive pre-fetcher that learns co-access patterns across sessions and combines them with static analysis for predictions.  Two prediction strategies: 1. **Static**: Parse imports, calls, and nami

## Functions
- `def extract_callees(source: str, language: str = "python") -> List[str]` —  Extract function/method names called from a source code fragment.  Returns a list of callee names (not fully qualified — just the function name as it appears in source).
- `def extract_imports(source: str, language: str = "python") -> List[str]` —  Extract import targets from a source code fragment.  Returns module/path strings that could be resolved to files.
- `def infer_test_files(file_path: str) -> List[str]` —  Infer likely test file paths from a source file path.  Heuristics: foo.py       → test_foo.py, foo_test.py, tests/test_foo.py utils/bar.py → tests/test_bar.py, utils/test_bar.py src/baz.rs   → tests/
- `def __init__(self, co_access_window: int = 5)`
- `def record_access(self, file_path: str, turn: int) -> None` —  Record that a file was accessed at a given turn.  Updates co-access counts with all files accessed within the co-access window.
- `def stats(self) -> dict`
