---
claim_id: 18a33f4c120395b803f521b8
entity: adaptive_pruner
status: inferred
confidence: 0.75
sources:
  - adaptive_pruner.py:38
  - adaptive_pruner.py:49
  - adaptive_pruner.py:58
  - adaptive_pruner.py:81
  - adaptive_pruner.py:127
  - adaptive_pruner.py:141
  - adaptive_pruner.py:147
  - adaptive_pruner.py:150
last_checked: 2026-04-04T19:51:14Z
derived_from:
  - cogops_compiler
  - sast
---

# Module: adaptive_pruner

**Language:** py
**Lines of code:** 161

## Types
- `class EntrolyPruner:` —  Adaptive RL pruner backed by ebbiforge_core.AdaptivePruner.  Extends entroly's Wilson-score feedback with a `historical_success` dimension: fragments that previously helped get boosted, those that di
- `class FragmentGuard:` —  Code quality scanner backed by ebbiforge_core.CodeQualityGuard.  Scans each ingested fragment for: - Hardcoded API secrets  (sk-..., API_KEY = "...") - unsafe Rust blocks - TODO comments - Console sp

## Functions
- `def __init__(self)`
- `def available(self) -> bool`
- `def apply_feedback(self, fragment_id: str, feedback: float) -> bool` —  Apply user feedback to update RL weights for this fragment's features.  Args: fragment_id: The fragment that received feedback. feedback:    +1.0 = helpful, -1.0 = not helpful, 0.0 = neutral.  Return
- `def __init__(self)`
- `def available(self) -> bool`
- `def scan(self, content: str, source: str = "") -> list[str]` —  Scan fragment content for code quality issues.  Returns list of issue strings (empty = clean).
