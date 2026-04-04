---
claim_id: 18a33f4c12f00b5004e19750
entity: evolution_logger
status: inferred
confidence: 0.75
sources:
  - evolution_logger.py:31
  - evolution_logger.py:40
  - evolution_logger.py:54
  - evolution_logger.py:125
last_checked: 2026-04-04T19:51:14Z
derived_from:
  - cogops_compiler
  - sast
---

# Module: evolution_logger

**Language:** py
**Lines of code:** 187

## Types
- `class MissRecord:` — A single recorded miss / failure.
- `class EvolutionLogger:` —  Tracks failures and identifies skill gap candidates.  Lives in the Evolution layer. Writes to vault/evolution/ when the system needs a new skill.

## Functions
- `def to_dict(self) -> Dict[str, Any]`
- `def stats(self) -> Dict[str, Any]` — Return evolution statistics.
