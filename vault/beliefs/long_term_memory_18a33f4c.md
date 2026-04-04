---
claim_id: 18a33f4c13170f3405089b34
entity: long_term_memory
status: inferred
confidence: 0.75
sources:
  - long_term_memory.py:70
  - long_term_memory.py:78
  - long_term_memory.py:107
  - long_term_memory.py:152
  - long_term_memory.py:155
  - long_term_memory.py:277
  - long_term_memory.py:308
last_checked: 2026-04-04T19:51:14Z
derived_from:
  - cogops_compiler
  - sast
---

# Module: long_term_memory

**Language:** py
**Lines of code:** 316

## Types
- `class SalienceProfile:` — Maps entroly fragment properties to hippocampus salience values.
- `class LongTermMemory:` —  Adapter between entroly's session-level context engine and hippocampus' cross-session memory.  Lifecycle: 1. Created once when EntrolyEngine starts (in server.py) 2. On each optimize_context(): recal

## Functions
- `def is_available() -> bool` — Check if hippocampus-sharp-memory is installed and available.
- `def active(self) -> bool`
- `def tick(self) -> None` — Advance the hippocampus clock by 1 tick (called on each advance_turn).
- `def stats(self) -> dict` — Get long-term memory statistics for the dashboard.
- `def consolidate(self) -> str` — Force a consolidation cycle (sleep-replay).
