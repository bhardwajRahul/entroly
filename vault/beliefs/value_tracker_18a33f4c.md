---
claim_id: 18a33f4c141e32e0060fbee0
entity: value_tracker
status: inferred
confidence: 0.75
sources:
  - value_tracker.py:64
  - value_tracker.py:98
  - value_tracker.py:110
  - value_tracker.py:255
  - value_tracker.py:260
  - value_tracker.py:267
  - value_tracker.py:274
  - value_tracker.py:281
  - value_tracker.py:290
  - value_tracker.py:320
last_checked: 2026-04-04T19:51:14Z
derived_from:
  - cogops_compiler
  - sast
---

# Module: value_tracker

**Language:** py
**Lines of code:** 344

## Types
- `class ValueTracker:` — Persistent, thread-safe tracker for lifetime Entroly value.  Stores cumulative stats + daily/weekly/monthly breakdowns. Survives proxy restarts via atomic JSON file writes.

## Functions
- `def estimate_cost(tokens_saved: int, model: str = "") -> float` — Estimate USD saved for a given number of tokens and model.
- `def __init__(self, data_dir: Optional[Path] = None)`
- `def get_lifetime(self) -> Dict[str, Any]` — Return lifetime cumulative stats.
- `def get_daily(self, last_n: int = 30) -> List[Dict[str, Any]]` — Return last N days of daily stats, sorted ascending.
- `def get_weekly(self, last_n: int = 12) -> List[Dict[str, Any]]` — Return last N weeks of stats, sorted ascending.
- `def get_monthly(self, last_n: int = 12) -> List[Dict[str, Any]]` — Return last N months of stats, sorted ascending.
- `def get_session(self) -> Dict[str, Any]` — Return current session stats (since proxy start).
- `def get_confidence(self) -> Dict[str, Any]` — Return real-time confidence snapshot for IDE widgets.  This is the single endpoint an IDE status bar polls.
- `def get_trends(self) -> Dict[str, Any]` — Return all trend data for dashboard charts.
- `def get_tracker() -> ValueTracker` — Get or create the global ValueTracker singleton.
