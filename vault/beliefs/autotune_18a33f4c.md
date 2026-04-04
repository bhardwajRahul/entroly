---
claim_id: 18a33f4c121d1f38040eab38
entity: autotune
status: inferred
confidence: 0.75
sources:
  - autotune.py:66
  - autotune.py:76
  - autotune.py:82
  - autotune.py:88
  - autotune.py:94
  - autotune.py:206
  - autotune.py:234
  - autotune.py:320
  - autotune.py:335
  - autotune.py:462
last_checked: 2026-04-04T19:51:14Z
derived_from:
  - cogops_compiler
  - sast
---

# Module: autotune

**Language:** py
**Lines of code:** 821

## Types
- `class BenchResult:` — Result of running the benchmark suite with a given config.
- `class FeedbackJournal:` — Persistent cross-session feedback journal (.jsonl).
- `class TaskProfileOptimizer:` — Per-task-type weight optimization from feedback journal.

## Functions
- `def load_cases() -> List[Dict[str, Any]]` — Load the fixed benchmark cases (read-only val set).
- `def load_config() -> Dict[str, Any]` — Load the current tuning config (the file we mutate).
- `def save_config(config: Dict[str, Any]) -> None` — Save tuning config (single-file mutation).
- `def evaluate(config: Dict[str, Any], cases: List[Dict[str, Any]],`
- `def mutate_config(config: Dict[str, Any]) -> Dict[str, Any]` — Mutate one parameter at a time (single-change experiments for interpretability).
- `def composite_score(result: BenchResult, config: Optional[Dict[str, Any]] = None,`
- `def log_result(iteration: int, config: Dict[str, Any], result: BenchResult,`
- `def run_autotune(iterations: int = 100,`
- `def main()`
- `def __init__(self, journal_dir: str)`
- `def log(self, *, weights: Dict[str, float], reward: float,`
- `def load(self, max_age: float = JOURNAL_MAX_AGE_S) -> List[Dict]` — Load episodes filtered by max age.
- `def prune(self, max_age: float = JOURNAL_MAX_AGE_S) -> None` — Remove episodes older than max_age.
- `def count(self) -> int`
- `def stats(self) -> Dict[str, Any]`
- `def classify_query(query: str) -> str` — Classify a query into a task type.
- `def __init__(self, journal: FeedbackJournal)`
- `def optimize_all(self) -> Dict[str, Dict]` — Classify episodes by task type and optimize each independently.
- `def get_profile_for_query(self, query: str) -> Tuple[Dict[str, float], str, float]` — Get optimal weights for a query. Returns (weights, task_type, confidence).
- `def apply_to_engine(self, engine, query: str) -> Tuple[str, float]` — Apply task-conditioned weights to an engine.
