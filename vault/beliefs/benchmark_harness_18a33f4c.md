---
claim_id: 18a33f4c123e6fbc042ffbbc
entity: benchmark_harness
status: inferred
confidence: 0.75
sources:
  - benchmark_harness.py:44
last_checked: 2026-04-04T19:51:14Z
derived_from:
  - cogops_compiler
  - sast
---

# Module: benchmark_harness

**Language:** py
**Lines of code:** 120


## Functions
- `def run_benchmark(engine: Any, budget_seconds: float = 10.0) -> Dict[str, Any]` —  Run the fixed evaluation payload and return the context_efficiency score.  READ ONLY — this function is the ground truth metric. autotune.py calls this but never modifies it. The engine and benchmark
