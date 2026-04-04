---
claim_id: 18a33f4c1402e2b005f46eb0
entity: skill_engine
status: inferred
confidence: 0.75
sources:
  - skill_engine.py:36
  - skill_engine.py:52
  - skill_engine.py:62
  - skill_engine.py:71
  - skill_engine.py:172
  - skill_engine.py:175
  - skill_engine.py:178
  - skill_engine.py:230
  - skill_engine.py:233
  - skill_engine.py:236
last_checked: 2026-04-04T19:51:14Z
derived_from:
  - cogops_compiler
  - sast
---

# Module: skill_engine

**Language:** py
**Lines of code:** 520

## Types
- `class SkillSpec:` — A skill specification.
- `class BenchmarkResult:` — Result of running a skill benchmark.
- `class SkillSynthesizer:` — Generates skill specs from gap reports and failure patterns.
- `class SandboxedRunner:` — Runs skill tools in isolation.
- `class SkillBenchmark:` — Evaluates skill fitness by running test cases.
- `class SkillEngine:` —  Full skill lifecycle manager.  Creates skills from gap reports, benchmarks them, promotes or prunes, and maintains the registry.

## Functions
- `def success_rate(self) -> float`
- `def __init__(self, timeout_seconds: float = 10.0)`
- `def run_tool(self, tool_code: str, query: str) -> Dict[str, Any]` — Execute a skill tool in a subprocess sandbox.
- `def __init__(self, runner: Optional[SandboxedRunner] = None)`
- `def benchmark(self, skill: SkillSpec) -> BenchmarkResult` — Run all test cases for a skill and compute fitness.
- `def __init__(self, vault: VaultManager)`
- `def benchmark_skill(self, skill_id: str) -> Dict[str, Any]` — Benchmark a skill and update its metrics.
- `def promote_or_prune(self, skill_id: str) -> Dict[str, Any]` — Evaluate a skill for promotion or pruning.
- `def list_skills(self) -> List[Dict[str, Any]]` — List all skills in the registry.
