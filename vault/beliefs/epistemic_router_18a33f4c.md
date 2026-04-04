---
claim_id: 18a33f4c12e0ccf804d258f8
entity: epistemic_router
status: inferred
confidence: 0.75
sources:
  - epistemic_router.py:39
  - epistemic_router.py:56
  - epistemic_router.py:65
  - epistemic_router.py:77
  - epistemic_router.py:88
  - epistemic_router.py:98
  - epistemic_router.py:172
  - epistemic_router.py:180
  - epistemic_router.py:197
  - epistemic_router.py:310
last_checked: 2026-04-04T19:51:14Z
derived_from:
  - cogops_compiler
  - sast
---

# Module: epistemic_router

**Language:** py
**Lines of code:** 570

## Types
- `class EpistemicIntent(str, Enum):` — What kind of answer is needed.
- `class EpistemicFlow(str, Enum):` — Which canonical flow to execute.
- `class RiskLevel(str, Enum):` — How dangerous is the answer domain.
- `class BeliefCoverage:` — Result of checking belief coverage in the vault.
- `class RoutingDecision:` — The router's output: which flow to execute and why.
- `class EpistemicRouter:` —  Epistemic Ingress Controller.  Inspects 4 signals (intent, coverage, freshness, risk) and selects one of 5 canonical flows.  Primary home: Action Layer (server.py + query_refiner.py) Inputs from:  Be

## Functions
- `def to_dict(self) -> Dict[str, Any]`
- `def classify_intent(query: str) -> EpistemicIntent` — Classify a query into an epistemic intent using keyword patterns.
- `def assess_risk(query: str, intent: EpistemicIntent) -> RiskLevel` — Assess the risk level of a query domain.
- `def record_miss(self, query: str) -> None` — Explicitly record a miss (system couldn't answer satisfactorily).
- `def stats(self) -> Dict[str, Any]` — Return routing statistics for observability.
