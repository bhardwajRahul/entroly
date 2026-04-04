---
claim_id: 18a33f4c12ff905c04f11c5c
entity: flow_orchestrator
status: inferred
confidence: 0.75
sources:
  - flow_orchestrator.py:35
  - flow_orchestrator.py:46
  - flow_orchestrator.py:59
last_checked: 2026-04-04T19:51:14Z
derived_from:
  - cogops_compiler
  - sast
---

# Module: flow_orchestrator

**Language:** py
**Lines of code:** 443

## Types
- `class FlowResult:` — Result of executing a canonical flow.
- `class FlowOrchestrator:` —  Executes the 5 canonical epistemic flows.  Takes a RoutingDecision from the EpistemicRouter and chains the appropriate pipeline steps together.

## Functions
- `def to_dict(self) -> Dict[str, Any]`
