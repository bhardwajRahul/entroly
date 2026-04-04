---
claim_id: 18a33f4c1361c04c05534c4c
entity: provenance
status: inferred
confidence: 0.75
sources:
  - provenance.py:25
  - provenance.py:36
  - provenance.py:46
  - provenance.py:63
  - provenance.py:69
  - provenance.py:75
  - provenance.py:80
  - provenance.py:85
  - provenance.py:97
last_checked: 2026-04-04T19:51:14Z
derived_from:
  - cogops_compiler
  - sast
---

# Module: provenance

**Language:** py
**Lines of code:** 184

## Types
- `class FragmentProvenance:` — Provenance record for a single selected context fragment.
- `class ContextProvenance:` —  Full provenance record for one optimize_context call.  The hallucination_risk is computed from: 1. Fraction of selected fragments with verified sources 2. Average confidence of selection 3. Whether a

## Functions
- `def risk_contribution(self) -> str` — Contribution to hallucination risk.
- `def verified_fraction(self) -> float`
- `def avg_confidence(self) -> float`
- `def source_set(self) -> set` — Set of verified source files — use to check LLM citations.
- `def quality_flagged_sources(self) -> List[str]` — Sources with code quality issues.
- `def hallucination_risk(self) -> str` —  low    — all fragments file-backed, high confidence medium — some low-confidence fragments, or 1-2 unverified high   — significant unverified content or very low confidence
- `def to_dict(self) -> Dict[str, Any]`
