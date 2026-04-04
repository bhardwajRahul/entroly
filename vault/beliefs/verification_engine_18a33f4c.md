---
claim_id: 18a33f4c14541a180645a618
entity: verification_engine
status: inferred
confidence: 0.75
sources:
  - verification_engine.py:35
  - verification_engine.py:46
  - verification_engine.py:57
  - verification_engine.py:65
  - verification_engine.py:75
  - verification_engine.py:87
  - verification_engine.py:118
  - verification_engine.py:136
  - verification_engine.py:180
  - verification_engine.py:260
last_checked: 2026-04-04T19:51:14Z
derived_from:
  - cogops_compiler
  - sast
---

# Module: verification_engine

**Language:** py
**Lines of code:** 485

## Types
- `class Contradiction:` — Two beliefs that disagree.
- `class StaleReport:` — A belief past its freshness window.
- `class CoverageGap:` — A file or module with no corresponding belief.
- `class BlastRadius:` — Impact analysis result for a change.
- `class VerificationReport:` — Complete verification pass result.
- `class VerificationEngine:` —  Runs automated verification passes against the belief vault.  This is the Belief CI engine — it challenges every belief in the vault and produces verification artifacts.

## Functions
- `def to_dict(self) -> Dict[str, Any]`
- `def full_verification_pass(self) -> VerificationReport` — Run a complete verification pass on all beliefs.
- `def check_belief(self, claim_id: str) -> Dict[str, Any]` — Verify a single belief by claim_id.
- `def coverage_gaps(self, source_dir: str) -> List[CoverageGap]` — Find source files with no corresponding belief.
