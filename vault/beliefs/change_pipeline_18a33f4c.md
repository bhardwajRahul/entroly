---
claim_id: 18a33f4c1262a9f4045435f4
entity: change_pipeline
status: inferred
confidence: 0.75
sources:
  - change_pipeline.py:36
  - change_pipeline.py:50
  - change_pipeline.py:60
  - change_pipeline.py:71
  - change_pipeline.py:82
  - change_pipeline.py:129
  - change_pipeline.py:208
  - change_pipeline.py:270
  - change_pipeline.py:277
  - change_pipeline.py:349
last_checked: 2026-04-04T19:51:14Z
derived_from:
  - cogops_compiler
  - sast
---

# Module: change_pipeline

**Language:** py
**Lines of code:** 425

## Types
- `class ChangeSet:` — A structured representation of a code change.
- `class BeliefDiff:` — How a code change affects existing beliefs.
- `class ReviewFinding:` — A code review finding.
- `class PRBrief:` — An auto-generated PR brief.
- `class ChangePipeline:` —  Processes code changes through the Change-Driven flow (④).  Pipeline: Diff → ChangeSet → BeliefDiff → PR Brief → Vault

## Functions
- `def to_markdown(self) -> str`
- `def parse_diff(diff_text: str, commit_message: str = "") -> ChangeSet` — Parse a unified diff into a structured ChangeSet.
- `def review_diff(diff_text: str) -> List[ReviewFinding]` — Review a diff for common issues.
- `def __init__(self, vault: VaultManager, verification: VerificationEngine)`
- `def refresh_docs(self, changed_files: List[str]) -> Dict[str, Any]` — Trigger belief refresh for changed files.
