---
claim_id: 18a33f4c132f4d740520d974
entity: multimodal
status: inferred
confidence: 0.75
sources:
  - multimodal.py:44
  - multimodal.py:768
last_checked: 2026-04-04T19:51:14Z
derived_from:
  - cogops_compiler
  - sast
---

# Module: multimodal

**Language:** py
**Lines of code:** 880

## Types
- `class ModalContent:` —  The normalized output of every multi-modal converter.  Always ingested via remember_fragment(text, source=source, ...). All metadata is embedded in `text` in a structured format.
- `class DiffHunk:`
