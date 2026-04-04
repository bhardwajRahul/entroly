---
claim_id: 18a33f4c13bf542805b0e028
entity: repo_map
status: inferred
confidence: 0.75
sources:
  - repo_map.py:18
  - repo_map.py:135
  - repo_map.py:158
last_checked: 2026-04-04T19:51:14Z
derived_from:
  - cogops_compiler
  - sast
---

# Module: repo_map

**Language:** py
**Lines of code:** 211

## Types
- `class FileMapEntry:`

## Functions
- `def build_repo_map(root: str | Path) -> Dict[str, List[FileMapEntry]]`
- `def render_repo_map_markdown(grouped: Dict[str, List[FileMapEntry]]) -> str`
