---
claim_id: 18a33f4c124f20640440ac64
entity: change_listener
status: inferred
confidence: 0.75
sources:
  - change_listener.py:42
  - change_listener.py:54
  - change_listener.py:69
  - change_listener.py:92
  - change_listener.py:152
  - change_listener.py:184
last_checked: 2026-04-04T19:51:14Z
derived_from:
  - cogops_compiler
  - sast
---

# Module: change_listener

**Language:** py
**Lines of code:** 239

## Types
- `class WorkspaceSyncResult:`
- `class WorkspaceChangeListener:` — Polls a workspace and feeds file changes into the belief pipeline.

## Functions
- `def to_dict(self) -> Dict[str, Any]`
- `def scan_once(self, force: bool = False, max_files: int = 100) -> WorkspaceSyncResult`
- `def start(self, interval_s: int = 120, max_files: int = 100, force_initial: bool = False) -> Dict[str, Any]`
- `def stop(self) -> Dict[str, Any]`
