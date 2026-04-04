---
claim_id: 18a33f4c1465ef5406577b54
entity: _docker_launcher
status: inferred
confidence: 0.75
sources:
  - _docker_launcher.py:91
last_checked: 2026-04-04T19:51:14Z
derived_from:
  - cogops_compiler
  - sast
---

# Module: _docker_launcher

**Language:** py
**Lines of code:** 194


## Functions
- `def launch() -> None` — Main entry point — docker launch or native fallback.  Routes CLI subcommands (init, dashboard, health, autotune, benchmark, status, proxy) to the local CLI handler. Only `serve` and bare `entroly` go 
