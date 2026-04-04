---
claim_id: 18a33f4c13870f1405789b14
entity: proxy_config
status: inferred
confidence: 0.75
sources:
  - proxy_config.py:52
  - proxy_config.py:145
  - proxy_config.py:180
  - proxy_config.py:243
last_checked: 2026-04-04T19:51:14Z
derived_from:
  - cogops_compiler
  - sast
---

# Module: proxy_config

**Language:** py
**Lines of code:** 382

## Types
- `class ProxyConfig:` — Configuration for the entroly prompt compiler proxy.  Supports two configuration modes: 1. Explicit: set each parameter individually via env vars 2. Single-dial: set ENTROLY_QUALITY=[0,1] to auto-deri

## Functions
- `def context_window_for_model(model: str) -> int` — Look up context window size for a model name, with fuzzy prefix matching.
- `def resolve_quality(value: str) -> float` — Accept either a named preset or a float 0.0-1.0.
- `def from_env(cls) -> ProxyConfig` — Create config from environment variables, with tuning_config.json overlay.  Supports single-dial mode: set ENTROLY_QUALITY=0.0–1.0 to auto-derive all numeric params from Pareto-interpolated profiles.
