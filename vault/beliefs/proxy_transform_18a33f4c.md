---
claim_id: 18a33f4c13988dfc058a19fc
entity: proxy_transform
status: inferred
confidence: 0.75
sources:
  - proxy_transform.py:62
  - proxy_transform.py:103
  - proxy_transform.py:121
  - proxy_transform.py:245
last_checked: 2026-04-04T19:51:14Z
derived_from:
  - cogops_compiler
  - sast
---

# Module: proxy_transform

**Language:** py
**Lines of code:** 860


## Functions
- `def extract_user_message(body: Dict[str, Any], provider: str) -> str` — Extract the latest user message text from the request body.
- `def extract_model(body: Dict[str, Any], path: str = "") -> str` — Extract the model name from the request body or URL path.  Gemini embeds the model in the URL path rather than the body: /v1beta/models/gemini-2.0-flash:generateContent
- `def compute_token_budget(model: str, config: ProxyConfig) -> int` — Compute the token budget for context injection.  When ECDB (Entropy-Calibrated Dynamic Budget) is disabled, uses the static context_fraction.  When enabled, use compute_dynamic_budget() instead — this
- `def calibrated_token_count(content: str, source: str = "") -> int` — Estimate token count using per-language char/token ratios.  More accurate than len/4 — saves 15-25% of wasted context budget.
