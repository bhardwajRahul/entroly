---
claim_id: 18a33f4c1375ccb8056758b8
entity: proxy
status: inferred
confidence: 0.75
sources:
  - proxy.py:104
  - proxy.py:111
  - proxy.py:119
  - proxy.py:130
  - proxy.py:135
  - proxy.py:143
  - proxy.py:148
  - proxy.py:154
  - proxy.py:161
  - proxy.py:173
last_checked: 2026-04-04T19:51:14Z
derived_from:
  - cogops_compiler
  - sast
---

# Module: proxy

**Language:** py
**Lines of code:** 1943

## Types
- `class _CircuitBreaker:` — SIRS-inspired circuit breaker: open after N consecutive failures, half-open after cooldown period, close on success.  Inspired by the refractory period in agentOS/scheduler.rs SIRS routing.
- `class _TokenBucket:` — Token bucket rate limiter.  Ported from agentOS/compliance.rs TokenBucket.
- `class _WelfordStats:` — Welford's online algorithm for streaming mean/variance.  Ported from agentOS/persona_manifold.rs Welford tracker. Tracks pipeline latency without storing all samples.
- `class ImplicitFeedbackTracker:` — Extract implicit RL feedback from proxy traffic.  Thread-safe. Per-client state tracks query trajectories for rephrase detection. Response text is scanned for confusion indicators to infer success/fai
- `class _CusumEmaDriftDetector:` — Dual online quality drift detector: CUSUM + EMA.  Combines two complementary algorithms from the change-point detection literature (Online Kernel CUSUM, arXiv 2025; RL drift detection, NeurIPS 2025): 
- `class PromptCompilerProxy:` — HTTP reverse proxy that optimizes every LLM request with entroly.

## Functions
- `def __init__(self, failure_threshold: int = 3, cooldown_s: float = 30.0)`
- `def allow_request(self) -> bool`
- `def record_success(self) -> None`
- `def record_failure(self) -> None`
- `def state(self) -> str`
- `def __init__(self, capacity: float, refill_per_second: float)`
- `def try_consume(self, cost: float = 1.0) -> bool`
- `def __init__(self)`
- `def add(self, x: float) -> None`
- `def variance(self) -> float`
- `def stddev(self) -> float`
- `def to_dict(self) -> dict`
- `def __init__(self)`
- `def assess_response(self, response_text: str) -> float` — Assess an LLM response for confusion vs confidence.  Returns a reward signal: -1.0  = strong confusion detected (multiple indicators) -0.5  = mild confusion detected (one indicator) 0.0  = ambiguous /
- `def record_assessment(self, reward: float) -> None` — Track assessment stats and feed the drift detector.
- `def quality_trend(self) -> str` — Return current quality trend: 'stable', 'declining', or 'improving'.
- `def stats(self) -> Dict[str, Any]` — Return feedback tracker statistics.
- `def __init__(self)`
- `def update(self, reward: float) -> None` — Feed a new reward observation.
- `def trend(self) -> str` — Return current quality trend.
- `def reset(self) -> None` — Reset detector state (e.g., on session restart).
- `def to_dict(self) -> Dict[str, Any]`
- `def __init__(self, engine: Any, config: Optional[ProxyConfig] = None)`
- `async def startup(self) -> None`
- `async def shutdown(self) -> None`
- `async def handle_proxy(self, request: Request) -> StreamingResponse | JSONResponse` — Main proxy handler — intercept, optimize, forward.  Uses pipelined async architecture: 1. Parse request + start HTTP connection warmup concurrently 2. Run Rust pipeline in thread pool (off event loop)
- `async def event_generator()`
