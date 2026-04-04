---
claim_id: 18a33f4c13e5a3f805d72ff8
entity: server
status: inferred
confidence: 0.75
sources:
  - server.py:73
  - server.py:83
  - server.py:133
  - server.py:136
  - server.py:142
  - server.py:163
  - server.py:174
  - server.py:340
  - server.py:363
  - server.py:367
last_checked: 2026-04-04T19:51:14Z
derived_from:
  - cogops_compiler
  - sast
---

# Module: server

**Language:** py
**Lines of code:** 2442

## Types
- `class _PyDedupIndex:` — Pure-Python SimHash-based deduplication index (LSH banding).
- `class _WilsonFeedbackTracker:` —  Python-exact port of the Rust FeedbackTracker in guardrails.rs.  Uses the Wilson score lower bound (the same formula Reddit uses for ranking) to produce a calibrated relevance multiplier for each fra
- `class EntrolyEngine:` —  Orchestrates all subsystems. Delegates math to Rust when available.  Rust handles: ingest, optimize, recall, stats, feedback, dep graph, ordering. Python handles: prefetch, checkpoint, MCP protocol.

## Functions
- `def py_analyze_query(query: str) -> dict:  # type: ignore[misc]` — Pure-Python stub when entroly_core is not available.
- `def py_refine_heuristic(query: str, context: str) -> str:  # type: ignore[misc]` — Pure-Python stub — returns query unchanged.
- `def __init__(self, hamming_threshold: int = 3)`
- `def insert(self, fragment_id: str, text: str) -> Optional[str]` — Insert a fragment. Returns duplicate_id if near-duplicate found.
- `def remove(self, fragment_id: str) -> None` — Remove a fragment from the index.
- `def stats(self) -> dict`
- `def __init__(self)`
- `def record_success(self, fragment_ids: list[str]) -> None`
- `def record_failure(self, fragment_ids: list[str]) -> None`
- `def learned_value(self, fragment_id: str) -> float`
- `def __init__(self, config: Optional[EntrolyConfig] = None)`
- `def advance_turn(self) -> None` — Advance the turn counter and apply Ebbinghaus decay.
- `def record_success(self, fragment_ids: List[str]) -> None` — Record that selected fragments led to a successful output.
- `def record_failure(self, fragment_ids: List[str]) -> None` — Record that selected fragments led to a failed output.
- `def record_reward(self, fragment_ids: List[str], reward: float) -> None` — Record a continuous reward signal for selected fragments.  Unlike record_success/failure (binary), this allows graded feedback: reward > 0 → positive signal (boost fragment weight) reward < 0 → negati
- `def set_model(self, model_name: str) -> None` — Auto-configure cache cost model from model name.  Covers 20+ models: OpenAI (gpt-4o, gpt-4, o1, o3), Anthropic (claude-3.5), Google (gemini), DeepSeek, Meta (llama), Mistral. Unknown models default to
- `def set_cache_cost_per_token(self, cost: float) -> None` — Set cost-per-token directly (power users only).  Most developers should use set_model() instead. Default is already $0.000015 (GPT-4o output) — no config needed.
- `def cache_clear(self) -> None` — Clear all cached LLM responses.  Useful when switching projects, after major refactors, or when cache correctness is suspect.
- `def cache_len(self) -> int` — Return the number of entries in the response cache.
- `def cache_is_empty(self) -> bool` — Check if the response cache is empty.
- `def cache_hit_rate(self) -> float` — Return the cache hit rate (0.0 to 1.0).  This is the primary observability metric for the EGSC cache. A healthy, warmed-up cache should show hit_rate > 0.3.
- `def checkpoint(self, metadata: Optional[Dict[str, Any]] = None) -> str` — Manually create a checkpoint.
- `def resume(self) -> Dict[str, Any]` — Resume from the latest checkpoint.
- `def get_stats(self) -> Dict[str, Any]` — Get comprehensive session statistics.
- `def explain_selection(self) -> Dict[str, Any]` — Explain why each fragment was included or excluded.
- `def create_mcp_server()` —  Create the MCP server with all tools registered.  Uses the FastMCP SDK for automatic tool schema generation from Python type hints and docstrings.
- `def explain_context() -> str` — Explain why each fragment was included or excluded in the last optimization.  Shows per-fragment scoring breakdowns with all dimensions visible: recency, frequency, semantic, entropy, feedback multipl
- `def resume_state() -> str` — Resume from the latest checkpoint.  Restores all context fragments, dedup index, co-access patterns, and custom metadata from the most recent checkpoint.
- `def get_stats() -> str` — Get comprehensive session statistics.  Shows token savings, duplicate detection counts, entropy distribution, dependency graph stats, checkpoint status, and cost estimates.
- `def entroly_dashboard() -> str` — Show the real, live value Entroly is providing to YOUR session right now.  Pulls from actual engine state — not synthetic data. Shows: Money saved: exact $ amounts from token optimization Performance:
- `def scan_for_vulnerabilities(content: str, source: str = "unknown") -> str` — Scan code content for security vulnerabilities (SAST analysis).  Uses a 55-rule engine with taint-flow simulation and CVSS-inspired scoring. Detects hardcoded secrets, SQL injection, path traversal, c
- `def security_report() -> str` — Generate a session-wide security audit across all ingested fragments.  Scans every fragment in the current session and returns an aggregated report showing: which fragments are most vulnerable, overal
- `def analyze_codebase_health() -> str` — Analyze the health of the ingested codebase.  Runs 5 analysis passes over all fragments in the current session: 1. Clone Detection — SimHash pairwise scan for Type-1/2/3 code clones 2. Dead Symbol Ana
- `def ingest_diagram(diagram_text: str, source: str, diagram_type: str = "auto") -> str` — Ingest an architecture or flow diagram into the context memory.  Converts Mermaid, PlantUML, DOT/Graphviz, or informal diagram text into a structured semantic fragment capturing nodes, edges, and rela
- `def ingest_voice(transcript: str, source: str) -> str` — Ingest a voice/meeting transcript into the context memory.  Converts pre-transcribed text (from Whisper, AssemblyAI, etc.) into a structured fragment capturing decisions, action items, open questions,
- `def ingest_diff(diff_text: str, source: str, commit_message: str = "") -> str` — Ingest a code diff/patch into the context memory.  Converts a unified diff (git diff output) into a structured change summary: intent classification (bug-fix/feature/refactor), symbols changed, files 
- `def vault_status() -> str` — Show the current state of the CogOps Knowledge Vault.  Initializes the vault directory structure if needed, then returns a coverage index: total beliefs, verification status, confidence distribution, 
- `def verify_beliefs() -> str` — Run a full verification pass on all beliefs in the vault.  Checks for: - Staleness (beliefs past their freshness window) - Contradictions (conflicting claims about the same entity) - Confidence diverg
- `def blast_radius(changed_files: str) -> str` — Analyze the blast radius of file changes on existing beliefs.  Given a list of changed files, determines which beliefs need re-verification, which may be invalidated, and the overall risk level (low/m
- `def main()` — Entry point for the entroly MCP server.
