"""
Entroly MCP Server
========================

Thin MCP wrapper around the Rust EntrolyEngine.

All computation (knapsack, entropy, SimHash, dep graph, feedback loop,
context ordering) runs in Rust via PyO3. Python only handles:
  - MCP protocol (FastMCP tool registration + JSON-RPC)
  - Predictive pre-fetching (static analysis + co-access learning)
  - Checkpoint I/O (gzipped JSON serialization)

Architecture:
  MCP Client → JSON-RPC → Python (FastMCP) → PyO3 → Rust Engine → Results

Supported clients:
  - Cursor (add to .cursor/mcp.json)
  - Claude Code (claude mcp add)
  - Cline (add to mcp settings)
  - Any MCP-compatible client

Run:
    entroly        # Start as STDIO server
    python -m entroly.server   # Alternative
"""

from __future__ import annotations

import gc
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import EntrolyConfig
from .prefetch import PrefetchEngine
from .checkpoint import CheckpointManager
from .query_refiner import QueryRefiner
from .adaptive_pruner import EntrolyPruner, FragmentGuard
from .provenance import build_provenance, ContextProvenance
from .multimodal import ingest_image as _mm_image, ingest_diagram as _mm_diagram
from .multimodal import ingest_voice as _mm_voice, ingest_diff as _mm_diff
# ── Rust engine import (required) ──────────────────────────────────
try:
    from entroly_core import EntrolyEngine as RustEngine
    from entroly_core import py_analyze_query, py_refine_heuristic
    _RUST_AVAILABLE = True
except ImportError as _rust_err:
    raise ImportError(
        "entroly-core Rust extension is required but not installed.\n"
        "Install with:  pip install entroly[native]\n"
        "Or build from source:  cd entroly-core && maturin develop --release\n"
        f"Original error: {_rust_err}"
    )

# Configure logging to stderr (MCP requires stdout for JSON-RPC)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [entroly] %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("entroly")


class EntrolyEngine:
    """
    Orchestrates all subsystems. Delegates math to Rust when available.

    Rust handles: ingest, optimize, recall, stats, feedback, dep graph, ordering.
    Python handles: prefetch, checkpoint, MCP protocol.
    """

    def __init__(self, config: Optional[EntrolyConfig] = None):
        self.config = config or EntrolyConfig()
        self._use_rust = True  # Rust engine is required (see import above)

        self._rust = RustEngine(
            w_recency=self.config.weight_recency,
            w_frequency=self.config.weight_frequency,
            w_semantic=self.config.weight_semantic_sim,
            w_entropy=self.config.weight_entropy,
            decay_half_life=self.config.decay_half_life_turns,
            min_relevance=self.config.min_relevance_threshold,
        )
        logger.info("Using Rust engine (entroly_core)")

        # Shared stats (used by both Rust and Python paths, e.g. SSSL filtering)
        self._total_tokens_saved: int = 0

        # Python-only subsystems
        self._prefetch = PrefetchEngine(co_access_window=5)
        self._checkpoint_mgr = CheckpointManager(
            checkpoint_dir=self.config.checkpoint_dir,
            auto_interval=self.config.auto_checkpoint_interval,
        )
        # Query refinement: vague queries are expanded using in-memory file
        # context before context selection, reducing hallucination from wrong files.
        self._refiner = QueryRefiner()

        # ebbiforge CodeQualityGuard: scans ingested fragments for secrets/TODO/unsafe
        self._guard = FragmentGuard()
        # ebbiforge AdaptivePruner: RL weight learning on feedback
        self._pruner = EntrolyPruner()
        # Turn counter for provenance
        self._turn_counter: int = 0

        # Fix #5: Validate that the checkpoint directory is writable at startup.
        # Fail fast with a clear error rather than a cryptic gzip/PermissionError
        # during the first auto-checkpoint (which could happen mid-session).
        self._validate_checkpoint_dir()

        # ── Persistent Repo-Level Indexing ──
        # On startup, try to load a previous session's index for instant warm retrieval.
        # Index is stored at <checkpoint_dir>/index.json.gz (gzip-compressed JSON).
        self._index_path = str(Path(self.config.checkpoint_dir) / "index.json.gz")
        try:
            loaded = self._rust.load_index(self._index_path)
            if loaded:
                n = self._rust.fragment_count()
                logger.info(f"Loaded persistent index: {n} fragments from {self._index_path}")
            else:
                logger.info("No persistent index found, starting fresh session")
        except Exception as e:
            logger.warning(f"Failed to load persistent index: {e}")

        # GC tuning: increase thresholds to reduce pause frequency.
        # Default (700, 10, 10) causes ~500ms stalls on large heaps.
        # We raise gen0 threshold and manually collect every N turns.
        self._gc_collect_interval = 50  # collect every 50 turns
        gc.collect()
        gc.set_threshold(5000, 15, 15)

    def advance_turn(self) -> None:
        """Advance the turn counter and apply Ebbinghaus decay."""
        # Periodic GC amortization: frozen at init, collect every N turns
        if self._turn_counter > 0 and self._turn_counter % self._gc_collect_interval == 0:
            gc.collect()

        self._rust.advance_turn()

    def ingest_fragment(
        self,
        content: str,
        source: str = "",
        token_count: int = 0,
        is_pinned: bool = False,
    ) -> Dict[str, Any]:
        """Ingest a new context fragment."""
        result = self._rust.ingest(content, source, token_count, is_pinned)
        # result is a dict from PyO3
        if source:
            self._prefetch.record_access(source, self._rust.get_turn())
        if self._checkpoint_mgr.should_auto_checkpoint():
            self._auto_checkpoint()
        return dict(result)

    def optimize_context(
        self,
        token_budget: int = 0,
        query: str = "",
    ) -> Dict[str, Any]:
        """Select the mathematically optimal subset of context fragments."""
        if token_budget <= 0:
            token_budget = self.config.default_token_budget

        # Query refinement: expand vague queries using in-memory file context.
        # This is the key fix for hallucination from incomplete context:
        # "fix the bug" → "bug fix in payments module (Python/Rust) involving
        # payment processing, error handling"
        refined_query = query
        refinement_info: Dict[str, Any] = {}
        if query:
            fragment_summaries = []
            try:
                recalled = list(self._rust.recall(query, 20))
                fragment_summaries = [r.get("content", "") for r in recalled]
            except Exception:
                pass
            # analyze() returns dict: vagueness_score, key_terms, needs_refinement, reason
            analysis_dict = self._refiner.analyze(query, fragment_summaries)
            # refine() returns the improved query string
            refined_query = self._refiner.refine(query, fragment_summaries)
            if analysis_dict.get("needs_refinement"):
                refinement_info = {
                    "original_query":    query,
                    "refined_query":     refined_query,
                    "vagueness_score":   analysis_dict["vagueness_score"],
                    "refinement_source": "rust_heuristic",
                    "key_terms":         analysis_dict.get("key_terms", []),
                }

        result = self._rust.optimize(token_budget, refined_query)
        result = dict(result)
        if refinement_info:
            result["query_refinement"] = refinement_info
        return self._apply_sssl_filtering(result)

    def _apply_sssl_filtering(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sliding Window Relevance Filter (SSSL pattern).
        Trims long tails of low-relevance context fragments that drag down LLM attention.
        """
        # Rust returns "selected", normalize to "selected_fragments" for MCP consumers
        selected = result.pop("selected", result.get("selected_fragments", []))
        if not selected or len(selected) < 3:
            result["selected_fragments"] = selected
            return result

        # Convert PyO3 dicts to plain dicts
        selected = [dict(f) for f in selected]

        relevances = [f.get("relevance", 0.0) for f in selected]
        max_rel = max(relevances) if relevances else 0.0

        # Dynamic cutoff based on the highest relevance in this specific query
        cutoff_threshold = max(0.05, max_rel * 0.3)

        filtered_selected = []
        tokens_purged = 0

        for f in selected:
            if f.get("relevance", 0.0) >= cutoff_threshold or f.get("is_pinned", False):
                filtered_selected.append(f)
            else:
                tokens_purged += f.get("token_count", 0)

        result["selected_fragments"] = filtered_selected

        # Update top-level total_tokens (Rust puts it at top level, not nested)
        if "total_tokens" in result:
            result["total_tokens"] = max(0, result["total_tokens"] - tokens_purged)

        result["sssl_tokens_purged"] = tokens_purged
        if tokens_purged > 0:
            self._total_tokens_saved += tokens_purged

        return result

    def recall_relevant(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """Semantic recall of relevant fragments."""
        result = self._rust.recall(query, top_k)
        return [dict(r) for r in result]

    def record_success(self, fragment_ids: List[str]) -> None:
        """Record that selected fragments led to a successful output."""
        self._rust.record_success(fragment_ids)
        # Fix #2: Wire AdaptivePruner RL feedback on every record_success call.
        # apply_feedback(+1.0) boosts the learned weights for features that
        # were present when the fragment was selected and led to success.
        for fid in fragment_ids:
            self._pruner.apply_feedback(fid, 1.0)

    def record_failure(self, fragment_ids: List[str]) -> None:
        """Record that selected fragments led to a failed output."""
        self._rust.record_failure(fragment_ids)
        # Fix #2: Wire AdaptivePruner RL feedback on every record_failure call.
        # apply_feedback(-1.0) down-weights feature combinations that led to
        # unhelpful context selections.
        for fid in fragment_ids:
            self._pruner.apply_feedback(fid, -1.0)


    def prefetch_related(
        self,
        file_path: str,
        source_content: str = "",
        language: str = "python",
    ) -> List[Dict[str, Any]]:
        """Predict and pre-load likely-needed context."""
        predictions = self._prefetch.predict(
            file_path, source_content, language
        )
        return [
            {
                "path": p.path,
                "reason": p.reason,
                "confidence": p.confidence,
            }
            for p in predictions
        ]

    def checkpoint(self, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Manually create a checkpoint."""
        return self._auto_checkpoint(metadata)

    def resume(self) -> Dict[str, Any]:
        """Resume from the latest checkpoint."""
        ckpt = self._checkpoint_mgr.load_latest()
        if ckpt is None:
            return {"status": "no_checkpoint_found"}

        # Try to restore from full engine state (preferred)
        engine_state = ckpt.metadata.get("engine_state") if ckpt.metadata else None
        if engine_state:
            self._rust.import_state(engine_state)
        else:
            # Fallback: re-create engine and re-ingest fragments
            self._rust = RustEngine(
                w_recency=self.config.weight_recency,
                w_frequency=self.config.weight_frequency,
                w_semantic=self.config.weight_semantic_sim,
                w_entropy=self.config.weight_entropy,
                decay_half_life=self.config.decay_half_life_turns,
                min_relevance=self.config.min_relevance_threshold,
            )
            for frag_data in ckpt.fragments:
                self._rust.ingest(
                    frag_data["content"],
                    frag_data.get("source", ""),
                    frag_data.get("token_count", 0),
                    frag_data.get("is_pinned", False),
                )

        # Restore co-access patterns
        from collections import Counter
        for src, targets in ckpt.co_access_data.items():
            self._prefetch._co_access[src] = Counter(targets)

        return {
            "status": "resumed",
            "checkpoint_id": ckpt.checkpoint_id,
            "restored_fragments": len(ckpt.fragments),
            "restored_turn": ckpt.current_turn,
            "metadata": ckpt.metadata,
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive session statistics."""
        rust_stats = dict(self._rust.stats())
        dep_stats = dict(self._rust.dep_graph_stats())
        rust_stats["dep_graph"] = dep_stats
        rust_stats["prefetch"] = self._prefetch.stats()
        rust_stats["checkpoint"] = self._checkpoint_mgr.stats()
        return rust_stats

    def explain_selection(self) -> Dict[str, Any]:
        """Explain why each fragment was included or excluded."""
        result = self._rust.explain_selection()
        return dict(result)

    def _auto_checkpoint(
        self,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Create an auto-checkpoint."""
        co_access = {
            k: dict(v)
            for k, v in self._prefetch._co_access.items()
        }
        # Export full engine state (not empty fragments)
        engine_state = self._rust.export_state()
        # Auto-persist repo-level index alongside checkpoint
        try:
            self._rust.persist_index(self._index_path)
        except Exception as e:
            logger.warning(f"Failed to persist index: {e}")
        return self._checkpoint_mgr.save(
            fragments=[],
            dedup_fingerprints={},
            co_access_data=co_access,
            current_turn=self._rust.get_turn(),
            metadata={**(metadata or {}), "engine_state": engine_state},
            stats=self.get_stats(),
        )

    def _validate_checkpoint_dir(self) -> None:
        """Fix #5: Validate checkpoint directory is writable at startup."""
        import os
        ckpt_dir = self.config.checkpoint_dir
        try:
            os.makedirs(ckpt_dir, exist_ok=True)
            # Write a probe file to confirm the directory is actually writable
            probe = os.path.join(str(ckpt_dir), ".entroly_write_probe")
            with open(probe, "w") as f:
                f.write("ok")
            os.unlink(probe)
        except OSError as e:
            raise RuntimeError(
                f"Entroly checkpoint directory '{ckpt_dir}' is not writable: {e}.\n"
                f"Set the ENTROLY_DIR env var or pass checkpoint_dir= to EntrolyConfig "
                f"to point to a writable location."
            ) from e



# ══════════════════════════════════════════════════════════════════════
# MCP Server Definition
# ══════════════════════════════════════════════════════════════════════

def create_mcp_server():
    """
    Create the MCP server with all tools registered.

    Uses the FastMCP SDK for automatic tool schema generation
    from Python type hints and docstrings.
    """
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError:
        logger.error(
            "MCP SDK not installed. Install with: pip install mcp"
        )
        raise

    mcp = FastMCP(
        "entroly",
        instructions=(
            "Information-theoretic context optimization for AI coding agents. "
            "Knapsack-optimal token budgeting, Shannon entropy scoring, "
            "SimHash deduplication, predictive pre-fetch, and checkpoint/resume."
        ),
    )

    # Shared engine instance
    engine = EntrolyEngine()

    @mcp.tool()
    def remember_fragment(
        content: str,
        source: str = "",
        token_count: int = 0,
        is_pinned: bool = False,
    ) -> str:
        """Store a context fragment with automatic dedup and entropy scoring.

        Fragments are fingerprinted via SimHash for O(1) duplicate detection.
        Each fragment's information density is scored using Shannon entropy.
        Duplicates are automatically merged with salience boosting.

        Args:
            content: The text content to store (code, tool output, etc.)
            source: Origin label (e.g., 'file:utils.py', 'tool:grep')
            token_count: Token count (auto-estimated if 0)
            is_pinned: If True, always include in optimized context
        """
        # NOTE: turn is NOT advanced here — turns advance on optimize/recall
        result = engine.ingest_fragment(content, source, token_count, is_pinned)
        # CodeQualityGuard: scan for secrets, TODOs, unsafe blocks
        issues = engine._guard.scan(content, source)
        if issues:
            result["quality_issues"] = issues
        return json.dumps(result, indent=2)

    @mcp.tool()
    def optimize_context(
        token_budget: int = 128000,
        query: str = "",
    ) -> str:
        """Select the mathematically optimal context subset for a token budget.

        Uses 0/1 Knapsack dynamic programming to maximize relevance within
        the budget. Scores fragments on four dimensions: recency (Ebbinghaus
        decay), access frequency (spaced repetition), semantic similarity
        (SimHash), and information density (Shannon entropy).

        QUERY REFINEMENT: Vague queries like "fix the bug" or "add feature"
        are automatically expanded into precise master prompts using the files
        already in memory. This improves context selection accuracy and reduces
        hallucination from selecting wrong files. The response includes
        query_refinement.refined_query so you can see what drove selection.

        Output is ordered for optimal LLM attention: pinned/critical first,
        high-dependency foundation files early, then by relevance.

        This is the core tool — call it before sending context to the LLM.

        Args:
            token_budget: Maximum tokens allowed (default: 128K)
            query: Current query/task for semantic relevance scoring (can be vague)
        """
        engine._turn_counter += 1
        engine.advance_turn()  # One turn per optimization request
        result = engine.optimize_context(token_budget, query)
        # Build ContextProvenance (hallucination_risk, source_set, per-fragment risk)
        provenance = build_provenance(
            optimize_result=result,
            query=result.get("query", query),
            refined_query=result.get("query_refinement", {}).get("refined_query") if isinstance(result.get("query_refinement"), dict) else None,
            turn=engine._turn_counter,
            token_budget=token_budget,
            quality_scan_fn=engine._guard.scan if engine._guard.available else None,
        )
        result["provenance"] = provenance.to_dict()
        return json.dumps(result, indent=2)

    @mcp.tool()
    def recall_relevant(
        query: str,
        top_k: int = 5,
    ) -> str:
        """Semantic recall of the most relevant stored fragments.

        Uses SimHash fingerprint distance + multi-dimensional scoring
        with feedback loop (fragments that previously led to successful
        outputs are boosted).

        Args:
            query: The search query
            top_k: Number of results to return
        """
        results = engine.recall_relevant(query, top_k)
        return json.dumps(results, indent=2)

    @mcp.tool()
    def record_outcome(
        fragment_ids: str,
        success: bool = True,
    ) -> str:
        """Record whether selected fragments led to a successful output.

        This feeds the reinforcement learning loop: fragments that
        contribute to successful outputs get boosted in future selections,
        while unhelpful fragments get suppressed.

        Args:
            fragment_ids: Comma-separated fragment IDs
            success: True if output was good, False if bad
        """
        ids = [fid.strip() for fid in fragment_ids.split(",") if fid.strip()]
        if success:
            engine.record_success(ids)
        else:
            engine.record_failure(ids)
        return json.dumps({
            "status": "recorded",
            "fragment_ids": ids,
            "outcome": "success" if success else "failure",
        }, indent=2)

    @mcp.tool()
    def explain_context() -> str:
        """Explain why each fragment was included or excluded in the last optimization.

        Shows per-fragment scoring breakdowns with all dimensions visible:
        recency, frequency, semantic, entropy, feedback multiplier,
        dependency boost, criticality, and composite score.

        Also shows context sufficiency (what % of referenced symbols
        have definitions included) and any exploration swaps.

        Call this after optimize_context to understand selection decisions.
        """
        result = engine.explain_selection()
        return json.dumps(result, indent=2)

    @mcp.tool()
    def checkpoint_state(
        task_description: str = "",
        current_step: str = "",
    ) -> str:
        """Save current state to disk for crash recovery and session resume.

        Checkpoints include all fragments, dedup index, co-access patterns,
        and custom metadata. Stored as gzipped JSON (~50-200 KB).

        Args:
            task_description: What the agent is working on
            current_step: Where in the task it currently is
        """
        metadata = {}
        if task_description:
            metadata["task"] = task_description
        if current_step:
            metadata["step"] = current_step

        path = engine.checkpoint(metadata)
        return json.dumps({
            "status": "checkpoint_saved",
            "path": path,
        }, indent=2)

    @mcp.tool()
    def resume_state() -> str:
        """Resume from the latest checkpoint.

        Restores all context fragments, dedup index, co-access patterns,
        and custom metadata from the most recent checkpoint.
        """
        result = engine.resume()
        return json.dumps(result, indent=2)

    @mcp.tool()
    def prefetch_related(
        file_path: str,
        source_content: str = "",
        language: str = "python",
    ) -> str:
        """Predict and pre-load context that will likely be needed next.

        Combines static analysis (imports, callees, test files) with
        learned co-access patterns to predict what the agent will need.

        Args:
            file_path: The file currently being accessed
            source_content: The source code content (for static analysis)
            language: Programming language (python, typescript, rust)
        """
        predictions = engine.prefetch_related(file_path, source_content, language)
        return json.dumps(predictions, indent=2)


    @mcp.tool()
    def get_stats() -> str:
        """Get comprehensive session statistics.

        Shows token savings, duplicate detection counts, entropy
        distribution, dependency graph stats, checkpoint status,
        and cost estimates.
        """
        stats = engine.get_stats()
        return json.dumps(stats, indent=2)

    @mcp.tool()
    def entroly_dashboard() -> str:
        """Show the real, live value Entroly is providing to YOUR session right now.

        Pulls from actual engine state — not synthetic data. Shows:
            Money saved: exact $ amounts from token optimization
            Performance: sub-millisecond selection speed vs API latency
            Bloat prevention: context compression ratio and memory footprint
            Selection quality: per-fragment scoring and context sufficiency
            Safety: duplicates caught, stale fragments filtered

        Call this anytime to see exactly what Entroly is doing for you.
        """
        stats = engine.get_stats()
        explanation = engine.explain_selection()

        # ── Real session metrics ──
        session = stats.get("session", {})
        savings = stats.get("savings", {})
        dep = stats.get("dep_graph", {})
        perf = stats.get("performance", {})
        mem = stats.get("memory", {})
        ctx_eff = stats.get("context_efficiency", {})
        checkpoint = stats.get("checkpoint", {})

        total_frags = session.get("total_fragments", 0)
        total_tokens = session.get("total_tokens_tracked", 0)
        current_turn = session.get("current_turn", 0)
        pinned = session.get("pinned", 0)

        tokens_saved = savings.get("total_tokens_saved", 0)
        dupes = savings.get("total_duplicates_caught", 0)
        total_opts = savings.get("total_optimizations", 0)
        total_ingested = savings.get("total_fragments_ingested", 0)

        # ── 💰 MONEY ──
        naive_cost = mem.get("naive_cost_per_call_usd", 0)
        optimized_cost = mem.get("optimized_cost_per_call_usd", 0)
        cost_saved_usd = savings.get("estimated_cost_saved_usd", 0)
        savings_pct = ((naive_cost - optimized_cost) / max(naive_cost, 1e-9)) * 100 if naive_cost > 0 else 0
        session_roi = naive_cost * total_opts - optimized_cost * total_opts

        # ── ⚡ PERFORMANCE ──
        avg_us = perf.get("avg_optimize_us", 0)
        peak_us = perf.get("peak_optimize_us", 0)
        avg_ms = avg_us / 1000
        # Typical API call is 500-3000ms; show the multiplier
        api_latency_ms = 2000  # typical GPT-4 API latency
        speedup = api_latency_ms / max(avg_ms, 0.001) if avg_ms > 0 else 0

        # ── 🧠 BLOAT PREVENTION ──
        compression = perf.get("context_compression", 1.0)
        bloat_prevented_pct = max(0, (1 - compression) * 100)
        mem_kb = mem.get("total_kb", 0)
        content_kb = mem.get("content_kb", 0)

        # ── 🎯 QUALITY ──
        info_efficiency = ctx_eff.get("context_efficiency", 0)
        dedup_rate = (dupes / max(total_ingested, 1)) * 100

        # ── Last optimization breakdown ──
        last_opt = None
        if not explanation.get("error"):
            included = [dict(f) for f in explanation.get("included", [])]
            excluded = [dict(f) for f in explanation.get("excluded", [])]
            sufficiency = explanation.get("sufficiency", 0)

            selected_summary = []
            for frag in included:
                scores = dict(frag.get("scores", {}))
                selected_summary.append({
                    "source": frag.get("source", ""),
                    "score": scores.get("composite", 0),
                    "top_signal": max(
                        [("recency", scores.get("recency", 0)),
                         ("semantic", scores.get("semantic", 0)),
                         ("entropy", scores.get("entropy", 0)),
                         ("frequency", scores.get("frequency", 0))],
                        key=lambda x: x[1]
                    )[0],
                    "reason": frag.get("reason", ""),
                })

            excluded_summary = []
            for frag in excluded[:5]:
                scores = dict(frag.get("scores", {}))
                excluded_summary.append({
                    "source": frag.get("source", ""),
                    "score": scores.get("composite", 0),
                    "reason": frag.get("reason", ""),
                })

            proto_dist = {}
            for f in included:
                p = f.get("prototype", "Unknown")
                proto_dist[p] = proto_dist.get(p, 0) + 1

            last_opt = {
                "context_sufficiency": f"{sufficiency:.0%}",
                "selected": len(included),
                "excluded": len(excluded),
                "pailitao_vl_prototype_distribution": proto_dist,
                "fragments_selected": selected_summary,
                "fragments_excluded": excluded_summary,
            }

        dashboard = {
            "money": {
                "tokens_saved_total": f"{tokens_saved:,}",
                "cost_saved_total_usd": f"${cost_saved_usd:.4f}",
                "cost_per_call_without_entroly": f"${naive_cost:.4f}",
                "cost_per_call_with_entroly": f"${optimized_cost:.4f}",
                "savings_pct": f"{savings_pct:.0f}%",
                "session_roi_usd": f"${session_roi:.4f}",
                "insight": (
                    f"Each optimize call costs ${optimized_cost:.4f} instead of ${naive_cost:.4f}. "
                    f"Over {total_opts} calls, that's ${session_roi:.4f} saved."
                    if total_opts > 0 else "Run optimize_context to see savings."
                ),
            },
            "performance": {
                "avg_optimize_latency": f"{avg_us:.0f}µs ({avg_ms:.2f}ms)",
                "peak_optimize_latency": f"{peak_us:.0f}µs",
                "vs_api_roundtrip": f"{speedup:.0f}x faster than a typical API call" if speedup > 0 else "N/A",
                "total_optimizations": total_opts,
                "insight": (
                    f"Context selection takes {avg_us:.0f}µs — that's {speedup:.0f}x faster "
                    f"than waiting for an API response."
                    if avg_us > 0 else "No optimizations run yet."
                ),
            },
            "bloat_prevention": {
                "total_tokens_in_memory": f"{total_tokens:,}",
                "context_compression": f"{compression:.2%}" if compression < 1 else "N/A (no optimize yet)",
                "bloat_filtered": f"{bloat_prevented_pct:.0f}% of context is noise that gets filtered",
                "duplicates_caught": f"{dupes} ({dedup_rate:.0f}% dedup rate)",
                "memory_footprint": f"{mem_kb} KB ({content_kb} KB content + {mem_kb - content_kb} KB metadata)",
                "insight": (
                    f"Entroly keeps {total_frags} fragments in {mem_kb} KB of memory. "
                    f"Without dedup, {dupes} duplicate fragments would bloat your context by "
                    f"~{dupes * (total_tokens // max(total_frags, 1)):,} extra tokens."
                    if total_frags > 0 else "Ingest some code to see memory stats."
                ),
            },
            "selection_quality": {
                "information_density": f"{info_efficiency:.4f} bits/token",
                "avg_entropy": f"{session.get('avg_entropy', 0):.4f}",
                "fragments_tracked": total_frags,
                "pinned_fragments": pinned,
                "dependency_edges": dep.get("edges", dep.get("total_edges", 0)),
                "turns_processed": current_turn,
                "insight": (
                    f"Entroly ranks {total_frags} fragments across {current_turn} turns. "
                    f"Information density: {info_efficiency:.4f} bits/token — higher = "
                    f"more valuable context per token spent."
                    if total_frags > 0 else "Ingest code to see quality metrics."
                ),
            },
            "safety": {
                "duplicates_blocked": dupes,
                "stale_fragments_deprioritized": f"Ebbinghaus decay active (half-life: 15 turns)",
                "persistent_index": "active" if hasattr(engine, '_index_path') else "disabled",
                "checkpoints": checkpoint.get("total_checkpoints", 0),
            },
        }

        if last_opt:
            dashboard["last_optimization"] = last_opt

        # ── 🔬 AUTOTUNE (live background self-tuning status) ──
        try:
            import csv
            from pathlib import Path as _p
            tc_path = _p(__file__).parent.parent / "tuning_config.json"
            tc = json.loads(tc_path.read_text()) if tc_path.exists() else {}
            at_cfg = tc.get("autotuner", {})
            strategy_name = at_cfg.get("strategy", "auto")
            time_budget_ms = at_cfg.get("time_budget_ms", 500)
            idle_only = at_cfg.get("idle_only", True)

            results_tsv = _p(__file__).parent.parent / "bench" / "results.tsv"
            at_total = 0
            at_improvements = 0
            at_best = 0.0
            at_last_strategy = strategy_name
            if results_tsv.exists():
                with open(results_tsv) as _f:
                    _rows = list(csv.DictReader(_f, delimiter="\t"))
                    at_total = len(_rows)
                    kept = [r for r in _rows if r.get("kept") == "keep"]
                    at_improvements = len(kept)
                    at_best = max((float(r["new_score"]) for r in kept), default=0.0)
                    if _rows:
                        at_last_strategy = _rows[-1].get("strategy", strategy_name)

            dashboard["autotune"] = {
                "strategy": strategy_name,
                "resolved_to": at_last_strategy if strategy_name == "auto" else strategy_name,
                "experiments_run": at_total,
                "improvements_kept": at_improvements,
                "best_score": round(at_best, 4),
                "time_budget_guarantee": f"{time_budget_ms}ms hard limit",
                "idle_only": idle_only,
                "insight": (
                    f"Self-tuning has run {at_total} experiments and found "
                    f"{at_improvements} improvements. Best score: {at_best:.4f}. "
                    f"Runs automatically in the background when your machine is idle."
                    if at_total > 0
                    else "Self-tuning starts automatically in the background when your machine is idle."
                ),
            }
        except Exception:
            pass  # Dashboard never fails due to autotune

        return json.dumps(dashboard, indent=2)


    @mcp.tool()
    def scan_for_vulnerabilities(content: str, source: str = "unknown") -> str:
        """Scan code content for security vulnerabilities (SAST analysis).

        Uses a 55-rule engine with taint-flow simulation and CVSS-inspired
        scoring. Detects hardcoded secrets, SQL injection, path traversal,
        command injection, insecure cryptography, unsafe deserialization,
        XSS, and authentication misconfigurations.

        Args:
            content: The source code to scan.
            source:  File path / identifier (used for language detection
                     and confidence scoring). E.g. "auth/login.py".

        Returns JSON with:
            - findings: [{rule_id, cwe, severity, line_number, description,
                          fix, confidence, taint_flow}]
            - risk_score: CVSS-inspired aggregate [0.0, 10.0]
            - critical_count, high_count, medium_count, low_count
            - top_fix: most impactful remediation action
        """
        from entroly_core import py_scan_content
        return py_scan_content(content, source)

    @mcp.tool()
    def security_report() -> str:
        """Generate a session-wide security audit across all ingested fragments.

        Scans every fragment in the current session and returns an aggregated
        report showing: which fragments are most vulnerable, overall risk posture,
        finding distribution by category, and the single most important fix.

        Returns JSON with:
            - fragments_scanned, fragments_with_findings
            - critical_total, high_total, max_risk_score
            - most_vulnerable_fragment (fragment_id)
            - findings_by_category: {category: count}
            - vulnerable_fragments: sorted list by risk_score
        """
        return engine._rust.security_report()

    @mcp.tool()
    def analyze_codebase_health() -> str:
        """Analyze the health of the ingested codebase.

        Runs 5 analysis passes over all fragments in the current session:
          1. Clone Detection — SimHash pairwise scan for Type-1/2/3 code clones
          2. Dead Symbol Analysis — defined but never referenced symbols
          3. God File Detection — files with > μ+2σ reverse dependencies
          4. Architecture Violation Detection — cross-layer imports
          5. Naming Convention Analysis — Python/Rust/React convention breaks

        Returns a JSON HealthReport with:
            - code_health_score [0–100] and health_grade (A/B/C/D/F)
            - Per-dimension scores: duplication, dead_code, coupling, arch, naming
            - clone_pairs, dead_symbols, god_files, arch_violations, naming_issues
            - summary (human-readable) and top_recommendation (most impactful action)
        """
        return engine._rust.analyze_health()

    @mcp.tool()
    def ingest_diagram(diagram_text: str, source: str, diagram_type: str = "auto") -> str:
        """Ingest an architecture or flow diagram into the context memory.

        Converts Mermaid, PlantUML, DOT/Graphviz, or informal diagram text into
        a structured semantic fragment capturing nodes, edges, and relationships.
        The result is stored as a normal context fragment and is retrievable
        by optimize_context and recall_relevant.

        Args:
            diagram_text: Raw diagram source (Mermaid/PlantUML/DOT/text description).
            source:       Identifier (e.g., 'arch_overview.mmd', 'db_schema.puml').
            diagram_type: 'mermaid', 'plantuml', 'dot', 'text', or 'auto' (default).

        Returns JSON with ingestion result (same as remember_fragment).
        """
        modal = _mm_diagram(diagram_text, source, diagram_type)
        result = engine.ingest_fragment(
            content=modal.text,
            source=source,
            token_count=modal.token_estimate,
            is_pinned=False,
        )
        if isinstance(result, str):
            data = json.loads(result)
        else:
            data = result
        data["modal_source_type"] = "diagram"
        data["diagram_type"] = diagram_type
        data["nodes_extracted"] = modal.metadata.get("node_count", 0)
        data["edges_extracted"] = modal.metadata.get("edge_count", 0)
        data["extraction_confidence"] = modal.confidence
        return json.dumps(data, indent=2)

    @mcp.tool()
    def ingest_voice(transcript: str, source: str) -> str:
        """Ingest a voice/meeting transcript into the context memory.

        Converts pre-transcribed text (from Whisper, AssemblyAI, etc.) into a
        structured fragment capturing decisions, action items, open questions,
        technical vocabulary, and key discussion excerpts.

        Args:
            transcript: The full transcript text.
            source:     Identifier (e.g., 'design_meeting_2026-03-07.txt').

        Returns JSON with ingestion result plus:
            - decisions, actions, open_questions (counts)
            - tech_terms_identified
        """
        modal = _mm_voice(transcript, source)
        result = engine.ingest_fragment(
            content=modal.text,
            source=source,
            token_count=modal.token_estimate,
            is_pinned=False,
        )
        if isinstance(result, str):
            data = json.loads(result)
        else:
            data = result
        data["modal_source_type"] = "voice"
        data["decisions_extracted"] = modal.metadata.get("decisions", 0)
        data["actions_extracted"] = modal.metadata.get("actions", 0)
        data["tech_terms"] = modal.metadata.get("tech_terms", 0)
        data["extraction_confidence"] = modal.confidence
        return json.dumps(data, indent=2)

    @mcp.tool()
    def ingest_diff(diff_text: str, source: str, commit_message: str = "") -> str:
        """Ingest a code diff/patch into the context memory.

        Converts a unified diff (git diff output) into a structured change summary:
        intent classification (bug-fix/feature/refactor), symbols changed,
        files modified, and line delta. Particularly useful for understanding
        recent changes and their architectural impact.

        Args:
            diff_text:      Raw unified diff text (git diff output).
            source:         Identifier (e.g., 'pr_42_auth_refactor.diff').
            commit_message: Optional commit message for better intent classification.

        Returns JSON with ingestion result plus:
            - intent: bug-fix/feature/refactor/test/security/performance
            - files_changed, added_lines, removed_lines
            - symbols_changed: functions/classes modified
        """
        modal = _mm_diff(diff_text, source, commit_message)
        result = engine.ingest_fragment(
            content=modal.text,
            source=source,
            token_count=modal.token_estimate,
            is_pinned=False,
        )
        if isinstance(result, str):
            data = json.loads(result)
        else:
            data = result
        data["modal_source_type"] = "diff"
        data["intent"] = modal.metadata.get("intent", "unknown")
        data["files_changed"] = modal.metadata.get("files_changed", 0)
        data["added_lines"] = modal.metadata.get("added_lines", 0)
        data["removed_lines"] = modal.metadata.get("removed_lines", 0)
        data["symbols_changed"] = modal.metadata.get("symbols_changed", [])
        return json.dumps(data, indent=2)

    @mcp.tool()
    def autotune_status() -> str:
        """Show the live status of Entroly's background self-tuning loop.

        Gives developers real-time visibility into what the autotuner is doing
        without them having to interact with it. Everything runs automatically
        in the background — this tool just surfaces the current state.

        Returns JSON with:
            - strategy: active tuning strategy (auto/balanced/latency/monorepo/quality)
            - best_score: highest composite score found so far
            - improvements: number of parameter improvements found this session
            - config_drift: how far current config has drifted from defaults (0=identical)
            - time_budget_guarantee: the hard latency ceiling in effect
            - recent_experiments: last 10 experiment results from results.tsv
            - idle_only: whether daemon only runs when system is idle
        """
        import csv
        from pathlib import Path as _Path

        # Read tuning config
        tc_path = _Path(__file__).parent.parent / "tuning_config.json"
        tc = {}
        if tc_path.exists():
            try:
                tc = json.loads(tc_path.read_text())
            except Exception:
                pass

        at = tc.get("autotuner", {})
        strategy = at.get("strategy", "balanced")
        time_budget_ms = at.get("time_budget_ms", 500)
        idle_only = at.get("idle_only", True)
        idle_threshold = at.get("idle_cpu_threshold", 0.30)

        # Read recent experiments from results.tsv
        recent = []
        results_path = _Path(__file__).parent.parent / "bench" / "results.tsv"
        if results_path.exists():
            try:
                with open(results_path) as f:
                    reader = csv.DictReader(f, delimiter="\t")
                    rows = list(reader)
                    for row in rows[-10:]:
                        recent.append({
                            "iter": int(row.get("iteration", 0)),
                            "param": row.get("param", ""),
                            "score": float(row.get("new_score", 0)),
                            "kept": row.get("kept", "discard") == "keep",
                            "drift": float(row.get("drift", 0)),
                            "strategy": row.get("strategy", strategy),
                        })
                kept_rows = [r for r in rows if r.get("kept") == "keep"]
                best_score = max(
                    (float(r["new_score"]) for r in kept_rows),
                    default=0.0
                )
                improvements = len(kept_rows)
                total_iters = len(rows)
            except Exception:
                best_score = 0.0
                improvements = 0
                total_iters = 0
        else:
            best_score = 0.0
            improvements = 0
            total_iters = 0

        # Config drift from defaults
        try:
            from bench.autotune import config_drift_score
            drift = config_drift_score(tc)
        except Exception:
            drift = 0.0

        return json.dumps({
            "strategy": strategy,
            "best_score_ever": round(best_score, 4),
            "improvements_found": improvements,
            "total_experiments_run": total_iters,
            "config_drift": round(drift, 4),
            "time_budget_guarantee": f"{time_budget_ms}ms hard limit per optimize() call",
            "idle_only": idle_only,
            "idle_cpu_threshold": f"{idle_threshold:.0%} CPU",
            "recent_experiments": recent,
            "insight": (
                f"Autotune has run {total_iters} experiments and kept {improvements} improvements. "
                f"Best score: {best_score:.4f}. Config drift: {drift:.4f} (0=identical to defaults)."
                if total_iters > 0
                else "Autotune hasn't run yet — it starts automatically in the background."
            ),
        }, indent=2)

    @mcp.tool()
    def autotune_history(n: int = 20) -> str:
        """Show the experiment history from the autotune background loop.

        Reads from bench/results.tsv — the structured log of every experiment
        the autotuner has run. Each entry shows the parameter mutated, the score
        delta, whether it was kept, the config drift, and the strategy in use.

        This is the equivalent of 'git log autotune/results' — full experiment
        auditability without needing git.

        Args:
            n: Number of recent experiments to show (default: 20)
        """
        import csv
        from pathlib import Path as _Path

        results_path = _Path(__file__).parent.parent / "bench" / "results.tsv"
        if not results_path.exists():
            return json.dumps({
                "status": "no_experiments_yet",
                "message": "Autotune hasn't run any experiments yet. It starts automatically in the background.",
                "experiments": [],
            }, indent=2)

        try:
            with open(results_path) as f:
                reader = csv.DictReader(f, delimiter="\t")
                rows = list(reader)
        except Exception as e:
            return json.dumps({"error": str(e)}, indent=2)

        kept = [r for r in rows if r.get("kept") == "keep"]
        skipped = [r for r in rows if r.get("kept") != "keep"]
        recent = rows[-n:]

        experiments = []
        for row in recent:
            experiments.append({
                "iter": int(row.get("iteration", 0)),
                "param": row.get("param", ""),
                "old_val": float(row.get("old_val", 0)),
                "new_val": float(row.get("new_val", 0)),
                "score": float(row.get("new_score", 0)),
                "kept": row.get("kept", "discard") == "keep",
                "drift": float(row.get("drift", 0)),
                "duration_ms": float(row.get("duration_ms", 0)),
                "strategy": row.get("strategy", "balanced"),
            })

        return json.dumps({
            "total_experiments": len(rows),
            "kept": len(kept),
            "discarded": len(skipped),
            "keep_rate": f"{len(kept)/max(len(rows),1):.0%}",
            "experiments": experiments,
        }, indent=2)

    @mcp.tool()
    def set_tuning_strategy(strategy: str) -> str:
        """Switch the autotuner to a different optimization strategy.

        Available strategies:
          - auto     (default) Auto-detects: monorepo for large repos, balanced otherwise
          - balanced Balanced defaults — good for most single-service repos
          - latency  Optimise for <50ms hard latency guarantee
          - monorepo Large codebases with deep dependency trees
          - quality  Maximum recall/precision (allows up to 500ms)

        The daemon picks up the new strategy within ~30 seconds.
        You don't need to restart anything.

        Args:
            strategy: One of: auto, balanced, latency, monorepo, quality
        """
        from pathlib import Path as _Path

        valid = {"auto", "balanced", "latency", "monorepo", "quality"}
        if strategy not in valid:
            return json.dumps({
                "error": f"Unknown strategy '{strategy}'. Valid: {sorted(valid)}",
            }, indent=2)

        tc_path = _Path(__file__).parent.parent / "tuning_config.json"
        if not tc_path.exists():
            return json.dumps({"error": "tuning_config.json not found"}, indent=2)

        try:
            tc = json.loads(tc_path.read_text())
            tc.setdefault("autotuner", {})["strategy"] = strategy
            tc_path.write_text(json.dumps(tc, indent=2) + "\n")
        except Exception as e:
            return json.dumps({"error": str(e)}, indent=2)

        # Load strategy description for response
        strategies_dir = _Path(__file__).parent.parent / "tuning_strategies"
        desc = f"{strategy} strategy"
        import re as _re
        recipe_path = strategies_dir / f"{strategy}.md"
        if recipe_path.exists():
            fm = _re.match(r"^---\n(.*?)\n---", recipe_path.read_text(), _re.DOTALL)
            if fm:
                for line in fm.group(1).splitlines():
                    if line.startswith("description:"):
                        desc = line.partition(":")[2].strip()
                        break

        return json.dumps({
            "status": "strategy_updated",
            "strategy": strategy,
            "description": desc,
            "note": "The background daemon picks up the new strategy within ~30 seconds. No restart needed.",
        }, indent=2)

    @mcp.tool()
    def get_tuning_strategy() -> str:
        """Show the current tuning strategy and its effect on the autotuner.

        Returns the active recipe details including what it optimizes for,
        the hard time budget in effect, and how it affects parameter search bounds.

        Call this to understand why the autotuner is exploring certain parameters.
        """
        from pathlib import Path as _Path
        import re as _re

        tc_path = _Path(__file__).parent.parent / "tuning_config.json"
        tc = json.loads(tc_path.read_text()) if tc_path.exists() else {}
        strategy_name = tc.get("autotuner", {}).get("strategy", "balanced")

        # If auto, determine what it resolved to
        resolved = strategy_name
        if strategy_name == "auto":
            resolved = _auto_detect_strategy()

        strategies_dir = _Path(__file__).parent.parent / "tuning_strategies"
        recipe_path = strategies_dir / f"{resolved}.md"

        recipe_content = {}
        if recipe_path.exists():
            text = recipe_path.read_text()
            fm = _re.match(r"^---\n(.*?)\n---", text, _re.DOTALL)
            if fm:
                for line in fm.group(1).splitlines():
                    if ":" in line:
                        k, _, v = line.partition(":")
                        recipe_content[k.strip()] = v.strip()

        return json.dumps({
            "configured_strategy": strategy_name,
            "resolved_strategy": resolved,
            "description": recipe_content.get("description", f"{resolved} strategy"),
            "optimize_for": recipe_content.get("optimize_for", "recall, precision, efficiency"),
            "time_budget_ms": recipe_content.get("time_budget_ms", "500"),
            "available_strategies": ["auto", "balanced", "latency", "monorepo", "quality"],
        }, indent=2)

    return mcp, engine



def _auto_detect_strategy() -> str:
    """
    Auto-detect the best tuning strategy based on the project structure.

    Heuristic:
      - If .git/modules or workspace files exist → 'monorepo'
      - Otherwise → 'balanced'

    This is called when strategy='auto' in tuning_config.json (the default).
    Developers install entroly and get the right strategy without configuring anything.
    """
    import subprocess
    from pathlib import Path

    cwd = Path.cwd()

    # Monorepo signals: workspace files, many top-level packages, git submodules
    monorepo_signals = [
        cwd / "pnpm-workspace.yaml",
        cwd / "lerna.json",
        cwd / "nx.json",
        cwd / "Cargo.toml",      # check if it's a workspace Cargo.toml below
        cwd / "go.work",
        cwd / "WORKSPACE",       # Bazel
        cwd / "WORKSPACE.bazel",
    ]

    for signal in monorepo_signals:
        if signal.exists():
            # Extra check for Cargo.toml — only count as monorepo if it has [workspace]
            if signal.name == "Cargo.toml":
                try:
                    content = signal.read_text()
                    if "[workspace]" in content:
                        return "monorepo"
                    continue
                except Exception:
                    continue
            return "monorepo"

    # Count top-level directories that look like packages
    try:
        dirs = [d for d in cwd.iterdir() if d.is_dir() and not d.name.startswith(".")]
        package_dirs = [d for d in dirs if (d / "package.json").exists()
                        or (d / "Cargo.toml").exists()
                        or (d / "pyproject.toml").exists()
                        or (d / "setup.py").exists()]
        if len(package_dirs) >= 3:
            return "monorepo"
    except Exception:
        pass

    return "balanced"


def _is_system_idle(cpu_threshold: float = 0.30) -> bool:
    """
    Check if the system is idle enough to safely run an autotune iteration.

    Safe for 16GB i5/i7/MacBook machines — returns False when developer is
    actively working (CPU > threshold). Autotune pauses until system is idle.

    Uses psutil if available, falls back to /proc/stat on Linux.
    """
    try:
        import psutil
        cpu_pct = psutil.cpu_percent(interval=0.1) / 100.0
        return cpu_pct < cpu_threshold
    except ImportError:
        pass

    # Fallback: read /proc/stat on Linux
    try:
        import time as _time
        def _cpu_times():
            with open("/proc/stat") as f:
                line = f.readline()
            parts = line.split()
            # user, nice, system, idle, iowait, irq, softirq
            nums = [int(x) for x in parts[1:8]]
            idle = nums[3]
            total = sum(nums)
            return idle, total

        idle1, total1 = _cpu_times()
        _time.sleep(0.1)
        idle2, total2 = _cpu_times()
        d_idle = idle2 - idle1
        d_total = total2 - total1
        cpu_pct = 1.0 - (d_idle / max(d_total, 1))
        return cpu_pct < cpu_threshold
    except Exception:
        return True  # Can't measure → assume idle (safe default)


def _start_autotune_daemon(engine: "EntrolyEngine") -> None:
    """
    Spawn the idle-aware autotune loop as a daemon background thread.

    Design for 16GB dev machines (i5/i7/MacBook):
      - Reads tuning_config.json every iteration (picks up strategy changes live)
      - Auto-detects monorepo vs single-service (no user config needed)
      - Pauses when CPU > idle_cpu_threshold (dev is actively working)
      - Sleeps iteration_sleep_secs between experiments (~0.5s default)
      - Runs at nice+10 OS priority — never competes with foreground work
      - Daemon thread: dies automatically when MCP server exits, no cleanup

    Controlled by tuning_config.json → autotuner.enabled (default: true).
    """
    import threading
    import os
    import time as _time
    from pathlib import Path

    config_path = Path(__file__).parent.parent / "tuning_config.json"

    def _read_autotune_config() -> dict:
        """Read current autotuner config fresh from disk each time."""
        try:
            import json as _json
            return _json.loads(config_path.read_text()).get("autotuner", {})
        except Exception:
            return {}

    # Check if enabled before starting the thread at all
    at_cfg = _read_autotune_config()
    if not at_cfg.get("enabled", True):
        logger.info("Autotune: disabled via tuning_config.json autotuner.enabled=false")
        return

    def _daemon_loop():
        # Lower OS scheduling priority: nice +10 on Linux/Mac
        try:
            os.nice(10)
        except (AttributeError, OSError):
            pass

        logger.info("Autotune: background self-tuning daemon started (nice+10)")

        iteration = 0
        best_score: Optional[float] = None
        improvements = 0

        while True:
            try:
                # Re-read config each iteration — picks up live changes (e.g., new strategy)
                at_cfg = _read_autotune_config()

                if not at_cfg.get("enabled", True):
                    logger.info("Autotune: disabled mid-session, daemon pausing")
                    _time.sleep(30)
                    continue

                idle_only = at_cfg.get("idle_only", True)
                idle_threshold = at_cfg.get("idle_cpu_threshold", 0.30)
                idle_sleep = at_cfg.get("idle_sleep_secs", 5)
                iter_sleep = at_cfg.get("iteration_sleep_secs", 0.5)
                strategy_name = at_cfg.get("strategy", "auto")
                drift_penalty = at_cfg.get("drift_penalty_weight", 0.1)

                # Auto-detect strategy (Feature 1: zero-config strategy selection)
                if strategy_name == "auto":
                    strategy_name = _auto_detect_strategy()

                # Idle-only mode: pause when developer is working (Feature: 16GB machine safety)
                if idle_only and not _is_system_idle(idle_threshold):
                    _time.sleep(idle_sleep)
                    continue

                # Run one iteration of autotune via bench module
                try:
                    from bench.autotune import autotune as _run_one_autotune
                    result = _run_one_autotune(
                        iterations=1,
                        seed=iteration,
                        verbose=False,
                        strategy_name=strategy_name,
                        use_git=False,  # git is opt-in via CLI, not daemon
                        drift_penalty_weight=drift_penalty,
                    )
                    new_score = result.get("final_score", 0.0)
                    if best_score is None or new_score > best_score:
                        if best_score is not None:
                            improvements += 1
                            logger.info(
                                f"Autotune: improvement #{improvements} "
                                f"score {best_score:.4f} → {new_score:.4f} "
                                f"[{strategy_name}]"
                            )
                        best_score = new_score
                except ImportError:
                    # bench module not available (e.g., installed as a package)
                    # Fall back to the simpler entroly/autotune.py
                    try:
                        from .autotune import run_autotune
                        run_autotune(iterations=1)
                    except Exception as e2:
                        logger.debug(f"Autotune: fallback also failed: {e2}")
                        _time.sleep(60)
                        continue

                iteration += 1
                # Sleep between iterations — 0.5s default keeps CPU load ~0% between runs
                _time.sleep(iter_sleep)

            except Exception as e:
                logger.debug(f"Autotune daemon iteration error (non-fatal): {e}")
                _time.sleep(30)  # Back off on unexpected errors

    t = threading.Thread(target=_daemon_loop, name="entroly-autotune", daemon=True)
    t.start()
    resolved_strategy = _auto_detect_strategy()
    logger.info(
        f"Autotune: daemon launched (tid={t.ident or 0}, "
        f"strategy=auto→{resolved_strategy}, idle_only=True)"
    )


def main():
    """Entry point for the entroly MCP server."""
    engine_type = "Rust" if _RUST_AVAILABLE else "Python"
    from entroly import __version__
    logger.info(f"Starting Entroly MCP server v{__version__} ({engine_type} engine)")
    mcp, engine = create_mcp_server()

    # Auto-index the project on startup (zero config)
    try:
        from entroly.auto_index import auto_index
        result = auto_index(engine)
        if result["status"] == "indexed":
            logger.info(
                f"Auto-indexed {result['files_indexed']} files "
                f"({result['total_tokens']:,} tokens) in {result['duration_s']}s"
            )
    except Exception as e:
        logger.warning(f"Auto-index failed (non-fatal): {e}")

    # Start the autotune daemon in the background — zero config needed.
    # It reads/writes only tuning_config.json and runs at nice+10 priority.
    try:
        _start_autotune_daemon(engine)
    except Exception as e:
        logger.warning("Autotune: failed to start daemon: %s", e)

    mcp.run()


if __name__ == "__main__":
    main()
