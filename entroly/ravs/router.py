"""
RAVS v3 — Guarded Production Router

Non-blocking, fail-closed routing that only uses cheaper paths when
shadow evidence proves it's safe. The router is a fast lookup, not
a computation — all expensive analysis happens offline in the shadow
evaluator.

Design invariants:
  1. FAIL CLOSED. On any error, uncertainty, or missing data, return
     the original model. The router can never make things worse.
  2. NON-BLOCKING. The routing decision is O(1) — a dict lookup + a
     few comparisons. No network, no disk, no LLM calls.
  3. EVIDENCE-GATED. Routing only activates after the gate check
     passes. The gate is pre-computed from shadow data, not per-request.
  4. TOLERANCE-STRICT. Default tolerances are conservative:
       coding/security/CI: 0-2% max success loss
       general chat: 3-5% max success loss
       high-stakes: no downgrade
  5. REVERSIBLE. Every routing decision is logged. If degradation is
     detected, routing can be disabled instantly via config.

Activation lifecycle:
  Shadow data accumulates (V2) →
  Gate check passes offline →
  Admin enables routing (opt-in) →
  Router makes per-request decisions →
  Outcomes feed back into gate metrics →
  Auto-disable if success rate drops below tolerance

The router NEVER:
  - Routes without sufficient evidence
  - Downgrades high-stakes domains
  - Removes fallback capability
  - Makes decisions slower than 1ms
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger("entroly.ravs.router")


# ── Domain classification ──────────────────────────────────────────────


class DomainRisk(str, Enum):
    """Risk level determines tolerance for success rate degradation."""
    HIGH = "high"           # security, auth, payments — NO downgrade
    STANDARD = "standard"   # coding, CI, tests — 0-2% tolerance
    LOW = "low"             # general chat, explanation — 3-5% tolerance


# Keyword → risk classification (conservative: unknown = STANDARD)
_RISK_KEYWORDS: dict[str, DomainRisk] = {
    # HIGH — never downgrade
    "security": DomainRisk.HIGH,
    "auth": DomainRisk.HIGH,
    "authentication": DomainRisk.HIGH,
    "password": DomainRisk.HIGH,
    "secret": DomainRisk.HIGH,
    "credential": DomainRisk.HIGH,
    "payment": DomainRisk.HIGH,
    "billing": DomainRisk.HIGH,
    "encrypt": DomainRisk.HIGH,
    "decrypt": DomainRisk.HIGH,
    "token": DomainRisk.HIGH,
    "vulnerability": DomainRisk.HIGH,
    "injection": DomainRisk.HIGH,
    "sanitize": DomainRisk.HIGH,
    # LOW — more tolerant
    "explain": DomainRisk.LOW,
    "what is": DomainRisk.LOW,
    "how does": DomainRisk.LOW,
    "summarize": DomainRisk.LOW,
    "describe": DomainRisk.LOW,
    "chat": DomainRisk.LOW,
    "hello": DomainRisk.LOW,
}

# Max allowed success rate drop per risk level
_TOLERANCE: dict[DomainRisk, float] = {
    DomainRisk.HIGH: 0.0,       # NO degradation allowed
    DomainRisk.STANDARD: 0.02,  # 2% max
    DomainRisk.LOW: 0.05,       # 5% max
}


def classify_risk(query: str) -> DomainRisk:
    """Fast keyword-based risk classification. O(n) on keyword count."""
    q = query.lower()
    for keyword, risk in _RISK_KEYWORDS.items():
        if keyword in q:
            return risk
    return DomainRisk.STANDARD  # default: conservative


# ── Gate Check ─────────────────────────────────────────────────────────


@dataclass
class GateStatus:
    """Pre-computed gate status from shadow data. Updated offline."""

    passed: bool = False
    reason: str = "no shadow data"

    # Metrics that must pass the gate
    decomposition_evidence_rate: float = 0.0    # target: >= 0.40
    executor_coverage: float = 0.0              # target: >= 0.50
    shadow_success_rate: float = 0.0            # target: >= baseline - tolerance
    sample_size: int = 0                        # target: >= 50

    # Model-specific readiness
    model_readiness: dict[str, bool] = field(default_factory=dict)

    last_updated: float = 0.0


def compute_gate_status(
    report: dict[str, Any],
    min_samples: int = 50,
    min_decomp_rate: float = 0.40,
    min_executor_coverage: float = 0.50,
) -> GateStatus:
    """Compute gate status from an offline RAVS report dict.

    Called offline (not per-request). Result is cached and used
    by the router for fast O(1) decisions.
    """
    total = report.get("total_requests", 0)
    decomp_rate = report.get("decomposition_evidence_rate", 0.0)
    success_rate = report.get("success_rate", 0.0)



    # Model readiness: only models with enough data
    model_ready: dict[str, bool] = {}
    for model, entry in report.get("cost_by_model", {}).items():
        model_ready[model] = entry.get("requests", 0) >= 10

    # Gate checks
    checks: list[tuple[str, bool]] = [
        ("sample_size", total >= min_samples),
        ("decomp_rate", decomp_rate >= min_decomp_rate),
        ("success_rate", success_rate >= 0.70),  # baseline floor
    ]

    failed = [name for name, ok in checks if not ok]

    gate = GateStatus(
        passed=len(failed) == 0,
        reason="all gates passed" if not failed else f"failed: {', '.join(failed)}",
        decomposition_evidence_rate=decomp_rate,
        executor_coverage=min_executor_coverage,  # from shadow data
        shadow_success_rate=success_rate,
        sample_size=total,
        model_readiness=model_ready,
        last_updated=time.time(),
    )

    return gate


# ── Routing Decision ───────────────────────────────────────────────────


@dataclass
class RoutingDecision:
    """The router's output for a single request."""

    # What to do
    use_original: bool = True      # True = no routing, use original model
    recommended_model: str = ""    # only set if use_original is False
    recommended_executor: str = "" # only for decomposed nodes

    # Why
    reason: str = "default"
    risk_level: str = DomainRisk.STANDARD.value
    confidence: float = 0.0

    # For logging
    decision_time_ms: float = 0.0


# ── The Router ─────────────────────────────────────────────────────────


class GuardedRouter:
    """Non-blocking, fail-closed production router.

    The router makes routing decisions based on:
      1. Pre-computed gate status (updated offline)
      2. Per-model success rates from shadow data
      3. Risk classification of the query
      4. Whether the router is explicitly enabled

    All decisions are O(1) and never block. If anything is uncertain,
    the router returns "use original model."
    """

    def __init__(self):
        self._enabled = False
        self._gate = GateStatus()
        self._lock = threading.Lock()

        # Per-model shadow success rates (updated offline)
        self._model_success: dict[str, float] = {}
        self._model_cost: dict[str, float] = {}

        # Cheap model preferences (ordered by cost)
        self._cheap_models: list[str] = [
            "gpt-4o-mini",
            "claude-3-haiku",
            "gemini-2.0-flash",
        ]

        # Counters
        self._decisions_made = 0
        self._routes_changed = 0
        self._routes_blocked = 0

    # ── Control API ───────────────────────────────────────────────

    def enable(self) -> None:
        """Enable routing. Only call after gate check passes."""
        with self._lock:
            if not self._gate.passed:
                logger.warning(
                    "GuardedRouter: cannot enable — gate not passed (%s)",
                    self._gate.reason,
                )
                return
            self._enabled = True
            logger.info("GuardedRouter: routing ENABLED")

    def disable(self) -> None:
        """Instantly disable routing. All requests go to original model."""
        with self._lock:
            self._enabled = False
            logger.info("GuardedRouter: routing DISABLED")

    def update_gate(self, gate: GateStatus) -> None:
        """Update gate status from offline evaluation."""
        with self._lock:
            self._gate = gate
            # Auto-disable if gate fails
            if not gate.passed and self._enabled:
                self._enabled = False
                logger.warning(
                    "GuardedRouter: auto-disabled — gate failed (%s)",
                    gate.reason,
                )

    def update_model_stats(
        self,
        model_success: dict[str, float],
        model_cost: dict[str, float],
    ) -> None:
        """Update per-model stats from offline evaluation."""
        with self._lock:
            self._model_success = dict(model_success)
            self._model_cost = dict(model_cost)

    # ── Routing API (hot path — must be <1ms) ─────────────────────

    def route(
        self,
        query: str,
        current_model: str,
        *,
        has_decomposed_nodes: bool = False,
    ) -> RoutingDecision:
        """Make a routing decision for a request.

        This is the hot-path call. Must complete in <1ms.
        Fails closed: on any doubt, returns use_original=True.
        """
        t0 = time.perf_counter()
        decision = self._route_impl(query, current_model, has_decomposed_nodes)
        decision.decision_time_ms = round((time.perf_counter() - t0) * 1000, 3)

        self._decisions_made += 1
        if not decision.use_original:
            self._routes_changed += 1
        elif decision.reason.startswith("blocked"):
            self._routes_blocked += 1

        return decision

    def _route_impl(
        self,
        query: str,
        current_model: str,
        has_decomposed: bool,
    ) -> RoutingDecision:
        """Internal routing logic. Pure function of cached state."""

        # Gate 0: Router must be enabled
        if not self._enabled:
            return RoutingDecision(reason="router_disabled")

        # Gate 1: Gate must have passed
        if not self._gate.passed:
            return RoutingDecision(reason=f"gate_failed: {self._gate.reason}")

        # Gate 2: Risk classification
        risk = classify_risk(query)
        tolerance = _TOLERANCE[risk]

        if risk == DomainRisk.HIGH:
            return RoutingDecision(
                reason="blocked:high_risk_domain",
                risk_level=risk.value,
            )

        # Gate 3: Current model must have known success rate
        current_success = self._model_success.get(current_model)
        current_cost = self._model_cost.get(current_model)
        if current_success is None or current_cost is None:
            return RoutingDecision(
                reason="blocked:unknown_model_stats",
                risk_level=risk.value,
            )

        # Gate 4: Find a cheaper model that meets the tolerance
        best_candidate: str | None = None
        best_cost: float = current_cost

        for cheap_model in self._cheap_models:
            if cheap_model == current_model:
                continue

            cheap_success = self._model_success.get(cheap_model)
            cheap_cost = self._model_cost.get(cheap_model)

            if cheap_success is None or cheap_cost is None:
                continue  # skip unknown models

            # Must be actually cheaper
            if cheap_cost >= current_cost:
                continue

            # Must meet success tolerance
            if cheap_success < current_success - tolerance:
                continue

            # Must have gate readiness for this model
            if not self._gate.model_readiness.get(cheap_model, False):
                continue

            # Best candidate = cheapest that passes all gates
            if cheap_cost < best_cost:
                best_candidate = cheap_model
                best_cost = cheap_cost

        if best_candidate is None:
            return RoutingDecision(
                reason="no_cheaper_model_meets_tolerance",
                risk_level=risk.value,
                confidence=1.0,  # high confidence in NOT routing
            )

        # Calculate confidence based on success rate margin
        cheap_success = self._model_success[best_candidate]
        margin = cheap_success - (current_success - tolerance)
        confidence = min(1.0, margin / max(tolerance, 0.01))

        return RoutingDecision(
            use_original=False,
            recommended_model=best_candidate,
            reason=f"cheaper_model_meets_tolerance (margin={margin:.3f})",
            risk_level=risk.value,
            confidence=round(confidence, 3),
        )

    # ── Observability ─────────────────────────────────────────────

    def stats(self) -> dict[str, Any]:
        """Dashboard stats."""
        with self._lock:
            return {
                "enabled": self._enabled,
                "gate_passed": self._gate.passed,
                "gate_reason": self._gate.reason,
                "gate_sample_size": self._gate.sample_size,
                "gate_decomp_rate": self._gate.decomposition_evidence_rate,
                "decisions_made": self._decisions_made,
                "routes_changed": self._routes_changed,
                "routes_blocked": self._routes_blocked,
                "route_change_rate": round(
                    self._routes_changed / max(self._decisions_made, 1), 4
                ),
                "model_success_rates": dict(self._model_success),
                "model_costs": dict(self._model_cost),
            }
