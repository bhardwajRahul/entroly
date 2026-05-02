"""
Shadow policy framework.

Every request runs N candidate routing policies in parallel. Each
policy returns a recommendation; recommendations are recorded into the
TraceEvent's ``shadow_recommendations`` field; **none of them route
production traffic**.

In v1, all four policies are stubs returning ``InsufficientData`` —
the framework exists so we can wire instrumentation cleanly today
and slot in real policies later (v2: embedding-conditioned failure
predictor, v3: conservative online router, v4: sequential controller).

Two principles encoded here:

  1. **No counterfactual regret in v1.** When shadow recommends model B
     but production used model A, we don't know what B would have done.
     The offline analyzer computes ``shadow_agreement`` (do shadows
     agree with the policy that ran?) — never ``counterfactual_regret``.

  2. **Insufficient-data is a first-class outcome.** Real policies will
     have feature-coverage gaps; saying "I don't know" is honest. The
     ``predicted_p_success`` field is None when the policy abstains.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Any, Protocol

logger = logging.getLogger(__name__)


@dataclass
class PolicyRecommendation:
    """A single policy's recommendation for one request."""

    policy_name: str
    model: str | None              # the policy's choice; None = abstain
    predicted_p_success: float | None  # in [0, 1] when the policy has data
    reason: str                    # short human-readable rationale
    insufficient_data: bool = False


class RoutingPolicy(Protocol):
    """Minimal interface for a shadow routing policy.

    A policy is a pure function of (query_features, candidate_models,
    current_model) → recommendation. State (training data, learned
    parameters) is held internally; the recommend method is read-only
    so shadow runs are deterministic and cheap.
    """

    name: str

    def recommend(
        self,
        *,
        query_text: str,
        query_features: dict[str, Any],
        current_model: str,
        candidates: list[str],
    ) -> PolicyRecommendation:
        ...


# ── v1 stubs — every policy returns InsufficientData ──────────────────────


class _InsufficientDataPolicy:
    """Base for v1 stubs. Records its name and returns 'no data'."""

    def __init__(self, name: str, future_version: str):
        self.name = name
        self._future_version = future_version

    def recommend(
        self,
        *,
        query_text: str,
        query_features: dict[str, Any],
        current_model: str,
        candidates: list[str],
    ) -> PolicyRecommendation:
        return PolicyRecommendation(
            policy_name=self.name,
            model=None,
            predicted_p_success=None,
            reason=(
                f"{self.name} not yet trained — scheduled for {self._future_version}; "
                "v1 only collects instrumentation"
            ),
            insufficient_data=True,
        )


def make_default_policies() -> list[RoutingPolicy]:
    """Construct the v1 set of shadow policies (all stubs)."""
    return [
        _InsufficientDataPolicy("current_heuristic", "v1.5"),
        _InsufficientDataPolicy("bucket_beta", "v2"),
        _InsufficientDataPolicy("embedding_knn", "v2"),
        _InsufficientDataPolicy("logistic_failure_predictor", "v2"),
    ]


# ── Runner ───────────────────────────────────────────────────────────────


class ShadowEvaluator:
    """Run all registered policies on a request; collect recommendations.

    Stateless w.r.t. requests (policies hold their own state). Errors
    in any single policy are caught and logged — they never block the
    request or pollute other policies' recommendations.
    """

    def __init__(self, policies: list[RoutingPolicy] | None = None):
        self._policies = policies if policies is not None else make_default_policies()

    @property
    def policy_names(self) -> list[str]:
        return [p.name for p in self._policies]

    def evaluate(
        self,
        *,
        query_text: str,
        query_features: dict[str, Any],
        current_model: str,
        candidates: list[str],
    ) -> dict[str, dict[str, Any]]:
        """Return a dict of policy_name → serialized recommendation.

        The output shape matches ``TraceEvent.shadow_recommendations``
        directly so the caller can splice it in.
        """
        out: dict[str, dict[str, Any]] = {}
        for policy in self._policies:
            try:
                rec = policy.recommend(
                    query_text=query_text,
                    query_features=query_features,
                    current_model=current_model,
                    candidates=list(candidates),
                )
                out[policy.name] = asdict(rec)
            except Exception as e:
                logger.debug(
                    "shadow policy %s raised — recording insufficient_data: %s",
                    getattr(policy, "name", "<unnamed>"), e,
                )
                out[getattr(policy, "name", "<unnamed>")] = asdict(
                    PolicyRecommendation(
                        policy_name=getattr(policy, "name", "<unnamed>"),
                        model=None,
                        predicted_p_success=None,
                        reason=f"policy raised {type(e).__name__}",
                        insufficient_data=True,
                    )
                )
        return out


# ── Offline metrics ──────────────────────────────────────────────────────
#
# These are NOT counterfactual regret. They're agreement statistics —
# quantifying how often each shadow agreed with the policy that ran.
# Combined with per-actual-model success rate (computed by callers from
# derive_label outputs), this gives us a faithful picture of which
# shadows would have been at least *non-disruptive* — without claiming
# we know what they'd have actually delivered.


def shadow_agreement(
    trace: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """For one trace, compute per-policy agreement with the production policy.

    Returns ``{policy_name: {agreed: bool, recommended_model: str|None,
    abstained: bool}}``.
    """
    prod_policy = trace.get("policy_decision", "")
    prod_model = trace.get("model", "")
    shadows = trace.get("shadow_recommendations", {}) or {}
    out: dict[str, dict[str, Any]] = {}
    for name, rec in shadows.items():
        if not isinstance(rec, dict):
            continue
        if rec.get("insufficient_data"):
            out[name] = {
                "abstained": True,
                "agreed": False,
                "recommended_model": None,
            }
            continue
        recommended = rec.get("model")
        if recommended is None:
            out[name] = {"abstained": True, "agreed": False, "recommended_model": None}
            continue
        out[name] = {
            "abstained": False,
            "agreed": (recommended == prod_model),
            "recommended_model": recommended,
            # Note we do NOT include predicted_p_success here — readers
            # of the agreement metric should NOT confuse "predicted"
            # with "would-have-happened".
        }
    # Decorate with which production policy ran (for readability)
    if out:
        out["_meta"] = {
            "production_policy": prod_policy,
            "production_model": prod_model,
        }
    return out


def aggregate_shadow_agreement(
    traces: list[dict[str, Any]],
) -> dict[str, dict[str, float]]:
    """Aggregate ``shadow_agreement`` across many traces.

    Returns ``{policy_name: {agree_rate, abstain_rate, n}}``. Excludes
    the ``_meta`` key from per-trace agreement dicts.
    """
    counters: dict[str, dict[str, int]] = {}
    for trace in traces:
        per_trace = shadow_agreement(trace)
        for name, info in per_trace.items():
            if name == "_meta":
                continue
            d = counters.setdefault(name, {"agree": 0, "abstain": 0, "n": 0})
            d["n"] += 1
            if info.get("abstained"):
                d["abstain"] += 1
            elif info.get("agreed"):
                d["agree"] += 1
    out: dict[str, dict[str, float]] = {}
    for name, d in counters.items():
        n = max(d["n"], 1)
        out[name] = {
            "agree_rate": round(d["agree"] / n, 4),
            "abstain_rate": round(d["abstain"] / n, 4),
            "n": float(d["n"]),
        }
    return out
