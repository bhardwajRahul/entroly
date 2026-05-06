"""
RAVS — Reasoning Amplification via Verified Scaffolds.

Production-ready subsystem with four tiers:

  V1: Instrumentation — Honest outcome signals, offline eval, PRISM bridge.
  V2: Shadow Compiler — Decompose & execute cheap paths (SymPy, AST, TF-IDF retrieval).
  V3: Guarded Router — Four learned policies (heuristic, Thompson, KNN, logistic).
  V4: Sequential Controller — Budget-bounded, escalation-aware step execution.
"""

from .events import (
    DECOMPOSITION_EXECUTORS,
    DECOMPOSITION_KINDS,
    DECOMPOSITION_SOURCES,
    DECOMPOSITION_VERIFIERS,
    EVENT_STRENGTHS,
    HONEST_OUTCOME_TYPES,
    AppendOnlyEventLog,
    DecompositionEvidence,
    OutcomeEvent,
    TraceEvent,
    derive_label,
)
from .report import generate_report, format_report_text
from .outcome_bridge import OutcomeBridge
from .compiler import PlanCompiler, Plan, PlanNode, NodeKind, detect_substeps
from .executors import (
    ExecutorRegistry, SymPyExecutor, PythonExecutor, ASTExecutor,
    TestRunnerExecutor, RetrievalExecutor,
)
from .verifiers import (
    VerifierRegistry, ExactVerifier, StructuralVerifier, CitationVerifier,
)
from .shadow import ShadowEvaluator
from .shadow_runner import ShadowRunner
from .router import GuardedRouter, GateStatus, compute_gate_status, classify_risk
from .controller import SequentialController, ControllerResult, EscalationPolicy

__all__ = [
    # V1 — Instrumentation
    "DECOMPOSITION_EXECUTORS",
    "DECOMPOSITION_KINDS",
    "DECOMPOSITION_SOURCES",
    "DECOMPOSITION_VERIFIERS",
    "EVENT_STRENGTHS",
    "HONEST_OUTCOME_TYPES",
    "AppendOnlyEventLog",
    "DecompositionEvidence",
    "OutcomeEvent",
    "TraceEvent",
    "derive_label",
    "generate_report",
    "format_report_text",
    # V1+ — PRISM Bridge
    "OutcomeBridge",
    # V2 — Shadow Compiler
    "PlanCompiler",
    "Plan",
    "PlanNode",
    "NodeKind",
    "detect_substeps",
    "ExecutorRegistry",
    "SymPyExecutor",
    "PythonExecutor",
    "ASTExecutor",
    "TestRunnerExecutor",
    "RetrievalExecutor",
    "VerifierRegistry",
    "ExactVerifier",
    "StructuralVerifier",
    "CitationVerifier",
    "ShadowRunner",
    "ShadowEvaluator",
    # V3 — Guarded Router
    "GuardedRouter",
    "GateStatus",
    "compute_gate_status",
    "classify_risk",
    # V4 — Sequential Controller
    "SequentialController",
    "ControllerResult",
    "EscalationPolicy",
]
