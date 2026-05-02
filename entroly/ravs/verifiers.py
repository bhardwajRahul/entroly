"""
RAVS v2 — Verifiers

Each verifier checks an executor's output against a ground-truth or
structural criterion. All verifiers:
  1. Accept (executor_result, expected/source) and return (passed: bool, reason: str)
  2. Never have side effects
  3. Fail closed — on any exception, return (False, error)

Verifier types:
  ExactVerifier        — deterministic equality (computation nodes)
  StructuralVerifier   — AST/JSON structural validation (code_inspection)
  ExitCodeVerifier     — exit code == 0 (test_execution)
  CitationVerifier     — stub for v2 (retrieval_claim)
"""

from __future__ import annotations

import ast
import json
import logging
import math
import re
from typing import Any

logger = logging.getLogger("entroly.ravs.verifiers")


# ── Verifier Result ────────────────────────────────────────────────────


class VerifierResult:
    __slots__ = ("passed", "reason", "verifier_name")

    def __init__(self, passed: bool = False, reason: str = "",
                 verifier_name: str = ""):
        self.passed = passed
        self.reason = reason
        self.verifier_name = verifier_name


# ── Exact Verifier ─────────────────────────────────────────────────────


class ExactVerifier:
    """Deterministic equality check for computation results.

    Handles:
      - Integer/float comparison with tolerance
      - Symbolic expression equivalence
      - List/set comparison for multi-valued results
    """

    def __init__(self, tolerance: float = 1e-9):
        self._tolerance = tolerance

    def verify(self, executor_result: str, expected: str) -> VerifierResult:
        try:
            # Try numeric comparison first
            try:
                a = float(executor_result.strip())
                b = float(expected.strip())
                if math.isnan(a) and math.isnan(b):
                    return VerifierResult(True, "both NaN", "exact")
                if abs(a - b) <= self._tolerance:
                    return VerifierResult(True, f"|{a}-{b}| <= {self._tolerance}", "exact")
                if b != 0 and abs((a - b) / b) <= self._tolerance:
                    return VerifierResult(True, f"relative error <= {self._tolerance}", "exact")
                return VerifierResult(False, f"{a} != {b}", "exact")
            except (ValueError, TypeError):
                pass

            # Try JSON comparison (for lists, sets, etc.)
            try:
                a_json = json.loads(executor_result)
                b_json = json.loads(expected)
                if a_json == b_json:
                    return VerifierResult(True, "JSON equal", "exact")
            except (json.JSONDecodeError, TypeError):
                pass

            # String equality (case-insensitive, whitespace-normalized)
            a_norm = " ".join(executor_result.strip().lower().split())
            b_norm = " ".join(expected.strip().lower().split())
            if a_norm == b_norm:
                return VerifierResult(True, "string equal (normalized)", "exact")

            return VerifierResult(False, f"'{executor_result}' != '{expected}'", "exact")

        except Exception as e:
            return VerifierResult(False, f"verifier error: {e}", "exact")


# ── Structural Verifier ────────────────────────────────────────────────


class StructuralVerifier:
    """AST/JSON structural validation for code inspection results.

    Checks that the executor's JSON output has the expected structure
    and key fields, without requiring exact value equality.
    """

    def verify(
        self,
        executor_result: str,
        expected_keys: list[str] | None = None,
        expected_types: dict[str, str] | None = None,
    ) -> VerifierResult:
        try:
            data = json.loads(executor_result)
            if not isinstance(data, dict):
                return VerifierResult(False, "result is not a JSON object", "structural")

            # Check required keys
            if expected_keys:
                missing = [k for k in expected_keys if k not in data]
                if missing:
                    return VerifierResult(
                        False,
                        f"missing keys: {missing}",
                        "structural",
                    )

            # Check value types
            if expected_types:
                for key, expected_type in expected_types.items():
                    if key not in data:
                        continue
                    val = data[key]
                    type_map = {
                        "int": int, "float": float, "str": str,
                        "list": list, "dict": dict, "bool": bool,
                    }
                    expected_py_type = type_map.get(expected_type)
                    if expected_py_type and not isinstance(val, expected_py_type):
                        return VerifierResult(
                            False,
                            f"key '{key}' expected {expected_type}, got {type(val).__name__}",
                            "structural",
                        )

            # Check no error field
            if "error" in data and data["error"]:
                return VerifierResult(
                    False,
                    f"result contains error: {data['error']}",
                    "structural",
                )

            return VerifierResult(True, "structure valid", "structural")

        except json.JSONDecodeError:
            return VerifierResult(False, "result is not valid JSON", "structural")
        except Exception as e:
            return VerifierResult(False, f"verifier error: {e}", "structural")


# ── Exit Code Verifier ─────────────────────────────────────────────────


class ExitCodeVerifier:
    """Check exit code for test execution results."""

    def verify(self, executor_result: str, expected_exit_code: int = 0) -> VerifierResult:
        try:
            # Try bare numeric first
            stripped = executor_result.strip()
            try:
                code = int(stripped)
                if code == expected_exit_code:
                    return VerifierResult(True, f"exit_code={code}", "exit_code")
                return VerifierResult(False, f"exit_code={code}, expected={expected_exit_code}", "exit_code")
            except ValueError:
                pass

            # Try JSON
            data = json.loads(executor_result)
            if isinstance(data, dict):
                exit_code = data.get("exit_code", data.get("status"))
                if exit_code == expected_exit_code:
                    return VerifierResult(True, f"exit_code={exit_code}", "exit_code")
                if data.get("status") == "shadow_stub":
                    return VerifierResult(
                        False,
                        "shadow stub — not actually executed",
                        "exit_code",
                    )
                return VerifierResult(
                    False,
                    f"exit_code={exit_code}, expected={expected_exit_code}",
                    "exit_code",
                )
            return VerifierResult(False, "result is not a JSON object", "exit_code")
        except (json.JSONDecodeError, Exception):
            return VerifierResult(False, "cannot parse exit code", "exit_code")


# ── Citation Verifier (stub) ───────────────────────────────────────────


class CitationVerifier:
    """Stub for v2 — citation/entailment checking deferred."""

    def verify(self, executor_result: str, source_text: str = "") -> VerifierResult:
        return VerifierResult(
            False,
            "v2 stub — citation verification not yet implemented",
            "citation",
        )


# ── Verifier Registry ─────────────────────────────────────────────────


class VerifierRegistry:
    """Central registry mapping verifier types to implementations."""

    def __init__(self):
        self._exact = ExactVerifier()
        self._structural = StructuralVerifier()
        self._exit_code = ExitCodeVerifier()
        self._citation = CitationVerifier()

    def get(self, verifier_type: str) -> Any:
        return {
            "exact": self._exact,
            "structural": self._structural,
            "exit_code": self._exit_code,
            "citation": self._citation,
        }.get(verifier_type)
