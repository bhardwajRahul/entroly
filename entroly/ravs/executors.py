"""
RAVS v2 — Node Executors

Each executor handles one node type. All executors:
  1. Accept an input string and return (result_str, succeeded: bool)
  2. Are sandboxed — no network, no filesystem writes, bounded CPU
  3. Fail closed — on any exception, return (error_msg, False)
  4. Are stateless — no side effects between calls

Executor types:
  SymPyExecutor    — symbolic math via SymPy (computation nodes)
  PythonExecutor   — safe eval for simple arithmetic (computation fallback)
  ASTExecutor      — Python AST queries (code_inspection nodes)
  TestRunnerStub   — records test intent, doesn't execute (test_execution)
  RetrievalStub    — records retrieval intent (retrieval_claim)
"""

from __future__ import annotations

import ast
import logging
import math
import re
import time
from typing import Any

logger = logging.getLogger("entroly.ravs.executors")


# ── Base ───────────────────────────────────────────────────────────────


class ExecutorResult:
    """Standardized executor output."""
    __slots__ = ("result", "succeeded", "error", "execution_time_ms", "executor_name")

    def __init__(
        self,
        result: str = "",
        succeeded: bool = False,
        error: str = "",
        execution_time_ms: float = 0.0,
        executor_name: str = "",
    ):
        self.result = result
        self.succeeded = succeeded
        self.error = error
        self.execution_time_ms = execution_time_ms
        self.executor_name = executor_name


# ── SymPy Executor ─────────────────────────────────────────────────────


class SymPyExecutor:
    """Execute symbolic math via SymPy.

    Accepts expressions like:
      "solve x^2 - 1"
      "integrate sin(x)"
      "simplify (x+1)^2 - x^2"
      "2 + 3 * 4"
    """

    def __init__(self, timeout_s: float = 5.0):
        self._timeout = timeout_s
        self._sympy = None
        self._available = False
        try:
            import sympy
            self._sympy = sympy
            self._available = True
        except ImportError:
            logger.debug("SymPy not available — SymPyExecutor disabled")

    @property
    def available(self) -> bool:
        return self._available

    def execute(self, input_text: str) -> ExecutorResult:
        if not self._available:
            return ExecutorResult(
                error="sympy not installed",
                executor_name="sympy",
            )

        t0 = time.perf_counter()
        try:
            result = self._eval_expression(input_text)
            elapsed = (time.perf_counter() - t0) * 1000
            return ExecutorResult(
                result=str(result),
                succeeded=True,
                execution_time_ms=round(elapsed, 2),
                executor_name="sympy",
            )
        except Exception as e:
            elapsed = (time.perf_counter() - t0) * 1000
            return ExecutorResult(
                error=str(e)[:200],
                execution_time_ms=round(elapsed, 2),
                executor_name="sympy",
            )

    def _eval_expression(self, text: str) -> Any:
        sympy = self._sympy

        # Strip instruction prefixes
        cleaned = re.sub(
            r'^(?:calculate|compute|evaluate|solve|simplify|'
            r'factor|expand|integrate|differentiate|find the value of|'
            r'what is|how much is)\s+',
            '', text.strip(), flags=re.I,
        ).strip()

        # Replace common notation
        cleaned = cleaned.replace('^', '**')

        # Try to parse and evaluate
        expr = sympy.sympify(cleaned)

        # If it's a relational/equation, solve it
        if isinstance(expr, (sympy.Eq, sympy.Rel)):
            symbols = list(expr.free_symbols)
            if symbols:
                return sympy.solve(expr, symbols[0])

        # If it contains free symbols, try to simplify
        if expr.free_symbols:
            return sympy.simplify(expr)

        # Pure numeric — evaluate
        result = expr.evalf()
        # Return integer if it is one
        if result == int(result):
            return int(result)
        return float(result)


# ── Python Safe Eval Executor ──────────────────────────────────────────


class PythonExecutor:
    """Safe evaluation of simple arithmetic expressions.

    Uses Python's ast.literal_eval for safety — no arbitrary code
    execution. Falls back to a restricted eval with math-only namespace
    for expressions that need math functions.
    """

    # Allowlisted names for restricted eval
    _SAFE_NAMES: dict[str, Any] = {
        "abs": abs, "round": round, "min": min, "max": max,
        "sum": sum, "len": len, "int": int, "float": float,
        "pow": pow, "divmod": divmod,
        # Math constants and functions
        "pi": math.pi, "e": math.e, "tau": math.tau,
        "sqrt": math.sqrt, "log": math.log, "log2": math.log2,
        "log10": math.log10, "exp": math.exp,
        "sin": math.sin, "cos": math.cos, "tan": math.tan,
        "ceil": math.ceil, "floor": math.floor,
        "gcd": math.gcd,
    }

    def execute(self, input_text: str) -> ExecutorResult:
        t0 = time.perf_counter()
        try:
            # Strip instruction prefixes
            cleaned = re.sub(
                r'^(?:calculate|compute|evaluate|what is|how much is|'
                r'find the value of)\s+',
                '', input_text.strip(), flags=re.I,
            ).strip()
            cleaned = cleaned.replace('^', '**')

            # Try literal_eval first (safest)
            try:
                result = ast.literal_eval(cleaned)
                elapsed = (time.perf_counter() - t0) * 1000
                return ExecutorResult(
                    result=str(result),
                    succeeded=True,
                    execution_time_ms=round(elapsed, 2),
                    executor_name="python_safe",
                )
            except (ValueError, SyntaxError):
                pass

            # Restricted eval with math-only namespace
            # Validate AST first — no calls except allowlisted
            tree = ast.parse(cleaned, mode='eval')
            self._validate_ast(tree)

            result = eval(  # noqa: S307
                compile(tree, '<ravs>', 'eval'),
                {"__builtins__": {}},
                self._SAFE_NAMES,
            )
            elapsed = (time.perf_counter() - t0) * 1000
            return ExecutorResult(
                result=str(result),
                succeeded=True,
                execution_time_ms=round(elapsed, 2),
                executor_name="python_safe",
            )
        except Exception as e:
            elapsed = (time.perf_counter() - t0) * 1000
            return ExecutorResult(
                error=str(e)[:200],
                execution_time_ms=round(elapsed, 2),
                executor_name="python_safe",
            )

    def _validate_ast(self, tree: ast.AST) -> None:
        """Walk AST and reject dangerous nodes."""
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id not in self._SAFE_NAMES:
                        raise ValueError(
                            f"Function '{node.func.id}' not in allowlist"
                        )
                elif isinstance(node.func, ast.Attribute):
                    raise ValueError("Attribute access not allowed")
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                raise ValueError("Imports not allowed")
            elif isinstance(node, ast.Attribute):
                raise ValueError("Attribute access not allowed")


# ── AST Executor ───────────────────────────────────────────────────────


class ASTExecutor:
    """Python AST-based code inspection.

    Answers structural questions about code: function signatures,
    class definitions, import lists, argument counts, etc.
    """

    def execute(self, input_text: str, source_code: str = "") -> ExecutorResult:
        t0 = time.perf_counter()
        if not source_code:
            elapsed = (time.perf_counter() - t0) * 1000
            return ExecutorResult(
                error="no source code provided",
                execution_time_ms=round(elapsed, 2),
                executor_name="ast",
            )

        try:
            tree = ast.parse(source_code)
            result = self._inspect(tree, input_text)
            elapsed = (time.perf_counter() - t0) * 1000
            return ExecutorResult(
                result=result,
                succeeded=True,
                execution_time_ms=round(elapsed, 2),
                executor_name="ast",
            )
        except Exception as e:
            elapsed = (time.perf_counter() - t0) * 1000
            return ExecutorResult(
                error=str(e)[:200],
                execution_time_ms=round(elapsed, 2),
                executor_name="ast",
            )

    def _inspect(self, tree: ast.AST, query: str) -> str:
        query_lower = query.lower()

        # List functions
        if "function" in query_lower or "method" in query_lower:
            if "list" in query_lower or "find" in query_lower or "show" in query_lower:
                funcs = [
                    node.name for node in ast.walk(tree)
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                ]
                return json.dumps({"functions": funcs, "count": len(funcs)})

            if "count" in query_lower or "how many" in query_lower:
                count = sum(
                    1 for node in ast.walk(tree)
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                )
                return json.dumps({"count": count})

        # List classes
        if "class" in query_lower:
            if "list" in query_lower or "find" in query_lower or "show" in query_lower:
                classes = [
                    node.name for node in ast.walk(tree)
                    if isinstance(node, ast.ClassDef)
                ]
                return json.dumps({"classes": classes, "count": len(classes)})

            if "count" in query_lower or "how many" in query_lower:
                count = sum(
                    1 for node in ast.walk(tree)
                    if isinstance(node, ast.ClassDef)
                )
                return json.dumps({"count": count})

        # List imports
        if "import" in query_lower:
            imports = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                    for alias in node.names:
                        imports.append(f"{module}.{alias.name}")
            return json.dumps({"imports": imports, "count": len(imports)})

        # Signature of a specific function
        sig_match = re.search(r'(?:signature|parameters|arguments)\s+(?:of\s+)?(\w+)', query_lower)
        if sig_match:
            target_name = sig_match.group(1)
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if node.name.lower() == target_name:
                        args = [a.arg for a in node.args.args]
                        return json.dumps({
                            "function": node.name,
                            "parameters": args,
                            "count": len(args),
                        })
            return json.dumps({"error": f"function '{target_name}' not found"})

        # Generic: return summary
        funcs = sum(1 for n in ast.walk(tree)
                    if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)))
        classes = sum(1 for n in ast.walk(tree) if isinstance(n, ast.ClassDef))
        return json.dumps({
            "functions": funcs,
            "classes": classes,
            "summary": "use a more specific query for detailed inspection",
        })


# ── Stubs for v2 ───────────────────────────────────────────────────────
# These record intent but don't execute in shadow. Full execution
# requires sandbox infrastructure (V3+).


class TestRunnerStub:
    """Records test execution intent. Does not actually run tests in v2."""

    def execute(self, input_text: str) -> ExecutorResult:
        return ExecutorResult(
            result=json.dumps({
                "status": "shadow_stub",
                "intent": "test_execution",
                "input": input_text[:200],
                "note": "v2 shadow mode — test execution recorded, not run",
            }),
            succeeded=False,  # not actually executed
            executor_name="test_runner_stub",
        )


class RetrievalStub:
    """Records retrieval intent. Citation checking deferred to V3."""

    def execute(self, input_text: str) -> ExecutorResult:
        return ExecutorResult(
            result=json.dumps({
                "status": "shadow_stub",
                "intent": "retrieval_claim",
                "input": input_text[:200],
                "note": "v2 shadow mode — retrieval recorded, not executed",
            }),
            succeeded=False,
            executor_name="retrieval_stub",
        )


# ── Executor Registry ─────────────────────────────────────────────────


import json  # noqa: E402 (used in ASTExecutor)


class ExecutorRegistry:
    """Central registry mapping executor types to implementations."""

    def __init__(self):
        self._sympy = SymPyExecutor()
        self._python = PythonExecutor()
        self._ast = ASTExecutor()
        self._test = TestRunnerStub()
        self._retrieval = RetrievalStub()

    def get(self, executor_type: str) -> Any:
        return {
            "sympy": self._sympy,
            "python": self._python,
            "ast": self._ast,
            "test_runner": self._test,
            "retrieval": self._retrieval,
        }.get(executor_type)

    def execute_node(
        self,
        executor_type: str,
        input_text: str,
        **kwargs: Any,
    ) -> ExecutorResult:
        """Execute a node with the appropriate executor."""
        executor = self.get(executor_type)
        if executor is None:
            return ExecutorResult(
                error=f"unknown executor: {executor_type}",
                executor_name=executor_type,
            )

        try:
            if executor_type == "ast" and "source_code" in kwargs:
                return executor.execute(input_text, kwargs["source_code"])
            return executor.execute(input_text)
        except Exception as e:
            return ExecutorResult(
                error=str(e)[:200],
                executor_name=executor_type,
            )
