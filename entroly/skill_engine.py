"""
Skill Engine — Evolution Layer
===============================

Handles the full skill lifecycle:
  1. Skill Synthesis:    Generate skill specs from gap reports
  2. Sandboxed Runner:   Execute skills in isolation
  3. Benchmark Harness:  Evaluate skill fitness
  4. Promotion Engine:   Promote, merge, or prune skills
  5. Registry Manager:   Maintain the skill index
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .vault import VaultManager

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════
# Data Structures
# ══════════════════════════════════════════════════════════════════════

@dataclass
class SkillSpec:
    """A skill specification."""
    skill_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    name: str = ""
    description: str = ""
    entity: str = ""
    trigger: str = ""  # pattern that triggers this skill
    procedure: str = ""  # step-by-step SOP
    tool_code: str = ""  # Python tool implementation
    test_cases: List[Dict[str, str]] = field(default_factory=list)
    status: str = "draft"  # draft, testing, promoted, pruned
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class BenchmarkResult:
    """Result of running a skill benchmark."""
    skill_id: str
    passed: int = 0
    failed: int = 0
    errors: List[str] = field(default_factory=list)
    fitness_score: float = 0.0  # 0.0-1.0
    duration_ms: float = 0

    @property
    def success_rate(self) -> float:
        total = self.passed + self.failed
        return self.passed / total if total > 0 else 0.0


# ══════════════════════════════════════════════════════════════════════
# Skill Synthesizer
# ══════════════════════════════════════════════════════════════════════

class SkillSynthesizer:
    """Generates skill specs from gap reports and failure patterns."""

    def synthesize_from_gap(
        self,
        entity_key: str,
        failing_queries: List[str],
        intent: str = "",
    ) -> SkillSpec:
        """Generate a skill spec from a gap report."""
        # Derive skill name and description from entity
        name = entity_key.replace(":", "_").replace("/", "_")

        # Generate trigger pattern from failing queries
        common_words = self._extract_common_terms(failing_queries)
        trigger = "|".join(common_words[:5]) if common_words else entity_key

        # Generate procedure
        procedure = self._generate_procedure(entity_key, intent, failing_queries)

        # Generate tool code template
        tool_code = self._generate_tool_template(name, entity_key, trigger)

        # Generate test cases
        tests = [
            {"input": q, "expected": "should_not_fail"}
            for q in failing_queries[:5]
        ]

        return SkillSpec(
            name=name,
            description=f"Skill for handling {entity_key} queries",
            entity=entity_key,
            trigger=trigger,
            procedure=procedure,
            tool_code=tool_code,
            test_cases=tests,
            status="draft",
        )

    def _extract_common_terms(self, queries: List[str]) -> List[str]:
        """Find common terms across failing queries."""
        import re
        word_counts: Dict[str, int] = {}
        for q in queries:
            words = set(
                w.lower() for w in re.findall(r'[a-zA-Z_]\w+', q) if len(w) > 3
            )
            for w in words:
                word_counts[w] = word_counts.get(w, 0) + 1

        # Return words that appear in >50% of queries
        threshold = max(1, len(queries) // 2)
        return sorted(
            [w for w, c in word_counts.items() if c >= threshold],
            key=lambda w: -word_counts[w],
        )

    def _generate_procedure(self, entity: str, intent: str, queries: List[str]) -> str:
        return (
            f"# Procedure for {entity}\n\n"
            f"## Trigger\n"
            f"This skill activates when a query relates to `{entity}`.\n\n"
            f"## Steps\n"
            f"1. Check if relevant source files exist for `{entity}`\n"
            f"2. Extract structural information (AST, dependencies)\n"
            f"3. Build a belief artifact with proper frontmatter\n"
            f"4. Cross-reference with existing beliefs for consistency\n"
            f"5. Generate an answer using the compiled understanding\n\n"
            f"## Evidence Required\n"
            f"- Source file references with line numbers\n"
            f"- Dependency graph edges\n"
            f"- Test coverage status\n"
        )

    def _generate_tool_template(self, name: str, entity: str, trigger: str) -> str:
        return (
            f'"""\n'
            f'Auto-generated skill tool: {name}\n'
            f'Entity: {entity}\n'
            f'"""\n\n'
            f'import re\n\n'
            f'TRIGGER_PATTERN = re.compile(r"\\b({trigger})\\b", re.I)\n\n\n'
            f'def matches(query: str) -> bool:\n'
            f'    """Check if this skill should handle the query."""\n'
            f'    return bool(TRIGGER_PATTERN.search(query))\n\n\n'
            f'def execute(query: str, context: dict) -> dict:\n'
            f'    """Execute the skill logic."""\n'
            f'    return {{\n'
            f'        "status": "executed",\n'
            f'        "skill": "{name}",\n'
            f'        "entity": "{entity}",\n'
            f'        "result": "Skill implementation needed",\n'
            f'    }}\n'
        )


# ══════════════════════════════════════════════════════════════════════
# Sandboxed Runner
# ══════════════════════════════════════════════════════════════════════

class SandboxedRunner:
    """Runs skill tools in isolation."""

    def __init__(self, timeout_seconds: float = 10.0):
        self._timeout = timeout_seconds

    def run_tool(self, tool_code: str, query: str) -> Dict[str, Any]:
        """Execute a skill tool in a subprocess sandbox."""
        # Write tool to temp file and run in subprocess
        import tempfile
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding="utf-8"
        ) as f:
            # Wrap the tool with execution harness
            harness = (
                f"{tool_code}\n\n"
                f'if __name__ == "__main__":\n'
                f'    import json, sys\n'
                f'    query = sys.argv[1] if len(sys.argv) > 1 else ""\n'
                f'    result = execute(query, {{}})\n'
                f'    print(json.dumps(result))\n'
            )
            f.write(harness)
            temp_path = f.name

        try:
            proc = subprocess.run(
                ["python", temp_path, query],
                capture_output=True, text=True,
                timeout=self._timeout,
            )
            if proc.returncode == 0:
                try:
                    result = json.loads(proc.stdout.strip())
                    return {"status": "success", "result": result}
                except json.JSONDecodeError:
                    return {"status": "success", "result": proc.stdout.strip()}
            else:
                return {
                    "status": "error",
                    "error": proc.stderr.strip(),
                    "returncode": proc.returncode,
                }
        except subprocess.TimeoutExpired:
            return {"status": "timeout", "timeout": self._timeout}
        except Exception as e:
            return {"status": "error", "error": str(e)}
        finally:
            try:
                os.unlink(temp_path)
            except OSError:
                pass


# ══════════════════════════════════════════════════════════════════════
# Benchmark Harness
# ══════════════════════════════════════════════════════════════════════

class SkillBenchmark:
    """Evaluates skill fitness by running test cases."""

    def __init__(self, runner: Optional[SandboxedRunner] = None):
        self._runner = runner or SandboxedRunner()

    def benchmark(self, skill: SkillSpec) -> BenchmarkResult:
        """Run all test cases for a skill and compute fitness."""
        t0 = time.time()
        result = BenchmarkResult(skill_id=skill.skill_id)

        for tc in skill.test_cases:
            query = tc.get("input", "")
            expected = tc.get("expected", "")
            try:
                run = self._runner.run_tool(skill.tool_code, query)
                if run["status"] == "success":
                    result.passed += 1
                else:
                    result.failed += 1
                    result.errors.append(f"Query '{query}': {run.get('error', 'unknown')}")
            except Exception as e:
                result.failed += 1
                result.errors.append(f"Query '{query}': {e}")

        result.duration_ms = (time.time() - t0) * 1000
        total = result.passed + result.failed
        result.fitness_score = result.passed / total if total > 0 else 0.0

        return result


# ══════════════════════════════════════════════════════════════════════
# Skill Engine (Promotion / Pruning / Registry)
# ══════════════════════════════════════════════════════════════════════

class SkillEngine:
    """
    Full skill lifecycle manager.

    Creates skills from gap reports, benchmarks them, promotes or prunes,
    and maintains the registry.
    """

    PROMOTION_THRESHOLD = 0.7  # fitness score to promote
    PRUNE_THRESHOLD = 0.3      # fitness score to prune

    def __init__(self, vault: VaultManager):
        self._vault = vault
        self._synthesizer = SkillSynthesizer()
        self._runner = SandboxedRunner()
        self._benchmark = SkillBenchmark(self._runner)

    def create_skill(
        self,
        entity_key: str,
        failing_queries: List[str],
        intent: str = "",
    ) -> Dict[str, Any]:
        """Create a new skill from a gap report."""
        self._vault.ensure_structure()
        spec = self._synthesizer.synthesize_from_gap(entity_key, failing_queries, intent)

        # Write skill package
        skill_dir = self._vault.config.path / "evolution" / "skills" / spec.skill_id
        skill_dir.mkdir(parents=True, exist_ok=True)

        # SKILL.md
        (skill_dir / "SKILL.md").write_text(
            f"---\n"
            f"skill_id: {spec.skill_id}\n"
            f"name: {spec.name}\n"
            f"entity: {spec.entity}\n"
            f"status: {spec.status}\n"
            f"created_at: {spec.created_at}\n"
            f"---\n\n"
            f"# {spec.name}\n\n"
            f"{spec.description}\n\n"
            f"{spec.procedure}\n",
            encoding="utf-8",
        )

        # tool.py
        (skill_dir / "tool.py").write_text(spec.tool_code, encoding="utf-8")

        # metrics.json
        (skill_dir / "metrics.json").write_text(
            json.dumps({
                "created_at": spec.created_at,
                "fitness_score": 0.0,
                "runs": 0,
                "successes": 0,
                "failures": 0,
            }, indent=2),
            encoding="utf-8",
        )

        # tests/
        tests_dir = skill_dir / "tests"
        tests_dir.mkdir(exist_ok=True)
        (tests_dir / "test_cases.json").write_text(
            json.dumps(spec.test_cases, indent=2),
            encoding="utf-8",
        )

        # Update registry
        self._update_registry(spec, "created")

        logger.info(f"SkillEngine: created skill {spec.skill_id} for {entity_key}")
        return {
            "status": "created",
            "skill_id": spec.skill_id,
            "name": spec.name,
            "path": str(skill_dir),
        }

    def benchmark_skill(self, skill_id: str) -> Dict[str, Any]:
        """Benchmark a skill and update its metrics."""
        spec = self._load_skill(skill_id)
        if not spec:
            return {"status": "not_found", "skill_id": skill_id}

        result = self._benchmark.benchmark(spec)

        # Update metrics
        self._update_metrics(skill_id, result)

        return {
            "status": "benchmarked",
            "skill_id": skill_id,
            "fitness": result.fitness_score,
            "passed": result.passed,
            "failed": result.failed,
            "duration_ms": result.duration_ms,
            "errors": result.errors[:5],
        }

    def promote_or_prune(self, skill_id: str) -> Dict[str, Any]:
        """Evaluate a skill for promotion or pruning."""
        spec = self._load_skill(skill_id)
        if not spec:
            return {"status": "not_found"}

        fitness = spec.metrics.get("fitness_score", 0.0)

        if fitness >= self.PROMOTION_THRESHOLD:
            action = "promoted"
            spec.status = "promoted"
        elif fitness <= self.PRUNE_THRESHOLD:
            action = "pruned"
            spec.status = "pruned"
        else:
            action = "kept"
            spec.status = "testing"

        # Update skill status
        skill_dir = self._vault.config.path / "evolution" / "skills" / skill_id
        skill_md = skill_dir / "SKILL.md"
        if skill_md.exists():
            content = skill_md.read_text(encoding="utf-8")
            import re
            content = re.sub(r"status: \w+", f"status: {spec.status}", content)
            skill_md.write_text(content, encoding="utf-8")

        self._update_registry(spec, action)

        logger.info(f"SkillEngine: {action} skill {skill_id} (fitness={fitness:.2f})")
        return {
            "status": action,
            "skill_id": skill_id,
            "fitness": fitness,
            "new_status": spec.status,
        }

    def list_skills(self) -> List[Dict[str, Any]]:
        """List all skills in the registry."""
        self._vault.ensure_structure()
        skills_dir = self._vault.config.path / "evolution" / "skills"
        results = []

        for skill_dir in sorted(skills_dir.iterdir()) if skills_dir.exists() else []:
            if not skill_dir.is_dir():
                continue
            metrics_file = skill_dir / "metrics.json"
            skill_md = skill_dir / "SKILL.md"

            info = {"skill_id": skill_dir.name, "path": str(skill_dir)}
            if metrics_file.exists():
                try:
                    info["metrics"] = json.loads(metrics_file.read_text(encoding="utf-8"))
                except Exception:
                    pass
            if skill_md.exists():
                try:
                    from .vault import _parse_frontmatter
                    content = skill_md.read_text(encoding="utf-8")
                    fm = _parse_frontmatter(content)
                    if fm:
                        info.update(fm)
                except Exception:
                    pass
            results.append(info)

        return results

    # ── Private ──────────────────────────────────

    def _load_skill(self, skill_id: str) -> Optional[SkillSpec]:
        """Load a skill spec from the vault."""
        skill_dir = self._vault.config.path / "evolution" / "skills" / skill_id
        if not skill_dir.exists():
            return None

        spec = SkillSpec(skill_id=skill_id)

        tool_file = skill_dir / "tool.py"
        if tool_file.exists():
            spec.tool_code = tool_file.read_text(encoding="utf-8")

        tests_file = skill_dir / "tests" / "test_cases.json"
        if tests_file.exists():
            try:
                spec.test_cases = json.loads(tests_file.read_text(encoding="utf-8"))
            except Exception:
                pass

        metrics_file = skill_dir / "metrics.json"
        if metrics_file.exists():
            try:
                spec.metrics = json.loads(metrics_file.read_text(encoding="utf-8"))
            except Exception:
                pass

        skill_md = skill_dir / "SKILL.md"
        if skill_md.exists():
            try:
                from .vault import _parse_frontmatter
                content = skill_md.read_text(encoding="utf-8")
                fm = _parse_frontmatter(content)
                if fm:
                    spec.name = fm.get("name", "")
                    spec.entity = fm.get("entity", "")
                    spec.status = fm.get("status", "draft")
            except Exception:
                pass

        return spec

    def _update_metrics(self, skill_id: str, result: BenchmarkResult) -> None:
        metrics_file = (
            self._vault.config.path / "evolution" / "skills" / skill_id / "metrics.json"
        )
        if metrics_file.exists():
            try:
                data = json.loads(metrics_file.read_text(encoding="utf-8"))
            except Exception:
                data = {}
        else:
            data = {}

        data["fitness_score"] = result.fitness_score
        data["runs"] = data.get("runs", 0) + 1
        data["successes"] = data.get("successes", 0) + result.passed
        data["failures"] = data.get("failures", 0) + result.failed
        data["last_benchmark"] = datetime.now(timezone.utc).isoformat()
        data["last_duration_ms"] = result.duration_ms

        metrics_file.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def _update_registry(self, spec: SkillSpec, action: str) -> None:
        """Update the registry.md index."""
        registry = self._vault.config.path / "evolution" / "registry.md"
        if not registry.exists():
            self._vault.ensure_structure()

        content = registry.read_text(encoding="utf-8")
        entry = f"| {spec.skill_id} | {action} | {spec.created_at[:10]} | {spec.description[:50]} |"

        # Check if already in registry
        if spec.skill_id in content:
            # Update existing line
            lines = content.splitlines()
            for i, line in enumerate(lines):
                if spec.skill_id in line:
                    lines[i] = entry
                    break
            content = "\n".join(lines)
        else:
            content = content.rstrip() + "\n" + entry + "\n"

        registry.write_text(content, encoding="utf-8")
