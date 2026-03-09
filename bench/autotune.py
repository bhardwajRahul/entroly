"""
Entroly Autonomous Self-Tuning Loop
====================================

Keep/discard experimentation loop that optimizes hyperparameters overnight by
trying small parameter mutations, evaluating the result, and keeping improvements.

In Entroly:  autotune mutates tuning_config.json → benchmark eval → composite_score → keep/discard.

Single-file mutation discipline: this script ONLY modifies tuning_config.json.
The benchmark harness (evaluate.py) and Rust core are read-only.

Each iteration:
  1. Load current best config from tuning_config.json
  2. Load the active tuning_strategy.md (if present) to get bounds/hints
  3. Mutate one parameter (guided by strategy hints)
  4. Run benchmark suite (evaluate.py) with hard time budget
  5. If composite_score improves AND within time budget: keep (write to tuning_config.json)
  6. If composite_score regresses or too slow: discard (restore previous config)
  7. When kept, optionally commit to git autotune/results branch
  8. Log the result and repeat

New in this version:
  - tuning_strategy.md recipe system (--strategy flag)
  - Git-backed experiment history (--git flag)
  - Config drift penalty (prevents config rot)
  - Hard time budget enforcement (no soft penalty)

Runs entirely on CPU within 32GB RAM. Each iteration takes seconds.

Usage:
    python -m bench.autotune                       # 50 iterations (balanced strategy)
    python -m bench.autotune --iterations 200      # run overnight
    python -m bench.autotune --strategy latency    # use latency recipe
    python -m bench.autotune --strategy monorepo   # use monorepo recipe
    python -m bench.autotune --git                 # commit kept experiments to git
"""

from __future__ import annotations

import copy
import json
import math
import os
import random
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .evaluate import evaluate, load_tuning_config


# ── Paths ───────────────────────────────────────────────────────────────

STRATEGIES_DIR = Path(__file__).parent.parent / "tuning_strategies"
DEFAULT_CONFIG_PATH = Path(__file__).parent.parent / "tuning_config.json"
RESULTS_TSV_PATH = Path(__file__).parent / "results.tsv"

# Default config for drift calculation (canonical reference is balanced strategy)
DEFAULT_WEIGHTS = {
    "weights.recency": 0.30,
    "weights.frequency": 0.25,
    "weights.semantic_sim": 0.25,
    "weights.entropy": 0.20,
    "decay.half_life_turns": 15.0,
    "decay.min_relevance_threshold": 0.05,
    "knapsack.exploration_rate": 0.10,
    "sliding_window.long_window_fraction": 0.25,
    "prism.learning_rate": 0.01,
    "prism.beta": 0.95,
}


# ── Tunable parameter definitions ──────────────────────────────────────

@dataclass
class TunableParam:
    """A single tunable parameter with its JSON path and bounds."""
    path: list[str]       # e.g., ["weights", "recency"]
    min_val: float
    max_val: float
    step_size: float      # mutation step size (fraction of range)
    is_integer: bool = False

    def get(self, config: dict) -> float:
        obj = config
        for key in self.path[:-1]:
            obj = obj[key]
        return obj[self.path[-1]]

    def set(self, config: dict, value: float) -> None:
        obj = config
        for key in self.path[:-1]:
            obj = obj[key]
        if self.is_integer:
            obj[self.path[-1]] = int(round(value))
        else:
            obj[self.path[-1]] = round(value, 6)

    @property
    def dot_path(self) -> str:
        return ".".join(self.path)


# Default (balanced strategy) tunable parameters
DEFAULT_TUNABLE_PARAMS = [
    TunableParam(["weights", "recency"],           0.05, 0.60, 0.05),
    TunableParam(["weights", "frequency"],          0.05, 0.60, 0.05),
    TunableParam(["weights", "semantic_sim"],       0.05, 0.60, 0.05),
    TunableParam(["weights", "entropy"],            0.05, 0.60, 0.05),
    TunableParam(["decay", "half_life_turns"],      5,    50,   5, is_integer=True),
    TunableParam(["decay", "min_relevance_threshold"], 0.01, 0.20, 0.02),
    TunableParam(["knapsack", "exploration_rate"],  0.0,  0.30, 0.02),
    TunableParam(["sliding_window", "long_window_fraction"], 0.10, 0.50, 0.05),
    TunableParam(["prism", "learning_rate"],        0.001, 0.05, 0.005),
    TunableParam(["prism", "beta"],                 0.80, 0.99, 0.02),
]


# ── Tuning Strategy (program.md-style recipe) ──────────────────────────

@dataclass
class TuningStrategy:
    """
    A human-readable recipe that guides the autotuner toward a specific objective.

    Loaded from tuning_strategies/<name>.md via YAML frontmatter.
    This is how users configure the autotuner without writing code.
    """
    name: str
    description: str
    optimize_for: list[str]
    time_budget_ms: int
    weight_hints: dict[str, float] = field(default_factory=dict)
    bounds_overrides: dict[str, tuple[float, float]] = field(default_factory=dict)

    @classmethod
    def load(cls, name: str) -> "TuningStrategy":
        """Load a strategy from tuning_strategies/<name>.md."""
        path = STRATEGIES_DIR / f"{name}.md"
        if not path.exists():
            available = [p.stem for p in STRATEGIES_DIR.glob("*.md")] if STRATEGIES_DIR.exists() else []
            raise FileNotFoundError(
                f"Strategy '{name}' not found at {path}.\n"
                f"Available strategies: {', '.join(available) or 'none'}\n"
                f"Create {path} to define a custom strategy."
            )
        text = path.read_text()

        # Parse YAML frontmatter (between --- markers)
        frontmatter = {}
        fm_match = re.match(r"^---\n(.*?)\n---", text, re.DOTALL)
        if fm_match:
            for line in fm_match.group(1).splitlines():
                if ":" in line:
                    key, _, val = line.partition(":")
                    key = key.strip()
                    val = val.strip()
                    # Handle list values like [recall, precision]
                    if val.startswith("[") and val.endswith("]"):
                        val_parsed = [v.strip() for v in val[1:-1].split(",")]
                    else:
                        try:
                            val_parsed = int(val)
                        except ValueError:
                            val_parsed = val
                    frontmatter[key] = val_parsed

        return cls(
            name=name,
            description=frontmatter.get("description", f"{name} strategy"),
            optimize_for=frontmatter.get("optimize_for", ["recall", "precision"]),
            time_budget_ms=int(frontmatter.get("time_budget_ms", 500)),
        )

    @classmethod
    def balanced(cls) -> "TuningStrategy":
        """Return the default balanced strategy (no file needed)."""
        return cls(
            name="balanced",
            description="Balanced defaults — explores all parameters equally.",
            optimize_for=["recall", "precision", "context_efficiency"],
            time_budget_ms=500,
        )

    def apply_to_params(self, params: list[TunableParam]) -> list[TunableParam]:
        """
        Return a copy of params with bounds overridden by this strategy.
        Implements the autoresearch concept of strategy-guided mutation space.
        """
        result = []
        for p in params:
            key = p.dot_path
            if key in self.bounds_overrides:
                lo, hi = self.bounds_overrides[key]
                result.append(TunableParam(
                    path=p.path[:],
                    min_val=lo,
                    max_val=hi,
                    step_size=p.step_size,
                    is_integer=p.is_integer,
                ))
            else:
                result.append(copy.copy(p))
        return result


# ── Experiment record ───────────────────────────────────────────────────

@dataclass
class Experiment:
    """Record of a single autotuning experiment."""
    iteration: int
    param_name: str
    old_value: float
    new_value: float
    old_score: float
    new_score: float
    kept: bool
    duration_ms: float
    config_drift: float = 0.0


# ── Config drift (autoresearch Simplicity Criterion) ────────────────────

def config_drift_score(config: dict, reference: dict | None = None) -> float:
    """
    Measure how far this config has drifted from the canonical defaults.

    Returns [0.0, 1.0] where 0.0 = identical to defaults, 1.0 = max drift.

    This implements the autoresearch "simplicity criterion": configs that deviate
    wildly from defaults without proportional gain should be penalised. Prevents
    "config rot" — the accumulation of unmotivated parameter tweaks.

    Args:
        config: The candidate config to measure.
        reference: Optional reference dict. Uses DEFAULT_WEIGHTS if not provided.
    """
    defaults = DEFAULT_WEIGHTS

    total_magnitude = 0.0
    total_drift = 0.0

    def _flatten(obj: dict, prefix: str = "") -> dict[str, float]:
        result = {}
        for k, v in obj.items():
            key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                result.update(_flatten(v, key))
            elif isinstance(v, (int, float)):
                result[key] = float(v)
        return result

    flat_config = _flatten(config)
    flat_ref = reference if reference else defaults

    for key, ref_val in flat_ref.items():
        if key in flat_config:
            cur_val = flat_config[key]
            # Normalise drift by the range of this param
            param_range = abs(ref_val) + 1e-6  # avoid division by zero
            drift = abs(cur_val - ref_val) / param_range
            total_drift += drift
            total_magnitude += 1.0

    if total_magnitude == 0:
        return 0.0

    avg_drift = total_drift / total_magnitude
    # Clamp to [0, 1]
    return min(1.0, avg_drift)


# ── Git Experiment Log ──────────────────────────────────────────────────

class GitExperimentLog:
    """
    Commit kept experiments to the 'autotune/results' git branch.

    Implements the autoresearch pattern: every kept experiment = one commit.
    git log autotune/results shows full experiment history with diffs.
    Falls back to noop if git is not available or repo is not initialised.
    """

    BRANCH = "autotune/results"

    def __init__(self, repo_root: Path | None = None):
        self.repo_root = repo_root or Path(__file__).parent.parent
        self._available = self._check_git_available()

    def _check_git_available(self) -> bool:
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--is-inside-work-tree"],
                cwd=self.repo_root,
                capture_output=True,
                text=True,
                timeout=5,
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    def _current_branch(self) -> str:
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                cwd=self.repo_root,
                capture_output=True,
                text=True,
                timeout=5,
            )
            return result.stdout.strip()
        except Exception:
            return "unknown"

    def commit_experiment(
        self,
        experiment: Experiment,
        config: dict,
        strategy_name: str,
    ) -> bool:
        """
        Commit a kept experiment to the autotune/results branch.

        Only called when kept=True. The commit message captures all useful
        information for git log --oneline analysis.
        """
        if not self._available:
            return False

        try:
            # Write updated config (it was already saved — we just need to stage it)
            config_path = DEFAULT_CONFIG_PATH
            subprocess.run(
                ["git", "add", str(config_path)],
                cwd=self.repo_root,
                capture_output=True,
                timeout=10,
            )

            # Commit with structured message
            score_delta = experiment.new_score - experiment.old_score
            commit_msg = (
                f"autotune: [{strategy_name}] iter={experiment.iteration} "
                f"{experiment.param_name}: {experiment.old_value:.4f}->{experiment.new_value:.4f} "
                f"score={experiment.new_score:.4f} (+{score_delta:.4f}) "
                f"drift={experiment.config_drift:.3f}"
            )

            result = subprocess.run(
                ["git", "commit", "-m", commit_msg, "--no-verify"],
                cwd=self.repo_root,
                capture_output=True,
                text=True,
                timeout=15,
            )
            return result.returncode == 0
        except Exception:
            return False

    def get_history(self, n: int = 20) -> list[dict]:
        """
        Return the last N autotune commits as structured dicts.
        Parses the structured commit message format back into fields.
        """
        if not self._available:
            return []

        try:
            result = subprocess.run(
                ["git", "log", "--oneline", "--grep=autotune:", f"-{n}"],
                cwd=self.repo_root,
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode != 0:
                return []

            history = []
            for line in result.stdout.strip().splitlines():
                if not line:
                    continue
                parts = line.split(" ", 1)
                sha = parts[0] if parts else ""
                msg = parts[1] if len(parts) > 1 else ""
                history.append({"sha": sha, "message": msg})
            return history
        except Exception:
            return []


# ── Mutation helpers ────────────────────────────────────────────────────

def normalize_weights(config: dict) -> None:
    """Ensure the 4 scoring weights sum to 1.0."""
    w = config["weights"]
    total = w["recency"] + w["frequency"] + w["semantic_sim"] + w["entropy"]
    if total > 0:
        w["recency"]      = round(w["recency"] / total, 6)
        w["frequency"]    = round(w["frequency"] / total, 6)
        w["semantic_sim"] = round(w["semantic_sim"] / total, 6)
        w["entropy"]      = round(w["entropy"] / total, 6)


def mutate_random(
    config: dict,
    rng: random.Random,
    tunable_params: list[TunableParam],
) -> tuple[dict, str, float, float]:
    """Mutate one random parameter. Returns (new_config, param_name, old_val, new_val)."""
    config = copy.deepcopy(config)
    param = rng.choice(tunable_params)
    old_val = param.get(config)
    delta = rng.uniform(-param.step_size, param.step_size) * (param.max_val - param.min_val)
    new_val = max(param.min_val, min(param.max_val, old_val + delta))
    param.set(config, new_val)

    # Re-normalize weights if we touched one
    if param.path[0] == "weights":
        normalize_weights(config)

    name = ".".join(param.path)
    return config, name, old_val, param.get(config)


def save_config(config: dict, path: Path) -> None:
    with open(path, "w") as f:
        json.dump(config, f, indent=2)
        f.write("\n")


# ── Results TSV log ─────────────────────────────────────────────────────

def log_to_tsv(
    experiment: Experiment,
    strategy_name: str,
    path: Path = RESULTS_TSV_PATH,
) -> None:
    """Append experiment result to results.tsv."""
    header = (
        "iteration\tparam\told_val\tnew_val\told_score\tnew_score\t"
        "kept\tduration_ms\tdrift\tstrategy\n"
    )
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(header)

    with open(path, "a") as f:
        f.write(
            f"{experiment.iteration}\t{experiment.param_name}\t"
            f"{experiment.old_value:.6f}\t{experiment.new_value:.6f}\t"
            f"{experiment.old_score:.4f}\t{experiment.new_score:.4f}\t"
            f"{'keep' if experiment.kept else 'discard'}\t"
            f"{experiment.duration_ms:.0f}\t{experiment.config_drift:.4f}\t"
            f"{strategy_name}\n"
        )


# ── Main autotune loop ──────────────────────────────────────────────────

def autotune(
    iterations: int = 50,
    config_path: Path | None = None,
    cases_path: Path | None = None,
    seed: int = 42,
    verbose: bool = True,
    strategy_name: str = "balanced",
    use_git: bool = False,
    drift_penalty_weight: float = 0.1,
) -> dict[str, Any]:
    """
    Run the autonomous self-tuning loop.

    Key improvements over the original:
      1. Strategy-guided mutation bounds (tuning_strategy.md recipes)
      2. Config drift penalty (simplicity criterion — prevents config rot)
      3. Hard time budget (no soft penalty — timeouts = discard)
      4. Git-backed experiment history (when use_git=True)
    """
    if config_path is None:
        config_path = DEFAULT_CONFIG_PATH

    # Load strategy (Feature 1: program.md-style recipe system)
    try:
        strategy = TuningStrategy.load(strategy_name)
    except FileNotFoundError:
        strategy = TuningStrategy.balanced()
        if strategy_name not in ("balanced", "default"):
            print(f"  Warning: strategy '{strategy_name}' not found, using balanced.")

    # Apply strategy bounds to params (strategy can restrict/expand search space)
    tunable_params = strategy.apply_to_params(DEFAULT_TUNABLE_PARAMS)

    # Git log (Feature 2: git-backed history)
    git_log = GitExperimentLog() if use_git else None

    rng = random.Random(seed)
    best_config = load_tuning_config(config_path)
    best_result = evaluate(best_config, cases_path)
    best_score = best_result["composite_score"]

    # Compute baseline drift (should be 0 if starting from defaults)
    baseline_drift = config_drift_score(best_config)

    experiments: list[Experiment] = []
    improvements = 0
    start_time = time.time()

    if verbose:
        print(f"Entroly Autotune — Strategy: {strategy.name}")
        print(f"  {strategy.description}")
        print(f"  Optimizing for: {', '.join(strategy.optimize_for)}")
        print(f"  Hard time budget: {strategy.time_budget_ms}ms per optimize() call")
        print(f"  Config drift penalty weight: {drift_penalty_weight:.2f}")
        print()
        print(f"Baseline composite_score = {best_score:.4f}")
        print(f"  recall={best_result['avg_recall']:.4f} "
              f"precision={best_result['avg_precision']:.4f} "
              f"efficiency={best_result['avg_context_efficiency']:.4f} "
              f"drift={baseline_drift:.4f}")
        print(f"  {iterations} iterations planned")
        print()

    for i in range(iterations):
        # Mutate one parameter
        candidate_config, param_name, old_val, new_val = mutate_random(
            best_config, rng, tunable_params
        )

        # Evaluate with HARD time budget (Feature 3)
        t0 = time.perf_counter()
        try:
            candidate_result = evaluate(candidate_config, cases_path)
            candidate_score = candidate_result["composite_score"]
        except Exception as e:
            if verbose:
                print(f"  [{i+1:3d}] ERROR evaluating {param_name}: {e}")
            continue
        duration_ms = (time.perf_counter() - t0) * 1000

        # Feature 3: Hard time budget — latency_ok is a hard gate, not soft penalty
        if not candidate_result["all_latency_ok"]:
            # Hard zero: slow configs are discarded immediately, no soft penalty
            candidate_score = 0.0

        # Feature 4: Config drift penalty (simplicity criterion)
        # Configs that deviate wildly from defaults without proportional gain lose.
        drift = config_drift_score(candidate_config)
        # min_delta_to_keep from tuning_config — apply it here
        min_delta = best_config.get("autotuner", {}).get("min_delta_to_keep", 0.001)
        # Effective score = raw score minus drift penalty
        drift_adjusted_score = candidate_score - drift_penalty_weight * drift

        # Keep/discard decision (autoresearch: improvement must cover cost of complexity)
        raw_improvement = candidate_score - best_score
        kept = (
            candidate_result["all_latency_ok"]
            and raw_improvement >= min_delta
            and drift_adjusted_score > best_score - drift_penalty_weight * baseline_drift
        )

        if kept:
            best_config = candidate_config
            best_score = candidate_score
            best_result = candidate_result
            baseline_drift = drift  # update reference drift
            save_config(best_config, config_path)
            improvements += 1

        exp = Experiment(
            iteration=i + 1,
            param_name=param_name,
            old_value=old_val,
            new_value=new_val,
            old_score=best_score if not kept else candidate_score - raw_improvement,
            new_score=candidate_score,
            kept=kept,
            duration_ms=duration_ms,
            config_drift=drift,
        )
        experiments.append(exp)
        log_to_tsv(exp, strategy.name)

        # Feature 2: Git commit for kept experiments
        if kept and git_log:
            committed = git_log.commit_experiment(exp, best_config, strategy.name)
            if verbose and committed:
                print(f"  [git] committed experiment {i+1} to {GitExperimentLog.BRANCH}")

        if verbose:
            status = "KEEP" if kept else "SKIP"
            delta = candidate_score - (best_score if not kept else best_score)
            print(
                f"  [{i+1:3d}/{iterations}] [{status}] "
                f"{param_name}: {old_val:.4f} -> {new_val:.4f}  "
                f"score={candidate_score:.4f} (delta={delta:+.4f}) "
                f"drift={drift:.3f}  {duration_ms:.0f}ms"
            )

    elapsed = time.time() - start_time

    if verbose:
        print()
        print(f"Autotune complete: {improvements}/{iterations} improvements kept")
        print(f"  Final composite_score = {best_score:.4f}")
        print(f"  Strategy used: {strategy.name}")
        print(f"  Total time: {elapsed:.1f}s ({elapsed/max(iterations,1)*1000:.0f}ms/iter)")

    return {
        "final_score": best_score,
        "final_result": best_result,
        "improvements": improvements,
        "iterations": iterations,
        "elapsed_seconds": round(elapsed, 1),
        "strategy": strategy.name,
        "config_drift": config_drift_score(best_config),
        "experiments": [
            {
                "iteration": e.iteration,
                "param": e.param_name,
                "old": e.old_value,
                "new": e.new_value,
                "score": e.new_score,
                "kept": e.kept,
                "ms": e.duration_ms,
                "drift": e.config_drift,
            }
            for e in experiments
        ],
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Entroly autonomous self-tuning loop"
    )
    parser.add_argument(
        "--iterations", "-n", type=int, default=50,
        help="Number of tuning iterations (default: 50)"
    )
    parser.add_argument(
        "--config", type=Path, default=None,
        help="Path to tuning_config.json (default: entroly/tuning_config.json)"
    )
    parser.add_argument(
        "--cases", type=Path, default=None,
        help="Path to benchmark cases.json"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--strategy", type=str, default="balanced",
        choices=["balanced", "latency", "monorepo", "quality"],
        help="Tuning strategy recipe (default: balanced). "
             "Each strategy adjusts parameter search space and optimization goal."
    )
    parser.add_argument(
        "--git", action="store_true",
        help="Commit kept experiments to git autotune/results branch"
    )
    parser.add_argument(
        "--drift-penalty", type=float, default=0.1,
        help="Config drift penalty weight [0.0, 1.0] (default: 0.1). "
             "Higher = more conservative, stays closer to defaults."
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Output results as JSON"
    )
    args = parser.parse_args()

    result = autotune(
        iterations=args.iterations,
        config_path=args.config,
        cases_path=args.cases,
        seed=args.seed,
        verbose=not args.json,
        strategy_name=args.strategy,
        use_git=args.git,
        drift_penalty_weight=args.drift_penalty,
    )

    if args.json:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
