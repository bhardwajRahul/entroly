---
claim_id: 18a33f4c128b208c047cac8c
entity: cli
status: inferred
confidence: 0.75
sources:
  - cli.py:59
  - cli.py:341
  - cli.py:393
  - cli.py:411
  - cli.py:444
  - cli.py:506
  - cli.py:564
  - cli.py:660
  - cli.py:707
  - cli.py:783
last_checked: 2026-04-04T19:51:14Z
derived_from:
  - cogops_compiler
  - sast
---

# Module: cli

**Language:** py
**Lines of code:** 2298

## Types
- `class C:`

## Functions
- `def cmd_init(args)` — entroly init — auto-detect and configure.
- `def cmd_serve(args)` — entroly serve — start MCP server with auto-indexing.
- `def cmd_dashboard(args)` — entroly dashboard — launch live web dashboard at localhost:9378.
- `def cmd_health(args)` — entroly health — analyze codebase health.
- `def cmd_autotune(args)` — entroly autotune — optimize engine hyperparameters.
- `def cmd_proxy(args)` — entroly proxy — start the invisible prompt compiler proxy.
- `def cmd_benchmark(args)` — entroly benchmark — run competitive comparison.
- `def cmd_status(args)` — entroly status — check if server/proxy is running.
- `def cmd_config(args)` — entroly config show — display current configuration.
- `def cmd_telemetry(args)` — entroly telemetry — manage anonymous usage statistics.
- `def is_telemetry_enabled() -> bool` — Check if opt-in telemetry is enabled. Always False by default.
- `def cmd_clean(args)` — entroly clean — clear cached state (checkpoints, index, pull cache).
- `def cmd_export(args)` — entroly export — export learned state for sharing (Gap #32).
- `def cmd_import(args)` — entroly import — import shared learned state (Gap #32).
- `def cmd_drift(args)` — entroly drift — detect weight drift / staleness (Gap #30).
- `def cmd_profile(args)` — entroly profile — manage per-project weight profiles (Gap #31).
- `def cmd_batch(args)` — entroly batch — headless/CI mode for batch optimization (Gap #33).
- `def cmd_go(args)` — entroly go — one command to rule them all: init + proxy + dashboard.
- `def cmd_demo(args)` — entroly demo — quick-win demo mode: before/after comparison (Gap #41).
- `def cmd_doctor(args)` — entroly doctor — diagnose common issues (Gap #52).
- `def cmd_digest(args)` — entroly digest — show weekly summary of entroly's value (Gap #44).
- `def cmd_migrate(args)` — entroly migrate — auto-migrate config/index to new format (Gap #53).
- `def cmd_role(args)` — entroly role — role-based weight presets for different developer types (Gap #49).
- `def cmd_completions(args)` — entroly completions {bash|zsh|fish} — output shell completion script.
- `def cmd_optimize(args)` — entroly optimize — generate an optimized context snapshot for a specific task.  This is the primary command for subagent-driven workflows. It indexes the codebase, selects the mathematically optimal f
- `def cmd_feedback(args)` — entroly feedback — signal outcome quality to improve future context selection.  After an agent completes a task using optimized context, run this to tell Entroly whether the context was helpful. Entro
- `def main()` — Main CLI entry point.
