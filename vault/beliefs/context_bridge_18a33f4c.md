---
claim_id: 18a33f4c12b79c7004a92870
entity: context_bridge
status: inferred
confidence: 0.75
sources:
  - context_bridge.py:52
  - context_bridge.py:67
  - context_bridge.py:76
  - context_bridge.py:86
  - context_bridge.py:147
  - context_bridge.py:240
  - context_bridge.py:257
  - context_bridge.py:267
  - context_bridge.py:290
  - context_bridge.py:300
last_checked: 2026-04-04T19:51:14Z
derived_from:
  - cogops_compiler
  - sast
---

# Module: context_bridge

**Language:** py
**Lines of code:** 2031

## Types
- `class SessionContext:` — Result of optimized context loading for an OpenClaw session.
- `class AgentBudget:` — Per-agent token budget from NKBE allocation.
- `class HeartbeatResult:` — Result of an optimized heartbeat check.
- `class NkbeAllocator:` —  Nash-KKT Budgetary Equilibrium allocator for multi-agent token budgets.  Given N agents sharing a global token budget B, finds the optimal split that maximizes weighted utility:  maximize  Σᵢ wᵢ · Uᵢ
- `class _AgentState:`
- `class CognitiveBus:` —  Information-Surprise-Adaptive (ISA) event routing between agents.  Routes events between OpenClaw agents with information-theoretic prioritization. Implements Poisson rate model per subscriber per ev
- `class _BusEvent:`
- `class _Subscriber:`
- `class _RateCell:` — Poisson rate model with Welford online variance for spike detection.
- `class AgentContext:` —  Entroly context intelligence for AI agents.  Wraps EntrolyEngine to optimize the agent's workspace file loading. Instead of naively reading all files into the LLM prompt, this class ingests workspace
- `class LodTier:` — Level-of-Detail tiers for agent lifecycle management.  Adapted from both ebbiforge (4-tier swarm) and agentOS (3-tier routing). Mapped to agent states: DORMANT:    Agent registered but idle. Minimal m
- `class AgentState:` — Per-agent lifecycle state for LOD management.
- `class LODManager:` — Level-of-Detail lifecycle manager for agent armies.  Prevents resource exhaustion when 100s of agents run simultaneously. Adapted from: - ebbiforge ProductionTensorSwarm: bitflag triggers, surprise-ba
- `class SubagentOrchestrator:` — Manages subagent spawning with context inheritance and NKBE budget splitting.  The system supports up to 5 subagents per parent, 3 depth levels. Each subagent inherits a subset of its parent's context
- `class CronSessionManager:` — Manages scheduled (cron) agent sessions with minimal context loading.  Cron agents run periodically (e.g., check email every 15 min, review calendar daily). They need: 1. Minimal context — just the re
- `class CronJob:`
- `class MemoryBridge:` — Connects the cognitive bus to hippocampus long-term memory.  Two-way bridge: 1. Bus → Hippocampus: High-salience events from the cognitive bus are automatically remembered via hippocampus.remember(). 
- `class CompressionLevel:` — 3-level content compression from agentOS HCC.  Full:      100% information, 100% tokens — verbatim content. Skeleton:  ~70% information, ~20% tokens — key lines only. Reference: ~15% information, ~2% 
- `class HCCFragment:` — A fragment at all 3 compression levels.
- `class HCCEngine:` — Rate-distortion optimizer for multi-level context compression.  Ported from agentOS context.rs HccEngine.  Algorithm: 1. All fragments start at Reference level (minimum viable) 2. Compute marginal gai
- `class AutoTune:` — Adaptive weight calibration for entropy scoring parameters.  Ported from agentOS autotune.rs. Uses: - EMA (Exponential Moving Average) for smooth tracking - Polyak averaging for stability - Drift pena
- `class MultiAgentContext(AgentContext):` — Full multi-agent context intelligence for the agent system.  Extends AgentContext with: - LOD lifecycle management (DORMANT ↔ ACTIVE ↔ SATURATED ↔ HEAVY) - Subagent orchestration (spawn/despawn with c

## Functions
- `def allocate(self) -> Dict[str, int]` —  Run NKBE allocation. Returns per-agent budgets.  Two-phase KKT bisection: 1. Bisect for global λ* such that Σ Bᵢ(λ*) = B 2. Each agent gets Bᵢ = max(Bᵢ_min, demand(λ*))
- `def reinforce(self, outcomes: Dict[str, float]) -> None` —  REINFORCE weight update based on agent outcomes.  Δwᵢ = η · (Rᵢ − R̄) · wᵢ
- `def __init__(self, novelty_threshold: float = 0.3, alpha: float = 0.1)`
- `def subscribe(self, agent_name: str, event_types: Optional[List[str]] = None) -> None` — Register an agent as event subscriber.
- `def drain(self, agent_name: str, limit: int = 10) -> List[Dict[str, Any]]` — Drain events for an agent, highest priority first.
- `def stats(self) -> Dict[str, Any]`
- `def __init__(self, alpha: float = 0.1)`
- `def observe(self, tick: int) -> None` — Record an event occurrence.
- `def kl_divergence(self) -> float` — KL(P(λ_obs) ‖ P(λ_exp)) for Poisson distributions.
- `def is_spike(self) -> bool` — Welford spike detection: event rate > mean + 3σ.
- `def ingest_workspace(self) -> Dict[str, Any]` —  Ingest all OpenClaw workspace files as Entroly fragments.  Reads SOUL.md, USER.md, MEMORY.md, IDENTITY.md, TOOLS.md, HEARTBEAT.md, and daily logs. Each file (or section within a file) becomes a fragm
- `def record_outcome(self, success: bool) -> None` — Record whether the LLM output was helpful (Wilson score update).
- `def drain_events(self, agent_name: str, limit: int = 10) -> List[Dict[str, Any]]` — Drain events for an agent from the cognitive bus.
- `def get_stats(self) -> Dict[str, Any]` — Get combined statistics.
- `def __init__(self)`
- `def unregister(self, agent_id: str) -> None` — Remove an agent (and orphan its children to parent or root).
- `def update_load(self, agent_id: str, load_factor: float) -> Optional[str]` — Update agent load and trigger tier transitions.  Returns new tier if transition occurred, else None.
- `def tick(self) -> List[str]` — Advance clock. Returns any alerts generated.
- `def get_active_agents(self) -> List[str]` — Get IDs of all ACTIVE or HEAVY agents (eligible for tasks).
- `def get_budget_weights(self) -> Dict[str, float]` — Get NKBE weight hints based on LOD tier.  HEAVY agents get 2x weight, ACTIVE get 1x, DORMANT/SATURATED get 0 (no budget allocation).
- `def stats(self) -> Dict[str, Any]`
- `def can_spawn(self, parent_id: str) -> Tuple[bool, str]` — Check if a parent agent can spawn a new subagent.
- `def despawn(self, child_id: str) -> None` — Despawn a subagent and release its resources.
- `def get_tree(self, root_id: str) -> Dict[str, Any]` — Get the subagent tree rooted at an agent.
- `def stats(self) -> Dict[str, Any]`
- `def unschedule(self, agent_id: str) -> None` — Remove a cron job.
- `def get_due_jobs(self, current_time: Optional[float] = None) -> List["CronSessionManager.CronJob"]` — Get jobs that are due for execution.
- `def stats(self) -> Dict[str, Any]`
- `def __init__(self, bus: CognitiveBus)`
- `def active(self) -> bool`
- `def bridge_events(self) -> int` — Transfer high-salience cognitive bus events to hippocampus.  Called periodically (e.g., after each optimize_context cycle).
- `def tick(self) -> None` — Advance hippocampus clock (triggers decay).
- `def stats(self) -> Dict[str, Any]`
- `def __init__(self)`
- `def optimize(self, token_budget: int) -> List[HCCFragment]` — Run rate-distortion optimization.  Returns fragments with assigned_level set to optimal compression.
- `def get_content(self, frag: HCCFragment) -> str` — Get content at the fragment's assigned compression level.
- `def update(self, outcome: float, metrics: Dict[str, float]) -> Dict[str, float]` — Update weights based on optimization outcome.  Args: outcome: Quality score of the last optimization (0-1). Higher = better (user found context useful). metrics: Per-weight contribution metrics from t
- `def get_weights(self) -> Dict[str, float]` — Get current Polyak-averaged weights (stable for production use).
- `def stats(self) -> Dict[str, Any]`
- `def __init__(self, **kwargs)`
- `def despawn_subagent(self, child_id: str) -> None` — Despawn a subagent.
- `def get_agent_tree(self, root_id: str = "main") -> Dict[str, Any]` — Get the full agent tree.
- `def unschedule_cron(self, agent_id: str) -> None` — Remove a cron job.
- `def get_due_cron_jobs(self) -> list` — Get cron jobs that are due.
- `def run_cron_job(self, job) -> SessionContext` — Execute a due cron job.
- `def bridge_memories(self) -> int` — Bridge high-salience bus events to hippocampus.
- `def update_agent_load(self, agent_id: str, load: float) -> Optional[str]` — Update agent load factor, returns new tier if changed.
- `def tick(self) -> None` — Advance all clocks (LOD + bus + memory).
- `def get_stats(self) -> Dict[str, Any]`
