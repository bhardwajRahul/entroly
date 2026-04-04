---
claim_id: 18a33f4c0fa0d81c0192641c
entity: cognitive_bus
status: inferred
confidence: 0.75
sources:
  - cognitive_bus.rs:61
  - cognitive_bus.rs:89
  - cognitive_bus.rs:109
  - cognitive_bus.rs:134
  - cognitive_bus.rs:148
  - cognitive_bus.rs:154
  - cognitive_bus.rs:161
  - cognitive_bus.rs:167
  - cognitive_bus.rs:179
  - cognitive_bus.rs:186
last_checked: 2026-04-04T19:51:14Z
derived_from:
  - cogops_compiler
  - sast
---

# Module: cognitive_bus

**Language:** rs
**Lines of code:** 897

## Types
- `pub enum EventType` — Event types routable on the cognitive bus. Maps to agentOS 25 event types, grouped into 4 zones.
- `pub struct BusEvent` — An event published on the cognitive bus.
- `pub struct PrioritizedEvent` — Prioritized event wrapper for the per-agent queue.
- `pub struct WelfordAccumulator` — Welford's online algorithm for running mean and variance. No windowing — tracks full history with O(1) memory.
- `pub struct PoissonRate` — Per-subscriber per-event-type Poisson rate estimation. EMA-smoothed: λ̂(n+1) = α · observed_rate + (1−α) · λ̂(n)
- `pub struct Subscriber` — A subscriber (agent) on the cognitive bus.
- `pub struct CognitiveBus` — - Entroly dedup (SimHash novelty scoring) - Hippocampus memory (salience-based remember/recall bridge) - NKBE allocator (events influence budget reallocation)  Memory-aware routing: - Events with sali

## Functions
- `fn from_str(s: &str) -> Self`
- `fn as_str(&self) -> &str`
- `fn eq(&self, other: &Self) -> bool`
- `fn partial_cmp(&self, other: &Self) -> Option<Ordering>`
- `fn cmp(&self, other: &Self) -> Ordering`
- `fn new() -> Self`
- `fn update(&mut self, x: f64) -> (f64, f64)` — Update with a new observation. Returns (new_mean, new_stddev).
- `fn is_spike(&self, x: f64, k: f64) -> bool` — Check if value is a spike (> k standard deviations from mean).
- `fn new() -> Self`
- `fn observe(&mut self, current_count: u64, current_time: f64)` — Update rate estimate with new observation.
- `fn kl_surprise(&self, observed_rate: f64) -> f64` — KL divergence: KL(λ_obs ‖ λ_exp) = λ_exp − λ_obs + λ_obs · ln(λ_obs / λ_exp) Measures information surprise — high KL = unexpected event rate.
- `fn new(agent_id: &str) -> Self`
- `fn compute_novelty(&self, event_hash: u64) -> f64` — Compute novelty of an event relative to this subscriber's history. novelty(msg) = 1 − max_{h ∈ history} (1 − d_H(msg, h) / 64) softcapped: CAP · tanh(novelty / CAP)
- `fn compute_mutual_info(&self, event_hash: u64) -> f64` — Estimate mutual information between event and subscriber's current task. Approximated via SimHash Hamming similarity as proxy for content overlap. I(event; task) ≈ similarity(event_hash, task_hash)
- `fn compute_priority(&mut self, event: &BusEvent, current_time: f64) -> f64` — Compute ISA priority for an event destined to this subscriber.
- `fn check_spike(&mut self, priority: f64) -> bool` — Check if a priority score represents a spike (Welford).
- `fn enqueue(&mut self, event: BusEvent, priority: f64)` — Enqueue an event with its computed priority.
- `fn drain(&mut self, limit: usize) -> Vec<BusEvent>` — Drain top-K events for delivery.
- `fn is_subscribed(&self, event_type: &EventType) -> bool` — Check if this subscriber is interested in an event type.
- `pub fn new(memory_salience_threshold: f64) -> Self`
- `pub fn subscribe(&mut self, agent_id: &str, event_types: Vec<String>)` — Register an agent as a subscriber.  Args: agent_id: Unique agent identifier. event_types: List of event type strings to subscribe to. Empty list = subscribe to all events.
- `pub fn unsubscribe(&mut self, agent_id: &str)` — Unsubscribe an agent.
- `pub fn set_task_context(&mut self, agent_id: &str, task_text: &str)` — Update a subscriber's current task context (for MI-based routing).  The task text is SimHashed and used to boost events with high mutual information toward the subscriber's current task.
- `pub fn tick(&mut self)` — Advance the bus clock by one tick.
- `pub fn set_tick(&mut self, tick: f64)` — Set the current tick explicitly (for sync with external clock).
- `pub fn publish(` — Args: source_agent: ID of the publishing agent. event_type: Event type string (e.g., "observation", "belief"). content: Event payload text. emotional_tag: 0=neutral, 1=positive, 2=negative, 3=critical
- `pub fn drain<'py>(&mut self, py: Python<'py>, agent_id: &str, limit: usize) -> PyResult<Vec<Bound<'py, PyDict>>>` — Returns events ordered by ISA priority (highest first). Spike events are always included first.  Args: agent_id: The subscriber agent ID. limit: Maximum number of events to return.  Returns: List of d
- `pub fn drain_memory_bridge<'py>(&mut self, py: Python<'py>) -> PyResult<Vec<Bound<'py, PyDict>>>` — Drain events flagged for hippocampus memory bridging.  Called by the Python layer to feed high-salience events to hippocampus.remember(). Returns and clears the bridge queue.
- `pub fn queue_depth(&self, agent_id: &str) -> usize` — Get queue depth for a subscriber.
- `pub fn stats<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>>` — Get bus statistics.
- `pub fn drain_raw(&mut self, agent_id: &str, limit: usize) -> Vec<BusEvent>` — Drain raw events (Rust-only, for internal use and testing).
- `pub fn drain_memory_bridge_raw(&mut self) -> Vec<BusEvent>` — Drain raw memory bridge events (Rust-only, for internal use and testing).
- `fn test_welford_basic()`
- `fn test_welford_needs_minimum_samples()`
- `fn test_poisson_rate_kl()`
- `fn test_subscriber_novelty_first_event()`
- `fn test_subscriber_novelty_duplicate()`
- `fn test_subscriber_mutual_info_no_task()`
- `fn test_subscriber_mutual_info_same_task()`
- `fn test_bus_publish_and_drain()`
- `fn test_bus_dedup_prevents_duplicate()`
- `fn test_bus_memory_bridge()`
- `fn test_bus_type_filtering()`
- `fn test_bus_no_self_routing()`
- `fn test_bus_task_context_boosts_priority()`
- `fn test_bus_stats_counters()`
