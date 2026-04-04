---
claim_id: 18a33f4c0fd34b5801c4d758
entity: conversation_pruner
status: inferred
confidence: 0.75
sources:
  - conversation_pruner.rs:96
  - conversation_pruner.rs:109
  - conversation_pruner.rs:120
  - conversation_pruner.rs:134
  - conversation_pruner.rs:147
  - conversation_pruner.rs:157
  - conversation_pruner.rs:167
  - conversation_pruner.rs:171
  - conversation_pruner.rs:181
  - conversation_pruner.rs:188
last_checked: 2026-04-04T19:51:14Z
derived_from:
  - cogops_compiler
  - sast
---

# Module: conversation_pruner

**Language:** rs
**Lines of code:** 1467

## Types
- `pub enum BlockKind` — The kind of conversation block.
- `pub enum Resolution` — Resolution level for a conversation block (LOD tier).
- `pub struct ConvBlock` — A single conversation block with causal metadata.
- `pub struct PruneResult` — Result of conversation pruning.
- `pub struct McItem` — Multi-choice knapsack item: one block with 4 resolution options.

## Functions
- `pub fn protection_weight(&self) -> f64` — Protection weight: how sacred is this block type? Higher = more protected from compression. User and system messages are nearly untouchable.
- `pub fn label(&self) -> &'static str`
- `pub fn token_fraction(&self) -> f64` — Token cost as fraction of original (lower = more savings).
- `pub fn info_retained(&self) -> f64` — Information retained as fraction of original.
- `pub fn info_loss(&self) -> f64` — Information lost = 1 - info_retained.
- `pub fn as_str(&self) -> &'static str`
- `pub fn all() -> &'static [Resolution]` — All levels in order of increasing compression.
- `pub fn classify_block(role: &str, content: &str, tool_name: Option<&str>) -> BlockKind` — Classify a message block from its role and content.
- `fn infer_dependencies(blocks: &[ConvBlock]) -> Vec<Vec<usize>>` — Infer causal dependencies between blocks.  Heuristics: 1. ToolResult[i] depends on ToolCall[i-1] (if adjacent) 2. AssistantMessage[i] depends on the most recent ToolResult before it 3. Every block dep
- `fn build_forward_refs(deps: &[Vec<usize>]) -> HashMap<usize, usize>` — Build forward reference counts from dependency edges.
- `fn score_block(` — Score a block's information value for pruning decisions.  w(v) = α·forward_value + β·ref_density + γ·recency + δ·kind_shield - ε·noise_penalty  Weights: α=0.25, β=0.20, γ=0.25, δ=0.25, ε=0.05  Returns
- `fn compute_all_forward_overlaps(blocks: &[ConvBlock]) -> Vec<f64>` — Pre-compute forward bigram overlap for ALL blocks in O(N·W) total.  Processes blocks right-to-left, accumulating a running union of "future" bigrams.  Each block's overlap = fraction of its bigrams fo
- `fn kkt_multichoice_bisect(` —  For each block i and resolution level l, define: tokens_after(i,l) = tokens(i) × level.token_fraction() info_cost(i,l)    = value(i) × level.info_loss() efficiency(i,l)   = info_cost(i,l) / tokens_sa
- `fn enforce_dag_coherence(` — Enforce causal coherence on the resolution assignments.  Rule: if block B depends on block A, then B.resolution ≥ A.resolution. A response that references a tool result can't be at higher fidelity tha
- `fn protect_recent(` — Protect recent blocks: last N non-user blocks stay at Skeleton or better.
- `pub fn compress_block(block: &ConvBlock, resolution: Resolution) -> String` — Generate compressed content for a block at a given resolution.
- `pub fn prune_conversation(` —  Uses multi-choice knapsack via KKT dual bisection with causal DAG coherence enforcement.  Protected blocks (user/system messages) are never compressed.  Recent blocks are kept at Skeleton or better. 
- `pub fn progressive_thresholds(` — Always-on mode — call before each request with current utilization. Returns recommended resolution per block at the current pressure level.  | Utilization | Action                                 | |-
- `fn make_block(index: usize, role: &str, content: &str, tokens: u32) -> ConvBlock`
- `fn make_tool_block(index: usize, role: &str, content: &str, tokens: u32, tool: &str) -> ConvBlock`
- `fn test_classify_block_types()`
- `fn test_prune_empty()`
- `fn test_prune_everything_fits()`
- `fn test_user_messages_never_pruned()`
- `fn test_system_messages_never_pruned()`
- `fn test_tool_results_pruned_first()`
- `fn test_thinking_blocks_most_aggressively_pruned()`
- `fn test_dag_coherence_propagates()`
- `fn test_dag_coherence_no_upward_propagation()`
- `fn test_infer_tool_result_depends_on_tool_call()`
- `fn test_infer_assistant_depends_on_tool_result()`
- `fn test_compress_skeleton_tool_result()`
- `fn test_compress_digest()`
- `fn test_compress_fingerprint()`
- `fn test_progressive_no_compression_below_70pct()`
- `fn test_progressive_tool_results_skeleton_at_75pct()`
- `fn test_progressive_aggressive_at_92pct()`
- `fn test_resolution_ordering()`
- `fn test_200_blocks_under_100ms()`
- `fn test_forward_overlap_recent_block_gets_default()`
- `fn test_noise_penalty_reduces_value_of_noisy_blocks()`
- `fn test_non_sequential_block_indices_no_panic()`
- `fn test_budget_zero()`
- `fn test_budget_exceeds_total()`
- `fn test_compress_block_empty_content()`
- `fn test_negative_decay_lambda_clamped()`
- `fn test_forward_overlap_multiple_blocks()`
- `fn test_progressive_thresholds_position_based_recency()`
- `fn stress_random_indices_never_panic()`
- `fn stress_budget_boundaries()`
- `fn stress_protected_messages_invariant()`
- `fn stress_compress_all_resolutions_all_kinds()`
- `fn stress_dag_coherence_post_pruning()`
- `fn stress_500_blocks_under_200ms()`
- `fn stress_progressive_thresholds_all_utilizations()`
- `fn stress_forward_overlap_correctness()`
