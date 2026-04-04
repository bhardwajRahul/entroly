---
claim_id: 18a33f4c1274750804660108
entity: checkpoint
status: inferred
confidence: 0.75
sources:
  - checkpoint.py:61
  - checkpoint.py:83
  - checkpoint.py:238
  - checkpoint.py:316
  - checkpoint.py:421
  - checkpoint.py:440
  - checkpoint.py:447
  - checkpoint.py:493
  - checkpoint.py:516
  - checkpoint.py:564
last_checked: 2026-04-04T19:51:14Z
derived_from:
  - cogops_compiler
  - sast
---

# Module: checkpoint

**Language:** py
**Lines of code:** 581

## Types
- `class ContextFragment:  # type: ignore[no-redef]` — Pure-Python fallback when entroly_core (Rust) is not installed.
- `class Checkpoint:` — A serialized snapshot of the Entroly state.
- `class CheckpointManager:` —  Manages saving and restoring Entroly state.  Checkpoints are stored as gzipped JSON files in the checkpoint directory. Each checkpoint includes the full state needed to resume a session without any d

## Functions
- `def should_auto_checkpoint(self) -> bool` — Check if an auto-checkpoint is due.
- `def load_latest(self) -> Optional[Checkpoint]` —  Load the most recent checkpoint (from this instance).  Returns None if no checkpoints exist or all are unreadable.
- `def load_by_id(self, checkpoint_id: str) -> Optional[Checkpoint]` — Load a specific checkpoint by its ID.
- `def merge_from_peers(self, local_fragments: List[Dict[str, Any]]) -> List[Dict[str, Any]]` — Scan for checkpoints from peer instances and merge their fragments.  Returns the merged fragment list combining local + peer knowledge. Uses most-recent-writer-wins conflict resolution.
- `def list_checkpoints(self) -> List[Dict[str, Any]]` — List all available checkpoints with metadata.
- `def restore_fragments(self, checkpoint: Checkpoint) -> List[ContextFragment]` — Extract ContextFragment objects from a checkpoint.
- `def stats(self) -> dict`
