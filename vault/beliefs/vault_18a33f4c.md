---
claim_id: 18a33f4c1437910406291d04
entity: vault
status: inferred
confidence: 0.75
sources:
  - vault.py:49
  - vault.py:55
  - vault.py:66
  - vault.py:80
  - vault.py:99
  - vault.py:113
  - vault.py:125
  - vault.py:143
  - vault.py:151
  - vault.py:156
last_checked: 2026-04-04T19:51:14Z
derived_from:
  - cogops_compiler
  - sast
---

# Module: vault

**Language:** py
**Lines of code:** 521

## Types
- `class VaultConfig:` — Configuration for the Obsidian vault.
- `class BeliefArtifact:` — A machine-auditable belief written to the vault.
- `class VerificationArtifact:` — A verification challenge against a belief.
- `class VaultManager:` —  Manages the Obsidian vault directory structure and artifact I/O.  This is the persistence layer for the Living Exocortex. All belief, verification, action, and evolution artifacts pass through here.

## Functions
- `def path(self) -> Path`
- `def to_markdown(self) -> str` — Render as markdown with YAML frontmatter.
- `def to_dict(self) -> Dict[str, Any]`
- `def to_markdown(self) -> str`
- `def __init__(self, config: Optional[VaultConfig] = None)`
- `def ensure_structure(self) -> Dict[str, Any]` — Create the vault directory structure if it doesn't exist.
- `def write_belief(self, artifact: BeliefArtifact) -> Dict[str, Any]` — Write a belief artifact to the vault.
- `def read_belief(self, entity: str) -> Optional[Dict[str, Any]]` — Read a belief artifact by entity name.
- `def list_beliefs(self) -> List[Dict[str, Any]]` — List all belief artifacts with their frontmatter.
- `def write_verification(self, artifact: VerificationArtifact) -> Dict[str, Any]` — Write a verification artifact to the vault.
- `def coverage_index(self) -> Dict[str, Any]` — Build a coverage index of all beliefs for the router.
- `def mark_beliefs_stale_for_files(self, changed_files: List[str]) -> Dict[str, Any]` — Mark beliefs stale when their sources overlap the changed files.
