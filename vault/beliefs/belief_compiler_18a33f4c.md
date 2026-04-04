---
claim_id: 18a33f4c122f7b9c0421079c
entity: belief_compiler
status: inferred
confidence: 0.75
sources:
  - belief_compiler.py:38
  - belief_compiler.py:47
  - belief_compiler.py:59
  - belief_compiler.py:65
  - belief_compiler.py:78
  - belief_compiler.py:112
  - belief_compiler.py:289
  - belief_compiler.py:292
  - belief_compiler.py:297
  - belief_compiler.py:304
last_checked: 2026-04-04T19:51:14Z
derived_from:
  - cogops_compiler
  - sast
---

# Module: belief_compiler

**Language:** py
**Lines of code:** 706

## Types
- `class CodeEntity:` — An entity extracted from source code.
- `class ModuleMap:` — A module-level view of the codebase.
- `class CompilationResult:` — Result of a belief compilation run.
- `class EntityResolver:` — Resolves and links entities across files.
- `class BeliefCompiler:` —  Compiles source code into machine-auditable belief artifacts.  This is the automated Truth → Belief pipeline. It: 1. Scans source files for code entities 2. Resolves cross-file dependencies 3. Genera

## Functions
- `def extract_skeleton(content: str, source: str) -> str:  # type: ignore`
- `def qualified_name(self) -> str`
- `def extract_entities(content: str, file_path: str) -> List[CodeEntity]` — Extract code entities from a source file.
- `def __init__(self)`
- `def add_entities(self, entities: List[CodeEntity]) -> None`
- `def resolve_dependencies(self) -> None` — Cross-link entities by matching import names to entity names.
- `def dependency_graph(self) -> Dict[str, List[str]]`
- `def get_modules(self) -> Dict[str, List[CodeEntity]]` — Group entities by file path (module).
- `def generate_module_diagram(modules: Dict[str, List[CodeEntity]]) -> str` — Generate a Mermaid module-level architecture diagram.
- `def __init__(self, vault: VaultManager)`
- `def compile_file(self, file_path: str, content: str) -> Optional[BeliefArtifact]` — Compile a single file into a belief artifact.
