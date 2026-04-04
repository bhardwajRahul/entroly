# CogOps Vault Contract
# =====================
# This vault is the machine-auditable knowledge graph for the Entroly codebase.
#
# Directory structure:
#   vault/beliefs/       - Durable system understanding (belief artifacts with YAML frontmatter)
#   vault/verification/  - Challenges to understanding (drift reports, contradiction reports)
#   vault/actions/       - Task outputs and reports (PR briefs, context packs, answers)
#   vault/evolution/     - Skill specs, trial results, promotion logs
#   vault/media/         - Shared render assets only (mermaid, charts, images)
#
# Enforcement:
#   beliefs/      → NO task outputs, test results
#   verification/ → NO raw data, skill specs
#   actions/      → NO belief definitions, skill code
#   evolution/    → NO belief pages, task outputs
#   media/        → NO markdown documents, code
#
# Every belief artifact MUST carry frontmatter:
#   claim_id, entity, status, confidence, sources, last_checked, derived_from
