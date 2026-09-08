---
name: project-verification
description: Verify significant FusionAgent changes and audit prospective files before each commit.
---

From the repository root, in its Python environment:

1. Run `python .agents/skills/project-verification/scripts/verify.py`.
2. Read outputs/verification.json for Ruff format/lint, Pytest, README demo commands, CLI and candidate-file audit. Fix failures; never remove a core test to obtain green status. Manually inspect data provenance and README claims: the secret scanner is heuristic. Output passed checks, failures, repairs and untested capabilities. Report live API and remote CI verification separately from offline tests.
3. Summarize actual observations and any unresolved failures.
