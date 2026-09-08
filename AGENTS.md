# FusionAgent contributor instructions

Build a truthful, reproducible infrared-visible fusion project.
Read docs/IMPLEMENTATION_PLAN.md when resuming work.

## Layout and commands
- src/fusion_agent/tools: deterministic algorithms and IO.
- workflow.py / schemas.py: execution evidence and validated contracts.
- agents/ and ui/: optional SDK and Gradio adapters.
- tests/, examples/, docs/, .agents/skills/: validation, synthetic demo, documentation, workflows.
- Install: `python -m pip install -e ".[dev,ui,agent]"` in a virtual environment.
- Demo: `python examples/generate_demo.py`, then
  `python -m fusion_agent.cli --visible examples/visible.png --infrared examples/infrared.png`.
- UI: `python app.py`. Test: `python -m pytest -q`.
- Style: `python -m ruff check .` and `python -m ruff format --check .`.

## Required workflows
- After data changes use `$dataset-audit` at .agents/skills/dataset-audit/SKILL.md.
- After fusion changes use `$fusion-experiment` at .agents/skills/fusion-experiment/SKILL.md;
  test identity, constants, odd dimensions and invalid arrays; inspect actual images.
- For experiment interpretation use `$result-analysis` at .agents/skills/result-analysis/SKILL.md.
- After significant changes and before every commit use `$project-verification` at
  .agents/skills/project-verification/SKILL.md, including candidate-file review.
- After Agent changes run offline SDK tool tests, missing-key fallback, invalid plans and
  execution-evidence checks. Only report live API verification if actually performed.

## Standards and privacy
English code/comments, typed core functions, Ruff, small modules, bilingual truthful READMEs.
No keys in code/logs/Git. Environment variables only; .env.example contains no secrets.
Never commit private research images, datasets, weights, caches or outputs. Do not alter source
data during audits. Demo images are generated synthetic shapes, not sensor benchmarks.
Resizing is explicit and is never described as registration. No single quality metric proves
perceptual superiority. Public releases require user authorization.
