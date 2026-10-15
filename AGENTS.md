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
- web/: browser-only demo. After kernel changes run tests/test_browser_parity.py with Node installed.
  Test JavaScript syntax with `node --check web/app.mjs` and `node --check web/fusion.mjs`.
  The free public demo is a static Hugging Face Space; deployment instructions are in docs/DEPLOYMENT.md.

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
Never commit private research images, full datasets, weights, caches or outputs. Do not alter source
data during audits. Bundled examples are synthetic shapes plus a small attributed CC BY 4.0 TNO
subset, plus 15 owner-authorized Jiandu RGB/IR/Y demonstration triplets with no general data license.
Preserve source manifests, pixel hashes and the TNO license; examples are not benchmarks.
Keep full test/, original research ZIP archives and local weights/ ignored. Research adapters are
optional; default baselines must work without PyTorch. Static neural previews are precomputed.
Resizing is explicit and is never described as registration. No single quality metric proves
perceptual superiority. Public releases require user authorization.
