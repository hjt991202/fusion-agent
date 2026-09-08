# Local verification

Date: 2026-09-08. Host: Windows, Python 3.12.4, CPU.

## Completed

- Editable installation in .venv with dev/ui/agent extras; pip check reports no broken requirements.
- Core initial stage: 17 tests passed, both-method CLI generated actual PNGs and JSON/Markdown.
- Agent/UI initial stage: 21 tests passed, including real SDK function-tool schema/invocation,
  no-key fallback, Gradio callback and missing-upload handling.
- Gradio build + launch on 127.0.0.1:7861 returned HTTP 200 and was closed cleanly.
- Dataset audit: one original synthetic pair, no issues.
- Fusion experiment and result-analysis scripts executed; both algorithms' outputs visually inspected.
- README comparison numbers are copied from measured report.json, not estimated.
- Final local suite: 26 tests passed. Ruff format and lint passed. Four Skill definitions passed
  quick_validate.py, and all four workflows were actually executed.
- Prospective Git candidate scan: no secret-pattern matches, large files, weights or output caches.

## Environment used

NumPy 2.5.3; OpenCV headless 4.14.0.94; PyWavelets 1.10.0; Pydantic 2.13.5;
Gradio 6.26.0; OpenAI Agents SDK 0.22.1; Pytest 9.1.1; Ruff 0.16.6.
Each experiment records core dependency versions in config.json. pyproject.toml uses compatible
major-version bounds so platform/Python resolvers can select appropriate wheels.

## Limits of verification

- No real API key/model call used; offline SDK tests cannot prove account access or model behavior.
- No real sensor dataset, training, GPU benchmark or statistical quality comparison.
- Remote CI status must be checked separately after publication.
- Secret scan is heuristic, complemented by candidate-file review; no private research data used.

Final full verification results are saved locally in outputs/verification.json (ignored by Git).
