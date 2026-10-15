# Local verification

## Jiandu integration — 2026-10-15

- Full project-verification passed: **115 tests passed**, no skips with the local ICIG checkpoint
  enabled; Ruff lint/format, README CLI, three JavaScript syntax checks and candidate audit pass.
  One warning is the original ICIG `torch.qr` deprecation retained for archive parity.
- Optional model contracts include identity-input/constant/odd-size execution, finite outputs,
  invalid arrays/masks/checkpoints, safe strict loading, NaN/Inf rejection and CPU repeatability.
- Exact original-code rerun parity on two ICIG examples and one full-resolution letters example:
  [ICIG evidence](research-validation/icig-parity.json), [letters evidence](research-validation/letters-parity.json).
- Read-only input audit: 327 complete triplets, 981 PNGs, no issues; every source hash unchanged.
  Fifteen triplets (14,009,494 bytes) were selected; published PNG bytes match original sources.
- Sixty actual method executions on fifteen original-resolution pairs; all published metrics
  equal independent recomputation from saved PNGs. Actual results and shared crops inspected.
  [Measured results and limits](JIANDU_RESULTS.md).
- Gradio real browser check: selected sample 517, selected `research`, executed successfully;
  four result images, both metric rows, weight evidence and report appeared. Prepared VIS_Y
  was loaded and both method metrics exactly matched the batch run.
- Static browser real check: 15 options, preview/metric table, source loading and browser baseline
  execution on sample 843; no console errors. Stage has exactly 80 allowlisted files / 22,579,242
  bytes, with all 79 copied files byte-identical to source plus one generated README.
- Wheel packaging verified: both model packages and letters config included, no checkpoints.
  pip check passed. Optional dependencies: torch 2.7.1+cpu, SciPy 1.13.1, einops 0.8.2,
  NumPy 2.2.6, OpenCV headless 4.14.0.94, Python 3.12.4; per-run configs record versions.
- CI includes separate research contracts with CPU PyTorch and small explicitly untrained test
  fixtures. Public CI does not contain the owner's checkpoints; trained-checkpoint parity is local.
  No GPU, training, recognition-quality benchmark, validated registration or live paid API call.

The following sections preserve the earlier project's historical verification record.

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
- Remote CI passed on Linux for Python 3.10 and 3.12, including dependency installation,
  Ruff, Pytest and CLI: [verified run](https://github.com/hjt991202/fusion-agent/actions/runs/34219080965).
  This run validates code commit a8b8922; the follow-up publication record changes only documentation.
- Secret scan is heuristic, complemented by candidate-file review; no private research data used.

Final full verification results are saved locally in outputs/verification.json (ignored by Git).

## Public examples and deployment preparation

- TNO official archive MD5 matched figshare. Three pairs imported with unchanged decoded pixels,
  CC BY 4.0 attribution and a source/output SHA256 manifest; dataset-audit found no issues.
- Updated local suite: 29 tests passed, including TNO provenance, public Agent rejection and
  a full public callback using a real sample. Ruff and the full verification script passed.
- CPU Gradio Space creation returned HTTP 402 (PRO required); this is a provider account
  restriction, not a successful online deployment. No paid resource was created.
- User chose the free static browser alternative. Hugging Face reports RUNNING with SDK static
  and no compute hardware: https://huggingface.co/spaces/hjt991202/fusion-agent .
- 32 local tests pass, including browser/Python parity across even/odd images. Weighted outputs
  match exactly; Haar tolerance is one grayscale level; metrics match on identical image arrays.
- JavaScript syntax checks pass. This does not claim pixel-identical browser/OpenCV color decoding.
- Anonymous HTTP checks confirm the hosted JavaScript, stylesheet and eight sample PNG hashes
  match the prepared package. Hosted index.html differs only by Hugging Face's injected platform
  variable script. Local static preview returned HTTP 200.
- Remote CI passed for browser/data code commit f63ff61 on Python 3.10 and 3.12:
  https://github.com/hjt991202/fusion-agent/actions/runs/34223914840 .
