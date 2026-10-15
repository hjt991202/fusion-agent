# Local verification

## PRCV addition — 2026-10-15

- Full project verification: **172 tests passed**, with local ICIG trained-checkpoint tests enabled;
  Ruff lint/format, CLI, three JavaScript syntax checks and candidate audit passed. The only warning
  remains the original ICIG torch.qr deprecation. PRCV has 30 additional architecture/adapter tests.
- PRCV RGB and Pillow L outputs exactly match fresh original infer.py runs on two full-resolution
  samples: [parity evidence](research-validation/prcv-parity.json). Source modules also match after
  documented import/type/mode cleanup: [source equivalence](research-validation/prcv-source-equivalence.json).
- Input audit: all 15 existing RGB/IR pairs valid; source RGB/IR/Y hashes unchanged. No new source
  examples were published. All 75 gray metric records and all 120 saved output hashes independently
  verified. Fifteen five-column plates and sample 843 details were visually inspected.
- Actual local Gradio PRCV callback on sample 517 returned gray/color PNGs, metrics, report and
  checkpoint trace; metric values and color PNG hash exactly match the final batch. No API call.
- Wheel build contains all nine PRCV package modules/config files and no weights. The importer
  accepts the reviewed nested or standalone archive, checks hashes/size bounds and refuses to
  overwrite a different checkpoint; 11 synthetic-archive tests cover these boundaries.
- Incremental evidence checks reject source, report, output and timing inconsistencies. The old
  four-method and new PRCV batches retain separate settings/versions/durations. Static staging
  has 80 allowlisted files / 24,381,391 bytes. Browser neural output remains precomputed.
- No CUDA, training, independent registration/recognition-quality benchmark or live paid API test.
- Public release verified: [GitHub CI](https://github.com/hjt991202/fusion-agent/actions/runs/38694898556)
  passed for code commit `0119b76`. Standard Git HTTPS was unavailable; the Git Data API published
  the exact verified tree/commit with a non-force update after checking the remote parent.
- After explicit approval for the new PRCV material, the existing static Space was updated to
  [ed32025b](https://huggingface.co/spaces/hjt991202/fusion-agent/commit/ed32025bffdba803777c60c5d7ae1ba9764734b0).
  Runtime is RUNNING; anonymous checks matched all 80 repository files and 79 served files to
  the reviewed package (only the platform-injected index.html script is removed for comparison).
  A real browser refresh showed the three-method heading, five-column preview, all five measured
  metric rows and separate-run timing records on sample 843.

## Jiandu integration — 2026-10-15

- Full project-verification passed: **116 tests passed**, no skips with the local ICIG checkpoint
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
- Explicit SDK tool error handling preserves fixed plan-constraint messages and redacts arbitrary
  IO/provider exception text. Existing guard tests and the added redaction regression pass.
- GitHub CI passed for code commit `be5b72c`: Python 3.10, Python 3.12 and research contracts.
  [Verified run](https://github.com/hjt991202/fusion-agent/actions/runs/38681680529).
- With explicit owner approval, the existing free static Space was updated to
  [commit a0f2b7f](https://huggingface.co/spaces/hjt991202/fusion-agent/commit/a0f2b7f72aa3f76b08faf82e859f67c8cff55dc9).
  Its runtime is RUNNING. Anonymous verification matched all 80 package file hashes to the Space
  repository and all 79 served files to the stage, removing only the platform-injected script
  from index.html. The generated Space README is checked through the repository.
- The live browser shows all 15 choices, previews and four-method metric tables; switching from
  sample 843 to 517 updates the image, metrics and original-source links correctly.

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
