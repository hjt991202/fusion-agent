# Implementation plan

## Decisions
- Build in the existing empty repository; preserve its Git metadata.
- Python 3.10+, src layout, isolated virtual environment.
- CPU grayscale baselines first. No claims of trained models or real sensor benchmarks.
- Resizing is explicit and is not geometric registration.
- Agent is optional; deterministic execution remains available without credentials.
- Synthetic demonstrations, three user-authorized CC BY 4.0 TNO pairs, and fifteen owner-authorized
  Jiandu RGB/IR/Y demo triplets. Full research datasets and checkpoints stay out of Git.

## Stages
1. Complete: packaging installed in isolated .venv; pip check passed.
2. Complete: validated IO, weighted/Haar fusion, four metrics, reports and CLI; 17 initial tests.
3. Complete: SDK inspection/execution tools, structured plan and summary, key-free fallback.
4. Complete: Gradio callback tested; local launch returned HTTP 200 and closed cleanly.
5. Complete: four skills created, validated and actually executed.
6. Local quality complete: 26 tests, Ruff format/lint, CLI and candidate audit pass.
   Bilingual READMEs, actual comparison montage and technical documentation complete.
   Published: https://github.com/hjt991202/fusion-agent (public, main).
   Remote CI passed on Linux with Python 3.10 and 3.12 for code commit a8b8922.

## Environment
Python 3.12.4. Portable official GitHub CLI installed under ignored outputs/tools.
User confirmed target hjt991202 and completed device authorization; CLI identity verified.
The unrelated connector account HanJintan is not used for publication.
Git requires per-command safe.directory because sandbox ownership differs.

## Next commands
All six stages and requested publication are complete. To demonstrate locally:
`.venv/Scripts/python.exe app.py` on Windows, or activate .venv then `python app.py`.
For future changes follow AGENTS.md and use project-verification before each commit.

## Known limits
No live paid API call or real sensor benchmark. Offline SDK tests do not prove model access.
Three pretrained research adapters are now implemented; no training, validated geometric registration
or reference metrics. CPU verified, CUDA untested. Static browser neural results are precomputed.
The earlier browser-automation limitation was resolved during the Jiandu update below.

## Jiandu integration update (2026-10-15)
- Extended the existing workflow, CLI, Gradio and bounded Agent dispatch with ICIG and letters.
  Default weighted/Haar behavior remains available without optional research dependencies.
- Imported only reviewed inference code; local ZIPs, full test/ and checkpoints stay ignored.
  A hash-checked helper imports local weights from the owner's original archives.
- Audited all 327 RGB/IR/Y triplets (981 PNGs) read-only and rechecked source hashes afterward.
  Selected 15 pairs deterministically by size/aspect/brightness coverage, without output scoring.
- Ran all four methods on all 15 original-resolution pairs. Recomputed all 60 metric records
  from actual output PNGs; curated preview plates, shared center crops and evidence are included.
- Exact CPU pixel parity against original-code reruns: ICIG two pairs; letters one masked pair.
- Full local project verification passed: 116 tests, Ruff lint/format, CLI, three JS syntax checks,
  browser/Python kernel parity and candidate audit. Original torch.qr deprecation is retained.
- Updated SDK error handling explicitly preserves fixed plan-constraint messages across SDK
  versions while redacting unexpected IO/provider errors. Offline tool tests cover both cases.
- Gradio was exercised through a real browser on sample 517 with both research methods and
  prepared VIS_Y; its metrics matched the batch. Static page loaded the 15-pair gallery and
  ran baselines on sample 843 with no browser errors. Local stage contains 80 allowlisted files.
- Reproduction and measured interpretation: docs/RESEARCH_METHODS.md and docs/JIANDU_RESULTS.md.
  Published the integration to GitHub as 3e5cdbb. The user also explicitly authorized synchronizing
  the 15 selected examples and precomputed results to the existing Hugging Face Space.
- Complete: code commit be5b72c passed all three remote CI jobs. The existing free static Space
  is RUNNING at a0f2b7f; 80 repository files and 79 served files match the reviewed package.
  Live browser verification confirmed the 15-pair gallery and switching between 843 and 517.

## PRCV integration update (2026-10-15)
- The user supplied JDFusion.zip and requested adding its new method to this existing project.
  Its ICIG/letters archives exactly match the earlier sources; PRCV_Fusion_submit.zip is new.
- Added the optional prcv adapter, original configuration, strict local checkpoint importer,
  CLI/Gradio selection and bounded Agent dispatch. research selects three methods; all selects five.
- Source-resolution RGB and gray output match fresh original PRCV runs exactly on two samples.
  Source parameter flags and full attention mode are retained; VFC is the active displacement.
- Reused and hash-verified the original 15 pairs and earlier four-method outputs; completed PRCV
  on all 15, then independently recomputed all 75 metric records and checked 120 output PNG hashes.
  Old/new run evidence stays separate. No new archive images or weights are published.
- Local verification: 172 tests, Ruff, CLI, JS syntax, data and candidate audit passed. Real PRCV
  Gradio callback matches batch sample 517; wheel includes all nine PRCV files and no checkpoints.
  Static package contains 80 allowlisted files / 24,381,391 bytes, with five-column comparisons.
- Details: docs/RESEARCH_METHODS.md, docs/JIANDU_RESULTS.md and docs/research-validation/prcv-parity.json.

## Online deployment update
- User requested an online demonstration and redistributable public examples.
- Three official TNO pairs imported with CC BY 4.0 attribution and source/output hashes; audit passed.
- Gradio now offers clickable sample pairs. Hosted mode disables model calls, bounds queue size
  and retains temporary run outputs for up to one hour before cleanup on a subsequent request.
- User registered Hugging Face hjt991202 and completed official device authentication.
- Deploy the unchanged Python algorithms using deployment/space_app.py and scripts/deploy_space.py.
- Sites alternative was registered but never published; its metadata is retained only under ignored
  outputs/deployment-notes. No browser implementation was created or published.
- Gradio upload attempt returned HTTP 402: CPU Gradio/Docker creation requires PRO.
  No paid resource was requested. User explicitly selected a free browser-compute demo.
- Static browser implementation created in web/, uploaded to the user's static Hugging Face Space.
  JavaScript/Python parity tests pass (weighted exact, Haar at most one level of roundoff).
- Hugging Face reports RUNNING, SDK static, no compute hardware requested.
  Live page: https://huggingface.co/spaces/hjt991202/fusion-agent
  Fullscreen: https://hjt991202-fusion-agent.static.hf.space/
- Complete: live links, licensed public examples and browser deployment instructions are on GitHub.
  Code commit f63ff61 matches the uploaded web assets; remote CI passed on Python 3.10 and 3.12.
  Git smart-HTTP was unavailable, so GitHub Git Data API published the identical verified commit
  after validating both tree and commit SHA; no force update was used.
