# Implementation plan

## Decisions
- Build in the existing empty repository; preserve its Git metadata.
- Python 3.10+, src layout, isolated virtual environment.
- CPU grayscale baselines first. No claims of trained models or real sensor benchmarks.
- Resizing is explicit and is not geometric registration.
- Agent is optional; deterministic execution remains available without credentials.
- Synthetic demonstrations plus three user-authorized CC BY 4.0 TNO pairs; no private research data.

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
No deep model, training, geometric registration or reference metric is implemented.
Browser automation timed out; Gradio tested by real HTTP startup and direct callbacks.

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
