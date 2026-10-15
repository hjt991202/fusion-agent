# Online deployment

## Free browser demo (selected deployment)

The public demo is a **static Hugging Face Space**, using web/index.html and JavaScript kernels.
It requires no CPU server, model API, account for visitors, or paid subscription. Images are decoded
and fused on the visitor's device. The app only fetches the bundled example images; it never uploads
user-selected images. Results can be downloaded as PNG and Markdown.

```bash
python scripts/deploy_static_space.py --stage outputs/static-release-NEW
python -m http.server 7863 --bind 127.0.0.1 --directory outputs/static-release-NEW
# After verification, prepare a new directory and publish:
python scripts/deploy_static_space.py --stage outputs/static-release-NEXT --repo hjt991202/fusion-agent --publish
```

Publication verifies the Hugging Face owner and the Space SDK type. The allowlist contains five
web source files, the code license, attributed TNO/synthetic examples, 15 explicitly authorized
Jiandu RGB/IR/Y triplets and their measured preview plates/manifests. Source hashes and matching
sample IDs are checked before staging. No API tokens, checkpoints, full test dataset or raw run
folders are copied. The project owner explicitly authorized publishing the selected Jiandu
examples and precomputed results to this existing Hugging Face Space on 2026-10-15.
The earlier Sites alternative remained unpublished; the chosen deployment uses the user's Space.

Browser scope: same-size pairs, weighted average and one-level symmetric Haar, four metrics,
task notes, input hashes, tool timings, PNG and report downloads. The expanded source also shows
15 precomputed ICIG/letters/PRCV comparisons with full-resolution measurements and shared crops.
Selecting a Jiandu example can load its original RGB/IR into browser baselines. The three learned
models do not run in the static browser. No online Agent or resizing. When adding PRCV incrementally,
the manifest retains the original four-method run and new PRCV run separately; displayed times
are measured across separate invocations, not a single contemporaneous performance benchmark.
PNG/JPEG/BMP/WebP up to 16 MiB and four million pixels. Transparent images are rejected;
convert high-bit-depth input to 8-bit explicitly before use. Canvas color/decoding may differ
from OpenCV. Tests compare grayscale kernels: weighted output is exact, Haar output may differ
by at most one level because of floating-point roundoff; metric definitions match Python.

## Original Python Gradio Space (optional, not deployed)

Current deployment attempt: the account returned HTTP 402 on 2026-09-08 because CPU Gradio
Space creation requires PRO. No running Gradio Space is claimed. The prepared files remain
available for a future eligible account; no subscription or paid hardware was purchased.

The hosted demonstration uses the same Python algorithms as the local CLI and Gradio app.
`deployment/space_app.py` binds to 0.0.0.0 and forces public deterministic mode, regardless of
checkbox/API input. It does not expose the owner's OpenAI key or allow paid model calls.

## Source preparation and upload

Authenticate with the official CLI (`hf auth login`) on your own machine. Never put tokens in Git,
logs, Space source or chat. Use the hf CLI installed with huggingface_hub, available through Gradio.

```bash
python scripts/deploy_space.py --stage outputs/space-release-NEW --repo hjt991202/fusion-agent
```

The stage must be a new directory. The script copies only src/, examples/, pyproject.toml, LICENSE
and the dedicated Space entrypoint/README; no outputs, private datasets, keys or local environments.
Review the prepared directory. To prepare and publish a subsequent new stage:

```bash
python scripts/deploy_space.py --stage outputs/space-release-NEXT --repo hjt991202/fusion-agent --publish
```

Publishing checks the authenticated owner, creates/reuses a public Gradio Space and uploads the
allowlisted files. Space metadata pins the tested Gradio 6.26.0 and Python 3.12. requirements.txt
installs the local project with its ui extra. No hardware upgrade is requested. Hugging Face may
impose account eligibility or plan requirements; the script never buys a plan.

## Runtime behavior

- Nineteen clickable local examples: three TNO, one synthetic and fifteen Jiandu pairs.
- Local Jiandu examples also load their prepared VIS_Y and clear previous masks.
- Optional local `research` extra and checkpoints enable ICIG/letters/PRCV; the hosted Python
  preparation installs only `ui` by default, so neural models need explicit additional setup.
- Uploads are sent to the hosted server. Do not upload confidential research data.
- One fusion request executes at a time; queue capacity is eight.
- Gradio cached files expire after an hour with checks every ten minutes.
- App run directories older than an hour are cleaned on subsequent requests; an idle instance may
  retain files until the next request/restart. New requests are rejected after 100 retained runs.
- The underlying sources are not modified. Use Download report to retain results locally.
- Higher metrics do not establish perceptual superiority. Size matching is not registration.

The CC BY 4.0 TNO image license remains separate from the MIT source-code license.
