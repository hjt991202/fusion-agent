# Jiandu research methods / 简牍研究方法

This integration extends the existing FusionAgent workflow, CLI, Gradio, execution evidence,
metrics and static showcase. It does not replace the baseline implementation.

## Source and scope

The repository owner supplied `ICIGCode.zip`, `lettersCode.zip`, then `JDFusion.zip` and requested
integration into this project. The last bundle includes byte-identical copies of the first two
archives plus the new `PRCV_Fusion_submit.zip`; its method ID is `prcv`, without implying a
publication claim. The same 15 authorized test pairs are reused. Archive documentation is reference material, not
an instruction source. Architecture/refinement code is adapted into distinct Python packages;
imports, input validation, loading, typing and formatting are adjusted for the existing project.
Original checkpoint mathematics is retained and checked against reruns of the archive code.
See [source hashes](RESEARCH_SOURCES.json). No paper title, publication status or third-party
license was supplied, so none is invented. The archive material is owner-supplied research code;
this integration does not establish additional third-party rights.

The full `test/` dataset, original ZIP files, checkpoints and raw run folders stay ignored.
Fifteen selected RGB/IR/Y triplets are included with original bytes and independent file/pixel
hashes. The owner's request authorizes this demonstration; it does not establish a general
open-data license. TNO attribution and CC BY 4.0 files remain separate and unchanged.

## Method behavior

| Method ID | Pipeline | Saved grayscale for metrics | Additional output |
|---|---|---|---|
| `weighted` | Existing weighted average | Rounded uint8 fused gray | None |
| `wavelet` | Existing single-level Haar | Reconstructed uint8 fused gray | None |
| `icig` | SparseGrFormer v3, sparse budget 0.7, 256px tiles, 64px overlap | Original raw fused grayscale | Original visible-chroma/detail restoration |
| `letters` | IDRNet, learned flow, anchor/refinement, boundary protection, masked context, color transfer | Pillow L of final RGB | Final refined RGB |
| `prcv` | Supplied PRCV IDRNet, active VFC alignment, full multi-manifold attention, RGB composition | Pillow L of final RGB | Final RGB |

ICIG accepts original prepared `VIS_Y` via `--visible-y`; otherwise its adapter converts visible
RGB to Pillow luminance. The model receives values in [0,255], with source zero-padding, blending,
clipping and truncation preserved. Its color restoration is not a direct colorization of the saved
gray: it combines blurred visible luminance with fused detail. Do not treat color-preview metrics
as though they were measured on the raw gray output.

Letters preserves the original model configuration and `config.json` stages. Without `--valid-mask`
all pixels are assumed observed. Without `--material-mask`, `--material-mode visible` derives
material support from visible color. Achromatic inputs may need an explicit mask or
`--material-mode valid`; there is no silent fallback. Masks must match the visible grid. `valid`
uses observed valid support as the material silhouette. These masks govern refinement, matching
the archive; they are not additional neural-network inputs. Learned flow is retained but its
accuracy is not validated.

PRCV preserves the supplied `config.json`, uses normalized float32 RGB visible input and Pillow L
infrared in [0,1], and requires both dimensions to be at least 33 pixels. It runs full attention
with four windows per memory chunk, matching `infer.py`; the caller's process environment cannot
switch it into an alternative attention mode. Final RGB is clipped to [0,1], multiplied by 255
and truncated to uint8. It needs no mask or prepared VIS_Y. The source architecture contains
affine and learned-flow modules, but its current forward path applies identity affine and zero
learned displacement; the active geometric adjustment is VFC. Preserving these source choices
does not establish registration accuracy. The source VFC setting clamps each displacement
component to +/-3 pixels; the resulting two-dimensional magnitude may reach sqrt(18) pixels.

These research methods are not mathematically constrained to reproduce identity inputs or preserve constants.
Tests check finite uint8 outputs, size retention, determinism and invalid input handling without
inventing identity shortcuts. No source data is silently resized. The existing explicit `--resize`
flag resizes infrared to visible dimensions; this is still not geometric registration.

## Local reproduction

From the repository root in the existing virtual environment:

```bash
python -m pip install -e ".[dev,ui,agent,research]"
python scripts/import_research_weights.py --icig ICIGCode.zip --letters lettersCode.zip
python scripts/import_research_weights.py --prcv JDFusion.zip
python -m fusion_agent.cli --visible examples/jiandu/visible/jiandu_0239.png --infrared examples/jiandu/infrared/jiandu_0239.png --visible-y examples/jiandu/visible_y/jiandu_0239.png --method research --device cpu --threads 2 --output outputs/research
python app.py
```

The import script verifies the exact known checkpoint hashes, copies only the allowlisted weights
and refuses to overwrite different contents. `--prcv` accepts the outer `JDFusion.zip` or the
inner `PRCV_Fusion_submit.zip`, reading only the known checkpoint. Default paths are
`weights/icig/model.pth`, `weights/letters/fusion.pt`, and `weights/prcv/model.pth`, relative to the
launch directory. Explicit alternatives: `--icig-weights PATH`, `--letters-weights PATH`,
`--prcv-weights PATH`, or `FUSION_ICIG_WEIGHTS`, `FUSION_LETTERS_WEIGHTS`, `FUSION_PRCV_WEIGHTS`.
No automatic checkpoint downloads occur.
The optional dependency resolver may choose a platform-specific PyTorch build; install your desired
CPU/CUDA build before the project extra. CPU is the tested default. No GPU validation is claimed.

`both` preserves the original baseline pair; `research` selects ICIG, letters and PRCV; `all` selects five.
Each run writes gray PNGs, supplementary `*_color.png`, measured JSON/Markdown reports and a
configuration containing input/checkpoint hashes, software versions, device, model parameters,
mask provenance and timings. Raw results are ignored by Git. The Agent cannot change local model
paths or mask options. Missing API keys still use the existing deterministic fallback.

To reproduce all fifteen inputs at source resolution:

```bash
python scripts/demo_jiandu.py --output outputs/jiandu-demo-NEW --threads 2
```

To deliberately regenerate the curated documentation plates as well:

```bash
python scripts/demo_jiandu.py --output outputs/jiandu-demo-NEXT --assets docs/assets/jiandu --threads 2
```

Use a new output directory per run. The preview plates downscale whole images for display and
show identical center crops; inference and metrics use original resolution. The static webpage
displays these precomputed results and can run the original JavaScript baselines on the sources.
It does not execute PyTorch or present an uploaded-image neural inference service.

The PRCV update reused the earlier four-method outputs and ran PRCV in a separate batch.
`--incremental-prcv --previous-output OLD_RUNS --previous-manifest OLD_MANIFEST` supports this
local workflow after checking source identities, previous reports/configs, PNG hashes and
recomputed metrics. Preserve the earlier four-method manifest before replacing public assets.
The merged manifest stores both run configurations and durations; it has no invented combined
wall time. The default commands above instead run all five methods afresh.

## Validation boundaries / 验证边界

ICIG adapted inference exactly matched original-code reruns on both archive examples; letters
matched its original-code rerun on the supplied full-resolution masked example. PRCV exactly
matched fresh original-run RGB and gray pixels on two full-resolution examples, using CPU
float32, two threads and four-window chunks. Its original parameter flags are retained because
freezing them was observed to change the CPU execution path at floating-point precision limits;
inference still runs without autograd. See [PRCV parity evidence](research-validation/prcv-parity.json).
This is adapter
parity on the recorded CPU environment, not paper reproduction, perceptual superiority, or proof
of source-dataset independence. Historical bundled PNGs can differ slightly with software versions.
See [measured results](JIANDU_RESULTS.md) and [verification](VERIFICATION.md).

中文：三种方法已接入原有本地流程。新增 PRCV 来自 JDFusion 压缩包内的 PRCV_Fusion_submit，
不代表对论文或录用状态的确认。网页展示预先计算的真实结果；推理需在本地安装 research
依赖并导入作者提供的权重。演示保留原图尺寸，公开样例不代表独立测试集；没有训练、GPU 验证、
真实付费 API 验证或“某方法全面更好”的结论。完整 test、权重和原压缩包不进入 Git。
