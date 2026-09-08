# GitHub and interview materials

## Repository description

Traceable infrared-visible image fusion with CPU baselines, measured quality metrics,
optional OpenAI Agents SDK orchestration, Gradio, CLI and reproducible experiment skills.

## Topics

`image-fusion`, `infrared-visible-fusion`, `computer-vision`, `python`, `opencv`,
`wavelet-transform`, `openai-agents`, `gradio`, `reproducible-research`, `pytest`

Do not add pytorch/deep-learning topics until a working model adapter is implemented.

## 中文简历描述

**FusionAgent：红外—可见光图像融合智能体系统 | Python / OpenCV / PyWavelets / Agents SDK**

- 实现加权平均与 Haar 小波两种 CPU 融合基线，封装图像校验、显式尺寸匹配及信息熵、
  标准差、平均梯度、空间频率四项指标，支持 CLI 与 Gradio 演示。
- 基于 Pydantic 与 OpenAI Agents SDK 实现可选任务规划和受约束工具调用，保留无密钥
  确定性流程；使用输入哈希、参数/版本配置、工具计时及 Markdown/JSON 报告追踪实验。
- 编写 Pytest 测试、Ruff 检查、GitHub Actions 与四个仓库级 Skills，覆盖输入异常、
  算法不变量、CLI、SDK 工具与界面回调，建立可复现验证流程。

请在能独立解释和复现代码后使用。不要写“提升 X%”“SOTA”“完成大模型训练”或“在线 Agent
全部验证”，除非后续有真实实验依据。当前示例是合成数据，不代表数据集级泛化性能。

## English resume description

**FusionAgent — Infrared/Visible Fusion Workflow | Python, OpenCV, PyWavelets, Agents SDK**

- Implemented weighted-average and Haar-wavelet CPU fusion baselines with validated image IO,
  explicit resizing, four no-reference metrics, a CLI and a Gradio demonstration.
- Integrated optional structured agent planning and bounded tool execution with a key-free
  deterministic fallback; recorded input hashes, dependency versions, parameters and tool timings.
- Built reproducible Markdown/JSON experiment reports, Pytest coverage, Ruff checks, GitHub Actions
  configuration and four repository skills for dataset auditing and experiment verification.

## Five interview questions and reference answers

### 1. Why use wavelets rather than simply averaging pixels?

Averaging is a stable baseline but attenuates modality-specific details. A one-level Haar transform
separates approximation and directional detail coefficients. This implementation averages low
frequencies and selects larger-magnitude signed details. It can preserve edges but may also select
noise or create discontinuities. The synthetic demo shows behavior, not universal superiority.

### 2. Why can't entropy or gradient alone establish better fusion?

Noise can raise entropy, contrast, gradient and spatial frequency. These no-reference statistics
do not establish target preservation or perceptual fidelity. I use an explicit gradient heuristic
for the demo, disclose that limitation, and would validate a real system with registered datasets,
human inspection and downstream detection or segmentation performance.

### 3. What does the Agent add, and how do you prevent fabricated results?

The Agent maps a natural-language preference to an allowed FusionPlan and invokes inspection and
execution tools. It does not generate image processing code. Pydantic validates methods and weights;
tools bind paths and constrain resizing. Python computes and saves metrics, records actual tools,
and returns the authoritative report. Model prose is separate. Missing tool execution raises an
error instead of trusting an invented response. Missing API keys use deterministic execution.

### 4. How do you make an experiment reproducible?

Save input SHA256 hashes, parameters, dependency and Python versions, generated PNGs and measured
statistics. Algorithms are deterministic, with seed 0 recorded for future stochastic extensions.
Use unique run directories and tests for known analytic metric values, constant images, identity
fusion, odd image dimensions and repeated runs. Timing is not bit-reproducible; record hardware
and repeated timing distributions before making speed claims.

### 5. How does this relate to your image-enhancement research?

Fusion and enhancement both require preserving useful structure while controlling noise and
artifacts. This project contributes a measurable engineering workflow around that research.
Retinex, frequency-domain methods, attention and Grassmann techniques could be evaluated later
as separate hypotheses against the existing baselines. I would not claim those techniques are
implemented here or upload private bamboo-slip research data.

## Future PyTorch interface (design only)

The `FusionMethod` protocol accepts same-size 2D uint8 arrays and returns a same-size uint8 array.
A future `TorchFusionMethod` should:

1. Load an explicitly supplied, licensed checkpoint from a user path; record its SHA256 and model
   configuration. Keep weights and datasets outside Git. Never download weights implicitly.
2. Convert images to float tensors `[1,1,H,W]` in [0,1], pad to the architecture's required stride,
   retain original dimensions, select CPU/CUDA explicitly and call `eval()` under `inference_mode()`.
3. Define modality order, range, output semantics and color policy; validate finite values and shape,
   unpad and quantize only once. Record device, precision and PyTorch/CUDA versions.
4. Add golden small-pair tests, deterministic CPU tests and a separate optional GPU test job.
   Record synchronization and warm-up rules for timing.
5. Add a typed method registry and checkpoint configuration to the workflow/plan schema; the Agent
   selects only registered methods. Benchmark on licensed registered data with target-aware metrics.

This release deliberately has no PyTorch dependency, training loop, neural checkpoint or adapter.
