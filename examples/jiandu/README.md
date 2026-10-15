# Jiandu demonstration inputs / 简牍演示图像

This directory contains **15 selected pairs** from the repository owner's local
`test/` inputs, plus the corresponding supplied luminance images: 15 RGB visible,
15 infrared and 15 visible-Y PNG files (14,009,494 image bytes, approximately
13.36 MiB). Every image is copied **byte for byte**; only filenames are normalized.
No resizing, geometric registration or color conversion was applied to these files.

本目录包含从项目所有者提供的 `test/` 中选取的 **15 对可见光/红外图像**，以及原有的
15 张可见光 Y 通道图像，共 45 个 PNG 文件。图像文件内容逐字节保持不变，只统一了
文件名；没有缩放、配准或颜色变换。`visible_y/` 使用原始提供的图像，未从 RGB 重新计算。

## Provenance and permission / 来源与使用范围

The user describes the inputs as their Jiandu fusion research data and explicitly
requested that about a dozen pairs be uploaded to their GitHub project for a
demonstration. That request authorizes this project demonstration; it does **not**
establish a general redistribution license for third parties. No separate dataset
license, original acquisition documentation or independently verified ownership
was supplied. These images are **not covered by the repository's MIT code license**.
Ask the repository owner about use beyond this demonstration.

用户将这些图像描述为其简牍融合研究数据，并明确要求在自己的 GitHub 项目中上传十几对
用于演示。这里记录的是本项目演示的授权，不代表向第三方授予通用再分发许可。
尚未提供单独的数据集许可证、原始采集说明，也未独立核实所有权。这些图像不适用
仓库代码的 MIT 许可证；超出演示范围的使用请向项目所有者确认。

The full local source has 327 complete RGB/IR/Y triplets. The initial read-only audit
found no missing counterparts, duplicate numeric IDs, decoding errors, unexpected
channel counts, dimension mismatches, constant images or duplicated pixel hashes
within a modality. Matching image dimensions do not establish geometric registration.
The full source dataset remains outside the published subset.

全量本地数据有 327 组 RGB/IR/Y 图像。首次只读审计没有发现缺失配对、重复编号、解码
错误、通道异常、配对尺寸不同、常量图像或单一模态内像素完全重复的文件。尺寸一致不能
证明几何配准正确。完整源数据不属于本次发布子集。

## Selection and integrity / 选样与完整性

`manifest.json` records each original relative path, encoded-file SHA-256, decoded
pixel SHA-256, dimensions, mode and input statistics. Pixel hashes use contiguous
row-major uint8 bytes, with RGB channel order for color images. `source_sha256` and
`sha256` must match because encoded files were copied unchanged.

The selection is reproducible: min-max normalize log pixel area, log height/width,
visible-Y mean and infrared mean; start nearest the feature median and repeatedly
choose the candidate farthest from its nearest selected sample. Ties use numeric
source ID. This spreads the examples across size, shape and input brightness;
no fusion outputs or quality scores are used for selection.

选样基于图像面积、长宽比和输入亮度的确定性覆盖，不按融合效果或质量评分挑选。
清单记录原始相对路径、文件和像素 SHA-256、尺寸、颜色模式及输入统计信息，可核对
副本与原图一致。该小规模选择性子集仅用于演示，不是基准测试集，训练/测试独立性
尚未确认，也不能据此证明算法优越性或泛化能力。

Recreate from the owner's local dataset (which is not included in Git):

```powershell
python scripts/prepare_jiandu_examples.py
python .agents/skills/dataset-audit/scripts/audit.py --root examples/jiandu --output outputs/jiandu-audit/subset-audit.json
```

The preparation script reads the source without changing it and writes its complete
audit and a review contact sheet under ignored `outputs/jiandu-audit/`. Existing
different image bytes are never overwritten. Reproduction requires the same 327
local source triplets; the manifest supports integrity checks without that dataset.

Selected original IDs: `239`, `406`, `517`, `606`, `705`, `843`, `844`, `964`,
`1116`, `1171`, `1232`, `1455`, `1589`, `1993`, `2116`.
