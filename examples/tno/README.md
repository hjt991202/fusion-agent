# TNO demonstration subset

Three visible/thermal pairs from **TNO Image Fusion Dataset**, Alexander Toet (2022),
version 2, TNO, Soesterberg, The Netherlands.

- Source and citation: https://doi.org/10.6084/m9.figshare.1008029.v2
- Official archive: https://ndownloader.figshare.com/files/1475454
- License: **Creative Commons Attribution 4.0 International (CC BY 4.0)**
  https://creativecommons.org/licenses/by/4.0/
- These images retain their CC BY 4.0 license; the repository's MIT code license does not replace it.
- No endorsement by the author or TNO is implied.

Pairs: `lake`, `helicopter`, `house`, from the archive's Athena_images directory.
Files are re-encoded from BMP to PNG without pixel changes and renamed descriptively.
The exact original member paths, original/output SHA256 hashes, dimensions and channels are in
manifest.json. Import checks every output's decoded pixels against the original.

Reproduce after downloading the official archive:

```bash
python scripts/import_tno_examples.py outputs/downloads/tno.zip
python .agents/skills/dataset-audit/scripts/audit.py --root examples/tno --output outputs/tno-audit.json
```

These three selected examples support demonstrations, not dataset-wide benchmark claims.
Equal dimensions do not prove perfect registration; inspect local alignment and sensor artifacts.
Do not mistake intensified visible imagery for normal daylight color photographs.

## 中文说明

本目录包含 TNO 官方公开数据集的三对可见光/热红外演示图片，作者 Alexander Toet，
权利人 TNO（荷兰 Soesterberg），许可为 CC BY 4.0。使用或再分发时保留上述署名、
来源、许可和修改说明；本项目代码的 MIT 许可不覆盖这些图片。
仅进行 BMP 到 PNG 的无损编码转换和文件重命名，没有缩放、裁剪或强度调整。
