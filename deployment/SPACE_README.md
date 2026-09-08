---
title: FusionAgent
emoji: 🔬
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 6.26.0
python_version: '3.12'
app_file: app.py
pinned: false
license: mit
short_description: Infrared-visible fusion with public TNO examples
---

# FusionAgent · 红外—可见光融合

Choose a TNO sample pair and click **Run fusion**, or upload a non-sensitive registered pair.
This Space runs the same deterministic Python workflow as the local project, without an API key.
It compares weighted-average and Haar-wavelet baselines, calculates four metrics and exports reports.
Uploaded images are processed on this server. Download results promptly; files are transient.
No model API calls are available on this public instance.

Source: https://github.com/hjt991202/fusion-agent

TNO images: Alexander Toet, TNO, Soesterberg, The Netherlands, **CC BY 4.0**,
https://doi.org/10.6084/m9.figshare.1008029.v2 . See examples/tno/README.md for provenance,
license and changes. The MIT license applies to code, not to these third-party images.

Four statistics are descriptive and do not prove perceptual quality. No geometric registration,
deep model or dataset-wide benchmark is included.
