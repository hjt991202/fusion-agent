# Jiandu measured demonstration / 简牍实测演示

Fifteen source-resolution pairs from the owner-supplied test folder were processed with all four methods on 2026-10-15. These are selected demonstrations, not a benchmark; training/test independence is unknown. No input resizing was used. Each figure shows full-image display thumbnails and identical source-coordinate center crops.

[Inputs and source hashes](../examples/jiandu/README.md) · [Method reproduction](RESEARCH_METHODS.md) · [Full measured evidence](assets/jiandu/manifest.json)

## Actual example: 843

![Visible, infrared, ICIG color and letters color; whole images and shared center crops](assets/jiandu/jiandu_0843.png)

| Method | Entropy | Std | Average gradient | Spatial frequency |
|---|---:|---:|---:|---:|
| weighted | 5.7139 | 58.0523 | 5.7193 | 12.8093 |
| wavelet | 5.9449 | 58.7535 | 9.4360 | 20.7770 |
| icig | 5.9798 | 51.7583 | 7.5581 | 17.2942 |
| letters | 5.3741 | 86.5089 | 3.5742 | 12.8090 |

For this pair, entropy is largest for ICIG, standard deviation for letters, and gradient/spatial frequency for Haar. These conflicting maxima are not perceptual rankings. In the actual color plate, ICIG emphasizes fine surface texture; letters produces darker strokes and a smoother background. The extent to which this preserves historical writing is not established by these metrics. Some source outlines differ between modalities; matching array dimensions is not proof of geometric alignment.

ICIG metrics refer to its raw fused grayscale, while its color restoration changes luminance. Letters metrics use Pillow grayscale of the final refined color. Whole-frame white/background areas affect statistics. The input IR is grayscale; ICIG uses the supplied VIS_Y; letters uses full valid support and derives material from visible color. These preprocessing/output differences limit direct interpretation.

## All fifteen pairs

| Source ID | Width x height | ICIG entropy | ICIG gradient | letters entropy | letters gradient | Preview |
|---|---:|---:|---:|---:|---:|---|
| 239 | 116 x 2696 | 6.2217 | 7.0948 | 5.6134 | 4.6578 | [View](assets/jiandu/jiandu_0239.png) |
| 406 | 355 x 1654 | 6.7113 | 9.9045 | 5.9760 | 3.1135 | [View](assets/jiandu/jiandu_0406.png) |
| 517 | 186 x 401 | 5.3812 | 6.7116 | 4.7442 | 4.3702 | [View](assets/jiandu/jiandu_0517.png) |
| 606 | 206 x 1367 | 5.7591 | 8.3255 | 4.9326 | 3.4900 | [View](assets/jiandu/jiandu_0606.png) |
| 705 | 163 x 376 | 6.3300 | 9.4393 | 5.5700 | 4.6158 | [View](assets/jiandu/jiandu_0705.png) |
| 843 | 591 x 648 | 5.9798 | 7.5581 | 5.3741 | 3.5742 | [View](assets/jiandu/jiandu_0843.png) |
| 844 | 462 x 661 | 7.0011 | 11.3929 | 6.6133 | 4.8445 | [View](assets/jiandu/jiandu_0844.png) |
| 964 | 246 x 2718 | 4.1292 | 4.3129 | 4.2701 | 3.9079 | [View](assets/jiandu/jiandu_0964.png) |
| 1116 | 196 x 2797 | 6.5955 | 8.1611 | 6.0326 | 5.3018 | [View](assets/jiandu/jiandu_1116.png) |
| 1171 | 439 x 1483 | 7.0199 | 10.9655 | 6.0749 | 4.6560 | [View](assets/jiandu/jiandu_1171.png) |
| 1232 | 223 x 596 | 3.9078 | 2.8793 | 3.0056 | 2.1396 | [View](assets/jiandu/jiandu_1232.png) |
| 1455 | 358 x 2783 | 6.7029 | 9.9123 | 6.5042 | 4.2739 | [View](assets/jiandu/jiandu_1455.png) |
| 1589 | 223 x 2569 | 6.0700 | 7.7210 | 5.9965 | 5.6183 | [View](assets/jiandu/jiandu_1589.png) |
| 1993 | 165 x 361 | 6.2656 | 10.5375 | 5.3697 | 5.7888 | [View](assets/jiandu/jiandu_1993.png) |
| 2116 | 226 x 710 | 6.9261 | 9.4542 | 6.2670 | 4.2008 | [View](assets/jiandu/jiandu_2116.png) |

## Execution evidence

All 60 method executions completed. The recorded total across the 15 sequential pair workflows was 720.49 seconds on CPU with two PyTorch threads. Other checks ran concurrently, model initialization/caching differs by method, and no controlled warm-up protocol was used; these times are operational observations, not a speed benchmark.

Versions: fusion-agent 0.1.0, numpy 2.2.6, opencv-python-headless 4.14.0.94, PyWavelets 1.10.0, pydantic 2.13.5, torch 2.7.1+cpu, scipy 1.13.1, einops 0.8.2, Pillow 12.3.0.

Both checkpoint hashes, source image hashes, output hashes, model settings, per-tool times and endpoint-pixel fractions are in the evidence manifest. Endpoint fractions measure how many final pixels equal 0/255; they cannot distinguish clipping from naturally black/white source content. Full output PNGs and per-run configs remain in local ignored outputs/jiandu-demo/. Only curated preview plates and metadata are published.

Archive-code parity is separate from the above sample demonstration: [ICIG, two pairs](research-validation/icig-parity.json) and [letters, one masked pair](research-validation/letters-parity.json) exactly match original-code reruns on the recorded CPU environment. The historical bundled letters PNG differs by at most one level in two pixels. GPU, training, character-recognition accuracy and live paid API behavior were not tested.

中文：15 对均以原尺寸完成四种方法运行。所有数值来自实际结果，展示图只为排版缩放并增加相同位置裁剪。指标存在冲突，不能据此宣布某算法全面更好；白背景、纹理和颜色恢复也会影响对比。公开的是选定演示子集，不是完整测试集，也没有确认其与训练数据相互独立。
