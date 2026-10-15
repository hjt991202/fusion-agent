# Jiandu measured demonstration / 简牍实测演示

Fifteen source-resolution pairs from the owner-supplied test folder have measured outputs for five methods on 2026-10-15. The earlier four-method outputs were retained after input, report, configuration, PNG-hash and metric verification; PRCV ran in a separate incremental batch. These selected demonstrations are not a benchmark, and training/test independence is unknown. No input resizing was used. Each figure shows whole-image display thumbnails and identical source-coordinate center crops.

[Inputs and source hashes](../examples/jiandu/README.md) · [Method reproduction](RESEARCH_METHODS.md) · [Full measured evidence](assets/jiandu/manifest.json)

## Actual example: 843

![Visible, infrared, ICIG, letters and PRCV color; whole images and shared center crops](assets/jiandu/jiandu_0843.png)

| Method | Entropy | Std | Average gradient | Spatial frequency |
|---|---:|---:|---:|---:|
| weighted | 5.7139 | 58.0523 | 5.7193 | 12.8093 |
| wavelet | 5.9449 | 58.7535 | 9.4360 | 20.7770 |
| icig | 5.9798 | 51.7583 | 7.5581 | 17.2942 |
| letters | 5.3741 | 86.5089 | 3.5742 | 12.8090 |
| prcv | 5.4795 | 75.3019 | 9.1946 | 20.6398 |

For this pair, entropy is largest for ICIG, standard deviation for letters, and gradient/spatial frequency for Haar. PRCV's gradient and spatial frequency are close to Haar, but these conflicting maxima are not perceptual rankings. In the actual color plate, ICIG emphasizes fine surface texture; letters produces darker strokes and a smoother background; PRCV retains dense surface texture and less stroke darkening than letters. Preservation or recognition of historical writing has not been measured. Some source outlines differ between modalities; matching array dimensions and successful model execution do not validate geometric alignment.

ICIG metrics refer to its raw fused grayscale, while its color restoration changes luminance. Letters and PRCV metrics use Pillow grayscale of final RGB. Whole-frame white/background areas affect statistics. Input IR is grayscale; ICIG uses supplied VIS_Y, letters uses full valid support with material derived from visible color, and PRCV uses the source configuration without external masks. These preprocessing/output differences limit interpretation.

PRCV retains the source VFC adjustment, full manifold attention and final RGB composition. Its VFC limit is +/-3 pixels per axis, not a 3-pixel Euclidean-magnitude limit. Its source forward path uses identity affine and zero learned raw/stroke-refinement flow. Registration accuracy is not independently established.

## All fifteen pairs

| Source ID | Width x height | ICIG entropy | ICIG gradient | letters entropy | letters gradient | PRCV entropy | PRCV gradient | Preview |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| 239 | 116 x 2696 | 6.2217 | 7.0948 | 5.6134 | 4.6578 | 5.6143 | 6.8033 | [View](assets/jiandu/jiandu_0239.png) |
| 406 | 355 x 1654 | 6.7113 | 9.9045 | 5.9760 | 3.1135 | 6.4130 | 12.0525 | [View](assets/jiandu/jiandu_0406.png) |
| 517 | 186 x 401 | 5.3812 | 6.7116 | 4.7442 | 4.3702 | 4.9558 | 7.6202 | [View](assets/jiandu/jiandu_0517.png) |
| 606 | 206 x 1367 | 5.7591 | 8.3255 | 4.9326 | 3.4900 | 5.2867 | 9.7562 | [View](assets/jiandu/jiandu_0606.png) |
| 705 | 163 x 376 | 6.3300 | 9.4393 | 5.5700 | 4.6158 | 5.8939 | 10.3346 | [View](assets/jiandu/jiandu_0705.png) |
| 843 | 591 x 648 | 5.9798 | 7.5581 | 5.3741 | 3.5742 | 5.4795 | 9.1946 | [View](assets/jiandu/jiandu_0843.png) |
| 844 | 462 x 661 | 7.0011 | 11.3929 | 6.6133 | 4.8445 | 7.0259 | 13.2109 | [View](assets/jiandu/jiandu_0844.png) |
| 964 | 246 x 2718 | 4.1292 | 4.3129 | 4.2701 | 3.9079 | 3.9093 | 5.8406 | [View](assets/jiandu/jiandu_0964.png) |
| 1116 | 196 x 2797 | 6.5955 | 8.1611 | 6.0326 | 5.3018 | 6.0615 | 8.8660 | [View](assets/jiandu/jiandu_1116.png) |
| 1171 | 439 x 1483 | 7.0199 | 10.9655 | 6.0749 | 4.6560 | 6.5915 | 11.9631 | [View](assets/jiandu/jiandu_1171.png) |
| 1232 | 223 x 596 | 3.9078 | 2.8793 | 3.0056 | 2.1396 | 3.0157 | 3.1653 | [View](assets/jiandu/jiandu_1232.png) |
| 1455 | 358 x 2783 | 6.7029 | 9.9123 | 6.5042 | 4.2739 | 6.6618 | 13.3796 | [View](assets/jiandu/jiandu_1455.png) |
| 1589 | 223 x 2569 | 6.0700 | 7.7210 | 5.9965 | 5.6183 | 6.0903 | 10.2891 | [View](assets/jiandu/jiandu_1589.png) |
| 1993 | 165 x 361 | 6.2656 | 10.5375 | 5.3697 | 5.7888 | 5.6059 | 10.2219 | [View](assets/jiandu/jiandu_1993.png) |
| 2116 | 226 x 710 | 6.9261 | 9.4542 | 6.2670 | 4.2008 | 6.4514 | 10.1957 | [View](assets/jiandu/jiandu_2116.png) |


## Execution evidence

All 75 method/output metric records were independently recomputed from saved gray PNGs. All 120 gray/color output hashes were checked. The prior four-method workflow times totalled 720.49 seconds; the separate final PRCV batch totalled 401.24 seconds. Both used CPU with two PyTorch threads. Other validation ran concurrently, initialization/caching differs by method, and there was no controlled warm-up protocol. The public manifest stores per-run durations, versions and configurations; it does not claim one combined wall time or a speed benchmark.

Recorded versions: fusion-agent 0.1.0, numpy 2.2.6, opencv-python-headless 4.14.0.94, PyWavelets 1.10.0, pydantic 2.13.5, torch 2.7.1+cpu, scipy 1.13.1, einops 0.8.2, Pillow 12.3.0.

All three checkpoint hashes, source image hashes, output hashes, model settings, per-tool times and endpoint-pixel fractions are in the evidence manifest. Endpoint fractions count final pixels equal to 0/255; they cannot distinguish clipping from naturally black/white content. Full PNG outputs and per-run configs remain in ignored local outputs/jiandu-demo/ and outputs/jiandu-prcv-demo-v2/. Only curated plates and metadata are published; no additional archive example images were added.

Archive-code parity is separate from the demonstration: [ICIG, two pairs](research-validation/icig-parity.json), [letters, one masked pair](research-validation/letters-parity.json), and [PRCV, two pairs](research-validation/prcv-parity.json) exactly match fresh original-code reruns in the recorded CPU environment. The historical bundled letters PNG differs by at most one level in two pixels. GPU, training, character-recognition accuracy and live paid API behavior were not tested.

中文：沿用同一批 15 对原尺寸样例，新增 PRCV 实测结果，并保留先前四种方法的真实输出。五种方法共 75 组指标均从结果图重新核对；新旧批次的时间和配置分别记录。展示图仅为排版缩放并增加同位置裁剪。指标存在冲突，不能据此宣布某算法全面更好。公开的是选定演示子集，未确认与训练数据相互独立。
