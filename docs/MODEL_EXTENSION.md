# Future PyTorch interface (design only)

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
