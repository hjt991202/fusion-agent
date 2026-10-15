"""Optional PRCV IDRNet inference from the supplied PRCV_Fusion_submit archive.

This preserves the complete source forward pass, including VFC infrared warping,
full multi-manifold attention, confidence/refinement and RGB composition. The
source runner forces full attention; this package fixes that mode internally
without modifying process environment variables. Internal registration accuracy
is not established merely by successful execution.
"""

from __future__ import annotations

import hashlib
import io
import json
import time
from collections.abc import Mapping
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from threading import RLock
from typing import Any, Literal, TypedDict

import numpy as np
import torch
from numpy.typing import NDArray
from PIL import Image

from .memory import enable_window_chunking
from .model.architecture import IDRNet

_LOCK = RLock()
_CONFIG_BYTES = Path(__file__).with_name("config.json").read_bytes()
_CONFIG = json.loads(_CONFIG_BYTES)["model"]


class FusionResult(TypedDict):
    gray: NDArray[np.uint8]
    color: NDArray[np.uint8]
    metadata: dict[str, Any]


def _validate_inputs(visible: NDArray[np.uint8], infrared: NDArray[np.uint8]) -> NDArray[np.uint8]:
    for image, name in [(visible, "visible"), (infrared, "infrared")]:
        if not isinstance(image, np.ndarray) or image.dtype != np.uint8:
            raise ValueError(f"PRCV {name} must be a uint8 NumPy array.")
    if visible.ndim != 3 or visible.shape[2] != 3:
        raise ValueError("PRCV visible must be an RGB array.")
    if infrared.ndim != 2 and not (infrared.ndim == 3 and infrared.shape[2] == 3):
        raise ValueError("PRCV infrared must be a grayscale or RGB array.")
    if min(visible.shape[:2]) < 33:
        raise ValueError("PRCV image height and width must both be at least 33 pixels.")
    if visible.shape[:2] != infrared.shape[:2]:
        raise ValueError("PRCV visible and infrared dimensions must match.")
    return np.asarray(Image.fromarray(infrared).convert("L"))


def _select_device(device: str) -> torch.device:
    if device not in {"cpu", "cuda", "auto"}:
        raise ValueError("PRCV device must be cpu, cuda or auto.")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable.")
    return torch.device(device)


@lru_cache(maxsize=2)
def _cached_model(payload: bytes, device: str, window_batch: int) -> IDRNet:
    state = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)
    if not isinstance(state, Mapping) or any(
        not isinstance(name, str) or not isinstance(value, torch.Tensor)
        for name, value in state.items()
    ):
        raise ValueError("PRCV checkpoint must map parameter names to tensors.")
    if any(not torch.isfinite(value).all() for value in state.values()):
        raise ValueError("PRCV checkpoint contains non-finite parameters.")
    # The architecture and checkpoint parameter names are unchanged.
    model = IDRNet(**_CONFIG)
    model.load_state_dict(state, strict=True)
    enable_window_chunking(model, chunk_size=window_batch)
    # Preserve source parameter flags: freezing can change CPU kernel rounding
    # by one quantized intensity even though inference_mode disables gradients.
    return model.to(device=device, dtype=torch.float32).eval()


def load_model(weights: str | Path, device: str = "cpu", window_batch: int = 4) -> IDRNet:
    """Load the reviewed architecture and an explicit local tensor-only checkpoint."""
    if type(window_batch) is not int or window_batch < 1:
        raise ValueError("window_batch must be a positive integer.")
    selected = _select_device(device)
    path = Path(weights)
    if not path.is_file():
        raise FileNotFoundError(f"Missing PRCV checkpoint: {path}")
    with _LOCK:
        return _cached_model(path.read_bytes(), str(selected), window_batch)


@torch.inference_mode()
def _predict(
    model: torch.nn.Module,
    visible: NDArray[np.uint8],
    infrared_luminance: NDArray[np.uint8],
    device: torch.device,
) -> tuple[NDArray[np.uint8], dict[str, float]]:
    rgb_array = visible.astype(np.float32) / np.float32(255.0)
    ir_array = infrared_luminance.astype(np.float32) / np.float32(255.0)
    rgb = torch.from_numpy(rgb_array.transpose(2, 0, 1).copy()).unsqueeze(0).to(device)
    ir = torch.from_numpy(ir_array.copy()).unsqueeze(0).unsqueeze(0).to(device)
    outputs = model(rgb, ir)
    fused = outputs["fused"]
    if tuple(fused.shape) != (1, 3, *visible.shape[:2]):
        raise RuntimeError("PRCV model returned unexpected output dimensions.")
    for name in ("fused", "flow", "vfc_flow", "affine_theta"):
        if not torch.isfinite(outputs[name]).all():
            raise RuntimeError(f"PRCV model returned non-finite {name} values.")
    result = fused[0].cpu().numpy().transpose(1, 2, 0)
    # Preserve source truncation; do not round to the nearest intensity.
    color = (np.clip(result, 0.0, 1.0) * np.float32(255.0)).astype(np.uint8)
    flow = outputs["flow"][0].cpu().numpy()
    magnitude = np.linalg.norm(flow, axis=0)
    diagnostics = {
        "flow_mean_pixels": float(magnitude.mean()),
        "flow_p95_pixels": float(np.percentile(magnitude, 95)),
        "flow_max_abs_component_pixels": float(np.abs(flow).max()),
    }
    return color, diagnostics


def fuse(
    visible: NDArray[np.uint8],
    infrared: NDArray[np.uint8],
    weights: str | Path,
    device: Literal["cpu", "cuda", "auto"] = "cpu",
    threads: int = 2,
    window_batch: int = 4,
) -> FusionResult:
    """Run full-resolution PRCV fusion without resizing or external masks.

    Image dimensions must both be at least 33, matching the source CLI. Gray is
    Pillow's luminance of the final RGB output, added for project metrics. The
    source's disabled TF32 precision and thread count apply during inference;
    previous process settings are restored even on failure. Weights are cached
    by their actual bytes, device and attention window batch size.
    """
    luminance = _validate_inputs(visible, infrared)
    if type(threads) is not int or not 1 <= threads <= 32:
        raise ValueError("threads must be an integer between 1 and 32.")
    if type(window_batch) is not int or window_batch < 1:
        raise ValueError("window_batch must be a positive integer.")
    selected = _select_device(device)
    path = Path(weights)
    if not path.is_file():
        raise FileNotFoundError(f"Missing PRCV checkpoint: {path}")
    payload = path.read_bytes()
    started = time.perf_counter()
    with _LOCK:
        previous_threads = torch.get_num_threads()
        previous_cudnn_tf32 = torch.backends.cudnn.allow_tf32
        previous_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
        try:
            torch.set_num_threads(threads)
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cuda.matmul.allow_tf32 = False
            model = _cached_model(payload, str(selected), window_batch)
            color, diagnostics = _predict(model, visible, luminance, selected)
        finally:
            torch.set_num_threads(previous_threads)
            torch.backends.cudnn.allow_tf32 = previous_cudnn_tf32
            torch.backends.cuda.matmul.allow_tf32 = previous_matmul_tf32
    return {
        "gray": np.asarray(Image.fromarray(color).convert("L")),
        "color": color,
        "metadata": {
            "implementation": "PRCV IDRNet with VFC and full multi-manifold attention",
            "weights_name": path.name,
            "weights_sha256": hashlib.sha256(payload).hexdigest(),
            "config_sha256": hashlib.sha256(_CONFIG_BYTES).hexdigest(),
            "model_config": deepcopy(_CONFIG),
            "device": str(selected),
            "torch_threads": threads,
            "window_batch": window_batch,
            "attention_mode": "full",
            "input_color_order": "RGB",
            "infrared_conversion": "Pillow RGB to L; grayscale unchanged",
            "input_normalization": "float32 divided by 255",
            "output_quantization": "clip to [0, 1], multiply by 255, truncate to uint8",
            "output_gray_conversion": "Pillow RGB to L of final color",
            "registration": "Source VFC infrared displacement; applied affine is identity; "
            "learned raw flow and stroke refinement flow are zero in this implementation",
            "external_mask": False,
            "tf32": False,
            "seconds": time.perf_counter() - started,
            **diagnostics,
        },
    }


__all__ = ["FusionResult", "fuse", "load_model"]
