"""Validated inference adapter preserving the archived ICIG pipeline.

The network consumes unnormalized luminance in [0, 255]. Its fixed 256-pixel
tiles, zero padding, weighted overlap and detail-based RGB restoration follow
``ICIGCode/infer.py``. RGB is a visualization using visible chroma; its
luminance is not identical to the learned grayscale prediction.
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image
from torch import nn

from .network import GrFormer

BLOCK_SIZE = 256
OVERLAP = 64


def _validate_gray(array: np.ndarray, name: str) -> None:
    if (
        not isinstance(array, np.ndarray)
        or array.dtype != np.uint8
        or array.ndim != 2
        or min(array.shape, default=0) < 1
    ):
        raise ValueError(f"{name} must be a nonempty 2-D uint8 array.")


def _prepare_inputs(
    visible: np.ndarray,
    infrared: np.ndarray,
    visible_luminance: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray | None]:
    _validate_gray(infrared, "Infrared")
    if not isinstance(visible, np.ndarray) or visible.dtype != np.uint8:
        raise ValueError("Visible must be a nonempty grayscale or RGB uint8 array.")
    color = None
    if visible.ndim == 3 and visible.shape[2] == 3:
        if min(visible.shape[:2]) < 1:
            raise ValueError("Visible must be nonempty.")
        color = visible
        luminance = np.asarray(Image.fromarray(visible).convert("L"))
    else:
        _validate_gray(visible, "Visible")
        luminance = visible
    if luminance.shape != infrared.shape:
        raise ValueError("Visible and infrared dimensions must match; align inputs first.")
    if visible_luminance is not None:
        _validate_gray(visible_luminance, "Visible luminance")
        if visible_luminance.shape != infrared.shape:
            raise ValueError("Prepared visible luminance dimensions must match the inputs.")
        luminance = visible_luminance
    return luminance, color


def tile_starts(length: int) -> list[int]:
    """Cover an axis using the archive's fixed tiles and final edge anchor."""
    if length < 1:
        raise ValueError("Tile axis length must be positive.")
    if length <= BLOCK_SIZE:
        return [0]
    starts = list(range(0, length - BLOCK_SIZE + 1, BLOCK_SIZE - OVERLAP))
    last = length - BLOCK_SIZE
    if starts[-1] != last:
        starts.append(last)
    return starts


def tile_weight(
    y: int, x: int, actual_h: int, actual_w: int, height: int, width: int
) -> np.ndarray:
    """Preserve the archived 32-pixel overlap fade."""
    vertical = np.ones(BLOCK_SIZE, dtype=np.float32)
    horizontal = np.ones(BLOCK_SIZE, dtype=np.float32)
    fade = OVERLAP // 2
    ramp = np.arange(fade, dtype=np.float32) / fade
    if y > 0:
        vertical[:fade] = ramp
    if y + actual_h < height:
        vertical[-fade:] = ramp[::-1]
    if x > 0:
        horizontal[:fade] = ramp
    if x + actual_w < width:
        horizontal[-fade:] = ramp[::-1]
    return vertical[:actual_h, None] * horizontal[None, :actual_w]


@torch.inference_mode()
def fuse_image(
    model: nn.Module, infrared: np.ndarray, visible_luminance: np.ndarray, device: torch.device
) -> tuple[np.ndarray, int]:
    """Fuse equal-size luminance arrays without resizing or registration."""
    _validate_gray(infrared, "Infrared")
    _validate_gray(visible_luminance, "Visible luminance")
    if infrared.shape != visible_luminance.shape:
        raise ValueError("Infrared and visible luminance dimensions must match.")
    height, width = infrared.shape
    result = np.zeros((height, width), dtype=np.float32)
    weights = np.zeros((height, width), dtype=np.float32)
    count = 0
    for y in tile_starts(height):
        for x in tile_starts(width):
            actual_h = min(BLOCK_SIZE, height - y)
            actual_w = min(BLOCK_SIZE, width - x)
            inputs = []
            for array in (infrared, visible_luminance):
                block = np.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=np.float32)
                block[:actual_h, :actual_w] = array[y : y + actual_h, x : x + actual_w]
                inputs.append(torch.from_numpy(block)[None, None].to(device))
            prediction = model(*inputs)[0][0, 0].cpu().numpy()
            if prediction.shape != (BLOCK_SIZE, BLOCK_SIZE):
                raise RuntimeError("ICIG model returned an unexpected tile shape.")
            if not np.isfinite(prediction).all():
                raise RuntimeError(f"Model produced non-finite values at tile ({y}, {x}).")
            weight = tile_weight(y, x, actual_h, actual_w, height, width)
            result[y : y + actual_h, x : x + actual_w] += prediction[:actual_h, :actual_w] * weight
            weights[y : y + actual_h, x : x + actual_w] += weight
            count += 1
    if np.any(weights <= 0):
        raise RuntimeError("Uncovered pixels in tiled output.")
    fused = (result / (weights + 1e-8)).clip(0, 255).astype(np.uint8)
    return fused, count


def color_restore_detail(
    fused_y: np.ndarray, visible_y: np.ndarray, visible_rgb: np.ndarray
) -> np.ndarray:
    """Apply the source's visible-color detail visualization, unchanged."""
    rgb = visible_rgb.astype(np.float32)
    cb = -0.169 * rgb[:, :, 0] - 0.331 * rgb[:, :, 1] + 0.5 * rgb[:, :, 2] + 128
    cr = 0.5 * rgb[:, :, 0] - 0.419 * rgb[:, :, 1] - 0.081 * rgb[:, :, 2] + 128
    fused = fused_y.astype(np.float32)
    detail = fused - cv2.GaussianBlur(fused, (31, 31), 0)
    visible_blur = cv2.GaussianBlur(visible_y.astype(np.float32), (31, 31), 0)
    y = np.clip(visible_blur + 1.2 * detail, 0, 255).astype(np.uint8).astype(np.float32)
    r = y + 1.402 * (cr - 128)
    g = y - 0.344 * (cb - 128) - 0.714 * (cr - 128)
    b = y + 1.772 * (cb - 128)
    return np.stack([r, g, b], axis=2).clip(0, 255).astype(np.uint8)


def _device(device: str | torch.device) -> torch.device:
    if str(device) not in {"cpu", "cuda", "auto"}:
        raise ValueError("ICIG device must be cpu, cuda or auto.")
    if str(device) == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if str(device) == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable; use cpu or install a compatible PyTorch build.")
    return torch.device(device)


def load_model(weights: Path, device: str | torch.device = "cpu") -> nn.Module:
    """Load an explicit local state dictionary with restricted unpickling."""
    weights = Path(weights)
    if not weights.is_file():
        raise ValueError(f"ICIG weights do not exist: {weights}")
    resolved_device = _device(device)
    model = GrFormer([16, 64, 32, 16], 1, 1, 1)
    state_dict = torch.load(weights, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict, strict=True)
    return model.to(resolved_device).eval()


def fuse_with_model(
    visible: np.ndarray,
    infrared: np.ndarray,
    model: nn.Module,
    device: str | torch.device = "cpu",
    visible_luminance: np.ndarray | None = None,
) -> dict[str, Any]:
    """Reuse a loaded model across pairs; return arrays and execution facts."""
    luminance, color = _prepare_inputs(visible, infrared, visible_luminance)
    resolved_device = _device(device)
    started = time.perf_counter()
    gray, tiles = fuse_image(model, infrared, luminance, resolved_device)
    result: dict[str, Any] = {
        "gray": gray,
        "metadata": {
            "method": "icig",
            "model": "SparseGrFormer v3",
            "sparse_budget": 0.7,
            "device": str(resolved_device),
            "tile_size": BLOCK_SIZE,
            "overlap": OVERLAP,
            "tiles": tiles,
            "input_range": [0, 255],
            "visible_luminance": (
                "provided"
                if visible_luminance is not None
                else "PIL L from RGB"
                if color is not None
                else "grayscale input"
            ),
            "color_restoration": (
                "visible chroma and visible blur plus 1.2 times fused detail"
                if color is not None
                else None
            ),
        },
    }
    if color is not None:
        result["color"] = color_restore_detail(gray, luminance, color)
    result["metadata"]["seconds"] = time.perf_counter() - started
    return result


def fuse(
    visible: np.ndarray,
    infrared: np.ndarray,
    weights: Path,
    device: str = "cpu",
    threads: int = 2,
    visible_luminance: np.ndarray | None = None,
) -> dict[str, Any]:
    """Validate inputs, load explicit weights and run the supplied ICIG method.

    Thread limits affect the process-wide PyTorch and OpenCV pools, matching
    the original command-line runner. Use ``load_model``/``fuse_with_model``
    when processing multiple pairs with one model instance.
    """
    _prepare_inputs(visible, infrared, visible_luminance)
    if isinstance(threads, bool) or not isinstance(threads, int) or threads < 1:
        raise ValueError("threads must be a positive integer.")
    torch.set_num_threads(threads)
    cv2.setNumThreads(threads)
    model = load_model(weights, device)
    result = fuse_with_model(visible, infrared, model, device, visible_luminance)
    digest = hashlib.sha256()
    with Path(weights).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    result["metadata"].update({"weights_sha256": digest.hexdigest(), "threads": threads})
    return result
