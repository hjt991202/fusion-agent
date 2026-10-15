"""Full letters model and refinement pipeline for RGB visible images.

This adapter preserves the supplied inference mathematics. Learned models do not
guarantee identity or constant preservation, including when both inputs agree.
Weights are local inputs and are never downloaded automatically.
"""

from __future__ import annotations

import hashlib
import io
from collections.abc import Mapping
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from threading import RLock
from typing import Any, Literal, TypedDict

import numpy as np
import torch
from numpy.typing import NDArray

from . import boundary_engine, color_transfer, masked_context
from .jdfusion.architecture import IDRNet

_INFERENCE_LOCK = RLock()


class FusionResult(TypedDict):
    gray: NDArray[np.uint8]
    color: NDArray[np.uint8]
    metadata: dict[str, Any]


def _image(value: NDArray[np.uint8], name: str, *, rgb_only: bool) -> NDArray[np.uint8]:
    if not isinstance(value, np.ndarray) or value.dtype != np.uint8:
        raise ValueError(f"{name} must be a uint8 NumPy array.")
    if value.ndim == 2 and not rgb_only:
        value = np.repeat(value[:, :, None], 3, axis=2)
    if value.ndim != 3 or value.shape[2] != 3:
        channels = "RGB" if rgb_only else "grayscale or RGB"
        raise ValueError(f"{name} must be {channels}.")
    if min(value.shape[:2]) < 2:
        raise ValueError(f"{name} height and width must both be at least 2.")
    return np.ascontiguousarray(value)


def _mask(value: NDArray[Any], shape: tuple[int, int], name: str) -> NDArray[np.bool_]:
    if not isinstance(value, np.ndarray) or value.shape != shape:
        raise ValueError(f"{name} must match the image height and width.")
    if value.dtype not in (np.dtype(bool), np.dtype(np.uint8)):
        raise ValueError(f"{name} must be a boolean or uint8 array; nonzero means included.")
    return value > 0


def _load_model(weights: Path, device: torch.device) -> tuple[IDRNet, dict[str, Any], str]:
    if not weights.is_file():
        raise FileNotFoundError(f"Missing letters checkpoint: {weights}")
    data = weights.read_bytes()
    model, config = _cached_model(data, str(device))
    return model, config, hashlib.sha256(data).hexdigest()


@lru_cache(maxsize=2)
def _cached_model(data: bytes, device: str) -> tuple[IDRNet, dict[str, Any]]:
    checkpoint = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, Mapping):
        raise ValueError("letters checkpoint must contain model_config and model_state.")
    config = checkpoint.get("model_config")
    state = checkpoint.get("model_state")
    if not isinstance(config, dict) or not isinstance(state, Mapping):
        raise ValueError("letters checkpoint must contain model_config and model_state mappings.")
    if set(config) - {"base_channels", "flow_channels", "ablation"}:
        raise ValueError("Unsupported letters model configuration keys.")
    for key in ("base_channels", "flow_channels"):
        if key in config and (type(config[key]) is not int or config[key] < 2):
            raise ValueError(f"letters {key} must be an integer of at least 2.")
    ablation = config.get("ablation")
    allowed = {
        "use_flow",
        "use_retinex",
        "use_saliency",
        "use_mask_guidance",
        "use_mask_attention",
        "use_enhancement",
    }
    if ablation is not None and (
        not isinstance(ablation, dict)
        or set(ablation) - allowed
        or any(type(value) is not bool for value in ablation.values())
    ):
        raise ValueError("Unsupported letters ablation configuration.")
    if any(
        not isinstance(key, str) or not isinstance(value, torch.Tensor)
        for key, value in state.items()
    ):
        raise ValueError("letters model_state must map parameter names to tensors.")
    if any(not torch.isfinite(value).all() for value in state.values()):
        raise ValueError("letters model_state contains non-finite tensors.")
    model = IDRNet(**config)
    model.load_state_dict(state, strict=True)
    model = model.to(device).eval()
    model.requires_grad_(False)
    return model, config


def fuse(
    visible: NDArray[np.uint8],
    infrared: NDArray[np.uint8],
    weights: str | Path,
    device: Literal["cpu", "cuda", "auto"] = "cpu",
    threads: int = 2,
    valid_mask: NDArray[Any] | None = None,
    material_mask: NDArray[Any] | None = None,
    material_mode: Literal["visible", "valid"] = "visible",
) -> FusionResult:
    """Fuse corresponding RGB visible and grayscale/RGB infrared uint8 arrays.

    Inputs must share a pixel grid; no resize is performed. The network includes
    its original learned optical flow stage, whose accuracy is not established
    by execution. Valid support and material masks affect source refinement;
    they are not fed to the neural model, matching the original inference code.
    Material is inferred from visible color unless supplied explicitly or mode
    is ``valid``. Gray is Pillow's integer luminance of the final RGB result.
    """
    vis = _image(visible, "visible", rgb_only=True)
    ir = _image(infrared, "infrared", rgb_only=False)
    if vis.shape != ir.shape:
        raise ValueError("Visible and infrared images must share the same pixel grid.")
    if material_mode not in ("visible", "valid"):
        raise ValueError("material_mode must be 'visible' or 'valid'.")
    if type(threads) is not int or not 1 <= threads <= 32:
        raise ValueError("threads must be an integer between 1 and 32.")
    if device not in ("cpu", "cuda", "auto"):
        raise ValueError("device must be 'cpu', 'cuda', or 'auto'.")
    shape = vis.shape[:2]
    valid = np.ones(shape, bool) if valid_mask is None else _mask(valid_mask, shape, "valid_mask")
    if not valid.any():
        raise ValueError("valid_mask must contain observed valid support.")
    if material_mask is not None:
        material = _mask(material_mask, shape, "material_mask") & valid
        material_source = "explicit_mask"
    elif material_mode == "valid":
        material = valid.copy()
        material_source = "valid_support"
    else:
        try:
            material = boundary_engine.trans.material(vis, valid)
        except ValueError as exc:
            raise ValueError(
                "Cannot derive colored material from visible input; supply material_mask "
                "or use material_mode='valid' for observed isolated material."
            ) from exc
        material_source = "visible_color"
    if not material.any():
        raise ValueError("Material support must contain pixels within valid support.")
    selected = "cuda" if device == "auto" and torch.cuda.is_available() else device
    if selected == "auto":
        selected = "cpu"
    if selected == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable.")
    weights = Path(weights)
    with _INFERENCE_LOCK:
        previous_threads = torch.get_num_threads()
        try:
            torch.set_num_threads(threads)
            model, model_config, weights_sha256 = _load_model(weights, torch.device(selected))
            raw, flow = boundary_engine.infer(model, vis, ir)
            reference = boundary_engine.stages(raw, vis, ir, valid, material)["final"]
            context = masked_context.prepare(raw, vis, ir, reference, valid, material)
            refined, _ = masked_context.render(context, boundary_engine.CFG["masked_context"])
            color = color_transfer.apply(vis, refined, material)
        finally:
            torch.set_num_threads(previous_threads)
    metadata = {
        "implementation": "letters_idrnet_with_boundary_refinement",
        "weights_name": weights.name,
        "weights_sha256": weights_sha256,
        "config_sha256": hashlib.sha256(
            Path(boundary_engine.__file__).with_name("config.json").read_bytes()
        ).hexdigest(),
        "model_config": deepcopy(model_config),
        "device": selected,
        "torch_threads": threads,
        "input_color_order": "RGB",
        "output_gray_conversion": "Pillow RGB to L",
        "material_source": material_source,
        "valid_pixels": int(valid.sum()),
        "material_pixels": int(material.sum()),
        "learned_flow_enabled": model.use_flow,
        "refinement": deepcopy(boundary_engine.CFG),
        **flow,
    }
    return {"gray": color_transfer.gray(color), "color": color, "metadata": metadata}


__all__ = ["FusionResult", "fuse"]
