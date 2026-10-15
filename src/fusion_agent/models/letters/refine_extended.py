"""Adapted from the user-supplied letters implementation."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from .refinement import features, smooth
from .refinement import process as original_process


def render(z: dict[str, Any], p: dict[str, Any]) -> NDArray[Any]:
    fl = z["fl"]
    ic = z["ic"]
    ratio = np.clip(z["il"] / np.maximum(z["ib"], 10), 0.05, 1.1)
    local_bg = np.minimum(z["bgL"] + p["lift"], 95.0)
    target = (
        local_bg - p["absolute"] * (z["ib"] - z["il"])
        if "absolute" in p
        else local_bg * ratio ** p["gamma"]
    )
    alpha = z["near"] * z["obj"] * (z["wb"] > 0.01)
    change = alpha * np.clip(target - fl[:, :, 0], -40, 18)
    out = fl.copy()
    out[:, :, 0] = np.clip(fl[:, :, 0] + change, 0, 100)
    out[:, :, 1:] *= (
        np.sqrt(np.clip(out[:, :, 0] / np.maximum(fl[:, :, 0], 1), 0.05, 1.5))
        * (1 - p["neutral"] * alpha * smooth(ic, 0.04, 0.18))
    )[:, :, None]
    changed = (np.abs(change) > 0.15) & z["obj"]
    bgr = np.clip(cv2.cvtColor(out, cv2.COLOR_LAB2BGR) * 255 + 0.5, 0, 255).astype(np.uint8)
    bgr[~changed] = z["fused"][~changed]
    return bgr


def process(
    fused: NDArray[Any], ir: NDArray[Any], vis: NDArray[Any], valid: NDArray[Any], p: dict[str, Any]
) -> NDArray[Any]:
    if p.get("family") == "lift":
        return render(features(fused, ir, vis, valid), p)
    return original_process(fused, ir, vis, valid, p)
