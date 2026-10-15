"""Adapted from the user-supplied letters implementation."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray
from PIL import Image

from .contour_core import lock_luminance


def gray(rgb: NDArray[Any]) -> NDArray[Any]:
    return np.asarray(Image.fromarray(rgb).convert("L"))


def apply(vis: NDArray[Any], fused: NDArray[Any], material: NDArray[Any]) -> NDArray[Any]:
    if vis.shape != fused.shape or material.shape != vis.shape[:2]:
        raise ValueError("Shape mismatch")
    proposal = fused.astype(np.float32) + material[:, :, None].astype(np.float32) * (
        vis.astype(np.float32) - fused.astype(np.float32)
    )
    out = lock_luminance(proposal, fused)
    bad = cv2.cvtColor(out, cv2.COLOR_RGB2GRAY) != cv2.cvtColor(fused, cv2.COLOR_RGB2GRAY)
    out[bad] = fused[bad]
    out[~material] = fused[~material]
    assert np.array_equal(gray(out), gray(fused))
    assert np.array_equal(
        cv2.cvtColor(out, cv2.COLOR_RGB2GRAY), cv2.cvtColor(fused, cv2.COLOR_RGB2GRAY)
    )
    return out
