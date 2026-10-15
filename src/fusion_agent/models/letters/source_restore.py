"""Adapted from the user-supplied letters implementation."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray


def prepare(
    vis: NDArray[Any], ir: NDArray[Any], fused: NDArray[Any], valid: NDArray[Any]
) -> dict[str, Any]:
    h, w = vis.shape[:2]
    gray = cv2.cvtColor(ir, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 5, 4, 2)
    k = max(25, min(101, int(w * 0.25))) | 1
    bg = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8)).astype(np.float32)
    bg = cv2.GaussianBlur(bg, (0, 0), 1)
    contrast = np.clip((bg - gray) / np.maximum(bg, 10), 0, 1)
    t = np.clip((contrast - 0.03) / 0.15, 0, 1)
    ink = t * t * (3 - 2 * t)
    rad = max(1, round(3 * w / 256))
    ink = cv2.dilate(ink, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * rad + 1, 2 * rad + 1)))
    ink = cv2.GaussianBlur(ink, (0, 0), max(0.6, w / 256))
    distance = np.minimum(np.arange(h), np.arange(h)[::-1])[:, None]
    return {
        "vis": vis,
        "fused": fused,
        "valid": valid > 0,
        "ink": ink,
        "distance": distance,
        "scale": w / 256,
    }


def render(z: dict[str, Any], p: dict[str, Any]) -> NDArray[Any]:
    if p is None:
        return z["fused"].copy()
    width = max(8, p["width_at256"] * z["scale"])
    t = np.clip((z["distance"] - width) / width, 0, 1)
    edge = 1 - t * t * (3 - 2 * t)
    alpha = p["strength"] * edge * (1 - z["ink"]) * z["valid"]
    out = z["fused"].astype(np.float32) + alpha[:, :, None] * (
        z["vis"].astype(np.float32) - z["fused"].astype(np.float32)
    )
    return np.uint8(np.clip(out + 0.5, 0, 255))
