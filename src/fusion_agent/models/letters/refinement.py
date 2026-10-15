"""Adapted from the user-supplied letters implementation."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray


def smooth(x: NDArray[Any], a: float, b: float) -> NDArray[Any]:
    t = np.clip((x - a) / (b - a), 0, 1)
    return t * t * (3 - 2 * t)


def blur(x: NDArray[Any], s: float) -> NDArray[Any]:
    return cv2.GaussianBlur(x, (0, 0), s)


def features(
    fused: NDArray[Any],
    ir: NDArray[Any],
    vis: NDArray[Any],
    valid: NDArray[Any],
    denoise: bool = False,
) -> dict[str, Any]:
    h, w = fused.shape[:2]
    fl = cv2.cvtColor(fused.astype(np.float32) / 255, cv2.COLOR_BGR2LAB)
    il = cv2.cvtColor(ir.astype(np.float32) / 255, cv2.COLOR_BGR2LAB)[:, :, 0]
    il = cv2.bilateralFilter(
        il, 7 if denoise else 5, 7.0 if denoise else 4.0, 3.0 if denoise else 2.0
    )
    obj = cv2.morphologyEx(
        (cv2.cvtColor(vis, cv2.COLOR_BGR2HSV)[:, :, 1] > 30).astype(np.uint8),
        cv2.MORPH_CLOSE,
        np.ones((9, 9), np.uint8),
    )
    n, labels, stats, _ = cv2.connectedComponentsWithStats(obj)
    if n > 1:
        obj = (labels == 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])).astype(np.uint8)
    obj = (cv2.erode(obj, np.ones((3, 3), np.uint8)) > 0) & (valid > 0)
    k = max(25, min(101, int(w * 0.25))) | 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    ib = blur(cv2.morphologyEx(il, cv2.MORPH_CLOSE, kernel), 1)
    ic = np.clip((ib - il) / np.maximum(ib, 10), 0, 0.95)
    ink = (ic > 0.1) * obj
    back = (ic < 0.025) * obj
    sigma = max(10, min(40, w * 0.16))
    wi = blur(ink.astype(np.float32), sigma)
    wb = blur(back.astype(np.float32), sigma)
    bgL = blur(fl[:, :, 0] * back, sigma) / np.maximum(wb, 1e-06)
    inkL = blur(fl[:, :, 0] * ink, sigma) / np.maximum(wi, 1e-06)
    near = smooth(blur(ink.astype(np.float32), max(3, w * 0.04)), 0.015, 0.1)
    return dict(
        fused=fused, fl=fl, il=il, ib=ib, ic=ic, obj=obj, bgL=bgL, inkL=inkL, near=near, wb=wb
    )


def render(z: dict[str, Any], p: dict[str, Any]) -> NDArray[Any]:
    fl = z["fl"]
    ic = z["ic"]
    ratio = np.clip(z["il"] / np.maximum(z["ib"], 10), 0.05, 1.1)
    target = z["bgL"] * ratio ** p["gamma"]
    alpha = p.get("strength", 1.0) * z["near"] * z["obj"] * (z["wb"] > 0.01)
    if p.get("protect"):
        alpha *= 1 - smooth((z["bgL"] - z["inkL"]) / np.maximum(z["bgL"], 10), 0.3, 0.65)
    if p.get("texture"):
        alpha *= 1 - p["texture"] * (1 - smooth(ic, 0.025, 0.1))
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
    return render(features(fused, ir, vis, valid, p.get("denoise", False)), p)
