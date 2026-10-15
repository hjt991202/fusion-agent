"""Adapted from the user-supplied letters implementation."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray
from PIL import Image
from scipy.ndimage import binary_fill_holes, distance_transform_edt


def material(vis: NDArray[Any], valid: NDArray[Any]) -> NDArray[Any]:
    hsv = cv2.cvtColor(vis, cv2.COLOR_RGB2HSV)
    _, labels, stats, _ = cv2.connectedComponentsWithStats(
        ((hsv[:, :, 1] > 30) & (hsv[:, :, 2] > 30)).astype(np.uint8)
    )
    return binary_fill_holes(labels == 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])) & (valid > 0)


def prepare(
    vis: NDArray[Any], ir: NDArray[Any], fused: NDArray[Any], valid: NDArray[Any]
) -> dict[str, Any]:
    valid = valid > 0
    h, w = valid.shape
    gray = cv2.cvtColor(ir, cv2.COLOR_RGB2GRAY)
    _, nearest = distance_transform_edt(~valid, return_indices=True)
    padded = cv2.bilateralFilter(gray[tuple(nearest)], 5, 4, 2)
    k = max(25, min(101, int(w * 0.25))) | 1
    bg = cv2.morphologyEx(padded, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8)).astype(np.float32)
    bg = cv2.GaussianBlur(bg, (0, 0), 1)
    contrast = np.clip((bg - padded) / np.maximum(bg, 10), 0, 1)
    t = np.clip((contrast - 0.03) / 0.15, 0, 1)
    ink = t * t * (3 - 2 * t)
    rad = max(1, round(3 * w / 256))
    ink = cv2.dilate(ink, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * rad + 1, 2 * rad + 1)))
    ink = cv2.GaussianBlur(ink, (0, 0), max(0.6, w / 256))
    dist = distance_transform_edt(np.pad(valid, 1))[1:-1, 1:-1]
    return dict(vis=vis, fused=fused, ink=ink, valid=valid, distance=dist, scale=w / 256)


def gray(a: NDArray[Any]) -> NDArray[Any]:
    return np.asarray(Image.fromarray(a).convert("L"))


def lock_luminance(proposal: NDArray[Any], reference: NDArray[Any]) -> NDArray[Any]:
    target = gray(reference).astype(np.float32)
    y = proposal @ np.array([0.299, 0.587, 0.114], np.float32)
    c = proposal - y[:, :, None]
    scale = np.ones_like(target)
    for k in range(3):
        pos = c[:, :, k] > 0
        neg = c[:, :, k] < 0
        scale[pos] = np.minimum(scale[pos], (255 - target[pos]) / c[:, :, k][pos])
        scale[neg] = np.minimum(scale[neg], -target[neg] / c[:, :, k][neg])
    out = np.uint8(np.clip(np.rint(target[:, :, None] + c * scale[:, :, None]), 0, 255))
    for _ in range(3):
        delta = target.astype(np.int16) - gray(out).astype(np.int16)
        if not np.any(delta):
            break
        out = np.uint8(np.clip(out.astype(np.int16) + delta[:, :, None], 0, 255))
    bad = gray(out) != target
    out[bad] = reference[bad]
    assert np.array_equal(gray(out), gray(reference))
    return out


def render(z: dict[str, Any], cfg: dict[str, Any]) -> NDArray[Any]:
    if cfg["mode"] == "none":
        return z["fused"].copy()
    width = max(8, cfg["width"] * z["scale"])
    t = np.clip((z["distance"] - width) / width, 0, 1)
    alpha = (1 - t * t * (3 - 2 * t)) * (1 - z["ink"]) * z["valid"]
    if cfg["mode"] == "protected":
        r = max(1, round(cfg["radius"] * z["scale"]))
        protected = (
            cv2.dilate(
                (z["ink"] > 0.1).astype(np.uint8),
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)),
            )
            > 0
        )
        alpha[protected] = 0
    proposal = z["fused"].astype(np.float32) + alpha[:, :, None] * (
        z["vis"].astype(np.float32) - z["fused"].astype(np.float32)
    )
    if cfg["mode"] == "chroma":
        out = lock_luminance(proposal, z["fused"])
        out[alpha == 0] = z["fused"][alpha == 0]
        return out
    return np.uint8(np.clip(proposal + 0.5, 0, 255))
