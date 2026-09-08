"""Interpretable grayscale fusion baselines."""

from typing import Protocol

import numpy as np
import pywt


class FusionMethod(Protocol):
    def __call__(self, visible: np.ndarray, infrared: np.ndarray) -> np.ndarray: ...


def validate_array(image: np.ndarray) -> None:
    if image.ndim != 2 or min(image.shape) < 2 or image.dtype != np.uint8:
        raise ValueError("Expected a 2D uint8 grayscale image of at least 2x2.")


def validate_pair(visible: np.ndarray, infrared: np.ndarray) -> None:
    validate_array(visible)
    validate_array(infrared)
    if visible.shape != infrared.shape:
        raise ValueError("Fusion requires equal image shapes.")


def weighted_average_fusion(
    visible: np.ndarray,
    infrared: np.ndarray,
    visible_weight: float = 0.5,
) -> np.ndarray:
    validate_pair(visible, infrared)
    if not np.isfinite(visible_weight) or not 0 <= visible_weight <= 1:
        raise ValueError("visible_weight must be between 0 and 1.")
    result = visible_weight * visible.astype(float) + (1 - visible_weight) * infrared
    return np.rint(result).clip(0, 255).astype(np.uint8)


def wavelet_fusion(visible: np.ndarray, infrared: np.ndarray) -> np.ndarray:
    """One-level Haar: mean approximation, signed maximum-magnitude details."""
    validate_pair(visible, infrared)
    va, vd = pywt.dwt2(visible.astype(float), "haar", mode="symmetric")
    ia, idetails = pywt.dwt2(infrared.astype(float), "haar", mode="symmetric")
    details = tuple(
        np.where(np.abs(v) >= np.abs(i), v, i) for v, i in zip(vd, idetails, strict=True)
    )
    result = pywt.idwt2(((va + ia) / 2, details), "haar", mode="symmetric")
    return np.rint(result[: visible.shape[0], : visible.shape[1]]).clip(0, 255).astype(np.uint8)
