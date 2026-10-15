"""Adapted from the user-supplied letters implementation."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from . import transverse as trans


def prepare(
    vis: NDArray[Any],
    ir: NDArray[Any],
    reference: NDArray[Any],
    padded: NDArray[Any],
    material: NDArray[Any],
) -> dict[str, Any]:
    z = trans.prepare(vis, ir, reference, material)
    z["padded"] = padded
    return z


def weight(z: dict[str, Any], cfg: dict[str, Any]) -> NDArray[Any]:
    if cfg is None:
        return np.zeros(z["valid"].shape, np.float32)
    width = max(8, cfg["width"] * z["scale"])
    t = np.clip((z["distance"] - width) / width, 0, 1)
    a = (1 - t * t * (3 - 2 * t)) * z["valid"]
    if cfg["ink_only"]:
        a = a * z["ink"]
    return a.astype(np.float32)


def render(z: dict[str, Any], cfg: dict[str, Any]) -> tuple[NDArray[np.uint8], NDArray[Any]]:
    a = weight(z, cfg)
    out = np.uint8(
        np.clip(
            z["fused"].astype(np.float32)
            + a[:, :, None] * (z["padded"].astype(np.float32) - z["fused"].astype(np.float32))
            + 0.5,
            0,
            255,
        )
    )
    assert np.array_equal(out[a == 0], z["fused"][a == 0])
    return (out, a)
