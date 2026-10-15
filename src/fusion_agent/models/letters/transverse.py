"""Adapted from the user-supplied letters implementation."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from . import contour_core as core

material = core.material
gray = core.gray
render = core.render


def prepare(
    vis: NDArray[Any], ir: NDArray[Any], fused: NDArray[Any], valid: NDArray[Any]
) -> dict[str, Any]:
    z = core.prepare(vis, ir, fused, valid)
    mask = valid > 0
    h, w = mask.shape
    top = np.argmax(mask, axis=0)
    bottom = h - 1 - np.argmax(mask[::-1], axis=0)
    has = mask.any(0)
    dt = np.arange(h)[:, None] - top[None, :]
    db = bottom[None, :] - np.arange(h)[:, None]
    dt = np.where((has & (top < h / 3))[None, :] & (dt >= 0), dt, np.inf)
    db = np.where((has & (bottom > 2 * h / 3))[None, :] & (db >= 0), db, np.inf)
    z["distance"] = np.minimum(dt, db)
    return z
