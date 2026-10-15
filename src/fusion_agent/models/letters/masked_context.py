"""Adapted from the user-supplied letters implementation."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from . import boundary_engine as engine
from . import local_boundary as local
from . import refinement as refine


def normalized_blur(a: NDArray[Any], mask: NDArray[Any], sigma: float) -> NDArray[Any]:
    m = mask.astype(np.float32)
    den = cv2.GaussianBlur(m, (0, 0), sigma, borderType=cv2.BORDER_CONSTANT)
    return cv2.GaussianBlur(a * m, (0, 0), sigma, borderType=cv2.BORDER_CONSTANT) / np.maximum(
        den, 1e-08
    )


def masked_close(a: NDArray[Any], mask: NDArray[Any], kernel: NDArray[Any]) -> NDArray[Any]:
    d = cv2.dilate(
        np.where(mask, a, -1000000.0).astype(np.float32),
        kernel,
        borderType=cv2.BORDER_CONSTANT,
        borderValue=-1000000.0,
    )
    c = cv2.erode(
        np.where(mask, d, 1000000.0).astype(np.float32),
        kernel,
        borderType=cv2.BORDER_CONSTANT,
        borderValue=1000000.0,
    )
    return np.where(mask, c, 0).astype(np.float32)


def features(fused: NDArray[Any], ir: NDArray[Any], material: NDArray[Any]) -> dict[str, Any]:
    mask = material > 0
    h, w = mask.shape
    fl = cv2.cvtColor(fused.astype(np.float32) / 255, cv2.COLOR_BGR2LAB)
    il = cv2.cvtColor(ir.astype(np.float32) / 255, cv2.COLOR_BGR2LAB)[:, :, 0]
    il = cv2.bilateralFilter(il, 5, 4.0, 2.0)
    k = max(25, min(101, int(w * 0.25))) | 1
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    ib = normalized_blur(masked_close(il, mask, ker), mask, 1)
    ic = np.clip((ib - il) / np.maximum(ib, 10), 0, 0.95)
    ink = (ic > 0.1) & mask
    back = (ic < 0.025) & mask
    sigma = max(10, min(40, w * 0.16))
    wb = normalized_blur(back.astype(np.float32), mask, sigma)
    near = refine.smooth(
        normalized_blur(ink.astype(np.float32), mask, max(3, w * 0.04)), 0.015, 0.1
    )
    bgL = normalized_blur(fl[:, :, 0], back, sigma)
    inkL = normalized_blur(fl[:, :, 0], ink, sigma)
    return dict(
        fused=fused, fl=fl, il=il, ib=ib, ic=ic, obj=mask, bgL=bgL, inkL=inkL, near=near, wb=wb
    )


def prepare(
    raw: NDArray[Any],
    vis: NDArray[Any],
    ir: NDArray[Any],
    reference: NDArray[Any],
    valid: NDArray[Any],
    material: NDArray[Any],
) -> dict[str, Any]:
    mask = material > 0
    idx = engine.distance_transform_edt(~mask, return_distances=False, return_indices=True)
    rp, vp, ip = [engine.nearest(a, mask, idx) for a in [raw, vis, ir]]
    anchor = engine.anchor(rp, vp, ip)
    f = features(engine.bgr(anchor), engine.bgr(ip), mask)
    r = engine.rgb(refine.render(f, engine.CFG["refinement"]))
    edge = engine.rgb(
        engine.bp_render(
            engine.bp_prepare(engine.bgr(vp), engine.bgr(ip), engine.bgr(r), valid),
            engine.CFG["boundary_protection"],
        )
    )
    target = engine.trans.render(
        engine.trans.prepare(vp, ip, edge, mask), {"mode": "soft", "width": 16}
    )
    target[~mask] = vis[~mask]
    z = local.prepare(vis, ir, reference, target, mask)
    return z


def render(z: dict[str, Any], cfg: dict[str, Any]) -> tuple[NDArray[np.uint8], NDArray[Any]]:
    a = local.weight(z, {"width": cfg["width"], "ink_only": False}) * cfg["strength"]
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
