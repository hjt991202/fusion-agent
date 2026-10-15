"""Adapted from the user-supplied letters implementation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from numpy.typing import NDArray
from PIL import Image
from scipy.ndimage import distance_transform_edt

from . import transverse as trans
from .refine_extended import process
from .source_restore import prepare as bp_prepare
from .source_restore import render as bp_render

CFG = json.loads(Path(__file__).with_name("config.json").read_text(encoding="utf8"))


def bgr(a: NDArray[Any]) -> NDArray[Any]:
    return cv2.cvtColor(a, cv2.COLOR_RGB2BGR)


def rgb(a: NDArray[Any]) -> NDArray[Any]:
    return cv2.cvtColor(a, cv2.COLOR_BGR2RGB)


def gray(a: NDArray[Any]) -> NDArray[Any]:
    return np.asarray(Image.fromarray(a).convert("L"))


def anchor(raw: NDArray[Any], vis: NDArray[Any], ir: NDArray[Any]) -> NDArray[Any]:
    p = CFG["anchor"]
    f = raw.astype(np.float32) / 255
    v = vis.astype(np.float32) / 255
    ii = gray(ir).astype(np.float32) / 255
    ink = np.clip((p["ink_threshold"] - ii) / p["ink_threshold"], 0, 1)
    k = p["ink_blur"]
    ink = cv2.GaussianBlur(ink, (k, k), 0)
    ink = np.clip(ink, 0, 1)
    bw = p["bg_rgb_keep"] * (1 - ink[:, :, None])
    base = f * (1 - bw) + v * bw
    lab = cv2.cvtColor(np.clip(base * 255, 0, 255).astype(np.uint8), cv2.COLOR_RGB2LAB).astype(
        np.float32
    )
    vl = cv2.cvtColor(np.clip(v * 255, 0, 255).astype(np.uint8), cv2.COLOR_RGB2LAB).astype(
        np.float32
    )
    w = p["chroma_from_rgb"]
    lab[:, :, 1:] = lab[:, :, 1:] * (1 - w) + vl[:, :, 1:] * w
    L = lab[:, :, 0] / 255
    loc = cv2.GaussianBlur(L, (15, 15), 0)
    enh = np.clip(loc + (L - loc) * (1 + p["ink_gain"] * ink) - p["ink_darken"] * ink, 0, 1)
    if p["ink_gamma"] > 0:
        enh = np.power(np.clip(enh, 1e-06, 1), 1 + p["ink_gamma"] * ink)
    lab[:, :, 0] = np.clip(L * (1 - ink) + enh * ink, 0, 1) * 255
    return cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)


def nearest(a: NDArray[Any], mask: NDArray[Any], idx: NDArray[Any] | None = None) -> NDArray[Any]:
    if idx is None:
        idx = distance_transform_edt(~mask, return_distances=False, return_indices=True)
    return a[tuple(idx)].copy()


def stages(
    raw: NDArray[Any],
    vis: NDArray[Any],
    ir: NDArray[Any],
    valid: NDArray[Any],
    material: NDArray[Any],
    pad: bool = False,
) -> dict[str, Any]:
    if pad:
        idx = distance_transform_edt(~material, return_distances=False, return_indices=True)
        raw, visp, irp = [nearest(a, material, idx) for a in [raw, vis, ir]]
    else:
        visp, irp = (vis, ir)
    a = anchor(raw, visp, irp)
    r = rgb(process(bgr(a), bgr(irp), bgr(visp), valid, CFG["refinement"]))
    edge = rgb(
        bp_render(bp_prepare(bgr(visp), bgr(irp), bgr(r), valid), CFG["boundary_protection"])
    )
    final = trans.render(trans.prepare(visp, irp, edge, material), {"mode": "soft", "width": 16})
    if pad:
        final[~material] = vis[~material]
    return {"raw": raw, "anchor": a, "reflectance": r, "edge": edge, "final": final}


@torch.inference_mode()
def infer(
    net: torch.nn.Module, vis: NDArray[Any], ir: NDArray[Any], mask: NDArray[Any] | None = None
) -> tuple[NDArray[np.uint8], dict[str, float]]:
    if mask is not None:
        idx = distance_transform_edt(~mask, return_distances=False, return_indices=True)
        vis, ir = [nearest(a, mask, idx) for a in [vis, ir]]
    v = (
        torch.from_numpy(vis.copy().transpose(2, 0, 1))
        .float()[None]
        .to(next(net.parameters()).device)
        / 255
    )
    i = (
        torch.from_numpy(gray(ir).copy()).float()[None, None].to(next(net.parameters()).device)
        / 255
    )
    result = net(v, i)
    for name in ("fused", "flow"):
        if not torch.isfinite(result[name]).all():
            raise RuntimeError(f"letters model produced non-finite {name} values.")
    out = result["fused"][0].clamp(0, 1).mul(255).add(0.5).byte().permute(1, 2, 0).cpu().numpy()
    flow = result["flow"][0].cpu().numpy()
    mag = np.linalg.norm(flow, axis=0)
    if not np.isfinite(mag).all():
        raise RuntimeError("letters model produced non-finite flow diagnostics.")
    diag = {"flow_mean_pixels": float(mag.mean()), "flow_p95_pixels": float(np.percentile(mag, 95))}
    del v, i, result
    return (out, diag)
