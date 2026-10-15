"""Inference equations adapted from the supplied PRCV_Fusion_submit archive."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .flow import warp


def _to_gray(x: torch.Tensor) -> torch.Tensor:
    if x.shape[1] == 1:
        return x
    return 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]


def _grad_mag(x: torch.Tensor) -> torch.Tensor:
    dx = F.pad(x[:, :, :, 1:] - x[:, :, :, :-1], (0, 1, 0, 0))
    dy = F.pad(x[:, :, 1:, :] - x[:, :, :-1, :], (0, 0, 0, 1))
    return torch.sqrt(dx.pow(2) + dy.pow(2) + 1e-06)


def _norm01(x: torch.Tensor) -> torch.Tensor:
    lo = x.amin(dim=(2, 3), keepdim=True)
    hi = x.amax(dim=(2, 3), keepdim=True)
    return (x - lo) / (hi - lo + 1e-06)


class VFCNonRigidRefiner(nn.Module):
    def __init__(
        self,
        enabled: bool = False,
        grid_size: int = 16,
        search_radius: int = 4,
        kernel_sigma: float = 32.0,
        max_disp: float = 3.0,
        iters: int = 3,
        topk: int = 128,
        use_mask: bool = True,
    ) -> None:
        super().__init__()
        self.enabled = bool(enabled)
        self.grid_size = int(grid_size)
        self.search_radius = int(search_radius)
        self.kernel_sigma = float(kernel_sigma)
        self.max_disp = float(max_disp)
        self.iters = int(iters)
        self.topk = int(topk)
        self.use_mask = bool(use_mask)

    @torch.no_grad()
    def forward(
        self,
        rgb: torch.Tensor,
        ir: torch.Tensor,
        mask: torch.Tensor | None = None,
        confidence: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        (b, _, h, w) = ir.shape
        zero = ir.new_zeros(b, 2, h, w)
        one = ir.new_ones(b, 1, h, w)
        if not self.enabled or self.max_disp <= 0 or self.search_radius <= 0:
            return (zero, one)
        rgb_y = _norm01(_to_gray(rgb))
        ir_y = _norm01(_to_gray(ir))
        rgb_g = _norm01(_grad_mag(rgb_y))
        ir_g = _norm01(_grad_mag(ir_y))
        weight = 0.35 + 0.65 * rgb_g
        if self.use_mask and mask is not None:
            weight = torch.clamp(weight + 0.35 * mask.float().clamp(0.0, 1.0), 0.0, 1.0)
        if confidence is not None:
            weight = weight * confidence.float().clamp(0.05, 1.0)
        gh = max(2, int(round(h / max(1, self.grid_size))))
        gw = max(2, int(round(w / max(1, self.grid_size))))
        best_err = ir.new_full((b, 1, gh, gw), 1000000.0)
        best_dx = ir.new_zeros(b, 1, gh, gw)
        best_dy = ir.new_zeros(b, 1, gh, gw)
        for dy in range(-self.search_radius, self.search_radius + 1):
            for dx in range(-self.search_radius, self.search_radius + 1):
                const_flow = zero.clone()
                const_flow[:, 0].fill_(float(dx))
                const_flow[:, 1].fill_(float(dy))
                ir_shift = warp(ir_y, const_flow)
                irg_shift = warp(ir_g, const_flow)
                err_map = (
                    0.35 * torch.abs(rgb_y - ir_shift) + 0.65 * torch.abs(rgb_g - irg_shift)
                ) * weight
                err = F.adaptive_avg_pool2d(err_map, (gh, gw))
                update = err < best_err
                best_err = torch.where(update, err, best_err)
                best_dx = torch.where(update, best_dx.new_full(best_dx.shape, float(dx)), best_dx)
                best_dy = torch.where(update, best_dy.new_full(best_dy.shape, float(dy)), best_dy)
        ctrl_flow = torch.cat([best_dx, best_dy], dim=1)
        ctrl_conf = torch.exp(-8.0 * best_err).clamp(0.02, 1.0)
        (ctrl_flow, ctrl_conf) = self._vector_field_consensus(ctrl_flow, ctrl_conf, h, w)
        ctrl_flow = ctrl_flow.clamp(-self.max_disp, self.max_disp)
        dense_flow = F.interpolate(
            ctrl_flow, size=(h, w), mode="bilinear", align_corners=True
        ).clamp(-self.max_disp, self.max_disp)
        dense_conf = F.interpolate(
            ctrl_conf, size=(h, w), mode="bilinear", align_corners=False
        ).clamp(0.02, 1.0)
        return (dense_flow, dense_conf)

    def _vector_field_consensus(
        self, ctrl_flow: torch.Tensor, ctrl_conf: torch.Tensor, height: int, width: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        (b, _, gh, gw) = ctrl_flow.shape
        n = gh * gw
        device = ctrl_flow.device
        dtype = ctrl_flow.dtype
        (yy, xx) = torch.meshgrid(
            torch.linspace(0, height - 1, gh, device=device, dtype=dtype),
            torch.linspace(0, width - 1, gw, device=device, dtype=dtype),
            indexing="ij",
        )
        pts = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1)
        dist2 = torch.cdist(pts, pts).pow(2)
        sigma2 = max(self.kernel_sigma, 1.0) ** 2
        kernel = torch.exp(-dist2 / (2.0 * sigma2)).unsqueeze(0).expand(b, -1, -1)
        vec = ctrl_flow.flatten(2).transpose(1, 2)
        conf = ctrl_conf.flatten(2).transpose(1, 2)
        if self.topk > 0 and n > self.topk:
            keep = torch.topk(conf.squeeze(-1), k=min(self.topk, n), dim=1).indices
            mask = torch.zeros_like(conf)
            mask.scatter_(1, keep.unsqueeze(-1), 1.0)
            conf = conf * mask
        inlier = torch.ones_like(conf)
        smooth = vec
        for _ in range(max(1, self.iters)):
            wgt = (conf * inlier).clamp_min(0.0001)
            numerator = torch.bmm(kernel, vec * wgt)
            denominator = torch.bmm(kernel, wgt).clamp_min(0.0001)
            smooth = numerator / denominator
            residual = torch.linalg.norm(vec - smooth, dim=-1, keepdim=True)
            scale = residual.detach().flatten(1).median(dim=1).values.view(b, 1, 1).clamp_min(0.5)
            inlier = torch.sigmoid((scale - residual) * 2.0)
        final_conf = (conf * inlier).view(b, gh, gw, 1).permute(0, 3, 1, 2).contiguous()
        final_flow = smooth.transpose(1, 2).view(b, 2, gh, gw).contiguous()
        return (final_flow, final_conf)
