"""Inference equations adapted from the supplied PRCV_Fusion_submit archive."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def _grad_mag(x: torch.Tensor) -> torch.Tensor:
    if x.shape[1] == 3:
        x = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
    dx = F.pad(x[:, :, :, 1:] - x[:, :, :, :-1], (0, 1, 0, 0))
    dy = F.pad(x[:, :, 1:, :] - x[:, :, :-1, :], (0, 0, 0, 1))
    g = torch.sqrt(dx.pow(2) + dy.pow(2) + 1e-06)
    return g / (g.amax(dim=(1, 2, 3), keepdim=True) + 1e-06)


class StrokeAwareControlRefiner(nn.Module):
    def __init__(
        self,
        control_grid: int = 8,
        radius: int = 2,
        max_pixels: float = 0.0,
        temperature: float = 0.1,
    ) -> None:
        super().__init__()
        self.control_grid = int(control_grid)
        self.radius = int(radius)
        self.max_pixels = float(max_pixels)
        self.temperature = float(temperature)
        self.conf_head = nn.Sequential(
            nn.Conv2d(5, 24, kernel_size=3, padding=1, bias=True),
            nn.GELU(),
            nn.Conv2d(24, 16, kernel_size=3, padding=1, bias=True),
            nn.GELU(),
            nn.Conv2d(16, 1, kernel_size=3, padding=1, bias=True),
            nn.Sigmoid(),
        )
        nn.init.constant_(self.conf_head[-2].bias, 1.2)

    def _local_match_conf(
        self, rgb_feat: torch.Tensor, ir_feat: torch.Tensor, out_size: tuple[int, int]
    ) -> torch.Tensor:
        (b, c, _, _) = rgb_feat.shape
        g = self.control_grid
        r = self.radius
        k = 2 * r + 1
        rgb_low = F.adaptive_avg_pool2d(rgb_feat, (g, g))
        ir_low = F.adaptive_avg_pool2d(ir_feat, (g, g))
        rgb_low = F.normalize(rgb_low, dim=1)
        ir_low = F.normalize(ir_low, dim=1)
        rgb_pad = F.pad(rgb_low, (r, r, r, r), mode="replicate")
        patches = F.unfold(rgb_pad, kernel_size=k).view(b, c, k * k, g, g)
        sim = (patches * ir_low.unsqueeze(2)).sum(dim=1)
        offsets = []
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                offsets.append((dx, dy))
        off = torch.tensor(offsets, device=rgb_feat.device, dtype=rgb_feat.dtype)
        dx = off[:, 0].view(1, k * k, 1, 1)
        dy = off[:, 1].view(1, k * k, 1, 1)
        sim = sim - 0.08 * (dx**2 + dy**2)
        prob = torch.softmax(sim / self.temperature, dim=1)
        local_conf = prob.max(dim=1, keepdim=True).values
        local_conf = F.interpolate(local_conf, size=out_size, mode="bilinear", align_corners=False)
        return local_conf.clamp(0.0, 1.0)

    def forward(
        self,
        rgb: torch.Tensor,
        ir_affine: torch.Tensor,
        mask_affine: torch.Tensor,
        hyper_conf_affine: torch.Tensor,
        feature_encoder,
        match_confidence: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        (b, _, h, w) = rgb.shape
        ir_rgb = ir_affine.repeat(1, 3, 1, 1) if ir_affine.shape[1] == 1 else ir_affine
        (rgb_feat, ir_feat) = feature_encoder.encode_shared_pair(rgb, ir_rgb)
        rgb_norm = F.normalize(rgb_feat, dim=1)
        ir_norm = F.normalize(ir_feat, dim=1)
        manifold_similarity = (rgb_norm * ir_norm).sum(dim=1, keepdim=True)
        manifold_similarity = ((manifold_similarity + 1.0) * 0.5).clamp(0.0, 1.0)
        manifold_similarity = F.interpolate(
            manifold_similarity, size=(h, w), mode="bilinear", align_corners=False
        )
        local_match_conf = self._local_match_conf(rgb_feat, ir_feat, (h, w))
        mask_roi = F.max_pool2d(
            mask_affine.float().clamp(0.0, 1.0), kernel_size=13, stride=1, padding=6
        )
        hyper = hyper_conf_affine.float().clamp(0.0, 1.0)
        if hyper.shape[-2:] != (h, w):
            hyper = F.interpolate(hyper, size=(h, w), mode="bilinear", align_corners=False)
        ir_saliency = _grad_mag(ir_affine)
        conf_input = torch.cat(
            [mask_roi, hyper, manifold_similarity, local_match_conf, ir_saliency], dim=1
        )
        learned_conf = self.conf_head(conf_input)
        registration_confidence = torch.clamp(
            learned_conf
            * (0.3 + 0.7 * manifold_similarity)
            * (0.35 + 0.65 * local_match_conf)
            * (0.25 + 0.75 * hyper),
            0.05,
            1.0,
        )
        roi_weight = torch.clamp(mask_roi * (0.2 + 0.8 * registration_confidence), 0.0, 1.0)
        zero_flow = torch.zeros(b, 2, h, w, device=rgb.device, dtype=rgb.dtype)
        ctrl_disp = torch.zeros(
            b, 2, self.control_grid, self.control_grid, device=rgb.device, dtype=rgb.dtype
        )
        return {
            "refine_flow": zero_flow,
            "ctrl_disp": ctrl_disp,
            "roi_weight": roi_weight,
            "local_match_confidence": F.adaptive_avg_pool2d(
                local_match_conf, (self.control_grid, self.control_grid)
            ),
            "local_match_confidence_full": local_match_conf,
            "ctrl_roi": F.adaptive_avg_pool2d(roi_weight, (self.control_grid, self.control_grid)),
            "registration_confidence": registration_confidence,
            "manifold_confidence": manifold_similarity,
            "reliable_ink": roi_weight,
            "refine_scale": torch.zeros(1, device=rgb.device, dtype=rgb.dtype),
        }
