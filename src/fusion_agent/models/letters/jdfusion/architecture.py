from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .flow import FlowEstimator, warp


def conv_block(in_channels: int, out_channels: int, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(0.2, inplace=True),
    )


class CoarseReflectanceEncoder(nn.Module):
    def __init__(self, in_channels: int = 3, base_channels: int = 32) -> None:
        super().__init__()
        self.layer1 = conv_block(in_channels, base_channels)
        self.layer2 = conv_block(base_channels, base_channels)
        self.layer3 = conv_block(base_channels, base_channels)
        self.head = nn.Conv2d(base_channels, 3, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feat = self.layer3(self.layer2(self.layer1(x)))
        reflectance = torch.sigmoid(self.head(feat))
        return (reflectance, feat)


class RetinexDecomposition(nn.Module):
    def __init__(self, in_channels: int = 3, base_channels: int = 32) -> None:
        super().__init__()
        self.enc1 = conv_block(in_channels, base_channels)
        self.enc2 = conv_block(base_channels, base_channels)
        self.enc3 = conv_block(base_channels, base_channels)
        self.reflectance = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels, 3, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )
        self.illumination = nn.Sequential(
            nn.Conv2d(base_channels, base_channels // 2, kernel_size=3, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels // 2, 1, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feat = self.enc3(self.enc2(self.enc1(x)))
        reflectance = self.reflectance(feat)
        illumination = self.illumination(feat)
        return (reflectance, illumination)


def _gradient_saliency(ir: torch.Tensor) -> torch.Tensor:
    sobel_x = (
        torch.tensor(
            [[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]], device=ir.device, dtype=ir.dtype
        ).view(1, 1, 3, 3)
        / 8.0
    )
    sobel_y = (
        torch.tensor(
            [[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]], device=ir.device, dtype=ir.dtype
        ).view(1, 1, 3, 3)
        / 8.0
    )
    grad_x = F.conv2d(ir, sobel_x, padding=1)
    grad_y = F.conv2d(ir, sobel_y, padding=1)
    saliency = torch.sqrt(grad_x**2 + grad_y**2 + 1e-08)
    max_val = saliency.amax(dim=(1, 2, 3), keepdim=True) + 1e-06
    return saliency / max_val


class ModalitySelectionNet(nn.Module):
    def __init__(self, in_channels: int = 7, base_channels: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            conv_block(in_channels, base_channels),
            conv_block(base_channels, base_channels),
            conv_block(base_channels, base_channels),
            nn.Conv2d(base_channels, 1, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class InkEnhancementNet(nn.Module):
    def __init__(self, in_channels: int = 4, base_channels: int = 32) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            conv_block(in_channels, base_channels), conv_block(base_channels, base_channels * 2)
        )
        self.enhance = nn.Sequential(
            conv_block(base_channels * 2, base_channels * 2),
            conv_block(base_channels * 2, base_channels),
            nn.Conv2d(base_channels, 3, kernel_size=3, padding=1),
            nn.Tanh(),
        )
        self.enhancement_scale = 0.3

    def forward(self, fused_base: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = torch.cat([fused_base, mask], dim=1)
        feat = self.encoder(x)
        enhancement = self.enhance(feat) * self.enhancement_scale
        enhanced = fused_base + mask * enhancement
        return torch.clamp(enhanced, 0.0, 1.0)


class SpatialAttention(nn.Module):
    def __init__(
        self, in_channels: int = 1, out_channels: int = 3, hidden_channels: int = 16
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class IDRNet(nn.Module):
    def __init__(
        self, base_channels: int = 32, flow_channels: int = 64, ablation: dict | None = None
    ) -> None:
        super().__init__()
        ablation = ablation or {}
        self.use_flow = ablation.get("use_flow", True)
        self.use_retinex = ablation.get("use_retinex", True)
        self.use_saliency = ablation.get("use_saliency", True)
        self.use_mask_guidance = ablation.get("use_mask_guidance", True)
        self.use_mask_attention = ablation.get("use_mask_attention", True)
        self.use_enhancement = ablation.get("use_enhancement", True)
        self.coarse_encoder = CoarseReflectanceEncoder(base_channels=base_channels)
        flow_in_channels = 3 + 3 + 3
        self.flow_estimator = FlowEstimator(
            in_channels=flow_in_channels, base_channels=flow_channels // 2
        )
        self.decomposition = RetinexDecomposition(base_channels=base_channels)
        self.selection = ModalitySelectionNet(in_channels=7, base_channels=base_channels)
        self.enhancement = InkEnhancementNet(in_channels=4, base_channels=base_channels)
        self.mask_attention = SpatialAttention(
            out_channels=3, hidden_channels=max(16, base_channels // 2)
        )

    def forward(
        self, rgb: torch.Tensor, ir: torch.Tensor, mask: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        if mask is None:
            mask = torch.ones_like(ir)
        mask = mask.float()
        ir_saliency = _gradient_saliency(ir)
        if self.use_saliency:
            mask_guidance = torch.clamp(torch.maximum(mask, ir_saliency), 0.0, 1.0)
        else:
            mask_guidance = mask
        ir_rgb = ir.repeat(1, 3, 1, 1)
        (coarse_rgb, _) = self.coarse_encoder(rgb)
        (coarse_ir, _) = self.coarse_encoder(ir_rgb)
        flow_input = torch.cat([coarse_rgb, coarse_ir, coarse_rgb - coarse_ir], dim=1)
        if self.use_flow:
            flow = self.flow_estimator(flow_input)
            rgb_aligned = warp(rgb, flow)
        else:
            flow = torch.zeros(
                rgb.size(0), 2, rgb.size(2), rgb.size(3), device=rgb.device, dtype=rgb.dtype
            )
            rgb_aligned = rgb
        if self.use_retinex:
            (reflect_rgb, illum_rgb) = self.decomposition(rgb_aligned)
            (reflect_ir, illum_ir) = self.decomposition(ir_rgb)
        else:
            reflect_rgb = rgb_aligned
            reflect_ir = ir_rgb
            illum_rgb = torch.ones_like(ir)
            illum_ir = torch.ones_like(ir)
        mask_resized = mask_guidance
        if mask_resized.shape[-2:] != reflect_rgb.shape[-2:]:
            mask_resized = F.interpolate(
                mask_resized, size=reflect_rgb.shape[-2:], mode="bilinear", align_corners=False
            )
        if self.use_mask_attention:
            attention_map = self.mask_attention(mask_resized)
            reflect_ir_enhanced = reflect_ir * (1.0 + attention_map)
        else:
            attention_map = torch.zeros_like(reflect_ir)
            reflect_ir_enhanced = reflect_ir
        selection_input = torch.cat([reflect_rgb, reflect_ir_enhanced, mask_resized], dim=1)
        alpha = self.selection(selection_input)
        if self.use_mask_guidance:
            alpha_guided = alpha * mask_resized + alpha * 0.1 * (1 - mask_resized)
        else:
            alpha_guided = alpha
        reflect_fused = alpha_guided * reflect_ir_enhanced + (1 - alpha_guided) * reflect_rgb
        reflect_fused = torch.clamp(reflect_fused, 0.0, 1.0)
        illum_rgb_expanded = illum_rgb.repeat(1, 3, 1, 1)
        fused_base = torch.clamp(reflect_fused * illum_rgb_expanded, 0.0, 1.0)
        if self.use_enhancement:
            fused_image = self.enhancement(fused_base, mask_resized)
        else:
            fused_image = fused_base
        return {
            "fused": fused_image,
            "fused_base": fused_base,
            "flow": flow,
            "rgb_aligned": rgb_aligned,
            "reflect_rgb": reflect_rgb,
            "reflect_ir": reflect_ir,
            "reflect_fused": reflect_fused,
            "illum_rgb": illum_rgb,
            "illum_ir": illum_ir,
            "mask_guidance": mask_guidance,
            "attention_map": attention_map,
            "alpha": alpha_guided,
        }
