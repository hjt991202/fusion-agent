from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class FlowEstimator(nn.Module):
    def __init__(
        self, in_channels: int, base_channels: int = 32, max_displacement: float = 10.0
    ) -> None:
        super().__init__()
        self.max_displacement = max_displacement
        self.enc1 = ConvBlock(in_channels, base_channels)
        self.enc2 = ConvBlock(base_channels, base_channels * 2, stride=2)
        self.enc3 = ConvBlock(base_channels * 2, base_channels * 4, stride=2)
        self.mid = ConvBlock(base_channels * 4, base_channels * 4)
        self.dec3 = ConvBlock(base_channels * 4 + base_channels * 2, base_channels * 2)
        self.dec2 = ConvBlock(base_channels * 2 + base_channels, base_channels)
        self.dec1 = nn.Sequential(
            nn.Conv2d(
                base_channels + base_channels, base_channels, kernel_size=3, padding=1, bias=False
            ),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels, 2, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        m = self.mid(e3)
        d3 = torch.cat(
            [F.interpolate(m, size=e2.shape[-2:], mode="bilinear", align_corners=False), e2], dim=1
        )
        d3 = self.dec3(d3)
        d2 = torch.cat(
            [F.interpolate(d3, size=e1.shape[-2:], mode="bilinear", align_corners=False), e1], dim=1
        )
        d2 = self.dec2(d2)
        flow = self.dec1(torch.cat([d2, e1], dim=1))
        flow = torch.tanh(flow) * self.max_displacement
        return flow


def generate_grid(height: int, width: int, device: torch.device) -> torch.Tensor:
    (y, x) = torch.meshgrid(
        torch.linspace(-1.0, 1.0, height, device=device),
        torch.linspace(-1.0, 1.0, width, device=device),
        indexing="ij",
    )
    grid = torch.stack([x, y], dim=-1)
    return grid


def warp(image: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    (b, _, h, w) = image.shape
    grid = generate_grid(h, w, image.device)
    grid = grid.unsqueeze(0).repeat(b, 1, 1, 1)
    norm_flow = torch.zeros_like(flow)
    norm_flow[:, 0] = flow[:, 0] / ((w - 1) / 2)
    norm_flow[:, 1] = flow[:, 1] / ((h - 1) / 2)
    sample_grid = grid + norm_flow.permute(0, 2, 3, 1)
    warped = F.grid_sample(
        image, sample_grid, mode="bilinear", padding_mode="reflection", align_corners=True
    )
    return warped
