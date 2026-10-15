"""Inference equations adapted from the supplied PRCV_Fusion_submit archive."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class ConvGNAct(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        groups = max(1, min(8, out_channels // 4))
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
            ),
            nn.GroupNorm(groups, out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SharedManifoldDescriptor(nn.Module):
    def __init__(self, kernel_size: int = 5) -> None:
        super().__init__()
        self.kernel_size = kernel_size

    @staticmethod
    def _gray(x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] == 1:
            return x
        return 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]

    def _grad(self, gray: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dtype = gray.dtype
        device = gray.device
        sobel_x = (
            torch.tensor(
                [[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]], device=device, dtype=dtype
            ).view(1, 1, 3, 3)
            / 8.0
        )
        sobel_y = (
            torch.tensor(
                [[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]], device=device, dtype=dtype
            ).view(1, 1, 3, 3)
            / 8.0
        )
        dx = F.conv2d(gray, sobel_x, padding=1)
        dy = F.conv2d(gray, sobel_y, padding=1)
        mag = torch.sqrt(dx.pow(2) + dy.pow(2) + 1e-06)
        return (dx, dy, mag)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        gray = self._gray(image)
        (dx, dy, mag) = self._grad(gray)
        k = self.kernel_size
        pad = k // 2
        base = torch.cat([gray, dx, dy, mag], dim=1)
        mean = F.avg_pool2d(base, kernel_size=k, stride=1, padding=pad)
        centered = base - mean
        cov_entries = []
        for i in range(base.shape[1]):
            for j in range(i, base.shape[1]):
                cov_entries.append(
                    F.avg_pool2d(
                        centered[:, i : i + 1] * centered[:, j : j + 1],
                        kernel_size=k,
                        stride=1,
                        padding=pad,
                    )
                )
        cov = torch.cat(cov_entries, dim=1)
        cov = torch.sign(cov) * torch.log1p(torch.abs(cov))
        orient_norm = torch.sqrt(dx.pow(2) + dy.pow(2) + 1e-06)
        ox = dx / orient_norm
        oy = dy / orient_norm
        g_xx = F.avg_pool2d(ox * ox, kernel_size=k, stride=1, padding=pad)
        g_xy = F.avg_pool2d(ox * oy, kernel_size=k, stride=1, padding=pad)
        g_yy = F.avg_pool2d(oy * oy, kernel_size=k, stride=1, padding=pad)
        grass = torch.cat([g_xx, g_xy, g_yy], dim=1)
        local_mean = mean[:, 0:1]
        local_contrast = F.avg_pool2d(
            torch.abs(gray - local_mean), kernel_size=k, stride=1, padding=pad
        )
        desc = torch.cat([gray, mag, dx, dy, local_contrast, cov, grass], dim=1)
        return desc


class PoincareBall:
    @staticmethod
    def project(x: torch.Tensor, eps: float = 1e-05) -> torch.Tensor:
        norm = torch.norm(x, dim=1, keepdim=True).clamp_min(1e-06)
        maxnorm = 1.0 - eps
        factor = torch.clamp(maxnorm / norm, max=1.0)
        return x * factor

    @staticmethod
    def expmap0(u: torch.Tensor) -> torch.Tensor:
        norm = torch.norm(u, dim=1, keepdim=True).clamp_min(1e-06)
        return PoincareBall.project(torch.tanh(norm) * u / norm)

    @staticmethod
    def logmap0(y: torch.Tensor) -> torch.Tensor:
        norm = torch.norm(y, dim=1, keepdim=True).clamp_min(1e-06).clamp(max=1.0 - 1e-05)
        return y / norm * torch.atanh(norm)

    @staticmethod
    def distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.norm(PoincareBall.logmap0(x) - PoincareBall.logmap0(y), dim=1, keepdim=True)


class HyperbolicHierarchyBranch(nn.Module):
    def __init__(self, in_channels: int, hyper_channels: int = 16) -> None:
        super().__init__()
        self.rgb_proj = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(max(1, min(8, in_channels // 4)), in_channels),
            nn.GELU(),
            nn.Conv2d(in_channels, hyper_channels, kernel_size=1),
        )
        self.ir_proj = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(max(1, min(8, in_channels // 4)), in_channels),
            nn.GELU(),
            nn.Conv2d(in_channels, hyper_channels, kernel_size=1),
        )
        self.conf_head = nn.Sequential(
            ConvGNAct(in_channels * 2 + 2, in_channels),
            nn.Conv2d(in_channels, 1, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    @staticmethod
    def _masked_proto(z: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if mask.shape[-2:] != z.shape[-2:]:
            mask = F.interpolate(mask, size=z.shape[-2:], mode="bilinear", align_corners=False)
        mask = mask.float().clamp(0.0, 1.0)
        denom = mask.sum(dim=(2, 3), keepdim=True).clamp_min(1.0)
        return (z * mask).sum(dim=(2, 3), keepdim=True) / denom

    def forward(
        self, rgb_feat: torch.Tensor, ir_feat: torch.Tensor, mask: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        if mask.shape[-2:] != rgb_feat.shape[-2:]:
            mask = F.interpolate(
                mask, size=rgb_feat.shape[-2:], mode="bilinear", align_corners=False
            )
        mask = mask.float().clamp(0.0, 1.0)
        fg_mask = F.max_pool2d(mask, kernel_size=13, stride=1, padding=6).clamp(0.0, 1.0)
        bg_mask = (1.0 - fg_mask).clamp(0.0, 1.0)
        rgb_h = PoincareBall.expmap0(0.55 * torch.tanh(self.rgb_proj(rgb_feat)))
        ir_h = PoincareBall.expmap0(0.55 * torch.tanh(self.ir_proj(ir_feat)))
        rgb_fg = self._masked_proto(rgb_h, fg_mask)
        ir_fg = self._masked_proto(ir_h, fg_mask)
        rgb_bg = self._masked_proto(rgb_h, bg_mask)
        ir_bg = self._masked_proto(ir_h, bg_mask)
        d_fg = PoincareBall.distance(ir_h, rgb_fg.expand_as(ir_h))
        d_bg = PoincareBall.distance(ir_h, rgb_bg.expand_as(ir_h))
        fg_bias = torch.sigmoid(2.5 * (d_bg - d_fg))
        confidence = self.conf_head(torch.cat([rgb_feat, ir_feat, fg_mask, fg_bias], dim=1))
        confidence = confidence * (0.25 + 0.75 * fg_bias)
        confidence = confidence.clamp(0.05, 1.0)
        return {
            "rgb_hyperbolic": rgb_h,
            "ir_hyperbolic": ir_h,
            "rgb_hyper_fg": rgb_fg,
            "ir_hyper_fg": ir_fg,
            "rgb_hyper_bg": rgb_bg,
            "ir_hyper_bg": ir_bg,
            "hyperbolic_confidence": confidence,
            "hyper_match_bias": fg_bias,
            "hyper_d_fg": d_fg,
            "hyper_d_bg": d_bg,
        }


def _coord_grid(
    batch: int, h: int, w: int, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    (y, x) = torch.meshgrid(
        torch.linspace(-1.0, 1.0, h, device=device, dtype=dtype),
        torch.linspace(-1.0, 1.0, w, device=device, dtype=dtype),
        indexing="ij",
    )
    coords = torch.stack([x, y], dim=-1).view(1, h * w, 2)
    return coords.repeat(batch, 1, 1)


def _solve_weighted_affine(
    src: torch.Tensor, tgt: torch.Tensor, weight: torch.Tensor
) -> torch.Tensor:
    (b, n, _) = src.shape
    ones = torch.ones(b, n, 1, device=src.device, dtype=src.dtype)
    p = torch.cat([src, ones], dim=-1)
    w = weight.clamp_min(0.0001).unsqueeze(-1)
    ptw = p.transpose(1, 2) * w.transpose(1, 2)
    a = ptw @ p
    eye = torch.eye(3, device=src.device, dtype=src.dtype).unsqueeze(0)
    a = a + 0.001 * eye
    rhs = ptw @ tgt
    sol = torch.linalg.solve(a, rhs)
    theta = sol.transpose(1, 2)
    return theta


def _solve_weighted_rigid(
    src: torch.Tensor, tgt: torch.Tensor, weight: torch.Tensor
) -> torch.Tensor:
    (b, n, _) = src.shape
    w = weight.clamp_min(0.0001)
    w = w / (w.sum(dim=1, keepdim=True) + 1e-06)
    src_mean = (src * w.unsqueeze(-1)).sum(dim=1, keepdim=True)
    tgt_mean = (tgt * w.unsqueeze(-1)).sum(dim=1, keepdim=True)
    x = src - src_mean
    y = tgt - tgt_mean
    a = (w * (x[..., 0] * y[..., 0] + x[..., 1] * y[..., 1])).sum(dim=1)
    bterm = (w * (x[..., 0] * y[..., 1] - x[..., 1] * y[..., 0])).sum(dim=1)
    angle = torch.atan2(bterm, a)
    angle = torch.clamp(angle, -0.052, 0.052)
    cos = torch.cos(angle)
    sin = torch.sin(angle)
    theta = src.new_zeros(b, 2, 3)
    theta[:, 0, 0] = cos
    theta[:, 0, 1] = -sin
    theta[:, 1, 0] = sin
    theta[:, 1, 1] = cos
    tx = tgt_mean[:, 0, 0] - (
        theta[:, 0, 0] * src_mean[:, 0, 0] + theta[:, 0, 1] * src_mean[:, 0, 1]
    )
    ty = tgt_mean[:, 0, 1] - (
        theta[:, 1, 0] * src_mean[:, 0, 0] + theta[:, 1, 1] * src_mean[:, 0, 1]
    )
    theta[:, 0, 2] = torch.clamp(tx, -0.25, 0.25)
    theta[:, 1, 2] = torch.clamp(ty, -0.25, 0.25)
    return theta


def _solve_weighted_translation(
    src: torch.Tensor, tgt: torch.Tensor, weight: torch.Tensor
) -> torch.Tensor:
    (b, n, _) = src.shape
    w = weight.clamp_min(0.0001)
    w = w / (w.sum(dim=1, keepdim=True) + 1e-06)
    delta = (tgt - src) * w.unsqueeze(-1)
    trans = delta.sum(dim=1)
    theta = src.new_zeros(b, 2, 3)
    theta[:, 0, 0] = 1.0
    theta[:, 1, 1] = 1.0
    theta[:, 0, 2] = torch.clamp(trans[:, 0], -0.25, 0.25)
    theta[:, 1, 2] = torch.clamp(trans[:, 1], -0.25, 0.25)
    return theta


def _solve_weighted_similarity(
    src: torch.Tensor, tgt: torch.Tensor, weight: torch.Tensor
) -> torch.Tensor:
    (b, n, _) = src.shape
    w = weight.clamp_min(0.0001)
    w = w / (w.sum(dim=1, keepdim=True) + 1e-06)
    src_mean = (src * w.unsqueeze(-1)).sum(dim=1, keepdim=True)
    tgt_mean = (tgt * w.unsqueeze(-1)).sum(dim=1, keepdim=True)
    x = src - src_mean
    y = tgt - tgt_mean
    a = (w * (x[..., 0] * y[..., 0] + x[..., 1] * y[..., 1])).sum(dim=1)
    bterm = (w * (x[..., 0] * y[..., 1] - x[..., 1] * y[..., 0])).sum(dim=1)
    denom = (w * (x[..., 0] ** 2 + x[..., 1] ** 2)).sum(dim=1).clamp_min(1e-06)
    scale = torch.sqrt(a**2 + bterm**2 + 1e-08) / denom
    scale = torch.clamp(scale, 0.92, 1.08)
    angle = torch.atan2(bterm, a)
    angle = torch.clamp(angle, -0.18, 0.18)
    cos = torch.cos(angle)
    sin = torch.sin(angle)
    theta = src.new_zeros(b, 2, 3)
    theta[:, 0, 0] = scale * cos
    theta[:, 0, 1] = -scale * sin
    theta[:, 1, 0] = scale * sin
    theta[:, 1, 1] = scale * cos
    tx = tgt_mean[:, 0, 0] - (
        theta[:, 0, 0] * src_mean[:, 0, 0] + theta[:, 0, 1] * src_mean[:, 0, 1]
    )
    ty = tgt_mean[:, 0, 1] - (
        theta[:, 1, 0] * src_mean[:, 0, 0] + theta[:, 1, 1] * src_mean[:, 0, 1]
    )
    theta[:, 0, 2] = torch.clamp(tx, -0.3, 0.3)
    theta[:, 1, 2] = torch.clamp(ty, -0.3, 0.3)
    return theta


def _clamp_affine(theta: torch.Tensor) -> torch.Tensor:
    identity = torch.tensor(
        [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]], device=theta.device, dtype=theta.dtype
    ).repeat(theta.size(0), 1, 1)
    limits = torch.tensor(
        [[[0.18, 0.16, 0.38], [0.16, 0.18, 0.38]]], device=theta.device, dtype=theta.dtype
    )
    delta = theta - identity
    return identity + limits * torch.tanh(delta / limits.clamp_min(1e-06))


class SharedManifoldRegistration(nn.Module):
    def __init__(
        self,
        feat_channels: int = 32,
        embed_dim: int = 64,
        heads: int = 4,
        token_grid: int = 16,
        hyper_channels: int = 16,
        temperature: float = 0.07,
    ) -> None:
        super().__init__()
        self.token_grid = token_grid
        self.temperature = temperature
        self.desc = SharedManifoldDescriptor(kernel_size=5)
        desc_channels = 18
        self.rgb_bridge = nn.Sequential(
            ConvGNAct(desc_channels, feat_channels),
            ConvGNAct(feat_channels, feat_channels),
            nn.Conv2d(feat_channels, feat_channels, kernel_size=3, padding=1, bias=False),
        )
        self.ir_bridge = nn.Sequential(
            ConvGNAct(desc_channels, feat_channels),
            ConvGNAct(feat_channels, feat_channels),
            nn.Conv2d(feat_channels, feat_channels, kernel_size=3, padding=1, bias=False),
        )
        self.shared_norm = nn.GroupNorm(max(1, min(8, feat_channels // 4)), feat_channels)
        self.hyper_branch = HyperbolicHierarchyBranch(feat_channels, hyper_channels=hyper_channels)
        self.token_proj = nn.Linear(feat_channels + 4, embed_dim)
        self.cross_attn_rgb = nn.MultiheadAttention(embed_dim, heads, batch_first=True)
        self.cross_attn_ir = nn.MultiheadAttention(embed_dim, heads, batch_first=True)
        self.token_ffn = nn.Sequential(
            nn.LayerNorm(embed_dim * 4),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
        )
        self.theta_residual_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.GELU(), nn.Linear(embed_dim, 6)
        )
        nn.init.zeros_(self.theta_residual_head[-1].weight)
        nn.init.zeros_(self.theta_residual_head[-1].bias)
        self.conf_head = nn.Sequential(
            ConvGNAct(feat_channels * 2 + 1, feat_channels),
            nn.Conv2d(feat_channels, 1, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def encode_single(self, image: torch.Tensor, branch: str) -> torch.Tensor:
        d = self.desc(image)
        if branch == "rgb":
            feat = self.rgb_bridge(d)
        elif branch == "ir":
            feat = self.ir_bridge(d)
        else:
            raise ValueError(f"branch must be 'rgb' or 'ir', got {branch}")
        return self.shared_norm(feat)

    def encode_shared_pair(
        self, rgb: torch.Tensor, ir_rgb: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return (self.encode_single(rgb, "rgb"), self.encode_single(ir_rgb, "ir"))

    def _tokens(self, feat: torch.Tensor, hyper_bias: torch.Tensor | None = None) -> torch.Tensor:
        pooled = F.adaptive_avg_pool2d(feat, (self.token_grid, self.token_grid))
        (b, c, h, w) = pooled.shape
        (y, x) = torch.meshgrid(
            torch.linspace(-1.0, 1.0, h, device=feat.device, dtype=feat.dtype),
            torch.linspace(-1.0, 1.0, w, device=feat.device, dtype=feat.dtype),
            indexing="ij",
        )
        pos = torch.stack([x, y, x * y], dim=0).unsqueeze(0).expand(b, -1, -1, -1)
        if hyper_bias is None:
            hyper_bias = torch.zeros(
                b, 1, feat.shape[-2], feat.shape[-1], device=feat.device, dtype=feat.dtype
            )
        hyper_bias = F.adaptive_avg_pool2d(hyper_bias, (h, w))
        token_map = torch.cat([pooled, pos, hyper_bias], dim=1)
        tokens = token_map.flatten(2).transpose(1, 2)
        return self.token_proj(tokens)

    def _small_residual_theta(
        self, rgb_feat: torch.Tensor, ir_feat: torch.Tensor, hyper_bias: torch.Tensor
    ) -> torch.Tensor:
        rgb_tok = self._tokens(rgb_feat, hyper_bias)
        ir_tok = self._tokens(ir_feat, hyper_bias)
        (rgb_ctx, _) = self.cross_attn_rgb(rgb_tok, ir_tok, ir_tok, need_weights=False)
        (ir_ctx, _) = self.cross_attn_ir(ir_tok, rgb_tok, rgb_tok, need_weights=False)
        rgb_mean = rgb_ctx.mean(dim=1)
        ir_mean = ir_ctx.mean(dim=1)
        pair = torch.cat(
            [
                rgb_mean,
                ir_mean,
                torch.abs(rgb_mean - ir_mean),
                rgb_tok.mean(dim=1) * ir_tok.mean(dim=1),
            ],
            dim=1,
        )
        delta = torch.tanh(self.theta_residual_head(self.token_ffn(pair)))
        theta = delta.new_zeros(delta.size(0), 2, 3)
        theta[:, 0, 0] = 0.03 * delta[:, 0]
        theta[:, 0, 1] = 0.03 * delta[:, 1]
        theta[:, 0, 2] = 0.06 * delta[:, 2]
        theta[:, 1, 0] = 0.03 * delta[:, 3]
        theta[:, 1, 1] = 0.03 * delta[:, 4]
        theta[:, 1, 2] = 0.06 * delta[:, 5]
        return theta

    def _coarse_contour_feature(self, image: torch.Tensor) -> torch.Tensor:
        if image.shape[1] == 1:
            gray = image
        else:
            gray = 0.299 * image[:, 0:1] + 0.587 * image[:, 1:2] + 0.114 * image[:, 2:3]
        low = F.avg_pool2d(gray, kernel_size=21, stride=1, padding=10)
        low = F.avg_pool2d(low, kernel_size=15, stride=1, padding=7)
        obj = torch.sigmoid(18.0 * (0.92 - low))
        dtype = low.dtype
        device = low.device
        sx = (
            torch.tensor(
                [[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]], device=device, dtype=dtype
            ).view(1, 1, 3, 3)
            / 8.0
        )
        sy = (
            torch.tensor(
                [[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]], device=device, dtype=dtype
            ).view(1, 1, 3, 3)
            / 8.0
        )
        dx = F.conv2d(obj, sx, padding=1)
        dy = F.conv2d(obj, sy, padding=1)
        contour = torch.sqrt(dx.pow(2) + dy.pow(2) + 1e-06)
        feat = torch.cat([obj, contour, low], dim=1)
        feat = F.avg_pool2d(feat, kernel_size=7, stride=1, padding=3)
        return feat

    def _global_correspondence_affine(
        self,
        rgb_feat: torch.Tensor,
        ir_feat: torch.Tensor,
        mask: torch.Tensor,
        hyper_conf: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        g = self.token_grid
        rgb_low = F.adaptive_avg_pool2d(rgb_feat, (g, g))
        ir_low = F.adaptive_avg_pool2d(ir_feat, (g, g))
        mask_low = F.adaptive_avg_pool2d(mask.float(), (g, g)).clamp(0.0, 1.0)
        _hyper_low = F.adaptive_avg_pool2d(hyper_conf.float(), (g, g)).clamp(0.05, 1.0)
        rgb_tok = F.normalize(rgb_low.flatten(2).transpose(1, 2), dim=-1)
        ir_tok = F.normalize(ir_low.flatten(2).transpose(1, 2), dim=-1)
        (b, n, _) = rgb_tok.shape
        rgb_coords = _coord_grid(b, g, g, rgb_feat.device, rgb_feat.dtype)
        ir_coords = _coord_grid(b, g, g, ir_feat.device, ir_feat.dtype)
        sim = torch.bmm(ir_tok, rgb_tok.transpose(1, 2)) / self.temperature
        dist = torch.cdist(ir_coords, rgb_coords, p=2)
        sim = sim - 0.15 * dist
        prob = torch.softmax(sim, dim=-1)
        matched_rgb = torch.bmm(prob, rgb_coords)
        peak_conf = prob.max(dim=-1).values
        roi_w = torch.ones_like(mask_low.flatten(2).squeeze(1))
        weight = (roi_w * (0.3 + 0.7 * peak_conf)).clamp_min(0.001)
        theta_corr = _solve_weighted_translation(matched_rgb, ir_coords, weight)
        theta_corr = _clamp_affine(theta_corr)
        return {
            "theta_corr": theta_corr,
            "matched_rgb_coords": matched_rgb,
            "ir_coords": ir_coords,
            "match_confidence": peak_conf.view(b, 1, g, g),
            "match_weight": weight.view(b, 1, g, g),
        }

    def local_correlation_flow(
        self,
        rgb: torch.Tensor,
        ir: torch.Tensor,
        mask: torch.Tensor,
        radius: int = 4,
        feat_size: int = 64,
        max_pixels: float = 4.0,
    ) -> torch.Tensor:
        ir_rgb = ir.repeat(1, 3, 1, 1) if ir.shape[1] == 1 else ir
        (rgb_feat, ir_feat) = self.encode_shared_pair(rgb, ir_rgb)
        (h0, w0) = rgb.shape[-2:]
        fh = min(feat_size, h0)
        fw = min(feat_size, w0)
        rgb_low = F.interpolate(rgb_feat, size=(fh, fw), mode="bilinear", align_corners=False)
        ir_low = F.interpolate(ir_feat, size=(fh, fw), mode="bilinear", align_corners=False)
        rgb_low = F.normalize(rgb_low, dim=1)
        ir_low = F.normalize(ir_low, dim=1)
        k = 2 * radius + 1
        rgb_pad = F.pad(rgb_low, (radius, radius, radius, radius), mode="replicate")
        patches = F.unfold(rgb_pad, kernel_size=k)
        (b, ck, hw) = patches.shape
        c = ir_low.shape[1]
        patches = patches.view(b, c, k * k, fh, fw)
        sim = (patches * ir_low.unsqueeze(2)).sum(dim=1)
        prob = torch.softmax(sim / 0.08, dim=1)
        offsets = []
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                offsets.append((dx, dy))
        off = torch.tensor(offsets, device=rgb.device, dtype=rgb.dtype)
        dx = off[:, 0].view(1, k * k, 1, 1)
        dy = off[:, 1].view(1, k * k, 1, 1)
        exp_dx = (prob * dx).sum(dim=1, keepdim=True)
        exp_dy = (prob * dy).sum(dim=1, keepdim=True)
        scale_x = float(w0) / float(fw)
        scale_y = float(h0) / float(fh)
        flow_low = torch.cat([-exp_dx * scale_x, -exp_dy * scale_y], dim=1)
        if mask.shape[-2:] != (fh, fw):
            mask_low = F.interpolate(
                mask.float(), size=(fh, fw), mode="bilinear", align_corners=False
            )
        else:
            mask_low = mask.float()
        roi = F.max_pool2d(mask_low.clamp(0.0, 1.0), kernel_size=9, stride=1, padding=4)
        flow_low = flow_low * (0.25 + 0.75 * roi)
        flow = F.interpolate(flow_low, size=(h0, w0), mode="bilinear", align_corners=False)
        flow = torch.clamp(flow, min=-max_pixels, max=max_pixels)
        return flow

    def forward(
        self, rgb: torch.Tensor, ir: torch.Tensor, mask: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        ir_rgb = ir.repeat(1, 3, 1, 1) if ir.shape[1] == 1 else ir
        if mask is None:
            mask = torch.ones(
                rgb.size(0), 1, rgb.size(2), rgb.size(3), device=rgb.device, dtype=rgb.dtype
            )
        mask = mask.float().clamp(0.0, 1.0)
        (rgb_feat, ir_feat) = self.encode_shared_pair(rgb, ir_rgb)
        hyper = self.hyper_branch(rgb_feat, ir_feat, mask)
        hyper_conf = hyper["hyperbolic_confidence"]
        _hyper_bias = hyper["hyper_match_bias"]
        rgb_fused = rgb_feat * (0.35 + 0.65 * hyper_conf)
        ir_fused = ir_feat * (0.35 + 0.65 * hyper_conf)
        rgb_coarse_feat = self._coarse_contour_feature(rgb)
        ir_coarse_feat = self._coarse_contour_feature(ir_rgb)
        coarse_hyper_neutral = torch.ones_like(mask)
        corr = self._global_correspondence_affine(
            rgb_coarse_feat, ir_coarse_feat, mask, coarse_hyper_neutral
        )
        theta_corr = corr["theta_corr"]
        residual = torch.zeros_like(theta_corr)
        theta = theta_corr
        diff = torch.abs(rgb_fused - ir_fused)
        conf = self.conf_head(torch.cat([rgb_fused, ir_fused, mask], dim=1)).clamp(0.05, 1.0)
        conf = conf * (0.25 + 0.75 * hyper_conf)
        conf = conf * torch.exp(-0.2 * diff.mean(dim=1, keepdim=True)).clamp(0.05, 1.0)
        out = {
            "theta": theta,
            "theta_corr": theta_corr,
            "theta_residual": residual,
            "rgb_manifold_pre": rgb_feat,
            "ir_manifold_pre": ir_feat,
            "rgb_fused_euclidean": rgb_fused,
            "ir_fused_euclidean": ir_fused,
            "manifold_pre_confidence": conf.clamp(0.05, 1.0),
        }
        out.update(hyper)
        out.update(corr)
        return out
