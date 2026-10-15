"""Inference equations adapted from the supplied PRCV_Fusion_submit archive."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .flow import FlowEstimator, warp
from .manifold_registration import SharedManifoldRegistration
from .stroke_refiner import StrokeAwareControlRefiner
from .vfc_registration import VFCNonRigidRefiner


def conv_block(in_channels: int, out_channels: int, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(0.2, inplace=True),
    )


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
    return saliency / (saliency.amax(dim=(1, 2, 3), keepdim=True) + 1e-06)


def _irboost_rgb_to_ycbcr(x: torch.Tensor):
    r, g, b = (x[:, 0:1], x[:, 1:2], x[:, 2:3])
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = -0.168736 * r - 0.331264 * g + 0.5 * b + 0.5
    cr = 0.5 * r - 0.418688 * g - 0.081312 * b + 0.5
    return (y, cb, cr)


def _irboost_ycbcr_to_rgb(y: torch.Tensor, cb: torch.Tensor, cr: torch.Tensor):
    cb = cb - 0.5
    cr = cr - 0.5
    r = y + 1.402 * cr
    g = y - 0.344136 * cb - 0.714136 * cr
    b = y + 1.772 * cb
    return torch.cat([r, g, b], dim=1).clamp(0.0, 1.0)


def _irboost_grad_norm(x: torch.Tensor):
    gx = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1])
    gy = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :])
    gx = F.pad(gx, (0, 1, 0, 0))
    gy = F.pad(gy, (0, 0, 0, 1))
    g = torch.sqrt(gx * gx + gy * gy + 1e-08)
    gmax = g.amax(dim=(1, 2, 3), keepdim=True) + 1e-06
    return g / gmax


def _no_mask_ir_ink_boost(
    rgb: torch.Tensor,
    ir_aligned: torch.Tensor,
    fused: torch.Tensor,
    strength: float = 2.0,
    prior_gate: torch.Tensor | None = None,
):
    rgb = rgb.clamp(0.0, 1.0)
    fused = fused.clamp(0.0, 1.0)
    ir = ir_aligned.clamp(0.0, 1.0)
    rgb_y, rgb_cb, rgb_cr = _irboost_rgb_to_ycbcr(rgb)
    fused_y, _, _ = _irboost_rgb_to_ycbcr(fused)
    local_bg = F.avg_pool2d(ir, kernel_size=21, stride=1, padding=10)
    ir_stroke = (local_bg - ir).clamp(0.0, 1.0)
    ir_stroke = ir_stroke / (ir_stroke.amax(dim=(1, 2, 3), keepdim=True) + 1e-06)
    ir_stroke = ir_stroke.pow(0.7)
    grad_ir = _irboost_grad_norm(ir)
    grad_rgb = _irboost_grad_norm(rgb_y)
    ir_better = torch.sigmoid(14.0 * (grad_ir - grad_rgb + 0.015))
    gate = torch.clamp(0.8 * ir_stroke + 0.45 * grad_ir, 0.0, 1.0)
    gate = gate * (0.35 + 0.65 * ir_better)
    gate = F.avg_pool2d(gate, kernel_size=3, stride=1, padding=1).clamp(0.0, 1.0)
    if prior_gate is not None:
        prior_gate = torch.nan_to_num(prior_gate, nan=0.5, posinf=1.0, neginf=0.0)
        if prior_gate.dim() == 3:
            prior_gate = prior_gate.unsqueeze(1)
        if prior_gate.size(1) != 1:
            prior_gate = prior_gate.mean(dim=1, keepdim=True)
        if prior_gate.shape[-2:] != gate.shape[-2:]:
            prior_gate = F.interpolate(
                prior_gate, size=gate.shape[-2:], mode="bilinear", align_corners=False
            )
        prior_gate = prior_gate.clamp(0.0, 1.0)
        gate = torch.clamp(gate * (0.75 + 0.5 * prior_gate), 0.0, 1.0)
    base_y = 0.9 * fused_y + 0.1 * rgb_y
    dark_amount = strength * gate * (0.055 + 0.28 * ir_stroke)
    out_y = (base_y - dark_amount).clamp(0.0, 1.0)
    out = _irboost_ycbcr_to_rgb(out_y, rgb_cb, rgb_cr)
    return (out, gate)


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


def warp_affine(image: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    grid = F.affine_grid(theta, image.size(), align_corners=True)
    warped = F.grid_sample(
        image, grid, mode="bilinear", padding_mode="reflection", align_corners=True
    )
    return warped


class AffineCoarseAligner(nn.Module):
    def __init__(self, in_channels: int = 9, base_channels: int = 16) -> None:
        super().__init__()
        self.enc = nn.Sequential(
            conv_block(in_channels, base_channels, stride=2),
            conv_block(base_channels, base_channels * 2, stride=2),
            conv_block(base_channels * 2, base_channels * 4, stride=2),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(base_channels * 4, base_channels * 2)
        self.fc2 = nn.Linear(base_channels * 2, 6)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.enc(x)
        feat = self.pool(feat).flatten(1)
        feat = F.leaky_relu(self.fc1(feat), negative_slope=0.2, inplace=False)
        delta = torch.tanh(self.fc2(feat))
        theta = x.new_zeros(x.size(0), 2, 3)
        theta[:, 0, 0] = 1.0 + 0.08 * delta[:, 0]
        theta[:, 0, 1] = 0.05 * delta[:, 1]
        theta[:, 0, 2] = 0.1 * delta[:, 2]
        theta[:, 1, 0] = 0.05 * delta[:, 3]
        theta[:, 1, 1] = 1.0 + 0.08 * delta[:, 4]
        theta[:, 1, 2] = 0.1 * delta[:, 5]
        return theta


class FeatureStem(nn.Module):
    def __init__(self, in_channels: int, feat_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            conv_block(in_channels, feat_channels),
            conv_block(feat_channels, feat_channels),
            conv_block(feat_channels, feat_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SharedPrivateProjector(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.shared = nn.Sequential(
            conv_block(channels, channels),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
        )
        self.private = nn.Sequential(
            conv_block(channels, channels),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        shared = self.shared(x)
        private = self.private(x)
        return (shared, private)


class SourceGate2D(nn.Module):
    def __init__(self, out_channels: int, hidden_channels: int = 16) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _attention_logit_norm(x: torch.Tensor) -> torch.Tensor:
    mean = x.mean(dim=(-2, -1), keepdim=True)
    std = x.std(dim=(-2, -1), keepdim=True).clamp_min(1e-06)
    return (x - mean) / std


def _token_covariance(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1]
    x = x - x.mean(dim=-1, keepdim=True)
    denom = float(max(d - 1, 1))
    return torch.matmul(x, x.transpose(-2, -1)) / denom


def _riemannian_token_attention(q: torch.Tensor, k: torch.Tensor, scale: float) -> torch.Tensor:
    q_cov = _token_covariance(q)
    k_cov = _token_covariance(k)
    if q_cov.shape[-1] == k_cov.shape[-1]:
        dots = torch.matmul(q_cov, k_cov.transpose(-2, -1))
        attn_spd = torch.linalg.norm(dots.unsqueeze(2), dim=2)
    else:
        q_center = q - q.mean(dim=-1, keepdim=True)
        k_center = k - k.mean(dim=-1, keepdim=True)
        denom = float(max(q.shape[-1] - 1, 1))
        attn_spd = torch.abs(torch.matmul(q_center, k_center.transpose(-2, -1)) / denom)
    return attn_spd * scale


class MultiManifoldAttention(nn.Module):
    def __init__(self, dim: int, heads: int = 4) -> None:
        super().__init__()
        if dim % heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by heads ({heads})")
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim ** (-0.5)
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.mix = nn.Conv2d(3, 1, kernel_size=1, bias=False)

    def _mode(self) -> str:
        return "full"

    def _select_attention(
        self,
        attn_euc: torch.Tensor,
        attn_spd: torch.Tensor,
        attn_grass: torch.Tensor,
        b: int,
        heads: int,
        nq: int,
        nk: int,
    ) -> torch.Tensor:
        mode = self._mode()
        if mode in ("full", "mma", "multi", "multi_manifold"):
            stack = torch.stack([attn_euc, attn_spd, attn_grass], dim=1)
            stack = stack.reshape(b * heads, 3, nq, nk)
            return self.mix(stack).reshape(b, heads, nq, nk)
        if mode in ("euc", "euc_only", "euclidean"):
            return attn_euc
        if mode in ("spd", "spd_only"):
            return attn_spd
        if mode in ("grass", "grass_only", "grassmann"):
            return attn_grass
        if mode in ("euc_spd", "spd_euc"):
            return 0.5 * (attn_euc + attn_spd)
        if mode in ("euc_grass", "grass_euc", "euc_grassmann"):
            return 0.5 * (attn_euc + attn_grass)
        if mode in ("spd_grass", "grass_spd", "spd_grassmann"):
            return 0.5 * (attn_spd + attn_grass)
        raise ValueError(
            f"Unsupported MMA_MODE={mode}. Use full/euc/spd/grass/euc_spd/euc_grass/spd_grass."
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.heads, self.head_dim)
        q = qkv[:, :, 0].permute(0, 2, 1, 3)
        k = qkv[:, :, 1].permute(0, 2, 1, 3)
        v = qkv[:, :, 2].permute(0, 2, 1, 3)
        attn_euc = q @ k.transpose(-2, -1) * self.scale
        q_cov = torch.matmul(q.transpose(-2, -1), q) / (n + 1e-06)
        k_cov = torch.matmul(k.transpose(-2, -1), k) / (n + 1e-06)
        spd_bias = -torch.norm(q_cov - k_cov, dim=(-2, -1), keepdim=True)
        attn_spd = attn_euc + spd_bias.expand_as(attn_euc)
        q_norm = F.normalize(q, dim=-1)
        k_norm = F.normalize(k, dim=-1)
        attn_grass = q_norm @ k_norm.transpose(-2, -1)
        attn = self._select_attention(attn_euc, attn_spd, attn_grass, b, self.heads, n, n)
        attn = torch.softmax(attn, dim=-1)
        out = attn @ v
        out = out.transpose(1, 2).reshape(b, n, c)
        return self.proj(out)


class CrossMultiManifoldAttention(nn.Module):
    def __init__(self, dim: int, heads: int = 4) -> None:
        super().__init__()
        if dim % heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by heads ({heads})")
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim ** (-0.5)
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.kv_proj = nn.Linear(dim, dim * 2, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.mix = nn.Conv2d(3, 1, kernel_size=1, bias=False)

    def _mode(self) -> str:
        return "full"

    def _select_attention(
        self,
        attn_euc: torch.Tensor,
        attn_spd: torch.Tensor,
        attn_grass: torch.Tensor,
        b: int,
        heads: int,
        nq: int,
        nk: int,
    ) -> torch.Tensor:
        mode = self._mode()
        if mode in ("full", "mma", "multi", "multi_manifold"):
            stack = torch.stack([attn_euc, attn_spd, attn_grass], dim=1)
            stack = stack.reshape(b * heads, 3, nq, nk)
            return self.mix(stack).reshape(b, heads, nq, nk)
        if mode in ("euc", "euc_only", "euclidean"):
            return attn_euc
        if mode in ("spd", "spd_only"):
            return attn_spd
        if mode in ("grass", "grass_only", "grassmann"):
            return attn_grass
        if mode in ("euc_spd", "spd_euc"):
            return 0.5 * (attn_euc + attn_spd)
        if mode in ("euc_grass", "grass_euc", "euc_grassmann"):
            return 0.5 * (attn_euc + attn_grass)
        if mode in ("spd_grass", "grass_spd", "spd_grassmann"):
            return 0.5 * (attn_spd + attn_grass)
        raise ValueError(
            f"Unsupported MMA_MODE={mode}. Use full/euc/spd/grass/euc_spd/euc_grass/spd_grass."
        )

    def forward(self, q_in: torch.Tensor, kv_in: torch.Tensor) -> torch.Tensor:
        b, nq, c = q_in.shape
        nk = kv_in.shape[1]
        q = self.q_proj(q_in).reshape(b, nq, self.heads, self.head_dim).permute(0, 2, 1, 3)
        kv = self.kv_proj(kv_in).reshape(b, nk, 2, self.heads, self.head_dim)
        k = kv[:, :, 0].permute(0, 2, 1, 3)
        v = kv[:, :, 1].permute(0, 2, 1, 3)
        attn_euc = q @ k.transpose(-2, -1) * self.scale
        q_cov = torch.matmul(q.transpose(-2, -1), q) / (nq + 1e-06)
        k_cov = torch.matmul(k.transpose(-2, -1), k) / (nk + 1e-06)
        spd_bias = -torch.norm(q_cov - k_cov, dim=(-2, -1), keepdim=True)
        attn_spd = attn_euc + spd_bias.expand_as(attn_euc)
        q_norm = F.normalize(q, dim=-1)
        k_norm = F.normalize(k, dim=-1)
        attn_grass = q_norm @ k_norm.transpose(-2, -1)
        attn = self._select_attention(attn_euc, attn_spd, attn_grass, b, self.heads, nq, nk)
        attn = torch.softmax(attn, dim=-1)
        out = attn @ v
        out = out.transpose(1, 2).reshape(b, nq, c)
        return self.proj(out)


class WindowSelfMMA(nn.Module):
    def __init__(
        self,
        in_channels: int,
        embed_dim: int = 64,
        heads: int = 4,
        window_size: int = 64,
        patch_size: int = 4,
    ) -> None:
        super().__init__()
        self.window_size = window_size
        self.patch_size = patch_size
        self.patch_embed = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size, bias=False
        )
        self.attn = MultiManifoldAttention(embed_dim, heads=heads)
        self.patch_recover = nn.ConvTranspose2d(
            embed_dim, in_channels, kernel_size=patch_size, stride=patch_size, bias=False
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        b, c, h, w = feat.shape
        ws = self.window_size
        pad_h = (ws - h % ws) % ws
        pad_w = (ws - w % ws) % ws
        if pad_h > 0 or pad_w > 0:
            feat = F.pad(feat, (0, pad_w, 0, pad_h), mode="reflect")
        _, _, hp, wp = feat.shape
        x = feat.view(b, c, hp // ws, ws, wp // ws, ws)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous().view(-1, c, ws, ws)
        x = self.patch_embed(x)
        e = x.shape[1]
        hs, ws2 = x.shape[-2:]
        x = x.flatten(2).transpose(1, 2)
        x = self.attn(x)
        x = x.transpose(1, 2).reshape(-1, e, hs, ws2)
        x = self.patch_recover(x)
        c_out = x.shape[1]
        x = x.view(b, hp // ws, wp // ws, c_out, ws, ws)
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous().view(b, c_out, hp, wp)
        if pad_h > 0 or pad_w > 0:
            x = x[:, :, :h, :w]
        return x


class WindowCrossMMA(nn.Module):
    def __init__(
        self,
        in_channels: int,
        embed_dim: int = 64,
        heads: int = 4,
        window_size: int = 64,
        patch_size: int = 4,
    ) -> None:
        super().__init__()
        self.window_size = window_size
        self.patch_size = patch_size
        self.q_patch_embed = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size, bias=False
        )
        self.kv_patch_embed = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size, bias=False
        )
        self.attn = CrossMultiManifoldAttention(embed_dim, heads=heads)
        self.patch_recover = nn.ConvTranspose2d(
            embed_dim, in_channels, kernel_size=patch_size, stride=patch_size, bias=False
        )

    def forward(self, q_feat: torch.Tensor, kv_feat: torch.Tensor) -> torch.Tensor:
        b, c, h, w = q_feat.shape
        ws = self.window_size
        pad_h = (ws - h % ws) % ws
        pad_w = (ws - w % ws) % ws
        if pad_h > 0 or pad_w > 0:
            q_feat = F.pad(q_feat, (0, pad_w, 0, pad_h), mode="reflect")
            kv_feat = F.pad(kv_feat, (0, pad_w, 0, pad_h), mode="reflect")
        _, _, hp, wp = q_feat.shape
        q = q_feat.view(b, c, hp // ws, ws, wp // ws, ws)
        q = q.permute(0, 2, 4, 1, 3, 5).contiguous().view(-1, c, ws, ws)
        kv = kv_feat.view(b, c, hp // ws, ws, wp // ws, ws)
        kv = kv.permute(0, 2, 4, 1, 3, 5).contiguous().view(-1, c, ws, ws)
        q = self.q_patch_embed(q)
        kv = self.kv_patch_embed(kv)
        e = q.shape[1]
        hs, ws2 = q.shape[-2:]
        q_tokens = q.flatten(2).transpose(1, 2)
        kv_tokens = kv.flatten(2).transpose(1, 2)
        x = self.attn(q_tokens, kv_tokens)
        x = x.transpose(1, 2).reshape(-1, e, hs, ws2)
        x = self.patch_recover(x)
        c_out = x.shape[1]
        x = x.view(b, hp // ws, wp // ws, c_out, ws, ws)
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous().view(b, c_out, hp, wp)
        if pad_h > 0 or pad_w > 0:
            x = x[:, :, :h, :w]
        return x


class GatedFusionDecoder(nn.Module):
    def __init__(self, feat_channels: int, base_channels: int) -> None:
        super().__init__()
        self.decode = nn.Sequential(
            conv_block(feat_channels, base_channels),
            conv_block(base_channels, base_channels),
            nn.Conv2d(base_channels, 3, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(x)


def _alignment_confidence(rgb_aligned: torch.Tensor, ir: torch.Tensor) -> torch.Tensor:
    rgb_y = rgb_aligned.mean(dim=1, keepdim=True)
    rgb_grad = _gradient_saliency(rgb_y)
    ir_grad = _gradient_saliency(ir)
    photometric = torch.abs(rgb_y - ir)
    structural = torch.abs(rgb_grad - ir_grad)
    conf = torch.exp(-1.5 * photometric - 6.0 * structural)
    return conf.clamp(0.05, 1.0)


def _rgb_to_ycbcr(image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r, g, b = (image[:, 0:1], image[:, 1:2], image[:, 2:3])
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = -0.168736 * r - 0.331264 * g + 0.5 * b + 0.5
    cr = 0.5 * r - 0.418688 * g - 0.081312 * b + 0.5
    return (y, cb, cr)


def _ycbcr_to_rgb(y: torch.Tensor, cb: torch.Tensor, cr: torch.Tensor) -> torch.Tensor:
    cb_shifted = cb - 0.5
    cr_shifted = cr - 0.5
    r = y + 1.402 * cr_shifted
    g = y - 0.344136 * cb_shifted - 0.714136 * cr_shifted
    b = y + 1.772 * cb_shifted
    return torch.cat([r, g, b], dim=1).clamp(0.0, 1.0)


class UnifiedFusionDecoder(nn.Module):
    def __init__(self, in_channels: int, base_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            conv_block(in_channels, base_channels),
            conv_block(base_channels, base_channels),
            conv_block(base_channels, base_channels),
            nn.Conv2d(base_channels, 3, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DualHeadReconstructor(nn.Module):
    def __init__(self, base_channels: int) -> None:
        super().__init__()
        self.bg_recon_head = nn.Sequential(
            conv_block(9, base_channels),
            conv_block(base_channels, base_channels),
            conv_block(base_channels, base_channels),
            nn.Conv2d(base_channels, 3, kernel_size=3, padding=1),
        )
        self.ink_map_head = nn.Sequential(
            conv_block(9, base_channels),
            conv_block(base_channels, base_channels),
            conv_block(base_channels, base_channels),
            nn.Conv2d(base_channels, 1, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )
        self.weight_head = nn.Sequential(
            conv_block(12, base_channels),
            conv_block(base_channels, base_channels),
            nn.Conv2d(base_channels, 3, kernel_size=3, padding=1),
        )
        nn.init.zeros_(self.weight_head[-1].weight)
        with torch.no_grad():
            self.weight_head[-1].bias[:] = torch.tensor([1.6, -0.4, -0.2])

    @staticmethod
    def _gray(x: torch.Tensor) -> torch.Tensor:
        if x.size(1) == 1:
            return x
        return 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]

    def forward(
        self,
        reflect_rgb: torch.Tensor,
        fused_base: torch.Tensor,
        reflect_fused: torch.Tensor,
        reflect_ir: torch.Tensor,
        gate_shared: torch.Tensor,
        gate_rgb: torch.Tensor,
        gate_ir: torch.Tensor,
        mask_guidance: torch.Tensor,
        align_confidence: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        bg_input = torch.cat(
            [reflect_rgb, fused_base, gate_rgb, gate_shared, align_confidence], dim=1
        )
        ir_hint = 0.22 * reflect_ir * (0.85 + 0.15 * align_confidence).repeat(1, 3, 1, 1)
        gate_hint = gate_ir * (0.8 + 0.2 * align_confidence)
        ink_input = torch.cat(
            [reflect_fused, ir_hint, gate_hint, mask_guidance, align_confidence], dim=1
        )
        bg_delta = 0.08 * torch.tanh(self.bg_recon_head(bg_input))
        bg_recon = torch.clamp(reflect_rgb + bg_delta, 0.0, 1.0)
        ink_map = self.ink_map_head(ink_input)
        reflect_ir_y = self._gray(reflect_ir).clamp(0.0, 1.0)
        ir_dark = torch.sigmoid(10.0 * (0.5 - reflect_ir_y))
        ink_strength = torch.clamp(
            ink_map
            * (0.35 + 0.65 * gate_hint)
            * (0.75 + 0.25 * align_confidence)
            * (0.35 + 0.65 * ir_dark),
            0.0,
            1.0,
        )
        bg_y, bg_cb, bg_cr = _rgb_to_ycbcr(bg_recon)
        dual_y = torch.clamp(bg_y - 0.36 * ink_strength, 0.0, 1.0)
        dual_candidate = _ycbcr_to_rgb(dual_y, bg_cb, bg_cr)
        weight_input = torch.cat(
            [reflect_rgb, dual_candidate, reflect_ir, ink_map, gate_hint, align_confidence], dim=1
        )
        weights = torch.softmax(self.weight_head(weight_input), dim=1)
        w_rgb = weights[:, 0:1]
        w_dual = weights[:, 1:2]
        w_ir = weights[:, 2:3]
        rgb_y, rgb_cb, rgb_cr = _rgb_to_ycbcr(reflect_rgb)
        ir_target_y = torch.minimum(dual_y, reflect_ir_y)
        fused_y = w_rgb * rgb_y + w_dual * dual_y + w_ir * ir_target_y
        fused_y = torch.clamp(fused_y - 0.1 * w_ir * ink_strength, 0.0, 1.0)
        fused_dual = _ycbcr_to_rgb(fused_y, rgb_cb, rgb_cr)
        return (fused_dual, bg_recon, ink_map, w_rgb, w_dual, w_ir)


class ReliabilityGuidedLuminanceComposer(nn.Module):
    def __init__(self, base_channels: int = 32) -> None:
        super().__init__()
        in_channels = 11
        self.reliability_head = nn.Sequential(
            conv_block(in_channels, base_channels),
            conv_block(base_channels, base_channels),
            nn.Conv2d(base_channels, 1, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )
        self.delta_head = nn.Sequential(
            conv_block(in_channels + 2, base_channels),
            conv_block(base_channels, base_channels),
            nn.Conv2d(base_channels, 1, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )
        rel_last = self.reliability_head[-2]
        delta_last = self.delta_head[-2]
        nn.init.zeros_(rel_last.weight)
        nn.init.constant_(rel_last.bias, -0.7)
        nn.init.zeros_(delta_last.weight)
        nn.init.constant_(delta_last.bias, -0.4)

    @staticmethod
    def _match_map(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(1)
        if x.size(1) != 1:
            x = x.mean(dim=1, keepdim=True)
        if x.shape[-2:] != ref.shape[-2:]:
            x = F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)
        return x

    @staticmethod
    def _match_img(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        if x.shape[-2:] != ref.shape[-2:]:
            x = F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)
        return x

    def forward(
        self,
        rgb: torch.Tensor,
        ir: torch.Tensor,
        fused_dual: torch.Tensor,
        bg_recon: torch.Tensor,
        ink_map: torch.Tensor,
        gate_ir: torch.Tensor,
        channel_gate_ir: torch.Tensor,
        registration_confidence: torch.Tensor,
        manifold_confidence: torch.Tensor,
        hyper_confidence: torch.Tensor,
        align_confidence: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        rgb = rgb.clamp(0.0, 1.0)
        ir = self._match_map(ir, rgb[:, :1]).clamp(0.0, 1.0)
        fused_dual = self._match_img(fused_dual, rgb).clamp(0.0, 1.0)
        bg_recon = self._match_img(bg_recon, rgb).clamp(0.0, 1.0)
        rgb_y, rgb_cb, rgb_cr = _rgb_to_ycbcr(rgb)
        dual_y, _, _ = _rgb_to_ycbcr(fused_dual)
        bg_y, _, _ = _rgb_to_ycbcr(bg_recon)
        ink_map = self._match_map(ink_map, rgb_y).clamp(0.0, 1.0)
        gate_ir = self._match_map(gate_ir, rgb_y).clamp(0.0, 1.0)
        channel_gate_ir = self._match_map(channel_gate_ir, rgb_y).clamp(0.0, 1.0)
        registration_confidence = self._match_map(registration_confidence, rgb_y).clamp(0.0, 1.0)
        manifold_confidence = self._match_map(manifold_confidence, rgb_y).clamp(0.0, 1.0)
        hyper_confidence = self._match_map(hyper_confidence, rgb_y).clamp(0.0, 1.0)
        align_confidence = self._match_map(align_confidence, rgb_y).clamp(0.0, 1.0)
        grad_rgb = _gradient_saliency(rgb_y)
        grad_ir = _gradient_saliency(ir)
        rgb_dark = torch.sigmoid(10.0 * (0.5 - rgb_y))
        ir_dark = torch.sigmoid(10.0 * (0.5 - ir))
        rgb_score = torch.clamp(0.45 * rgb_dark + 0.55 * grad_rgb, 0.0, 1.0)
        ir_score = torch.clamp(0.45 * ir_dark + 0.55 * grad_ir, 0.0, 1.0)
        ir_better = torch.sigmoid(10.0 * (ir_score - rgb_score + 0.02))
        ir_evidence = torch.clamp(0.6 * ir_dark + 0.4 * grad_ir, 0.0, 1.0)
        rule_gate = ir_better * ir_evidence
        rule_gate = F.avg_pool2d(rule_gate, kernel_size=5, stride=1, padding=2)
        rgb_keep = torch.clamp(rgb_score - ir_score + 0.1, 0.0, 1.0)
        rule_gate = rule_gate * (1.0 - 0.55 * rgb_keep)
        rule_gate = rule_gate.clamp(0.0, 0.95)
        reliability_input = torch.cat(
            [
                rule_gate,
                ink_map,
                gate_ir,
                channel_gate_ir,
                registration_confidence,
                manifold_confidence,
                hyper_confidence,
                align_confidence,
                ir_dark,
                grad_ir,
                grad_rgb,
            ],
            dim=1,
        )
        learned_reliability = self.reliability_head(reliability_input)
        source_gate = rule_gate
        source_gate = source_gate * (0.35 + 0.65 * learned_reliability)
        source_gate = source_gate * (0.7 + 0.3 * ink_map)
        source_gate = source_gate * (0.75 + 0.25 * gate_ir)
        source_gate = source_gate * (0.75 + 0.25 * channel_gate_ir)
        source_gate = source_gate * (0.7 + 0.3 * registration_confidence)
        source_gate = source_gate * (0.75 + 0.25 * manifold_confidence)
        source_gate = source_gate * (0.75 + 0.25 * hyper_confidence)
        source_gate = source_gate * (0.75 + 0.25 * align_confidence)
        source_gate = torch.clamp(source_gate, 0.0, 0.8)
        dual_valid = torch.sigmoid(20.0 * (dual_y.mean(dim=(1, 2, 3), keepdim=True) - 0.08))
        bg_valid = torch.sigmoid(20.0 * (bg_y.mean(dim=(1, 2, 3), keepdim=True) - 0.08))
        rgb_weight = 1.0 - 0.12 * dual_valid - 0.08 * bg_valid
        base_y = torch.clamp(
            rgb_weight * rgb_y + 0.12 * dual_valid * dual_y + 0.08 * bg_valid * bg_y, 0.0, 1.0
        )
        delta_input = torch.cat([reliability_input, base_y, ir], dim=1)
        learned_delta = self.delta_head(delta_input)
        target_y = torch.minimum(rgb_y, ir)
        target_y = torch.clamp(target_y - 0.045 * ir_evidence * learned_delta, 0.0, 1.0)
        fused_y = torch.lerp(base_y, target_y, source_gate)
        fused_y = torch.clamp(fused_y - 0.015 * source_gate * ir_evidence * learned_delta, 0.0, 1.0)
        fused = _ycbcr_to_rgb(fused_y, rgb_cb, rgb_cr)
        return {
            "fused": fused,
            "fused_y": fused_y,
            "base_y": base_y,
            "target_y": target_y,
            "source_gate": source_gate,
            "rule_source_gate": rule_gate,
            "learned_reliability": learned_reliability,
            "learned_delta": learned_delta,
            "ir_evidence": ir_evidence,
            "rgb_score": rgb_score,
            "ir_score": ir_score,
        }


class OutputRefiner(nn.Module):
    def __init__(self, base_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            conv_block(11, base_channels),
            conv_block(base_channels, base_channels),
            nn.Conv2d(base_channels, 3, kernel_size=3, padding=1),
            nn.Tanh(),
        )
        self.scale = 0.04

    def forward(
        self,
        fused_base: torch.Tensor,
        guidance: torch.Tensor,
        reflect_ir: torch.Tensor,
        reflect_rgb: torch.Tensor,
        align_confidence: torch.Tensor,
    ) -> torch.Tensor:
        ir_hint = 0.1 * reflect_ir * (0.25 + 0.75 * align_confidence).repeat(1, 3, 1, 1)
        x = torch.cat([fused_base, ir_hint, reflect_rgb, guidance, align_confidence], dim=1)
        delta = self.net(x)
        refiner_gain = guidance.repeat(1, 3, 1, 1) * (
            0.15 + 0.55 * align_confidence.repeat(1, 3, 1, 1)
        )
        fused = torch.clamp(fused_base + self.scale * refiner_gain * delta, 0.0, 1.0)
        y, cb, cr = _rgb_to_ycbcr(fused)
        style_mask = guidance * (0.25 + 0.75 * align_confidence)
        y = torch.clamp(y - 0.06 * style_mask, 0.0, 1.0)
        chroma_suppress = 0.82 * style_mask
        cb = 0.5 + (cb - 0.5) * (1.0 - chroma_suppress)
        cr = 0.5 + (cr - 0.5) * (1.0 - chroma_suppress)
        return _ycbcr_to_rgb(y, cb, cr)


class SharedPrivateFusionBlock(nn.Module):
    def __init__(
        self,
        feat_channels: int,
        embed_dim: int = 64,
        heads: int = 4,
        window_size: int = 64,
        patch_size: int = 4,
    ) -> None:
        super().__init__()
        hidden = max(16, feat_channels // 2)
        self.rgb_self = WindowSelfMMA(feat_channels, embed_dim, heads, window_size, patch_size)
        self.ir_self = WindowSelfMMA(feat_channels, embed_dim, heads, window_size, patch_size)
        self.rgb_proj = SharedPrivateProjector(feat_channels)
        self.ir_proj = SharedPrivateProjector(feat_channels)
        self.rgb_private_from_ir = WindowCrossMMA(
            feat_channels, embed_dim, heads, window_size, patch_size
        )
        self.ir_private_from_rgb = WindowCrossMMA(
            feat_channels, embed_dim, heads, window_size, patch_size
        )
        self.shared_from_rgb = WindowCrossMMA(
            feat_channels, embed_dim, heads, window_size, patch_size
        )
        self.shared_from_ir = WindowCrossMMA(
            feat_channels, embed_dim, heads, window_size, patch_size
        )
        self.ir_gate_raw = SourceGate2D(feat_channels, hidden)
        self.ir_gate_inv = SourceGate2D(feat_channels, hidden)
        self.rgb_gate_raw = SourceGate2D(feat_channels, hidden)
        self.rgb_gate_inv = SourceGate2D(feat_channels, hidden)
        self.fuse_gate = nn.Sequential(
            conv_block(feat_channels * 4 + 1, feat_channels),
            nn.Conv2d(feat_channels, 3, kernel_size=3, padding=1),
        )
        self.fuse_update = nn.Sequential(
            conv_block(feat_channels * 4 + 1, feat_channels),
            conv_block(feat_channels, feat_channels),
            nn.Conv2d(feat_channels, feat_channels, kernel_size=3, padding=1),
        )
        self.channel_pool = nn.AdaptiveAvgPool2d(1)
        self.channel_gate = nn.Sequential(
            nn.Conv2d(feat_channels * 4, feat_channels, kernel_size=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feat_channels, feat_channels * 3, kernel_size=1),
        )
        self.joint_reduce = nn.Sequential(
            conv_block(feat_channels * 3 + 2, feat_channels),
            conv_block(feat_channels, feat_channels),
            nn.Conv2d(feat_channels, feat_channels, kernel_size=3, padding=1),
        )

    def forward(
        self,
        rgb_feat: torch.Tensor,
        ir_feat: torch.Tensor,
        fused_latent: torch.Tensor,
        rgb_raw: torch.Tensor,
        rgb_inv: torch.Tensor,
        ir_raw: torch.Tensor,
        ir_inv: torch.Tensor,
        guidance: torch.Tensor,
        align_confidence: torch.Tensor,
    ):
        rgb_feat = rgb_feat + 0.2 * self.rgb_self(rgb_feat)
        ir_feat = ir_feat + 0.2 * self.ir_self(ir_feat)
        rgb_shared, rgb_private = self.rgb_proj(rgb_feat)
        ir_shared, ir_private = self.ir_proj(ir_feat)
        g_ir_raw = self.ir_gate_raw(ir_raw)
        g_ir_inv = self.ir_gate_inv(ir_inv)
        g_rgb_raw = self.rgb_gate_raw(rgb_raw)
        g_rgb_inv = self.rgb_gate_inv(rgb_inv)
        rgb_priv_cross = 0.5 * (
            self.rgb_private_from_ir(rgb_private, ir_private * (1.0 + g_ir_raw))
            + self.rgb_private_from_ir(rgb_private, ir_private * (1.0 + g_ir_inv))
        )
        ir_priv_cross = 0.5 * (
            self.ir_private_from_rgb(ir_private, rgb_private * (1.0 + g_rgb_raw))
            + self.ir_private_from_rgb(ir_private, rgb_private * (1.0 + g_rgb_inv))
        )
        shared_query = 0.5 * (rgb_shared + ir_shared) + fused_latent
        shared_consensus = 0.5 * (
            self.shared_from_rgb(shared_query, rgb_shared)
            + self.shared_from_ir(shared_query, ir_shared)
        )
        gate_in = torch.cat(
            [shared_consensus, rgb_priv_cross, ir_priv_cross, fused_latent, guidance], dim=1
        )
        gate_logits = F.softplus(self.fuse_gate(gate_in)) + 0.0001
        gate_shared, gate_rgb, gate_ir = torch.chunk(gate_logits, 3, dim=1)
        gate_sum = gate_shared + gate_rgb + gate_ir + 1e-06
        gate_shared = gate_shared / gate_sum
        gate_rgb = gate_rgb / gate_sum
        gate_ir = gate_ir / gate_sum
        gate_shared_c = gate_shared.repeat(1, shared_consensus.shape[1], 1, 1)
        gate_rgb_c = gate_rgb.repeat(1, rgb_priv_cross.shape[1], 1, 1)
        gate_ir_c = gate_ir.repeat(1, ir_priv_cross.shape[1], 1, 1)
        ir_conf_c = (0.7 + 0.3 * align_confidence).repeat(1, shared_consensus.shape[1], 1, 1)
        fused_mix = (
            gate_shared_c * shared_consensus
            + gate_rgb_c * rgb_priv_cross
            + 0.55 * ir_conf_c * gate_ir_c * ir_priv_cross
        )
        channel_in = torch.cat(
            [shared_consensus, rgb_priv_cross, ir_priv_cross, fused_latent], dim=1
        )
        channel_logits = self.channel_gate(self.channel_pool(channel_in))
        channel_logits = channel_logits.view(fused_latent.size(0), 3, fused_latent.size(1), 1, 1)
        channel_weights = torch.softmax(channel_logits, dim=1)
        ch_shared = channel_weights[:, 0]
        ch_rgb = channel_weights[:, 1]
        ch_ir = channel_weights[:, 2]
        fused_channel = (
            ch_shared * shared_consensus
            + ch_rgb * rgb_priv_cross
            + 0.55 * ch_ir * ir_conf_c * ir_priv_cross
        )
        joint_in = torch.cat(
            [fused_mix, fused_channel, fused_latent, guidance, align_confidence], dim=1
        )
        fused_latent = (
            fused_latent
            + 0.16 * fused_mix
            + 0.16 * fused_channel
            + 0.1 * self.joint_reduce(joint_in)
            + 0.06 * self.fuse_update(gate_in)
        )
        rgb_feat = rgb_feat + 0.08 * rgb_priv_cross
        ir_feat = ir_feat + 0.05 * ir_priv_cross
        h, w = gate_shared.shape[-2:]
        channel_gate_shared = ch_shared.mean(dim=1, keepdim=True).expand(-1, 1, h, w)
        channel_gate_rgb = ch_rgb.mean(dim=1, keepdim=True).expand(-1, 1, h, w)
        channel_gate_ir = ch_ir.mean(dim=1, keepdim=True).expand(-1, 1, h, w)
        gate_maps = {
            "gate_shared": gate_shared,
            "gate_rgb": gate_rgb,
            "gate_ir": gate_ir,
            "channel_gate_shared": channel_gate_shared,
            "channel_gate_rgb": channel_gate_rgb,
            "channel_gate_ir": channel_gate_ir,
            "shared_response": shared_consensus.abs().mean(dim=1, keepdim=True),
            "rgb_from_ir_response": rgb_priv_cross.abs().mean(dim=1, keepdim=True),
            "ir_from_rgb_response": ir_priv_cross.abs().mean(dim=1, keepdim=True),
        }
        return (
            rgb_feat,
            ir_feat,
            fused_latent,
            rgb_shared,
            ir_shared,
            rgb_private,
            ir_private,
            shared_consensus,
            gate_maps,
        )


class ResidualDepthGate(nn.Module):
    def __init__(self, channels: int, hidden_channels: int | None = None) -> None:
        super().__init__()
        hidden = hidden_channels or max(16, channels // 2)
        self.score_net = nn.Sequential(
            nn.Conv2d(channels * 2 + 1, hidden, kernel_size=3, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden, 1, kernel_size=3, padding=1),
        )

    def forward(
        self, current: torch.Tensor, history: list[torch.Tensor], guidance: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        candidates = history + [current]
        if len(candidates) == 1:
            ones = torch.ones(
                current.size(0),
                1,
                current.size(2),
                current.size(3),
                device=current.device,
                dtype=current.dtype,
            )
            return (current, ones, torch.zeros_like(ones))
        scores = []
        for latent in candidates:
            score_in = torch.cat([current, latent, guidance], dim=1)
            scores.append(self.score_net(score_in))
        weights = torch.softmax(torch.cat(scores, dim=1), dim=1)
        fused = 0.0
        depth_gate_map = torch.zeros_like(weights[:, :1])
        denom = max(1, len(candidates) - 1)
        for i, latent in enumerate(candidates):
            w = weights[:, i : i + 1]
            fused = fused + w * latent
            depth_gate_map = depth_gate_map + w * (float(i) / float(denom))
        return (fused, weights, depth_gate_map)


class StructureReliabilityPriorGate(nn.Module):
    def __init__(self, hidden_channels: int = 16) -> None:
        super().__init__()
        in_channels = 10
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden_channels, 1, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )
        nn.init.normal_(self.net[-2].weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.net[-2].bias)

    @staticmethod
    def _to_gray_rgb(rgb: torch.Tensor) -> torch.Tensor:
        if rgb.size(1) == 1:
            return rgb
        return 0.299 * rgb[:, 0:1] + 0.587 * rgb[:, 1:2] + 0.114 * rgb[:, 2:3]

    @staticmethod
    def _match_map(x, ref: torch.Tensor, default: float = 0.0) -> torch.Tensor:
        if x is None or not torch.is_tensor(x):
            return torch.full_like(ref, float(default))
        x = torch.nan_to_num(
            x.detach() if not x.requires_grad else x, nan=0.0, posinf=1.0, neginf=0.0
        )
        if x.dim() == 3:
            x = x.unsqueeze(1)
        if x.size(1) != 1:
            x = x.mean(dim=1, keepdim=True)
        if x.shape[-2:] != ref.shape[-2:]:
            x = F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)
        return x.clamp(0.0, 1.0)

    def forward(
        self,
        rgb: torch.Tensor,
        ir: torch.Tensor,
        registration_confidence=None,
        manifold_confidence=None,
        align_confidence=None,
        mask_guidance=None,
        ink_map=None,
        gate_ir=None,
        channel_gate_ir=None,
    ) -> torch.Tensor:
        rgb = torch.nan_to_num(rgb, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
        ir = torch.nan_to_num(ir, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
        if ir.size(1) != 1:
            ir = ir.mean(dim=1, keepdim=True)
        rgb_y = self._to_gray_rgb(rgb).clamp(0.0, 1.0)
        ir_dark = torch.sigmoid(10.0 * (0.5 - ir))
        grad_ir = _gradient_saliency(ir)
        grad_rgb = _gradient_saliency(rgb_y)
        reg_conf = self._match_map(registration_confidence, ir, default=0.5)
        mani_conf = self._match_map(manifold_confidence, ir, default=0.5)
        align_conf = self._match_map(align_confidence, ir, default=0.5)
        mask_g = self._match_map(mask_guidance, ir, default=0.0)
        ink = self._match_map(ink_map, ir, default=0.0)
        g_ir = self._match_map(gate_ir, ir, default=0.0)
        ch_g_ir = self._match_map(channel_gate_ir, ir, default=0.0)
        x = torch.cat(
            [
                ir_dark,
                grad_ir,
                grad_rgb,
                reg_conf,
                mani_conf,
                align_conf,
                mask_g,
                ink,
                g_ir,
                ch_g_ir,
            ],
            dim=1,
        )
        return self.net(x).clamp(0.0, 1.0)


class IDRNet(nn.Module):
    def __init__(
        self,
        base_channels: int = 32,
        flow_channels: int = 64,
        mma_embed_dim: int = 64,
        mma_heads: int = 4,
        mma_window_size: int = 64,
        mma_patch_size: int = 4,
        fusion_blocks: int = 2,
        vfc_enabled: bool = False,
        vfc_grid_size: int = 16,
        vfc_search_radius: int = 4,
        vfc_kernel_sigma: float = 32.0,
        vfc_max_disp: float = 3.0,
        vfc_iters: int = 3,
        vfc_topk: int = 128,
    ) -> None:
        super().__init__()
        self.structure_prior_gate = StructureReliabilityPriorGate(hidden_channels=16)
        feat_channels = base_channels
        self.coarse_encoder = CoarseReflectanceEncoder(base_channels=base_channels)
        self.affine_aligner = AffineCoarseAligner(
            in_channels=9, base_channels=max(16, flow_channels // 4)
        )
        self.manifold_registration = SharedManifoldRegistration(
            feat_channels=base_channels, embed_dim=mma_embed_dim, heads=mma_heads, token_grid=16
        )
        self.flow_estimator = FlowEstimator(in_channels=9, base_channels=flow_channels // 2)
        self.stroke_refiner = StrokeAwareControlRefiner(
            control_grid=8, radius=2, max_pixels=0.0, temperature=0.1
        )
        self.vfc_refiner = VFCNonRigidRefiner(
            enabled=vfc_enabled,
            grid_size=vfc_grid_size,
            search_radius=vfc_search_radius,
            kernel_sigma=vfc_kernel_sigma,
            max_disp=vfc_max_disp,
            iters=vfc_iters,
            topk=vfc_topk,
            use_mask=True,
        )
        self.rgb_stem = FeatureStem(3, feat_channels)
        self.ir_stem = FeatureStem(3, feat_channels)
        self.fusion_blocks = nn.ModuleList(
            [
                SharedPrivateFusionBlock(
                    feat_channels=feat_channels,
                    embed_dim=mma_embed_dim,
                    heads=mma_heads,
                    window_size=mma_window_size,
                    patch_size=mma_patch_size,
                )
                for _ in range(fusion_blocks)
            ]
        )
        self.rgb_decoder = GatedFusionDecoder(feat_channels, base_channels)
        self.ir_decoder = GatedFusionDecoder(feat_channels, base_channels)
        self.shared_decoder = GatedFusionDecoder(feat_channels, base_channels)
        self.fused_decoder = GatedFusionDecoder(feat_channels, base_channels)
        self.unified_decoder = UnifiedFusionDecoder(in_channels=11, base_channels=base_channels)
        self.dual_head_reconstructor = DualHeadReconstructor(base_channels)
        self.luminance_composer = ReliabilityGuidedLuminanceComposer(base_channels)
        self.output_refiner = OutputRefiner(base_channels)
        self.depth_residual_gate = ResidualDepthGate(feat_channels)

    def forward(
        self, rgb: torch.Tensor, ir: torch.Tensor, mask: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        no_external_mask = mask is None
        if mask is None:
            mask = torch.zeros_like(ir)
        mask = mask.float().clamp(0.0, 1.0)
        if no_external_mask:
            ir_prior = ir.clamp(0.0, 1.0)
            ir_prior_dark = torch.sigmoid(10.0 * (0.45 - ir_prior))
            ir_prior_grad = _gradient_saliency(ir_prior)
            reg_mask_prior = torch.clamp(0.6 * ir_prior_dark + 0.4 * ir_prior_grad, 0.0, 1.0)
        else:
            reg_mask_prior = mask
        coarse_rgb, _ = self.coarse_encoder(rgb)
        reg_out = self.manifold_registration(rgb, ir, reg_mask_prior)
        affine_theta = torch.zeros(rgb.size(0), 2, 3, device=rgb.device, dtype=rgb.dtype)
        affine_theta[:, 0, 0] = 1.0
        affine_theta[:, 1, 1] = 1.0
        ir_affine = warp_affine(ir, affine_theta)
        mask_affine = warp_affine(reg_mask_prior, affine_theta).clamp(0.0, 1.0)
        hyper_conf_affine = warp_affine(reg_out["hyperbolic_confidence"], affine_theta).clamp(
            0.0, 1.0
        )
        ir_affine_rgb = ir_affine.repeat(1, 3, 1, 1)
        vfc_flow, vfc_confidence = self.vfc_refiner(
            rgb=rgb, ir=ir_affine, mask=mask_affine, confidence=hyper_conf_affine
        )
        ir_vfc = warp(ir_affine, vfc_flow)
        mask_vfc = warp(mask_affine, vfc_flow).clamp(0.0, 1.0)
        hyper_conf_vfc = warp(hyper_conf_affine, vfc_flow).clamp(0.0, 1.0)
        ir_refine_base = ir_vfc
        mask_refine_base = mask_vfc
        hyper_conf_refine_base = hyper_conf_vfc
        coarse_ir_affine, _ = self.coarse_encoder(ir_refine_base.repeat(1, 3, 1, 1))
        hyper_focus = F.interpolate(
            hyper_conf_refine_base,
            size=mask_refine_base.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        flow_focus = (0.2 + 0.5 * mask_refine_base + 0.3 * hyper_focus).repeat(1, 3, 1, 1)
        flow_input = torch.cat(
            [
                coarse_ir_affine * flow_focus,
                coarse_rgb * flow_focus,
                (coarse_ir_affine - coarse_rgb) * flow_focus,
            ],
            dim=1,
        )
        raw_flow = torch.zeros(
            rgb.size(0), 2, rgb.size(2), rgb.size(3), device=rgb.device, dtype=rgb.dtype
        )
        corr_flow = vfc_flow
        refine_out = self.stroke_refiner(
            rgb=rgb,
            ir_affine=ir_refine_base,
            mask_affine=mask_refine_base,
            hyper_conf_affine=hyper_conf_refine_base,
            feature_encoder=self.manifold_registration,
            match_confidence=reg_out.get("match_confidence", None),
        )
        refine_flow = refine_out.get("refine_flow", torch.zeros_like(raw_flow))
        flow = vfc_flow + refine_flow
        ir_aligned = warp(ir_affine, flow)
        mask_aligned = warp(mask_affine, flow).clamp(0.0, 1.0)
        hyper_conf_aligned = warp(hyper_conf_affine, flow).clamp(0.0, 1.0)
        registration_confidence = refine_out.get(
            "registration_confidence", torch.ones_like(mask_aligned)
        )
        if registration_confidence.shape[-2:] != mask_aligned.shape[-2:]:
            registration_confidence = F.interpolate(
                registration_confidence,
                size=mask_aligned.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        registration_confidence = torch.minimum(registration_confidence, vfc_confidence).clamp(
            0.05, 1.0
        )
        ir_rgb = ir_aligned.repeat(1, 3, 1, 1)
        rgb_aligned = rgb
        align_confidence = torch.clamp(
            (0.85 + 0.15 * hyper_conf_aligned) * (0.75 + 0.25 * vfc_confidence), 0.05, 1.0
        )
        rgb_manifold, ir_manifold = self.manifold_registration.encode_shared_pair(rgb, ir_rgb)
        ir_saliency = _gradient_saliency(ir_aligned)
        ir_ink = torch.sigmoid(10.0 * (0.45 - ir_aligned))
        if no_external_mask:
            mask_guidance = torch.clamp(
                0.55 * ir_ink + 0.35 * ir_saliency + 0.1 * hyper_conf_aligned, 0.0, 1.0
            )
        else:
            mask_guidance = torch.clamp(
                0.68 * mask_aligned + 0.16 * ir_ink + 0.1 * ir_saliency + 0.06 * hyper_conf_aligned,
                0.0,
                1.0,
            )
        rgb_feat = self.rgb_stem(rgb_aligned)
        ir_feat = self.ir_stem(ir_rgb)
        fused_latent = 0.5 * (rgb_feat + ir_feat)
        latent_history = [fused_latent]
        depth_gate_weights = None
        mask_guidance = torch.clamp(mask_guidance * (0.9 + 0.1 * align_confidence), 0.0, 1.0)
        bg_guidance = 1.0 - mask_guidance
        depth_gate_map = torch.zeros_like(mask_guidance)
        rgb_raw = rgb_aligned.mean(dim=1, keepdim=True)
        rgb_inv = 1.0 - rgb_raw
        ir_raw = ir_aligned
        ir_inv = 1.0 - ir_aligned
        last_rgb_shared = rgb_feat
        last_ir_shared = ir_feat
        last_rgb_private = rgb_feat
        last_ir_private = ir_feat
        last_shared_consensus = fused_latent
        gate_maps = {
            "gate_shared": mask_guidance,
            "gate_rgb": bg_guidance,
            "gate_ir": torch.clamp(mask_guidance * (0.1 + 0.5 * align_confidence), 0.0, 1.0),
        }
        for blk in self.fusion_blocks:
            (
                rgb_feat,
                ir_feat,
                fused_latent,
                last_rgb_shared,
                last_ir_shared,
                last_rgb_private,
                last_ir_private,
                last_shared_consensus,
                gate_maps,
            ) = blk(
                rgb_feat,
                ir_feat,
                fused_latent,
                rgb_raw,
                rgb_inv,
                ir_raw,
                ir_inv,
                mask_guidance,
                align_confidence,
            )
            gate_maps["gate_ir"] = torch.clamp(
                gate_maps["gate_ir"] * (0.85 + 0.15 * align_confidence), 0.0, 1.0
            )
            gate_sum = (
                gate_maps["gate_shared"] + gate_maps["gate_rgb"] + gate_maps["gate_ir"] + 1e-06
            )
            gate_maps["gate_shared"] = gate_maps["gate_shared"] / gate_sum
            gate_maps["gate_rgb"] = gate_maps["gate_rgb"] / gate_sum
            gate_maps["gate_ir"] = gate_maps["gate_ir"] / gate_sum
            raw_fused_latent = fused_latent
            fused_latent, depth_gate_weights, depth_gate_map = self.depth_residual_gate(
                raw_fused_latent, latent_history, mask_guidance
            )
            latent_history.append(raw_fused_latent)
        reflect_rgb = self.rgb_decoder(rgb_feat)
        reflect_ir = self.ir_decoder(ir_feat)
        reflect_shared = self.shared_decoder(last_shared_consensus)
        reflect_fused = self.fused_decoder(fused_latent)
        ir_direct_gain = 0.35 + 0.15 * align_confidence
        fused_base = torch.clamp(
            gate_maps["gate_shared"].repeat(1, 3, 1, 1) * reflect_shared
            + gate_maps["gate_rgb"].repeat(1, 3, 1, 1) * reflect_rgb
            + gate_maps["gate_ir"].repeat(1, 3, 1, 1)
            * ir_direct_gain.repeat(1, 3, 1, 1)
            * reflect_ir,
            0.0,
            1.0,
        )
        ir_aligned_rgb_for_dual = (
            ir_aligned.repeat(1, 3, 1, 1) if ir_aligned.size(1) == 1 else ir_aligned
        )
        dual_rgb_base = rgb_aligned.clamp(0.0, 1.0)
        dual_reflect_fused = torch.clamp(0.55 * reflect_fused + 0.45 * rgb_aligned, 0.0, 1.0)
        dual_reflect_ir = torch.clamp(0.45 * reflect_ir + 0.55 * ir_aligned_rgb_for_dual, 0.0, 1.0)
        fused_dual, bg_recon, ink_map, w_rgb, w_dual, w_ir = self.dual_head_reconstructor(
            reflect_rgb=dual_rgb_base,
            fused_base=fused_base,
            reflect_fused=dual_reflect_fused,
            reflect_ir=dual_reflect_ir,
            gate_shared=gate_maps["gate_shared"],
            gate_rgb=gate_maps["gate_rgb"],
            gate_ir=gate_maps["gate_ir"],
            mask_guidance=mask_guidance,
            align_confidence=align_confidence,
        )
        ir_y_for_dual = ir_aligned.clamp(0.0, 1.0)
        ir_dark_for_dual = torch.sigmoid(10.0 * (0.45 - ir_y_for_dual))
        ir_grad_for_dual = _gradient_saliency(ir_y_for_dual)
        ink_prior_for_dual = torch.clamp(
            0.65 * ir_dark_for_dual + 0.35 * ir_grad_for_dual, 0.0, 1.0
        )
        ink_map = torch.maximum(
            torch.nan_to_num(ink_map, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0),
            0.28 * ink_prior_for_dual.detach(),
        )
        fb_w_ir = torch.clamp(0.04 + 0.34 * ink_prior_for_dual * align_confidence, 0.02, 0.42)
        fb_w_dual = torch.clamp(0.08 + 0.24 * ink_prior_for_dual * align_confidence, 0.04, 0.36)
        fb_w_rgb = torch.clamp(1.0 - fb_w_ir - fb_w_dual, 0.3, 0.94)
        fb_sum = fb_w_rgb + fb_w_dual + fb_w_ir + 1e-06
        fb_w_rgb = fb_w_rgb / fb_sum
        fb_w_dual = fb_w_dual / fb_sum
        fb_w_ir = fb_w_ir / fb_sum
        w_sum = torch.nan_to_num(w_rgb + w_dual + w_ir, nan=0.0, posinf=0.0, neginf=0.0)
        learned_w_valid = (w_sum > 0.5).float()
        w_rgb = learned_w_valid * w_rgb + (1.0 - learned_w_valid) * fb_w_rgb
        w_dual = learned_w_valid * w_dual + (1.0 - learned_w_valid) * fb_w_dual
        w_ir = learned_w_valid * w_ir + (1.0 - learned_w_valid) * fb_w_ir
        bg_y_dual, bg_cb_dual, bg_cr_dual = _rgb_to_ycbcr(bg_recon.clamp(0.0, 1.0))
        dual_ink_strength = torch.clamp(ink_map * (0.45 + 0.55 * w_ir), 0.0, 1.0)
        dual_y_rebuild = torch.clamp(bg_y_dual - 0.3 * dual_ink_strength, 0.0, 1.0)
        fused_dual = _ycbcr_to_rgb(dual_y_rebuild, bg_cb_dual, bg_cr_dual)
        gate_ir_for_composer = gate_maps.get("gate_ir", torch.zeros_like(ir_aligned))
        channel_gate_ir_for_composer = gate_maps.get(
            "channel_gate_ir", torch.zeros_like(ir_aligned)
        )
        registration_conf_for_composer = registration_confidence
        manifold_conf_for_composer = reg_out.get("manifold_pre_confidence", align_confidence)
        hyper_conf_for_composer = hyper_conf_aligned
        composer_out = self.luminance_composer(
            rgb=rgb_aligned.clamp(0.0, 1.0),
            ir=ir_aligned.clamp(0.0, 1.0),
            fused_dual=fused_dual,
            bg_recon=bg_recon,
            ink_map=ink_map,
            gate_ir=gate_ir_for_composer,
            channel_gate_ir=channel_gate_ir_for_composer,
            registration_confidence=registration_conf_for_composer,
            manifold_confidence=manifold_conf_for_composer,
            hyper_confidence=hyper_conf_for_composer,
            align_confidence=align_confidence,
        )
        composer_candidate = composer_out["fused"]
        composer_candidate_y = composer_out["fused_y"]
        source_gate = composer_out["source_gate"]
        rule_source_gate = composer_out["rule_source_gate"]
        learned_reliability = composer_out["learned_reliability"]
        learned_delta = composer_out["learned_delta"]
        ir_evidence = composer_out["ir_evidence"]
        rgb_score = composer_out["rgb_score"]
        ir_score = composer_out["ir_score"]
        target_y = composer_out["target_y"]
        base_y = composer_out["base_y"]
        rgb_safe = rgb_aligned.clamp(0.0, 1.0)
        ir_safe = ir_aligned.clamp(0.0, 1.0)
        rgb_y_safe, rgb_cb_safe, rgb_cr_safe = _rgb_to_ycbcr(rgb_safe)
        source_gate = torch.clamp(source_gate, 0.0, 0.9)
        learned_delta = torch.clamp(learned_delta, 0.0, 1.0)
        ir_evidence = torch.clamp(ir_evidence, 0.0, 1.0)
        dark_amount = source_gate * ir_evidence * (0.1 + 0.3 * learned_delta)
        fused_y = torch.clamp(rgb_y_safe - dark_amount, 0.0, 1.0)
        cand = composer_candidate.clamp(0.0, 1.0)
        cand_y, _, _ = _rgb_to_ycbcr(cand)
        cand_mean = cand_y.mean(dim=(1, 2, 3), keepdim=True)
        cand_valid = torch.sigmoid(40.0 * (cand_mean - 0.12))
        fused_y = torch.lerp(fused_y, cand_y, 0.08 * cand_valid * source_gate)
        fused_y = torch.clamp(fused_y, 0.0, 1.0)
        fused = _ycbcr_to_rgb(fused_y, rgb_cb_safe, rgb_cr_safe)
        fused_before_ir_boost = composer_candidate
        ir_ink_gate = source_gate
        fused_base = fused
        illum_rgb = torch.ones_like(ir)
        illum_ir = torch.ones_like(ir)
        gate_maps_for_prior = locals().get("gate_maps", {})
        if not isinstance(gate_maps_for_prior, dict):
            gate_maps_for_prior = {}
        structure_prior_gate = self.structure_prior_gate(
            rgb=rgb_aligned.clamp(0.0, 1.0),
            ir=ir_aligned.clamp(0.0, 1.0),
            registration_confidence=registration_confidence,
            manifold_confidence=reg_out.get("manifold_pre_confidence", registration_confidence),
            align_confidence=align_confidence,
            mask_guidance=mask_guidance,
            ink_map=ink_map,
            gate_ir=gate_maps_for_prior.get("gate_ir", None),
            channel_gate_ir=gate_maps_for_prior.get("channel_gate_ir", None),
        )
        rgb_safe = rgb_aligned.clamp(0.0, 1.0)
        rgb_y_safe, rgb_cb_safe, rgb_cr_safe = _rgb_to_ycbcr(rgb_safe)
        w_rgb_safe = torch.nan_to_num(w_rgb, nan=1.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
        w_dual_safe = torch.nan_to_num(w_dual, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
        w_ir_safe = torch.nan_to_num(w_ir, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
        source_gate = torch.clamp(
            source_gate * (0.75 + 0.5 * structure_prior_gate) * (0.85 + 0.15 * w_ir_safe), 0.0, 0.9
        )
        cand = composer_candidate.clamp(0.0, 1.0)
        cand_y, _, _ = _rgb_to_ycbcr(cand)
        cand_mean = cand_y.mean(dim=(1, 2, 3), keepdim=True)
        cand_valid = torch.sigmoid(40.0 * (cand_mean - 0.1))
        fused_y = torch.clamp(
            rgb_y_safe - source_gate * ir_evidence * (0.1 + 0.3 * learned_delta), 0.0, 1.0
        )
        dual_blend = torch.clamp(0.04 + 0.18 * w_dual_safe * structure_prior_gate, 0.0, 0.22)
        fused_y = torch.lerp(fused_y, cand_y, cand_valid * dual_blend)
        fused_y = torch.clamp(fused_y, 0.0, 1.0)
        model_candidate = _ycbcr_to_rgb(fused_y, rgb_cb_safe, rgb_cr_safe)
        stable_fused, stable_gate = _no_mask_ir_ink_boost(
            rgb_aligned.clamp(0.0, 1.0),
            ir_aligned.clamp(0.0, 1.0),
            rgb_aligned.clamp(0.0, 1.0),
            strength=2.15,
            prior_gate=structure_prior_gate,
        )
        fused = torch.lerp(model_candidate, stable_fused, 0.55)
        fused = torch.where(torch.isfinite(fused), fused, stable_fused)
        fused = torch.nan_to_num(fused, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
        source_gate = torch.maximum(source_gate, 0.7 * stable_gate).clamp(0.0, 0.95)
        fused_before_ir_boost = model_candidate
        ir_ink_gate = source_gate
        fused_base = fused

        def _safe_optional_tensor(x, ref):
            if x is None:
                return torch.zeros_like(ref)
            if not torch.is_tensor(x):
                return torch.zeros_like(ref)
            return torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0)

        bg_recon = _safe_optional_tensor(locals().get("bg_recon", None), rgb_aligned)
        fused_dual = _safe_optional_tensor(locals().get("fused_dual", None), rgb_aligned)
        ink_map = _safe_optional_tensor(locals().get("ink_map", None), ir_aligned)
        gate_maps = locals().get("gate_maps", {})
        if isinstance(gate_maps, dict):
            gate_maps["gate_ir"] = _safe_optional_tensor(gate_maps.get("gate_ir", None), ir_aligned)
            gate_maps["channel_gate_ir"] = _safe_optional_tensor(
                gate_maps.get("channel_gate_ir", None), ir_aligned
            )
        rgb_eff = rgb_aligned.clamp(0.0, 1.0)
        ir_eff = ir_aligned.clamp(0.0, 1.0)
        if ir_eff.size(1) != 1:
            ir_eff_y = 0.299 * ir_eff[:, 0:1] + 0.587 * ir_eff[:, 1:2] + 0.114 * ir_eff[:, 2:3]
        else:
            ir_eff_y = ir_eff
        rgb_eff_y, rgb_eff_cb, rgb_eff_cr = _rgb_to_ycbcr(rgb_eff)
        ir_dark_eff = torch.sigmoid(10.0 * (0.45 - ir_eff_y))
        ir_grad_eff = _gradient_saliency(ir_eff_y)
        ink_prior_eff = torch.clamp(0.6 * ir_dark_eff + 0.4 * ir_grad_eff, 0.0, 1.0)
        ink_map = torch.nan_to_num(ink_map, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
        ink_map = torch.maximum(ink_map, 0.22 * ink_prior_eff.detach()).clamp(0.0, 1.0)
        bg_recon = torch.nan_to_num(bg_recon, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
        bg_mean = bg_recon.mean(dim=(1, 2, 3), keepdim=True)
        bg_valid = (bg_mean > 0.08).float()
        bg_recon = bg_valid * (0.8 * rgb_eff + 0.2 * bg_recon) + (1.0 - bg_valid) * rgb_eff
        bg_recon = bg_recon.clamp(0.0, 1.0)
        sp_gate_eff = torch.nan_to_num(structure_prior_gate, nan=0.5, posinf=1.0, neginf=0.0).clamp(
            0.0, 1.0
        )
        align_eff = torch.nan_to_num(align_confidence, nan=0.5, posinf=1.0, neginf=0.0).clamp(
            0.0, 1.0
        )
        fb_w_ir = torch.clamp(0.04 + 0.36 * ink_map * (0.6 + 0.4 * align_eff), 0.02, 0.42)
        fb_w_dual = torch.clamp(0.08 + 0.26 * ink_map * (0.6 + 0.4 * sp_gate_eff), 0.04, 0.36)
        fb_w_rgb = torch.clamp(1.0 - fb_w_ir - fb_w_dual, 0.3, 0.94)
        fb_sum = fb_w_rgb + fb_w_dual + fb_w_ir + 1e-06
        fb_w_rgb = fb_w_rgb / fb_sum
        fb_w_dual = fb_w_dual / fb_sum
        fb_w_ir = fb_w_ir / fb_sum
        w_rgb = torch.nan_to_num(w_rgb, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
        w_dual = torch.nan_to_num(w_dual, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
        w_ir = torch.nan_to_num(w_ir, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
        w_sum = w_rgb + w_dual + w_ir
        w_valid = (
            (w_sum.mean(dim=(1, 2, 3), keepdim=True) > 0.5)
            & (w_sum.std(dim=(1, 2, 3), keepdim=True) > 1e-05)
        ).float()
        w_rgb = w_valid * (0.35 * w_rgb + 0.65 * fb_w_rgb) + (1.0 - w_valid) * fb_w_rgb
        w_dual = w_valid * (0.35 * w_dual + 0.65 * fb_w_dual) + (1.0 - w_valid) * fb_w_dual
        w_ir = w_valid * (0.35 * w_ir + 0.65 * fb_w_ir) + (1.0 - w_valid) * fb_w_ir
        w_sum2 = w_rgb + w_dual + w_ir + 1e-06
        w_rgb = w_rgb / w_sum2
        w_dual = w_dual / w_sum2
        w_ir = w_ir / w_sum2
        bg_y_eff, bg_cb_eff, bg_cr_eff = _rgb_to_ycbcr(bg_recon)
        dual_ink_strength = torch.clamp(ink_map * (0.4 + 0.6 * w_ir), 0.0, 1.0)
        dual_y_eff = torch.clamp(bg_y_eff - 0.32 * dual_ink_strength, 0.0, 1.0)
        fused_dual = _ycbcr_to_rgb(dual_y_eff, bg_cb_eff, bg_cr_eff)
        fused = torch.lerp(
            fused.clamp(0.0, 1.0),
            fused_dual.clamp(0.0, 1.0),
            torch.clamp(0.1 + 0.18 * w_dual, 0.0, 0.22),
        ).clamp(0.0, 1.0)
        rgb_struct_eff = _gradient_saliency(rgb_eff_y)
        ir_struct_eff = _gradient_saliency(ir_eff_y)
        shared_response_out = torch.maximum(rgb_struct_eff, ir_struct_eff).clamp(0.0, 1.0)
        rgb_from_ir_response_out = torch.clamp(ink_map * (0.5 + 0.5 * ir_struct_eff), 0.0, 1.0)
        ir_from_rgb_response_out = torch.clamp(rgb_struct_eff * (0.5 + 0.5 * sp_gate_eff), 0.0, 1.0)
        fused_latent_response_out = torch.maximum(
            shared_response_out, rgb_from_ir_response_out
        ).clamp(0.0, 1.0)
        effective_ir_gate = torch.maximum(
            torch.maximum(w_ir, structure_prior_gate.clamp(0.0, 1.0)),
            torch.maximum(ink_map, ir_ink_gate.clamp(0.0, 1.0)),
        ).clamp(0.0, 1.0)
        return {
            "fused": fused,
            "fused_before_ir_boost": fused_before_ir_boost,
            "ir_ink_gate": ir_ink_gate,
            "effective_ir_gate": effective_ir_gate,
            "fused_base": fused_base,
            "flow": flow,
            "vfc_flow": vfc_flow,
            "vfc_confidence": vfc_confidence,
            "ir_vfc": ir_vfc,
            "mask_vfc": mask_vfc,
            "rgb_aligned": rgb_aligned,
            "ir_affine": ir_affine,
            "ir_aligned": ir_aligned,
            "mask_affine": mask_affine,
            "mask_aligned": mask_aligned,
            "hyper_conf_affine": hyper_conf_affine,
            "hyper_conf_aligned": hyper_conf_aligned,
            "raw_flow": raw_flow,
            "refine_flow": refine_flow,
            "ctrl_disp": refine_out["ctrl_disp"],
            "local_match_confidence": refine_out["local_match_confidence"],
            "ctrl_roi": refine_out["ctrl_roi"],
            "corr_flow": corr_flow,
            "theta_corr": reg_out["theta_corr"],
            "match_confidence": reg_out["match_confidence"],
            "match_weight": reg_out["match_weight"],
            "rgb_manifold": rgb_manifold,
            "ir_manifold": ir_manifold,
            "rgb_manifold_pre": reg_out["rgb_manifold_pre"],
            "ir_manifold_pre": reg_out["ir_manifold_pre"],
            "manifold_pre_confidence": reg_out["manifold_pre_confidence"],
            "rgb_hyperbolic": reg_out["rgb_hyperbolic"],
            "ir_hyperbolic": reg_out["ir_hyperbolic"],
            "rgb_hyper_fg": reg_out["rgb_hyper_fg"],
            "ir_hyper_fg": reg_out["ir_hyper_fg"],
            "rgb_hyper_bg": reg_out["rgb_hyper_bg"],
            "ir_hyper_bg": reg_out["ir_hyper_bg"],
            "hyperbolic_confidence": reg_out["hyperbolic_confidence"],
            "hyper_match_bias": reg_out.get("hyper_match_bias", hyper_conf_aligned),
            "hyper_d_fg": reg_out.get("hyper_d_fg", torch.zeros_like(hyper_conf_aligned)),
            "hyper_d_bg": reg_out.get("hyper_d_bg", torch.zeros_like(hyper_conf_aligned)),
            "affine_theta": affine_theta,
            "reflect_rgb": reflect_rgb,
            "reflect_ir": reflect_ir,
            "reflect_shared": reflect_shared,
            "reflect_fused": reflect_fused,
            "bg_recon": bg_recon,
            "ink_map": ink_map,
            "fused_dual": fused_dual,
            "w_rgb": w_rgb,
            "w_dual": w_dual,
            "w_ir": w_ir,
            "illum_rgb": illum_rgb,
            "illum_ir": illum_ir,
            "source_gate": source_gate,
            "structure_prior_gate": structure_prior_gate,
            "rule_source_gate": rule_source_gate,
            "learned_reliability": learned_reliability,
            "learned_delta": learned_delta,
            "ir_evidence": ir_evidence,
            "mask_guidance": mask_guidance,
            "reg_mask_prior": reg_mask_prior,
            "registration_confidence": registration_confidence,
            "manifold_confidence": refine_out.get("manifold_confidence", registration_confidence),
            "reliable_ink": refine_out.get("reliable_ink", mask_guidance),
            "roi_weight": refine_out.get("roi_weight", mask_guidance),
            "local_match_confidence_full": refine_out.get(
                "local_match_confidence_full", registration_confidence
            ),
            "bg_guidance": bg_guidance,
            "align_confidence": align_confidence,
            "gate_shared": gate_maps["gate_shared"],
            "gate_rgb": gate_maps["gate_rgb"],
            "gate_ir": gate_maps["gate_ir"],
            "channel_gate_shared": gate_maps["channel_gate_shared"],
            "channel_gate_rgb": gate_maps["channel_gate_rgb"],
            "channel_gate_ir": gate_maps["channel_gate_ir"],
            "shared_response": shared_response_out,
            "rgb_from_ir_response": rgb_from_ir_response_out,
            "ir_from_rgb_response": ir_from_rgb_response_out,
            "fused_latent_response": fused_latent_response_out,
            "shared_rgb_latent": last_rgb_shared,
            "shared_ir_latent": last_ir_shared,
            "private_rgb_latent": last_rgb_private,
            "private_ir_latent": last_ir_private,
            "shared_consensus_latent": last_shared_consensus,
            "fused_latent": fused_latent,
            "depth_gate_map": depth_gate_map,
            "depth_gate_weights": depth_gate_weights,
            "alpha": gate_maps["gate_ir"],
        }
