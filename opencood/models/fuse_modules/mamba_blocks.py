import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict


class SimpleMamba1D(nn.Module):
    """
    A lightweight Mamba-style 1D mixing block without external deps.
    Operates on [B, C, T]. Uses gated depthwise conv + 1x1 conv projections.
    """
    def __init__(self, channels: int, kernel_size: int = 7, drop_path: float = 0.0):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        padding = kernel_size // 2
        self.in_proj = nn.Conv1d(channels, channels * 2, kernel_size=1, bias=True)
        self.dwconv = nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding, groups=channels, bias=True)
        self.out_proj = nn.Conv1d(channels, channels, kernel_size=1, bias=True)
        self.norm = nn.BatchNorm1d(channels)
        self.dropout = nn.Dropout(p=drop_path) if drop_path and drop_path > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T]
        residual = x
        x = self.norm(x)
        x = self.in_proj(x)
        x1, x2 = x.chunk(2, dim=1)
        x = x1 * torch.sigmoid(x2)
        x = self.dwconv(x)
        x = self.out_proj(x)
        x = self.dropout(x)
        return x + residual


class BiMamba1D(nn.Module):
    """Bidirectional 1D Mamba: run forward and backward, then average."""
    def __init__(self, channels: int, kernel_size: int = 7, drop_path: float = 0.0):
        super().__init__()
        self.fwd = SimpleMamba1D(channels, kernel_size, drop_path)
        self.bwd = SimpleMamba1D(channels, kernel_size, drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T]
        out_f = self.fwd(x)
        # reverse sequence length dim
        out_b = self.bwd(torch.flip(x, dims=[-1]))
        out_b = torch.flip(out_b, dims=[-1])
        return 0.5 * (out_f + out_b)


def _next_pow2(n: int) -> int:
    return 1 if n <= 1 else 1 << (int(n - 1).bit_length())


def _rot(n: int, x: int, y: int, rx: int, ry: int) -> Tuple[int, int]:
    if ry == 0:
        if rx == 1:
            x = n - 1 - x
            y = n - 1 - y
        x, y = y, x
    return x, y


def d2xy(n: int, d: int) -> Tuple[int, int]:
    # Convert Hilbert distance d to (x, y) for an n x n grid, n is power of 2
    x = y = 0
    t = d
    s = 1
    while s < n:
        rx = 1 & (t // 2)
        ry = 1 & (t ^ rx)
        x, y = _rot(s, x, y, rx, ry)
        x += s * rx
        y += s * ry
        t //= 4
        s *= 2
    return x, y


class HilbertSerializer:
    """
    Generates and caches Hilbert permutation for (H, W) grids.
    """
    def __init__(self):
        self.cache: Dict[Tuple[int, int], Tuple[torch.Tensor, torch.Tensor]] = {}

    def get_perm(self, H: int, W: int, device=None) -> Tuple[torch.Tensor, torch.Tensor]:
        key = (H, W)
        if key in self.cache:
            perm, inv = self.cache[key]
            return perm.to(device=device), inv.to(device=device)
        n = _next_pow2(max(H, W))
        idx_list = []
        for d in range(n * n):
            x, y = d2xy(n, d)
            if x < W and y < H:
                idx_list.append(y * W + x)
        perm = torch.tensor(idx_list, dtype=torch.long)
        inv = torch.empty(H * W, dtype=torch.long)
        inv[perm] = torch.arange(len(perm))
        self.cache[key] = (perm, inv)
        return perm.to(device=device), inv.to(device=device)


class LocalMamba2D(nn.Module):
    """
    Apply bidirectional Mamba within non-overlapping WxW windows on 2D feature maps.
    Input: [B, C, H, W] -> Output: [B, C, H, W]
    """
    def __init__(self, channels: int, window_size: int = 8, kernel_size: int = 7, drop_path: float = 0.0):
        super().__init__()
        self.window = window_size
        self.mix = BiMamba1D(channels, kernel_size, drop_path)
        self.norm2d = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        w = self.window
        # pad to multiples of w if necessary
        pad_h = (w - H % w) % w
        pad_w = (w - W % w) % w
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        Hp, Wp = x.shape[-2:]
        nh, nw = Hp // w, Wp // w
        # [B, C, nh, w, nw, w] -> [B*nh*nw, C, w*w]
        x_blocks = x.view(B, C, nh, w, nw, w).permute(0, 2, 4, 1, 3, 5).contiguous().view(B * nh * nw, C, w * w)
        x_blocks = self.mix(x_blocks)
        # reshape back
        x_blocks = x_blocks.view(B, nh, nw, C, w, w).permute(0, 3, 1, 4, 2, 5).contiguous().view(B, C, Hp, Wp)
        if pad_h or pad_w:
            x_blocks = x_blocks[:, :, :H, :W]
        x_blocks = self.norm2d(x_blocks)
        return x_blocks


class GlobalMamba2D(nn.Module):
    """
    Serialize 2D features with Hilbert curve then apply bidirectional Mamba.
    """
    def __init__(self, channels: int, kernel_size: int = 7, drop_path: float = 0.0):
        super().__init__()
        self.mix = BiMamba1D(channels, kernel_size, drop_path)
        self.hilbert = HilbertSerializer()
        self.norm2d = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        B, C, H, W = x.shape
        device = x.device
        perm, inv = self.hilbert.get_perm(H, W, device=device)
        seq = x.view(B, C, H * W).index_select(dim=2, index=perm)
        seq = self.mix(seq)
        seq = seq.index_select(dim=2, index=inv)
        out = seq.view(B, C, H, W)
        out = self.norm2d(out)
        return out
