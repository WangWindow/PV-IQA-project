from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


class LayerNorm2d(nn.LayerNorm):
    """LayerNorm for BCHW tensors."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalized = x.permute(0, 2, 3, 1).contiguous()
        normalized = F.layer_norm(
            normalized,
            self.normalized_shape,
            self.weight,
            self.bias,
            self.eps,
        )
        return normalized.permute(0, 3, 1, 2).contiguous()


class FeedForward2d(nn.Module):
    def __init__(self, channels: int, *, mlp_ratio: float) -> None:
        super().__init__()
        hidden_channels = max(channels, int(channels * mlp_ratio))
        self.block = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class WaveOperator2D(nn.Module):
    """A lightweight WaveFormer-style frequency-time operator."""

    def __init__(self, channels: int, *, base_resolution: int) -> None:
        super().__init__()
        self.channels = channels
        self.base_resolution = base_resolution
        self.depthwise = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            groups=channels,
        )
        self.proj_in = nn.Linear(channels, channels * 2)
        self.proj_out = nn.Linear(channels, channels)
        self.output_norm = nn.LayerNorm(channels)
        self.to_phase = nn.Sequential(
            nn.Linear(channels, channels),
            nn.GELU(),
        )
        self.wave_speed = nn.Parameter(torch.ones(1))
        self.damping = nn.Parameter(torch.full((1,), 0.1))
        self.frequency_embed = nn.Parameter(
            torch.zeros(base_resolution, base_resolution, channels)
        )
        nn.init.trunc_normal_(self.frequency_embed, std=0.02)

    @staticmethod
    def _cosine_basis(
        size: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        positions = (torch.arange(size, device=device, dtype=dtype).view(1, -1) + 0.5) / size
        frequencies = torch.arange(size, device=device, dtype=dtype).view(-1, 1)
        basis = torch.cos(frequencies * positions * math.pi) * math.sqrt(2.0 / size)
        basis[0, :] = basis[0, :] / math.sqrt(2.0)
        return basis

    @staticmethod
    def _dct2d(
        x: torch.Tensor,
        cos_h: torch.Tensor,
        cos_w: torch.Tensor,
    ) -> torch.Tensor:
        transformed = torch.einsum("bhwc,nh->bnwc", x, cos_h)
        return torch.einsum("bnwc,mw->bnmc", transformed, cos_w)

    @staticmethod
    def _idct2d(
        x: torch.Tensor,
        cos_h: torch.Tensor,
        cos_w: torch.Tensor,
    ) -> torch.Tensor:
        reconstructed = torch.einsum("bnmc,wm->bnwc", x, cos_w.transpose(0, 1))
        return torch.einsum("bnwc,hn->bhwc", reconstructed, cos_h.transpose(0, 1))

    def _resize_frequency_embed(
        self,
        height: int,
        width: int,
    ) -> torch.Tensor:
        freq = self.frequency_embed.permute(2, 0, 1).unsqueeze(0)
        if freq.shape[-2:] != (height, width):
            freq = F.interpolate(
                freq,
                size=(height, width),
                mode="bicubic",
                align_corners=False,
            )
        return freq.squeeze(0).permute(1, 2, 0).contiguous()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, _, height, width = x.shape
        mixed = self.depthwise(x).permute(0, 2, 3, 1).contiguous()
        carrier, gate = self.proj_in(mixed).chunk(2, dim=-1)

        cos_h = self._cosine_basis(height, device=x.device, dtype=x.dtype)
        cos_w = self._cosine_basis(width, device=x.device, dtype=x.dtype)
        spectral = self._dct2d(carrier, cos_h, cos_w)

        phase_embed = self._resize_frequency_embed(height, width)
        phase = self.to_phase(phase_embed).unsqueeze(0).expand(batch, -1, -1, -1)
        wave_phase = self.wave_speed * phase

        wave_term = torch.cos(wave_phase) * spectral
        velocity_scale = torch.sin(wave_phase) / self.wave_speed.abs().clamp_min(1e-6)
        velocity_term = velocity_scale * (spectral + 0.5 * self.damping * spectral)

        fused = self._idct2d(wave_term + velocity_term, cos_h, cos_w)
        fused = self.output_norm(fused)
        fused = fused * F.silu(gate)
        return self.proj_out(fused).permute(0, 3, 1, 2).contiguous()


class WaveFormerLayer(nn.Module):
    """A residual WaveFormer plugin layer for IQA feature refinement."""

    def __init__(
        self,
        channels: int,
        *,
        base_resolution: int,
        mlp_ratio: float,
    ) -> None:
        super().__init__()
        self.wave_norm = LayerNorm2d(channels)
        self.wave = WaveOperator2D(channels, base_resolution=base_resolution)
        self.ffn_norm = LayerNorm2d(channels)
        self.ffn = FeedForward2d(channels, mlp_ratio=mlp_ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.wave(self.wave_norm(x))
        return x + self.ffn(self.ffn_norm(x))


__all__ = ["WaveFormerLayer"]
