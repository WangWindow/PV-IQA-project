from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


class ChannelLayerNorm2d(nn.Module):
    def __init__(self, channels: int, *, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=1, keepdim=True)
        centered = x - mean
        variance = centered.square().mean(dim=1, keepdim=True)
        normalized = centered / torch.sqrt(variance + self.eps)
        return normalized * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)


class ChannelLastLayerNorm(nn.Module):
    def __init__(self, channels: int, *, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        centered = x - mean
        variance = centered.square().mean(dim=-1, keepdim=True)
        normalized = centered / torch.sqrt(variance + self.eps)
        return normalized * self.weight.view(1, 1, 1, -1) + self.bias.view(1, 1, 1, -1)


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
        self.output_norm = ChannelLastLayerNorm(channels)
        self.to_phase = nn.Sequential(
            nn.Linear(channels, channels),
            nn.GELU(),
        )
        self.wave_speed = nn.Parameter(torch.ones(1))
        self.damping = nn.Parameter(torch.full((1,), 0.1))
        self.frequency_embed = nn.Parameter(
            torch.zeros(base_resolution, base_resolution, channels)
        )
        self.register_buffer(
            "cos_h",
            self._build_cosine_basis(base_resolution),
            persistent=False,
        )
        self.register_buffer(
            "cos_w",
            self._build_cosine_basis(base_resolution),
            persistent=False,
        )
        nn.init.trunc_normal_(self.frequency_embed, std=0.02)

    @staticmethod
    def _build_cosine_basis(size: int) -> torch.Tensor:
        positions = (torch.arange(size, dtype=torch.float32).view(1, -1) + 0.5) / size
        frequencies = torch.arange(size, dtype=torch.float32).view(-1, 1)
        basis = torch.cos(frequencies * positions * math.pi) * math.sqrt(2.0 / size)
        basis[0, :] = basis[0, :] / math.sqrt(2.0)
        return basis

    def _resolve_basis(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        height, width = x.shape[1:3]
        if height != self.base_resolution or width != self.base_resolution:
            raise ValueError(
                f"WaveFormer expects {self.base_resolution}x{self.base_resolution} features, "
                f"received {height}x{width}."
            )
        return self.cos_h.to(dtype=x.dtype), self.cos_w.to(dtype=x.dtype)

    def _dct2d(self, x: torch.Tensor) -> torch.Tensor:
        batch, height, width, channels = x.shape
        cos_h, cos_w = self._resolve_basis(x)
        transformed = (
            x.permute(0, 2, 3, 1)
            .contiguous()
            .reshape(batch * width * channels, height)
            .matmul(cos_h.transpose(0, 1))
            .reshape(batch, width, channels, height)
            .permute(0, 3, 1, 2)
            .contiguous()
        )
        return (
            transformed.permute(0, 1, 3, 2)
            .contiguous()
            .reshape(batch * height * channels, width)
            .matmul(cos_w.transpose(0, 1))
            .reshape(batch, height, channels, width)
            .permute(0, 1, 3, 2)
            .contiguous()
        )

    def _idct2d(self, x: torch.Tensor) -> torch.Tensor:
        batch, height, width, channels = x.shape
        cos_h, cos_w = self._resolve_basis(x)
        reconstructed = (
            x.permute(0, 1, 3, 2)
            .contiguous()
            .reshape(batch * height * channels, width)
            .matmul(cos_w)
            .reshape(batch, height, channels, width)
            .permute(0, 1, 3, 2)
            .contiguous()
        )
        return (
            reconstructed.permute(0, 2, 3, 1)
            .contiguous()
            .reshape(batch * width * channels, height)
            .matmul(cos_h)
            .reshape(batch, width, channels, height)
            .permute(0, 3, 1, 2)
            .contiguous()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        mixed = self.depthwise(x).permute(0, 2, 3, 1).contiguous()
        carrier, gate = self.proj_in(mixed).chunk(2, dim=-1)
        spectral = self._dct2d(carrier)
        phase = self.to_phase(self.frequency_embed).unsqueeze(0).expand(batch, -1, -1, -1)
        wave_phase = self.wave_speed * phase

        wave_term = torch.cos(wave_phase) * spectral
        velocity_scale = torch.sin(wave_phase) / self.wave_speed.abs().clamp_min(1e-6)
        velocity_term = velocity_scale * (spectral + 0.5 * self.damping * spectral)

        fused = self._idct2d(wave_term + velocity_term)
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
        self.wave_norm = ChannelLayerNorm2d(channels)
        self.wave = WaveOperator2D(channels, base_resolution=base_resolution)
        self.ffn_norm = ChannelLayerNorm2d(channels)
        self.ffn = FeedForward2d(channels, mlp_ratio=mlp_ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.wave(self.wave_norm(x))
        return x + self.ffn(self.ffn_norm(x))


__all__ = ["WaveFormerLayer"]
