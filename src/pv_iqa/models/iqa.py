from __future__ import annotations

import timm
import torch
from torch import nn


class LocalWindowMixer(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=5, padding=2, groups=channels),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class TransposedAttentionBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = nn.BatchNorm2d(channels)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        qkv = self.qkv(self.norm(x))
        q, k, v = qkv.chunk(3, dim=1)
        q = q.flatten(2)
        k = k.flatten(2)
        v = v.flatten(2)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = torch.softmax((q @ k.transpose(1, 2)) * self.temperature, dim=-1)
        fused = (attn @ v).view(batch, channels, height, width)
        return x + self.proj(fused)


class LightweightIQARegressor(nn.Module):
    def __init__(self, backbone_name: str, pretrained: bool) -> None:
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=(1, 2, 4),
        )
        channels = self.backbone.feature_info.channels()
        self.local_mixer = LocalWindowMixer(channels[0])
        self.deep_attention = TransposedAttentionBlock(channels[-1])
        self.align_local = nn.Sequential(
            nn.Conv2d(channels[0], 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
        )
        self.align_deep = nn.Sequential(
            nn.Conv2d(channels[-1], 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.GELU(),
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(384, 192, kernel_size=1),
            nn.BatchNorm2d(192),
            nn.GELU(),
            nn.Conv2d(192, 96, kernel_size=3, padding=1),
            nn.BatchNorm2d(96),
            nn.GELU(),
        )
        self.score_head = nn.Conv2d(96, 1, kernel_size=1)
        self.weight_head = nn.Sequential(nn.Conv2d(96, 1, kernel_size=1), nn.Sigmoid())

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        low_feature, _, high_feature = self.backbone(image)
        low_feature = self.local_mixer(low_feature)
        high_feature = self.deep_attention(high_feature)

        local = self.align_local(low_feature)
        deep = self.align_deep(high_feature)
        local = torch.nn.functional.adaptive_avg_pool2d(local, deep.shape[-2:])
        fused = self.fusion(torch.cat([local, deep], dim=1))

        score_map = self.score_head(fused)
        weight_map = self.weight_head(fused)
        weighted_score = (score_map * weight_map).sum(dim=(2, 3))
        normalizer = weight_map.sum(dim=(2, 3)).clamp_min(1e-6)
        score = (weighted_score / normalizer).squeeze(1)
        return {
            "score": score,
            "score_map": score_map.squeeze(1),
            "weight_map": weight_map.squeeze(1),
        }
