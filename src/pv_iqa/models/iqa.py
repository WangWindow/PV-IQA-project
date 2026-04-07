from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .backbone import IQABackbone
from .layers.attention import TransposedAttentionBlock
from .layers.mixer import LocalWindowMixer


class LightweightIQARegressor(nn.Module):
    def __init__(self, backbone_name: str, pretrained: bool) -> None:
        super().__init__()
        self.backbone = IQABackbone(
            backbone_name,
            pretrained=pretrained,
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
        # 低层特征保留纹理细节，高层特征提供更稳定的结构语义。
        low_feature, _, high_feature = self.backbone(image)
        low_feature = self.local_mixer(low_feature)
        high_feature = self.deep_attention(high_feature)

        local = self.align_local(low_feature)
        deep = self.align_deep(high_feature)
        local = F.adaptive_avg_pool2d(local, deep.shape[-2:])
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
