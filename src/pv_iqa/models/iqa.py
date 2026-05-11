from __future__ import annotations

import timm
import torch
from torch import nn


class IQABackbone(nn.Module):
    """Multi-scale feature extractor for IQA. Wraps timm with features_only=True."""

    def __init__(
        self,
        backbone_name: str,
        *,
        pretrained: bool,
        out_indices: tuple[int, ...],
    ):
        super().__init__()
        self.out_indices = out_indices
        self.model = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=out_indices,
        )

    @property
    def feature_info(self):
        return self.model.feature_info

    def forward(self, image: torch.Tensor):
        return self.model(image)


class SEAttention(nn.Module):
    """Squeeze-Excitation channel attention."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        reduced = max(8, channels // reduction)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, reduced, 1),
            nn.ReLU(),
            nn.Conv2d(reduced, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.se(x)


class PalmVeinIQARegressor(nn.Module):
    """Multi-scale IQA regressor.

    backbone → {stage1, stage3, stage4}
      stage1 → Conv3×3 → GAP → 32d      (texture/sharpness)
      stage3 → Conv1×1 → GAP → 64d      (mid-level structure)
      stage4 → SE-Attn → GAP → ch4      (semantic quality)
    concat → FC(128) → FC(1)   (linear output, no activation)
    """

    def __init__(
        self,
        backbone_name: str,
        *,
        pretrained: bool = True,
        out_indices: tuple[int, ...] = (1, 3, 4),
    ):
        super().__init__()
        self.backbone = IQABackbone(
            backbone_name, pretrained=pretrained, out_indices=out_indices,
        )

        with torch.no_grad():
            dims = [
                f.shape[1]
                for f in self.backbone(torch.zeros(1, 3, 224, 224))
            ]

        self.branch1 = nn.Sequential(
            nn.Conv2d(dims[0], 32, 3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(dims[1], 64, 1),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        )
        self.branch4 = nn.Sequential(
            SEAttention(dims[2]),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        )

        self.head = nn.Sequential(
            nn.Linear(32 + 64 + dims[2], 128),
            nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        f1 = self.branch1(feats[0])
        f3 = self.branch3(feats[1])
        f4 = self.branch4(feats[2])
        return self.head(torch.cat([f1, f3, f4], dim=1)).squeeze(-1)
