from __future__ import annotations

import timm
import torch
from torch import nn


class IQABackbone(nn.Module):
    """IQA 分支使用的多尺度特征 backbone 封装。"""

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
            model_name=backbone_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=out_indices,
        )

    @property
    def feature_info(self):
        return self.model.feature_info

    def forward(self, image: torch.Tensor):
        return self.model(image)


class PalmVeinIQARegressor(nn.Module):
    def __init__(self, backbone_name: str, pretrained: bool = True) -> None:
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name, pretrained=pretrained, num_classes=0, global_pool="avg"
        )
        dim = self._infer_dim()
        self.head = nn.Linear(dim, 1)

    def _infer_dim(self) -> int:
        with torch.no_grad():
            return int(self.backbone(torch.zeros(1, 3, 224, 224)).shape[1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 100.0 * torch.sigmoid(self.head(self.backbone(x)).squeeze(-1))
