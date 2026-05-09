from __future__ import annotations

import timm
import torch
from torch import nn


class IQARegressor(nn.Module):
    def __init__(self, backbone_name: str, pretrained: bool = True) -> None:
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0, global_pool="avg")
        dim = self._infer_dim()
        self.head = nn.Linear(dim, 1)

    def _infer_dim(self) -> int:
        with torch.no_grad():
            return int(self.backbone(torch.zeros(1, 3, 224, 224)).shape[1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 100.0 * torch.sigmoid(self.head(self.backbone(x)).squeeze(-1))
