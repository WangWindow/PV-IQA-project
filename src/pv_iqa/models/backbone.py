from __future__ import annotations

from dataclasses import dataclass

import timm
import torch
from torch import nn


@dataclass(slots=True)
class TimmBackboneSpec:
    """使用 timm 创建 backbone 时的规格"""

    name: str
    pretrained: bool


class RecognitionBackbone(nn.Module):
    """识别分支使用的 timm 全局特征 backbone 封装。"""

    def __init__(self, backbone_name: str, *, pretrained: bool) -> None:
        super().__init__()
        self.spec = TimmBackboneSpec(name=backbone_name, pretrained=pretrained)
        self.model = timm.create_model(
            model_name=backbone_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.model(image)


class IQABackbone(nn.Module):
    """IQA 分支使用的多尺度特征 backbone 封装。"""

    def __init__(
        self,
        backbone_name: str,
        *,
        pretrained: bool,
        out_indices: tuple[int, ...],
    ) -> None:
        super().__init__()
        self.spec = TimmBackboneSpec(name=backbone_name, pretrained=pretrained)
        self.out_indices = out_indices
        self.model = timm.create_model(
            model_name=backbone_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=out_indices,
        )

    @property
    def feature_info(self):  # noqa: ANN201 - timm feature_info is dynamic
        return self.model.feature_info

    def forward(self, image: torch.Tensor):
        return self.model(image)
