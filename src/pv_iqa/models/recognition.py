from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from .backbone import RecognitionBackbone
from .layers.metric_head import ArcMarginHead


def infer_feature_dim(
    backbone: nn.Module,
    *,
    image_size: int = 224,
    channels: int = 3,
) -> int:
    """用一次 dummy forward 推断识别 backbone 输出维度。"""
    with torch.no_grad():
        dummy = torch.zeros(1, channels, image_size, image_size)
        features = backbone(dummy)
    return int(features.shape[1])


@dataclass(slots=True)
class RecognitionOutputs:
    embeddings: torch.Tensor
    logits: torch.Tensor


class PalmVeinRecognizer(nn.Module):
    def __init__(
        self,
        backbone_name: str,
        num_classes: int,
        embedding_dim: int,
        dropout: float,
        margin: float,
        scale: float,
        pretrained: bool,
        image_size: int = 224,
    ) -> None:
        super().__init__()
        self.backbone = RecognitionBackbone(backbone_name, pretrained=pretrained)
        feature_dim = infer_feature_dim(self.backbone, image_size=image_size)
        self.embedding = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
        )
        self.head = ArcMarginHead(
            in_features=embedding_dim,
            out_features=num_classes,
            scale=scale,
            margin=margin,
        )

    def forward(
        self,
        image: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> RecognitionOutputs:
        # 先提取全局身份特征，再做归一化嵌入和 ArcFace 分类。
        features = self.backbone(image)
        embeddings = F.normalize(self.embedding(features), dim=1)
        logits = self.head(embeddings, labels)
        return RecognitionOutputs(embeddings=embeddings, logits=logits)
