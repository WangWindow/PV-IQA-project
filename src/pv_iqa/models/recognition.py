from __future__ import annotations

import math

import timm
import torch
import torch.nn.functional as F
from torch import nn


class RecognitionBackbone(nn.Module):
    """识别分支使用的全局特征 backbone 封装。"""

    def __init__(self, backbone_name="mobilenetv3_large_100", *, pretrained: bool):
        super().__init__()
        self.model = timm.create_model(
            model_name=backbone_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
        )

    def forward(self, image: torch.Tensor):
        return self.model(image)


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


class ArcMarginHead(nn.Module):
    """识别分支使用的 ArcFace 分类头。"""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        scale: float,
        margin: float,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.scale = scale
        self.margin = margin
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

    def forward(
        self, embeddings: torch.Tensor, labels: torch.Tensor | None
    ) -> torch.Tensor:
        cosine = F.linear(F.normalize(embeddings), F.normalize(self.weight))
        if labels is None:
            return cosine * self.scale

        sine = torch.sqrt(torch.clamp(1.0 - cosine.pow(2), min=0.0))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)
        logits = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        return logits * self.scale


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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # 先提取全局身份特征，再做归一化嵌入和 ArcFace 分类。
        features = self.backbone(image)
        embeddings = F.normalize(self.embedding(features), dim=1)
        logits = self.head(embeddings, labels)
        return (embeddings, logits)
