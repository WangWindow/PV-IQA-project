from __future__ import annotations

import math
from dataclasses import dataclass

import timm
import torch
import torch.nn.functional as F
from torch import nn


class ArcMarginHead(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
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

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor | None) -> torch.Tensor:
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
    ) -> None:
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
        )
        feature_dim = self._infer_feature_dim()
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

    def _infer_feature_dim(self) -> int:
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            features = self.backbone(dummy)
        return int(features.shape[1])

    def forward(
        self,
        image: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> RecognitionOutputs:
        features = self.backbone(image)
        embeddings = F.normalize(self.embedding(features), dim=1)
        logits = self.head(embeddings, labels)
        return RecognitionOutputs(embeddings=embeddings, logits=logits)
