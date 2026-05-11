from __future__ import annotations

import math

import timm
import torch
import torch.nn.functional as F
from torch import nn


# ---------------------------------------------------------------------------
# ArcFace (Additive Angular Margin Loss)
#   Deng et al., "ArcFace: Additive Angular Margin Loss for Deep Face
#   Recognition", CVPR 2019.
#
#   logits_i = s × ( onehot_i × cos(θ + m) + (1 − onehot_i) × cos(θ) )
#
#   where:
#     θ = arccos(cos) = angle between embedding and weight vector
#     m = angular margin penalty (default 0.5)
#     s = feature scale (default 64)
#
#   The margin penalty forces intra-class embeddings closer and inter-class
#   embeddings farther apart on the hypersphere, which enables Q^P (intra-class
#   cosine similarity) to serve as a quality metric.
# ---------------------------------------------------------------------------
class ArcMarginHead(nn.Module):
    """ArcFace classification head (Deng et al., CVPR 2019)."""

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
        # Cosine similarity between L2-normalized embedding and weight vectors
        cosine = F.linear(F.normalize(embeddings), F.normalize(self.weight))

        if labels is None:
            return cosine * self.scale

        # cos(θ + m) = cos(θ)cos(m) − sin(θ)sin(m)
        sine = torch.sqrt(torch.clamp(1.0 - cosine.pow(2), min=0.0))
        phi = cosine * self.cos_m - sine * self.sin_m

        # Numeric stability: when cos(θ + m) < cos(π − m), use cos(θ) − m·sin(m)
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # One-hot mask: use cos(θ + m) for target class, cos(θ) for others
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)
        logits = (one_hot * phi) + ((1.0 - one_hot) * cosine)

        return logits * self.scale


class RecognitionBackbone(nn.Module):
    """Global feature backbone for palm vein recognition."""

    def __init__(
        self, backbone_name: str = "mobilenetv3_large_100", *, pretrained: bool
    ):
        super().__init__()
        self.model = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.model(image)


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
        channels: int = 3,
    ) -> None:
        super().__init__()
        self.backbone = RecognitionBackbone(backbone_name, pretrained=pretrained)

        with torch.no_grad():
            feature_dim = int(
                self.backbone(torch.zeros(1, channels, image_size, image_size)).shape[1]
            )

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
        features = self.backbone(image)
        embeddings = F.normalize(self.embedding(features), dim=1)
        logits = self.head(embeddings, labels)
        return embeddings, logits
