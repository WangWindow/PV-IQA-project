from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(slots=True)
class IQALossOutput:
    """IQA 训练目标的分项输出。"""

    total: torch.Tensor
    huber: torch.Tensor
    ranking: torch.Tensor


@dataclass(slots=True)
class RecognitionLossOutput:
    """识别分支训练目标输出。"""

    total: torch.Tensor


class IQATrainingObjective(nn.Module):
    """将回归项和排序项显式封装为一个训练目标。"""

    def __init__(
        self,
        *,
        delta: float,
        ranking_margin: float,
        ranking_weight: float,
        min_ranking_gap: float,
    ) -> None:
        super().__init__()
        self.huber = nn.HuberLoss(delta=delta)
        self.ranking_margin = ranking_margin
        self.ranking_weight = ranking_weight
        self.min_ranking_gap = min_ranking_gap

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> IQALossOutput:
        huber_loss = self.huber(prediction, target)

        pair_gap = target[:, None] - target[None, :]
        prediction_gap = prediction[:, None] - prediction[None, :]
        valid_pairs = pair_gap > self.min_ranking_gap

        if valid_pairs.any():
            ranking_loss = torch.relu(
                self.ranking_margin - prediction_gap[valid_pairs]
            ).mean()
        else:
            ranking_loss = torch.zeros((), device=prediction.device)

        total_loss = huber_loss + self.ranking_weight * ranking_loss
        return IQALossOutput(
            total=total_loss,
            huber=huber_loss,
            ranking=ranking_loss,
        )


def build_iqa_objective(
    *,
    delta: float,
    ranking_margin: float,
    ranking_weight: float,
    min_ranking_gap: float,
) -> IQATrainingObjective:
    return IQATrainingObjective(
        delta=delta,
        ranking_margin=ranking_margin,
        ranking_weight=ranking_weight,
        min_ranking_gap=min_ranking_gap,
    )


class RecognitionTrainingObjective(nn.Module):
    """识别分支的基础分类目标封装。"""

    def __init__(self) -> None:
        super().__init__()
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> RecognitionLossOutput:
        total_loss = self.cross_entropy(logits, target)
        return RecognitionLossOutput(total=total_loss)


def build_recognition_objective() -> RecognitionTrainingObjective:
    return RecognitionTrainingObjective()

__all__ = [
    "IQALossOutput",
    "IQATrainingObjective",
    "RecognitionLossOutput",
    "RecognitionTrainingObjective",
    "build_iqa_objective",
    "build_recognition_objective",
]
