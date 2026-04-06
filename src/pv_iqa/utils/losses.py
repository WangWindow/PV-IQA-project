from __future__ import annotations

import torch
from torch import nn


class IQALoss(nn.Module):
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

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
        huber_loss = self.huber(prediction, target)
        pair_gap = target[:, None] - target[None, :]
        pred_gap = prediction[:, None] - prediction[None, :]
        valid_pairs = pair_gap > self.min_ranking_gap
        if valid_pairs.any():
            ranking_loss = torch.relu(self.ranking_margin - pred_gap[valid_pairs]).mean()
        else:
            ranking_loss = torch.zeros((), device=prediction.device)
        total_loss = huber_loss + self.ranking_weight * ranking_loss
        return {
            "loss": total_loss,
            "huber_loss": huber_loss,
            "ranking_loss": ranking_loss,
        }
