from __future__ import annotations

import timm
import torch
import torch.nn.functional as F
from torch import nn

from pv_iqa.utils.degradation import DEGRADE_TYPES


class IQABackbone(nn.Module):
    """IQA 多尺度特征提取器，封装 timm 并启用 features_only=True。"""

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
    """Squeeze-Excitation 通道注意力。"""

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


# ---------------------------------------------------------------------------
# Degradation-aware Mixture of Experts (soft routing)
# ---------------------------------------------------------------------------
class DegradationMoE(nn.Module):
    """Soft-routing MoE，各 expert 对应一种退化类型。

    gate:     Linear(features → hidden) → ReLU → Linear(hidden → K)
    experts:  K × [Linear(features → hidden) → ReLU → Linear(hidden → 1)]

    输出 = Σ_k softmax(gate(x))[k] · expert_k(x)

    Parameters
    ----------
    in_features : 输入特征维度（e.g. 128）
    num_experts : expert 数量（与 DEGRADE_TYPES 一致）
    hidden : expert / gate 隐藏层维度（默认 64）
    """

    def __init__(self, in_features: int, num_experts: int, *, hidden: int = 64) -> None:
        super().__init__()
        self.num_experts = num_experts

        self.gate = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_experts),
        )

        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(in_features, hidden),
                    nn.ReLU(),
                    nn.Linear(hidden, 1),
                )
                for _ in range(num_experts)
            ]
        )

    def forward(
        self, x: torch.Tensor, *, return_gate_logits: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        gate_logits = self.gate(x)  # (B, K)
        weights = F.softmax(gate_logits, dim=-1)  # (B, K)

        expert_outs = torch.stack(
            [expert(x).squeeze(-1) for expert in self.experts], dim=-1
        )  # (B, K)

        output = (weights * expert_outs).sum(dim=-1)  # (B,)

        if return_gate_logits:
            return output, gate_logits
        return output


# ---------------------------------------------------------------------------
# PalmVein IQA Regressor
# ---------------------------------------------------------------------------
class PalmVeinIQARegressor(nn.Module):
    """多尺度 IQA 回归器（MoE 头）。

    backbone → {stage1, stage3, stage4}
      stage1 → Conv3×3 → GAP → 32d      (纹理/清晰度)
      stage3 → Conv1×1 → GAP → 64d      (中层结构)
      stage4 → GAP → SE-Attn → ch4      (全局语义)
    concat → FC(128)→ReLU→Dropout → MoE
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
            backbone_name,
            pretrained=pretrained,
            out_indices=out_indices,
        )

        with torch.no_grad():
            dims = [f.shape[1] for f in self.backbone(torch.zeros(1, 3, 224, 224))]

        self.branch1 = nn.Sequential(
            nn.Conv2d(dims[0], 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(dims[1], 64, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.branch4 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            SEAttention(dims[2]),
            nn.Flatten(),
        )

        feature_dim = 32 + 64 + dims[2]

        self.feature = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        self.moe = DegradationMoE(
            in_features=128,
            num_experts=len(DEGRADE_TYPES),
            hidden=64,
        )

    def forward(
        self, x: torch.Tensor, *, return_gate_logits: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        feats = self.backbone(x)
        f1 = self.branch1(feats[0])
        f3 = self.branch3(feats[1])
        f4 = self.branch4(feats[2])
        fused = self.feature(torch.cat([f1, f3, f4], dim=1))
        return self.moe(fused, return_gate_logits=return_gate_logits)
