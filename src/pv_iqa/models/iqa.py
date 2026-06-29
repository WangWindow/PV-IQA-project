from __future__ import annotations

import math

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
# Generalized Hebbian Algorithm — online principal subspace learning
# ---------------------------------------------------------------------------
class GHAUpdater(nn.Module):
    """Generalized Hebbian Algorithm (Sanger, 1989) — 在线估计 top-K 主成分方向。

    维护正交基 V ∈ R^{K×d}，迭代逼近输入分布协方差矩阵的前 K 个特征向量。
    所有更新在 ``torch.no_grad()`` 下执行，V 为 buffer（非可训练参数，但参与
    state_dict 保存/恢复）。

    Parameters
    ----------
    d : 输入特征维度
    K : 主成分数量（= expert 数量）
    lr : GHA 学习率 η（默认 2e-5）
    """

    V: torch.Tensor  # type: ignore[assignment]  — register_buffer, pyright narrows incorrectly

    def __init__(self, d: int, K: int, *, lr: float = 2e-5) -> None:
        super().__init__()
        self.K = K
        self.lr = lr
        # 初始化为随机正交矩阵
        V_init = torch.randn(K, d)
        Q, _ = torch.linalg.qr(V_init.T)
        self.register_buffer("V", Q.T)  # (K, d)

    @torch.no_grad()
    def update(self, x: torch.Tensor, m: int = 1) -> None:
        """GHA 迭代更新 V。

        Parameters
        ----------
        x : (B, d) 批量特征向量
        m : 每次调用执行的 GHA 迭代次数
        """
        for _ in range(m):
            Y = x @ self.V.T  # (B, K) — 所有投影系数
            for k in range(self.K):
                y_k = Y[:, k]  # (B,)
                # 前 k+1 个分量的重构: Σ_{i=0}^k y_i · V[i]
                recon = Y[:, : k + 1] @ self.V[: k + 1]  # (B, d)
                residual = x - recon  # (B, d)
                delta = (y_k.unsqueeze(1) * residual).mean(dim=0)  # (d,)
                self.V[k] += self.lr * delta
                self.V[k] /= self.V[k].norm(p=2)


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
# STAR-inspired Structure-Aware Mixture of Experts
# ---------------------------------------------------------------------------
class StructureAwareMoE(nn.Module):
    """STAR 风格的结构感知 MoE 路由。

    路由 logits = σ(α) ⊙ (W_g · x) + (1 − σ(α)) ⊙ ((R · V) · x)

    其中：
      W_g  — 可学习 gate 矩阵 (K×d)，任务监督
      V    — GHA 在线估计的主成分子空间 (K×d)，捕捉输入结构
      R    — 基础混合矩阵 (K×K)，解耦 expert 与主成分方差层级
      α    — 插值系数 (K,)，训练中自适应下降

    与 DegradationMoE 保持接口兼容：``forward(x, return_gate_logits=True)``
    返回 ``(output, l_learn)``，其中 l_learn 用于 gate CE 监督。
    """

    def __init__(
        self,
        in_features: int,
        num_experts: int,
        *,
        hidden: int = 64,
        m: int = 3,
        gha_lr: float = 2e-5,
    ) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.m = m
        self.in_features = in_features

        self.gha = GHAUpdater(d=in_features, K=num_experts, lr=gha_lr)

        self.W_g = nn.Parameter(
            torch.randn(num_experts, in_features) * (1.0 / math.sqrt(in_features))
        )

        self.R = nn.Parameter(
            torch.eye(num_experts) + torch.randn(num_experts, num_experts) * 0.01
        )

        self.alpha = nn.Parameter(torch.full((num_experts,), 3.0))

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
        self,
        x: torch.Tensor,
        *,
        return_gate_logits: bool = False,
        update_gha: bool = True,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if update_gha and self.training:
            self.gha.update(x.detach(), m=self.m)

        Z = self.R @ self.gha.V
        l_learn = F.linear(x, self.W_g)
        l_struct = F.linear(x, Z)

        alpha_s = torch.sigmoid(self.alpha)
        l_combined = alpha_s * l_learn + (1.0 - alpha_s) * l_struct

        weights = F.softmax(l_combined, dim=-1)

        expert_outs = torch.stack(
            [expert(x).squeeze(-1) for expert in self.experts], dim=-1
        )

        output = (weights * expert_outs).sum(dim=-1)

        if return_gate_logits:
            return output, l_learn
        return output


# ---------------------------------------------------------------------------
# PalmVein IQA Regressor
# ---------------------------------------------------------------------------
class PalmVeinIQARegressor(nn.Module):
    """多尺度 IQA 回归器（支持标准 MoE 和 STAR 结构感知 MoE）。

    backbone → {stage1, stage3, stage4}
      stage1 → Conv3×3 → GAP → 32d      (纹理/清晰度)
      stage3 → Conv1×1 → GAP → 64d      (中层结构)
      stage4 → GAP → SE-Attn → ch4      (全局语义)
    concat → FC(128)→ReLU→Dropout → MoE

    Parameters
    ----------
    backbone_name : timm 模型名
    pretrained : 是否加载 ImageNet 预训练
    out_indices : 多尺度输出 stage 索引
    use_structure_aware : 启用 STAR 结构感知 MoE（默认 False）
    gha_iterations : GHA 每 forward 迭代次数 m
    gha_lr : GHA 学习率 η
    """

    def __init__(
        self,
        backbone_name: str,
        *,
        pretrained: bool = True,
        out_indices: tuple[int, ...] = (1, 3, 4),
        use_structure_aware: bool = False,
        gha_iterations: int = 3,
        gha_lr: float = 2e-5,
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

        if use_structure_aware:
            self.moe = StructureAwareMoE(
                in_features=128,
                num_experts=len(DEGRADE_TYPES),
                hidden=64,
                m=gha_iterations,
                gha_lr=gha_lr,
            )
        else:
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
