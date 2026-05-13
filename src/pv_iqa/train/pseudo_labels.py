from pathlib import Path

import numpy as np
import pandas as pd
import torch
from safetensors.torch import load_file
from scipy.stats import wasserstein_distance
from sklearn.preprocessing import minmax_scale

from pv_iqa.config import Config
from pv_iqa.utils.common import ensure_dir


# ---------------------------------------------------------------------------
# 核心伪标签计算
#   Q = 100 × minmax( Q^P + β·WD )
#
#
# 分量（逐样本 i）：
#   Q^P_i = mean(cos(e_i, e_j))  ∀j: class[j]=class[i], j≠i  (PGRG Eq.2)
#   WD_i  = Wasserstein(S^P_i, top-k S^N_i)                  (SDD-FIQA)
# ---------------------------------------------------------------------------
def compute_pseudo_labels(
    embeddings: torch.Tensor,
    classifier_weight: torch.Tensor,
    class_ids: torch.Tensor,
    *,
    beta: float = 0.0,
    mode: str = "ours",
) -> np.ndarray:
    """无监督掌静脉 IQA 的双分量伪标签融合。

    Args:
        embeddings: (N, D) L2 归一化特征向量。
        classifier_weight: (C, D) ArcFace 分类器权重向量。
        class_ids: (N,) 整数类别标签。
        beta: WD 权重（默认 0.0，0 = 禁用）。

    Returns:
        (N,) float32 数组，质量分数范围 [0, 100]。
    """

    # -- 归一化与转换 ----------------------------------------------------------
    emb = torch.nn.functional.normalize(embeddings.float(), dim=1).cpu().numpy()
    w = torch.nn.functional.normalize(classifier_weight.float(), dim=1).cpu().numpy()  # noqa: F841
    labels = class_ids.cpu().numpy().astype(np.int64)
    N = emb.shape[0]

    # -- 分量 1: Q^P — 类内余弦相似度均值 (PGRG Eq.2) --------------------------
    cos = emb @ emb.T
    qp = np.zeros(N, dtype=np.float32)

    for i, cls in enumerate(labels):
        pos_mask = labels == cls
        pos_mask[i] = False
        pos_scores = cos[i, pos_mask]
        qp[i] = float(np.mean(pos_scores)) if len(pos_scores) > 0 else 0.0

    # -- 分量 2: WD — Wasserstein 距离 (SDD-FIQA) -----------------------------
    qwd = np.zeros(N, dtype=np.float32)

    if beta > 0.0:
        for i in range(N):
            cls = labels[i]
            is_same = labels == cls
            is_same[i] = False
            s_pos = cos[i, is_same]
            if len(s_pos) == 0:
                continue
            is_diff = ~is_same
            s_neg = cos[i, is_diff]
            k = min(len(s_pos), len(s_neg))
            if k == 0:
                continue
            top_neg = np.partition(s_neg, -k)[-k:]
            qwd[i] = float(wasserstein_distance(s_pos, top_neg))

    # -- 加权融合 → 统一 minmax → [0, 100] -----------------------------------
    if mode == "sdd":
        blended = qwd
    elif mode == "qp_only":
        blended = qp
    else:
        blended = qp + beta * qwd
    scores = 100.0 * minmax_scale(blended)
    return scores.astype(np.float32)


# ---------------------------------------------------------------------------
# 公开 API — 伪标签生成流水线
# ---------------------------------------------------------------------------
#
# 流程：
#   1. 加载预计算识别特征 (safetensors)
#   2. 按 split 筛选（类别隔离：排除测试集类别）
#   3. 计算双分量伪标签 (Q^P + β·WD)
#   4. 将伪标签附加到元数据，供下游 IQA 训练使用
# ---------------------------------------------------------------------------
def generate_pseudo_labels(config: Config) -> Path:
    feature_dir = config.experiment_dir / "recognizer"
    tensors = load_file(str(feature_dir / "features.safetensors"))
    meta = pd.read_csv(feature_dir / "feature_metadata.csv")

    # 伪标签仅用于 IQA 训练集（train + val），排除识别器训练集与测试集。
    meta = pd.read_csv(feature_dir / "feature_metadata.csv")
    idx = meta[meta["split"].isin(["train", "val"])].index.to_numpy()
    subset = meta.iloc[idx].reset_index(drop=True)

    scores = compute_pseudo_labels(
        embeddings=tensors["embeddings"][idx].clone(),
        classifier_weight=tensors["classifier_weight"].clone(),
        class_ids=tensors["class_ids"][idx].clone(),
        beta=config.pseudo_beta,
        mode=config.pseudo_mode,
    )

    pseudo_df = pd.DataFrame(
        {
            "sample_id": subset["sample_id"].tolist(),
            "quality_score": scores,
        }
    )

    # 附加到元数据 — 使用 merge 安全处理可能的重复 sample_id
    full_meta = pd.read_csv(config.metadata_path)
    quality_map = pseudo_df.set_index("sample_id")["quality_score"]
    full_meta["quality_score"] = full_meta["sample_id"].map(quality_map)
    full_meta.to_csv(config.metadata_path, index=False)

    out = ensure_dir(config.experiment_dir / "pseudo_labels") / "pseudo_labels.csv"
    pseudo_df.to_csv(out, index=False)
    return out
