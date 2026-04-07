from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from safetensors.torch import load_file
from scipy.stats import wasserstein_distance
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import minmax_scale

from pv_iqa.config import AppConfig
from pv_iqa.utils.common import ensure_dir, save_frame
from pv_iqa.utils.datasets import load_metadata


def filter_positive_scores(
    values: np.ndarray,
    *,
    components: int,
    min_positive_count: int,
) -> np.ndarray:
    """用 GMM 过滤明显偏离的正样本相似度，降低噪声。"""
    if len(values) < max(min_positive_count, components + 1):
        return values

    mixture = GaussianMixture(
        n_components=components,
        covariance_type="full",
        random_state=42,
    )
    labels = mixture.fit_predict(values.reshape(-1, 1))
    best_component = int(np.argmax(mixture.means_.reshape(-1)))
    filtered = values[labels == best_component]
    return filtered if len(filtered) >= 2 else values


def compute_dual_branch_labels(
    embeddings: torch.Tensor,
    classifier_weight: torch.Tensor,
    class_ids: torch.Tensor,
    *,
    alpha: float,
    adaptive_alpha: bool,
    negative_samples: int,
    gmm_components: int,
    min_positive_count: int,
    eps: float,
    seed: int,
) -> dict[str, np.ndarray]:
    """根据 SDD + CR 双分支约束生成无监督质量伪标签。"""
    rng = np.random.default_rng(seed)

    embeddings_np = (
        torch.nn.functional.normalize(embeddings.float(), dim=1).cpu().numpy()
    )
    weights_np = (
        torch.nn.functional.normalize(classifier_weight.float(), dim=1).cpu().numpy()
    )
    labels_np = class_ids.cpu().numpy()
    cosine_matrix = embeddings_np @ embeddings_np.T

    q_sdd: list[float] = []
    q_cr: list[float] = []

    for index, class_id in enumerate(labels_np):
        positive_mask = labels_np == class_id
        positive_mask[index] = False

        negative_mask = ~positive_mask
        negative_mask[index] = False

        positive_scores = cosine_matrix[index, positive_mask]
        negative_scores = cosine_matrix[index, negative_mask]
        if len(negative_scores) > negative_samples:
            negative_scores = rng.choice(
                negative_scores,
                size=negative_samples,
                replace=False,
            )

        # QSDD: 比较同类和异类分布的 Wasserstein 距离。
        filtered_positive = filter_positive_scores(
            positive_scores,
            components=gmm_components,
            min_positive_count=min_positive_count,
        )
        q_sdd.append(float(wasserstein_distance(filtered_positive, negative_scores)))

        # QCR: 当前样本与正确类别中心、最难负类中心的相对夹角关系。
        embedding = embeddings_np[index]
        positive_center = weights_np[class_id]
        negative_centers = np.delete(weights_np, class_id, axis=0)
        positive_cos = float(
            np.clip(embedding @ positive_center, a_min=0.0, a_max=None)
        )
        negative_cos = float(
            np.clip(np.max(negative_centers @ embedding), a_min=0.0, a_max=None)
        )
        q_cr.append(positive_cos / (negative_cos + eps))

    q_sdd_np = minmax_scale(np.asarray(q_sdd, dtype=np.float32))
    q_cr_np = minmax_scale(np.asarray(q_cr, dtype=np.float32))

    if adaptive_alpha:
        sdd_var = float(np.var(q_sdd_np))
        cr_var = float(np.var(q_cr_np))
        blend = sdd_var / max(sdd_var + cr_var, eps)
    else:
        blend = alpha

    quality = blend * q_sdd_np + (1.0 - blend) * q_cr_np
    return {"q_sdd": q_sdd_np, "q_cr": q_cr_np, "quality_score": quality}


def attach_quality_scores(config: AppConfig, pseudo_frame: pd.DataFrame) -> Path:
    metadata = load_metadata(config).set_index("sample_id")
    metadata.loc[pseudo_frame["sample_id"], "quality_score"] = pseudo_frame.set_index(
        "sample_id"
    )["quality_score"]
    metadata = metadata.reset_index()
    save_frame(metadata, config.data.metadata_path)
    return Path(config.data.metadata_path)


def generate_pseudo_labels(config: AppConfig) -> Path:
    feature_dir = config.experiment_dir / "recognizer"
    tensors = load_file(str(feature_dir / "features.safetensors"))
    feature_metadata = pd.read_csv(feature_dir / "feature_metadata.csv")

    if config.pseudo_labels.split == "all":
        selected_indices = feature_metadata.index.to_numpy()
    else:
        selected_mask = (
            feature_metadata["split"].eq(config.pseudo_labels.split).to_numpy()
        )
        selected_indices = selected_mask.nonzero()[0]

    subset = feature_metadata.iloc[selected_indices].reset_index(drop=True)
    dual_branch = compute_dual_branch_labels(
        tensors["embeddings"][selected_indices],
        tensors["classifier_weight"],
        tensors["class_ids"][selected_indices],
        alpha=config.pseudo_labels.alpha,
        adaptive_alpha=config.pseudo_labels.adaptive_alpha,
        negative_samples=config.pseudo_labels.negative_samples,
        gmm_components=config.pseudo_labels.gmm_components,
        min_positive_count=config.pseudo_labels.min_positive_count,
        eps=config.pseudo_labels.eps,
        seed=config.runtime.seed,
    )

    pseudo_frame = subset.copy()
    pseudo_frame["q_sdd"] = dual_branch["q_sdd"]
    pseudo_frame["q_cr"] = dual_branch["q_cr"]
    pseudo_frame["quality_score"] = dual_branch["quality_score"]

    output_path = (
        ensure_dir(config.experiment_dir / "pseudo_labels") / "pseudo_labels.csv"
    )
    save_frame(pseudo_frame, output_path)
    attach_quality_scores(config, pseudo_frame[["sample_id", "quality_score"]])
    return output_path
