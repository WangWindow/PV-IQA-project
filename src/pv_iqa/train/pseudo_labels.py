from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from safetensors.torch import load_file
from scipy.stats import wasserstein_distance
from sklearn.preprocessing import minmax_scale
from tqdm.auto import tqdm

from pv_iqa.config import Config
from pv_iqa.utils.common import ensure_dir

# ---------------------------------------------------------------------------
# Degradation-aware penalty
# Multipliers encode per-type severity: overexposed/underexposed = severe,
# incomplete = moderate, enhanced_extreme = mild.
# Applied in generate_pseudo_labels when pseudo_degrade_penalty > 0.
# ---------------------------------------------------------------------------
DEGRADE_KEYWORDS = ["enhanced_extreme", "overexposed", "underexposed", "incomplete"]

_DEGRADE_TYPE_MULTIPLIER = {
    "enhanced_extreme": 0.2,
    "incomplete": 0.7,
    "overexposed": 1.0,
    "underexposed": 1.0,
}


def _get_degrade_factor(sample_id: str) -> float:
    for kw, mult in _DEGRADE_TYPE_MULTIPLIER.items():
        if kw in sample_id:
            return mult
    return 0.0


def compute_visual_quality(
    image_paths: list[str], weights: tuple[float, float, float] = (1.0, 1.0, 1.0)
) -> np.ndarray:
    """Compute per-image visual quality Q^V from raw pixels.

    Three sub-metrics, each normalized to [0, 1]:
      - Laplacian variance (sharpness)
      - RMS contrast (std dev / 128)
      - Exposure balance (1 − |mean − 128| / 128)

    Returns (N,) float32 array where higher = better visual quality.
    """
    w_sum = sum(weights)
    qv = np.zeros(len(image_paths), dtype=np.float32)
    for i, p in enumerate(tqdm(image_paths, desc="qv", leave=False)):
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img_f = img.astype(np.float32)

        laplacian_var = cv2.Laplacian(img, cv2.CV_64F).var()
        sharpness = np.clip(laplacian_var / 500.0, 0.0, 1.0)

        rms_contrast = np.clip(np.std(img_f) / 128.0, 0.0, 1.0)

        mean_val = np.mean(img_f)
        exposure = 1.0 - abs(mean_val - 128.0) / 128.0

        qv[i] = (
            weights[0] * sharpness + weights[1] * rms_contrast + weights[2] * exposure
        ) / w_sum
    return qv


# ---------------------------------------------------------------------------
# Core pseudo-label computation
#   Q = 100 × minmax( δ·Q^P_norm + β·WD_norm + γ·Q^V_norm )
# Components: PGRG (Q^P) + SDD-FIQA (WD) + Visual Quality (Q^V)
#
# Note: CR-FIQA component deliberately excluded — CR is designed for
# joint training (ArcFace + regression head), not as a static pseudo-label
# (Boutros et al., CVPR 2023, Sec.3.4).
# ---------------------------------------------------------------------------


def compute_pseudo_labels(
    embeddings: torch.Tensor,
    classifier_weight: torch.Tensor,
    class_ids: torch.Tensor,
    *,
    delta: float = 1.0,
    beta: float = 0.0,
    gamma: float = 0.0,
    qv: np.ndarray | None = None,
    per_component_norm: bool = False,
) -> np.ndarray:
    """3-component pseudo-label fusion for unsupervised palm vein IQA.

        Q = 100 × minmax( δ·minmax(Q^P) + β·minmax(WD) + γ·minmax(Q^V) )

    Components (per-sample i):
      Q^P_i = mean(cos(e_i, e_j))  ∀j: class[j]=class[i], j≠i  (PGRG Eq.2)
      WD_i  = Wasserstein(S^P_i, top-k S^N_i)                  (SDD-FIQA)
      Q^V_i = (sharpness + contrast + exposure_balance) / 3    (pixel-based)

    Args:
        embeddings: (N, D) L2-normalized feature vectors.
        classifier_weight: (C, D) ArcFace classifier weight vectors.
        class_ids: (N,) integer class labels.
        delta: Q^P weight (default 1.0, 0 = disable).
        beta: WD weight (default 0.0, 0 = disable).
        gamma: Q^V weight (default 0.0, 0 = disable).
        qv: pre-computed visual quality scores (N,), optional.
        per_component_norm: if True, minmax each component before fusion.

    Returns:
        (N,) float32 array of quality scores in [0, 100].
    """

    # -- Normalise & convert --------------------------------------------------
    emb = torch.nn.functional.normalize(embeddings.float(), dim=1).cpu().numpy()
    labels = class_ids.cpu().numpy().astype(np.int64)
    N = emb.shape[0]

    # -- Component 1: Q^P — mean intra-class cosine similarity (PGRG Eq.2) ---
    cos = emb @ emb.T
    qp = np.zeros(N, dtype=np.float32)

    for i, cls in enumerate(labels):
        pos_mask = labels == cls
        pos_mask[i] = False
        pos_scores = cos[i, pos_mask]
        qp[i] = float(np.mean(pos_scores)) if len(pos_scores) > 0 else 0.0

    # -- Component 2: WD — Wasserstein distance (SDD-FIQA) --------------------
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

    # -- 3-component weighted fusion ------------------------------------------
    if per_component_norm:
        qp_norm = minmax_scale(qp)
        qwd_norm = minmax_scale(qwd) if beta > 0 else qwd
        qv_norm = (
            minmax_scale(qv)
            if gamma > 0 and qv is not None
            else (qv if qv is not None else np.zeros(N, dtype=np.float32))
        )
        blended = delta * qp_norm + beta * qwd_norm + gamma * qv_norm
    else:
        qv_term = gamma * qv if (gamma > 0 and qv is not None) else 0.0
        blended = delta * qp + beta * qwd + qv_term

    # -- Final minmax → [0, 100] ----------------------------------------------
    scores = 100.0 * minmax_scale(blended)
    return scores.astype(np.float32)


# ---------------------------------------------------------------------------
# Public API — pseudo-label generation pipeline
# ---------------------------------------------------------------------------
#
# Flow:
#   1. Load pre-computed recognition features (safetensors)
#   2. Filter by split (class-disjoint: exclude test classes)
#   3. Optionally compute Q^V from raw images (if gamma > 0)
#   4. Compute 4-component pseudo-labels via weighted fusion
#   5. Apply degradation penalty to known low-quality images
#   6. Attach pseudo-labels to metadata for downstream IQA training
# ---------------------------------------------------------------------------


def generate_pseudo_labels(config: Config) -> Path:
    feature_dir = config.experiment_dir / "recognizer"
    tensors = load_file(str(feature_dir / "features.safetensors"))
    meta = pd.read_csv(feature_dir / "feature_metadata.csv")

    idx = (
        meta.index.to_numpy()
        if config.pseudo_split == "all"
        else meta["split"].eq(config.pseudo_split).to_numpy().nonzero()[0]
    )
    # In class-disjoint mode, exclude test-class samples from pseudo-label generation.
    # feature_metadata.csv lacks 'split', so join with full metadata on sample_id.
    if config.split_mode == "class" and config.pseudo_split == "all":
        full_meta = pd.read_csv(config.metadata_path)
        test_sids = set(full_meta[full_meta["split"] == "test"]["sample_id"])
        idx = meta[~meta["sample_id"].isin(test_sids)].index.to_numpy()
    subset = meta.iloc[idx].reset_index(drop=True)

    qv = None
    if config.pseudo_gamma > 0:
        full_meta = pd.read_csv(config.metadata_path)
        sid_to_path = dict(zip(full_meta["sample_id"], full_meta["image_path"]))
        # Only include sample_ids present in the current metadata
        valid_mask = subset["sample_id"].isin(sid_to_path.keys())
        if not valid_mask.all():
            subset = subset[valid_mask].reset_index(drop=True)
            idx = idx[valid_mask.to_numpy()]
        image_paths = [sid_to_path[sid] for sid in subset["sample_id"]]
        w_str = config.pseudo_qv_weights
        w_parts = [float(x.strip()) for x in w_str.split(",")]
        qv_weights = (
            (w_parts[0], w_parts[1], w_parts[2])
            if len(w_parts) >= 3
            else (1.0, 1.0, 1.0)
        )
        qv = compute_visual_quality(image_paths, weights=qv_weights)

    scores = compute_pseudo_labels(
        embeddings=tensors["embeddings"][idx],
        classifier_weight=tensors["classifier_weight"],
        class_ids=tensors["class_ids"][idx],
        delta=config.pseudo_delta,
        beta=config.pseudo_beta,
        gamma=config.pseudo_gamma,
        qv=qv,
        per_component_norm=config.pseudo_per_component_norm,
    )

    pseudo_df = pd.DataFrame(
        {
            "sample_id": subset["sample_id"].tolist(),
            "quality_score": scores,
        }
    )

    # Apply degradation penalty: multiply score by (1 - penalty) for known degraded images
    if config.pseudo_degrade_penalty > 0:
        factors = pseudo_df["sample_id"].apply(_get_degrade_factor)
        n_degraded = (factors > 0).sum()
        if n_degraded > 0:
            effective_penalty = config.pseudo_degrade_penalty * factors
            pseudo_df["quality_score"] *= 1.0 - effective_penalty

    # Attach to metadata
    full_meta = pd.read_csv(config.metadata_path).set_index("sample_id")
    full_meta.loc[pseudo_df["sample_id"], "quality_score"] = pseudo_df.set_index(
        "sample_id"
    )["quality_score"]
    full_meta.reset_index().to_csv(config.metadata_path, index=False)

    out = ensure_dir(config.experiment_dir / "pseudo_labels") / "pseudo_labels.csv"
    pseudo_df.to_csv(out, index=False)
    return out
