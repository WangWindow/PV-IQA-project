from pathlib import Path

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


# ---------------------------------------------------------------------------
# Core pseudo-label computation
#   Q = 100 × minmax( Q^P + β·WD − λ·degrade )
# Components: PGRG (Q^P) + SDD-FIQA (WD) + degradation penalty
# ---------------------------------------------------------------------------


def compute_pseudo_labels(
    embeddings: torch.Tensor,
    classifier_weight: torch.Tensor,
    class_ids: torch.Tensor,
    *,
    beta: float = 0.0,
) -> np.ndarray:
    """2-component pseudo-label fusion for unsupervised palm vein IQA.

        Q = 100 × minmax( Q^P + β·WD − λ·degrade )

    Components (per-sample i):
      Q^P_i = mean(cos(e_i, e_j))  ∀j: class[j]=class[i], j≠i  (PGRG Eq.2)
      WD_i  = Wasserstein(S^P_i, top-k S^N_i)                  (SDD-FIQA)

    Args:
        embeddings: (N, D) L2-normalized feature vectors.
        classifier_weight: (C, D) ArcFace classifier weight vectors.
        class_ids: (N,) integer class labels.
        beta: WD weight (default 0.0, 0 = disable).

    Returns:
        (N,) float32 array of quality scores in [0, 100].
    """

    # -- Normalise & convert --------------------------------------------------
    emb = torch.nn.functional.normalize(embeddings.float(), dim=1).cpu().numpy()
    w = torch.nn.functional.normalize(classifier_weight.float(), dim=1).cpu().numpy()  # noqa: F841
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

    # -- Weighted fusion → unified minmax → [0, 100] ---------------------------
    #   Q = 100 × minmax( Q^P + β·WD )   (PGRG Eq.5)
    blended = qp + beta * qwd
    scores = 100.0 * minmax_scale(blended)
    return scores.astype(np.float32)


# ---------------------------------------------------------------------------
# Public API — pseudo-label generation pipeline
# ---------------------------------------------------------------------------
#
# Flow:
#   1. Load pre-computed recognition features (safetensors)
#   2. Filter by split (class-disjoint: exclude test classes)
#   3. Compute 2-component pseudo-labels via weighted fusion
#   4. Apply degradation penalty to known low-quality images
#   5. Attach pseudo-labels to metadata for downstream IQA training
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
    # Exclude test-class samples from pseudo-label generation.
    # feature_metadata.csv lacks 'split', so join with full metadata on sample_id.
    if config.pseudo_split == "all":
        full_meta = pd.read_csv(config.metadata_path)
        test_sids = set(full_meta[full_meta["split"] == "test"]["sample_id"])
        idx = meta[~meta["sample_id"].isin(test_sids)].index.to_numpy()
    subset = meta.iloc[idx].reset_index(drop=True)

    scores = compute_pseudo_labels(
        embeddings=tensors["embeddings"][idx].clone(),
        classifier_weight=tensors["classifier_weight"].clone(),
        class_ids=tensors["class_ids"][idx].clone(),
        beta=config.pseudo_beta,
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

    # Attach to metadata — use merge to handle potential duplicate sample_ids safely
    full_meta = pd.read_csv(config.metadata_path)
    quality_map = pseudo_df.set_index("sample_id")["quality_score"]
    full_meta["quality_score"] = full_meta["sample_id"].map(quality_map)
    full_meta.to_csv(config.metadata_path, index=False)

    out = ensure_dir(config.experiment_dir / "pseudo_labels") / "pseudo_labels.csv"
    pseudo_df.to_csv(out, index=False)
    return out
