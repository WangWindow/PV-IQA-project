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
# Core pseudo-label computation (PGRG + CR-FIQA + SDD-FIQA)
# ---------------------------------------------------------------------------


def compute_pseudo_labels(
    embeddings: torch.Tensor,
    classifier_weight: torch.Tensor,
    class_ids: torch.Tensor,
    *,
    alpha: float = 0.5,
    beta: float = 0.0,
    seed: int,
) -> np.ndarray:
    """3-way pseudo-label fusion: PGRG (Q^P) + CR-FIQA (CR) + SDD-FIQA (WD).

    Three independent quality components are computed, independently minmax'd,
    then fused via weighted sum, and finally minmax'd to [0, 100]:

        Q = 100 × minmax(minmax(Q^P) + α×minmax(CR) + β×minmax(WD))

    Components (per-sample i):
      Q^P_i = mean(cos(e_i, e_j))  ∀j: class[j] = class[i], j≠i        (PGRG Eq.2)
      CR_i  = CCS / (NNCCS + 1 + ε)                                    (CR-FIQA Eq.4)
               where CCS  = cos(e_i, w_{class[i]})
                     NNCCS = max_{k≠class[i]} cos(e_i, w_k)
                     ε = 0.001
      WD_i  = 1D Wasserstein distance between S^P_i and top-n S^N_i    (PGRG Eq.4)

    Edge cases:
      - Single-sample class → Q^P_i = 0 (no intra-class pairs).
      - Single-class dataset → NNCCS = 0, CR_i = CCS / (0 + 1 + ε) ≈ CCS.
      - Empty S^P or S^N → WD_i = 0.

    References:
      PGRG:      Zou et al., IEEE TIM 2023 — "Unsupervised Palmprint IQA …"
      CR-FIQA:   Boutros et al., 2021 — "CR-FIQA: Face Image Quality …"
      SDD-FIQA:  Ou et al., CVPR 2021 — "SDD-FIQA: Unsupervised Face IQA …"

    Args:
        embeddings: (N, D) feature embeddings.
        classifier_weight: (C, D) classifier weight vectors.
        class_ids: (N,) integer class labels (0 … C-1).
        alpha: weight for the CR-FIQA term (default 0.5).
        beta: weight for the WD term (default 0.0 — disabled for b/w compat).
        seed: unused; reserved for future reproducibility.

    Returns:
        (N,) numpy float32 array of quality scores in [0, 100].
    """
    del seed  # reserved

    # -- Normalise & convert --------------------------------------------------
    emb = torch.nn.functional.normalize(embeddings.float(), dim=1).cpu().numpy()
    w = torch.nn.functional.normalize(classifier_weight.float(), dim=1).cpu().numpy()
    labels = class_ids.cpu().numpy().astype(np.int64)
    N, C = emb.shape[0], w.shape[0]

    # -- Component 1: Q^P — mean intra-class cosine similarity (PGRG Eq.2) ---
    cos = emb @ emb.T
    qp = np.zeros(N, dtype=np.float32)

    for i, cls in enumerate(labels):
        pos_mask = labels == cls
        pos_mask[i] = False
        pos_scores = cos[i, pos_mask]
        qp[i] = float(np.mean(pos_scores)) if len(pos_scores) > 0 else 0.0

    # -- Component 2: CR — Certainty Ratio (CR-FIQA Eq.4) ---------------------
    EPS = 0.001
    qcr = np.zeros(N, dtype=np.float32)

    if C > 1:
        for i, cls in enumerate(labels):
            ccs = float(np.dot(emb[i], w[cls]))
            neg_centers = np.delete(w, cls, axis=0)
            nnccs = float(np.max(neg_centers @ emb[i]))
            qcr[i] = ccs / (nnccs + 1.0 + EPS)
    else:
        # Single-class dataset: no negative centers → NNCCS = 0
        for i, cls in enumerate(labels):
            ccs = float(np.dot(emb[i], w[0]))
            qcr[i] = ccs / (0.0 + 1.0 + EPS)

    # -- Component 3: WD — Wasserstein distance (PGRG Eq.4 / SDD-FIQA) --------
    qwd = np.zeros(N, dtype=np.float32)

    if beta > 0.0:  # skip if WD not used in fusion
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

    # -- 3-way fusion (raw scales, no per-component normalisation) ---------
    # α/β must be tuned to balance the different component ranges:
    #   Q^P ∈ [-1, 1],  CR ∈ [~0, ~∞),  WD ∈ [0, 2]
    # Start with small α (e.g. 0.01-0.05) since CR is unbounded.
    blended = qp + alpha * qcr + beta * qwd

    # -- Final minmax → [0, 100] ----------------------------------------------
    scores = 100.0 * minmax_scale(blended)
    return scores.astype(np.float32)


# ---------------------------------------------------------------------------
# Public API
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
    subset = meta.iloc[idx].reset_index(drop=True)

    scores = compute_pseudo_labels(
        embeddings=tensors["embeddings"][idx],
        classifier_weight=tensors["classifier_weight"],
        class_ids=tensors["class_ids"][idx],
        alpha=config.pseudo_alpha,
        beta=config.pseudo_beta,
        seed=config.seed,
    )

    pseudo_df = pd.DataFrame({
        "sample_id": subset["sample_id"].tolist(),
        "quality_score": scores,
    })

    # Attach to metadata
    full_meta = pd.read_csv(config.metadata_path).set_index("sample_id")
    full_meta.loc[pseudo_df["sample_id"], "quality_score"] = pseudo_df.set_index(
        "sample_id"
    )["quality_score"]
    full_meta.reset_index().to_csv(config.metadata_path, index=False)

    out = ensure_dir(config.experiment_dir / "pseudo_labels") / "pseudo_labels.csv"
    pseudo_df.to_csv(out, index=False)
    return out
