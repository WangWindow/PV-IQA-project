from __future__ import annotations

import numpy as np
import torch
from scipy.stats import wasserstein_distance
from sklearn.mixture import GaussianMixture


def normalize_scores(values: np.ndarray) -> np.ndarray:
    minimum = values.min()
    maximum = values.max()
    if maximum - minimum < 1e-12:
        return np.ones_like(values)
    return (values - minimum) / (maximum - minimum)


def filter_positive_scores(
    values: np.ndarray,
    *,
    components: int,
    min_positive_count: int,
) -> np.ndarray:
    if len(values) < max(min_positive_count, components + 1):
        return values
    mixture = GaussianMixture(n_components=components, covariance_type="full", random_state=42)
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
    rng = np.random.default_rng(seed)
    embeddings_np = torch.nn.functional.normalize(embeddings.float(), dim=1).cpu().numpy()
    weights_np = torch.nn.functional.normalize(classifier_weight.float(), dim=1).cpu().numpy()
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
            negative_scores = rng.choice(negative_scores, size=negative_samples, replace=False)

        filtered_positive = filter_positive_scores(
            positive_scores,
            components=gmm_components,
            min_positive_count=min_positive_count,
        )
        q_sdd.append(float(wasserstein_distance(filtered_positive, negative_scores)))

        embedding = embeddings_np[index]
        positive_center = weights_np[class_id]
        negative_centers = np.delete(weights_np, class_id, axis=0)
        positive_cos = float(np.clip(embedding @ positive_center, a_min=0.0, a_max=None))
        negative_cos = float(np.clip(np.max(negative_centers @ embedding), a_min=0.0, a_max=None))
        q_cr.append(positive_cos / (negative_cos + eps))

    q_sdd_np = normalize_scores(np.asarray(q_sdd, dtype=np.float32))
    q_cr_np = normalize_scores(np.asarray(q_cr, dtype=np.float32))
    if adaptive_alpha:
        sdd_var = float(np.var(q_sdd_np))
        cr_var = float(np.var(q_cr_np))
        blend = sdd_var / max(sdd_var + cr_var, eps)
    else:
        blend = alpha

    quality = blend * q_sdd_np + (1.0 - blend) * q_cr_np
    return {"q_sdd": q_sdd_np, "q_cr": q_cr_np, "quality_score": quality}
