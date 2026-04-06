from __future__ import annotations

from itertools import combinations

import numpy as np
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, roc_curve


def classification_accuracy(targets: list[int], predictions: list[int]) -> float:
    return float(accuracy_score(targets, predictions))


def regression_summary(targets: list[float], predictions: list[float]) -> dict[str, float]:
    mae = mean_absolute_error(targets, predictions)
    rmse = mean_squared_error(targets, predictions) ** 0.5
    return {"mae": float(mae), "rmse": float(rmse)}


def _sample_verification_pairs(
    embeddings: np.ndarray,
    class_ids: np.ndarray,
    *,
    max_impostor_pairs: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    positive_scores: list[float] = []
    negative_scores: list[float] = []

    indices_by_class: dict[int, list[int]] = {}
    for index, class_id in enumerate(class_ids):
        indices_by_class.setdefault(int(class_id), []).append(index)

    for indices in indices_by_class.values():
        for left, right in combinations(indices, 2):
            positive_scores.append(float(embeddings[left] @ embeddings[right]))

    all_indices = np.arange(len(class_ids))
    sampled = 0
    while sampled < max_impostor_pairs and len(all_indices) > 1:
        left, right = rng.choice(all_indices, size=2, replace=False)
        if class_ids[left] == class_ids[right]:
            continue
        negative_scores.append(float(embeddings[left] @ embeddings[right]))
        sampled += 1

    labels = np.concatenate(
        [
            np.ones(len(positive_scores), dtype=np.int64),
            np.zeros(len(negative_scores), dtype=np.int64),
        ]
    )
    scores = np.concatenate([positive_scores, negative_scores])
    return labels, scores


def verification_metrics(
    embeddings: np.ndarray,
    class_ids: np.ndarray,
    *,
    far_targets: list[float],
    max_impostor_pairs: int,
    seed: int,
) -> dict[str, float]:
    labels, scores = _sample_verification_pairs(
        embeddings,
        class_ids,
        max_impostor_pairs=max_impostor_pairs,
        seed=seed,
    )
    if len(np.unique(labels)) < 2:
        metrics = {"eer": 1.0}
        for far in far_targets:
            metrics[f"tar@far={far:.0e}"] = 0.0
        return metrics

    fpr, tpr, _ = roc_curve(labels, scores)
    fnr = 1.0 - tpr
    eer_index = int(np.nanargmin(np.abs(fnr - fpr)))
    metrics = {"eer": float((fnr[eer_index] + fpr[eer_index]) / 2.0)}
    for far in far_targets:
        valid = np.where(fpr <= far)[0]
        metrics[f"tar@far={far:.0e}"] = float(tpr[valid[-1]]) if len(valid) else 0.0
    return metrics
