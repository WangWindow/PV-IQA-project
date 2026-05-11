from __future__ import annotations

import warnings
from dataclasses import dataclass
from itertools import combinations

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (
    accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    roc_curve,
)


@dataclass(slots=True)
class ClassificationReport:
    accuracy: float


def evaluate_classification(
    targets: list[int],
    predictions: list[int],
) -> ClassificationReport:
    """识别任务当前只使用准确率，统一封装方便后续扩展。"""
    return ClassificationReport(accuracy=float(accuracy_score(targets, predictions)))


@dataclass(slots=True)
class RegressionReport:
    mae: float
    rmse: float
    pearson: float
    spearman: float
    ranking_accuracy: float


def _safe_correlation(
    function,
    targets: np.ndarray,
    predictions: np.ndarray,
) -> float:
    if len(targets) < 2:
        return 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            result = function(targets, predictions)
        except ValueError:
            return 0.0
    value = float(result.statistic)
    return 0.0 if np.isnan(value) else value


def pairwise_ranking_accuracy(
    targets: list[float] | np.ndarray,
    predictions: list[float] | np.ndarray,
    *,
    min_gap: float = 0.0,
) -> float:
    target_array = np.asarray(targets, dtype=np.float32)
    prediction_array = np.asarray(predictions, dtype=np.float32)
    pair_gap = target_array[:, None] - target_array[None, :]
    prediction_gap = prediction_array[:, None] - prediction_array[None, :]
    valid_pairs = pair_gap > min_gap
    if not np.any(valid_pairs):
        return 0.0
    return float(np.mean(prediction_gap[valid_pairs] > 0.0))


def evaluate_regression(
    targets: list[float],
    predictions: list[float],
    *,
    min_ranking_gap: float = 0.0,
) -> RegressionReport:
    """IQA 回归阶段统一输出 MAE / RMSE。"""
    targets_array = np.asarray(targets, dtype=np.float32)
    predictions_array = np.asarray(predictions, dtype=np.float32)
    mae = mean_absolute_error(targets, predictions)
    rmse = mean_squared_error(targets, predictions) ** 0.5
    pearson = _safe_correlation(pearsonr, targets_array, predictions_array)
    spearman = _safe_correlation(spearmanr, targets_array, predictions_array)
    ranking_accuracy = pairwise_ranking_accuracy(
        targets_array,
        predictions_array,
        min_gap=min_ranking_gap,
    )
    return RegressionReport(
        mae=float(mae),
        rmse=float(rmse),
        pearson=pearson,
        spearman=spearman,
        ranking_accuracy=ranking_accuracy,
    )


class VerificationEvaluator:
    """生物识别验证指标封装器，统一输出 EER 与 TAR@FAR。"""

    def __init__(
        self,
        *,
        far_targets: list[float],
        max_impostor_pairs: int,
        seed: int,
    ) -> None:
        self.far_targets = far_targets
        self.max_impostor_pairs = max_impostor_pairs
        self.seed = seed

    def _sample_pairs(
        self,
        embeddings: np.ndarray,
        class_ids: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(self.seed)
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
        while sampled < self.max_impostor_pairs and len(all_indices) > 1:
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

    def evaluate(
        self, embeddings: np.ndarray, class_ids: np.ndarray
    ) -> dict[str, float]:
        labels, scores = self._sample_pairs(embeddings, class_ids)
        if len(np.unique(labels)) < 2:
            metrics = {"eer": 1.0}
            for far in self.far_targets:
                metrics[f"tar@far={far:.0e}"] = 0.0
            return metrics

        fpr, tpr, _ = roc_curve(labels, scores)
        fnr = 1.0 - tpr
        eer_index = int(np.nanargmin(np.abs(fnr - fpr)))

        metrics = {"eer": float((fnr[eer_index] + fpr[eer_index]) / 2.0)}
        for far in self.far_targets:
            valid = np.where(fpr <= far)[0]
            metrics[f"tar@far={far:.0e}"] = float(tpr[valid[-1]]) if len(valid) else 0.0
        return metrics


def classification_accuracy(targets: list[int], predictions: list[int]) -> float:
    return evaluate_classification(targets, predictions).accuracy


def regression_summary(
    targets: list[float], predictions: list[float]
) -> dict[str, float]:
    report = evaluate_regression(targets, predictions)
    return {
        "mae": report.mae,
        "rmse": report.rmse,
        "pearson": report.pearson,
        "spearman": report.spearman,
        "ranking_accuracy": report.ranking_accuracy,
    }


def verification_metrics(
    embeddings,
    class_ids,
    *,
    far_targets: list[float],
    max_impostor_pairs: int,
    seed: int,
) -> dict[str, float]:
    evaluator = VerificationEvaluator(
        far_targets=far_targets,
        max_impostor_pairs=max_impostor_pairs,
        seed=seed,
    )
    return evaluator.evaluate(embeddings, class_ids)


__all__ = [
    "ClassificationReport",
    "RegressionReport",
    "VerificationEvaluator",
    "classification_accuracy",
    "evaluate_classification",
    "evaluate_regression",
    "pairwise_ranking_accuracy",
    "regression_summary",
    "verification_metrics",
]
