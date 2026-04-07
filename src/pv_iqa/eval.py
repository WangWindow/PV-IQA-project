from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from safetensors.torch import load_file
from tqdm.auto import tqdm

from pv_iqa.config import AppConfig
from pv_iqa.models.iqa import LightweightIQARegressor
from pv_iqa.utils.common import (
    ensure_dir,
    move_batch_to_device,
    resolve_device,
    save_frame,
    save_json,
)
from pv_iqa.utils.datasets import PalmVeinDataset, create_dataloader, load_metadata
from pv_iqa.utils.metrics import VerificationEvaluator


def load_iqa_checkpoint(
    config: AppConfig,
    checkpoint_path: str | Path,
) -> tuple[LightweightIQARegressor, torch.device]:
    device = resolve_device(config)
    model = LightweightIQARegressor(
        backbone_name=config.iqa.backbone,
        pretrained=False,
    ).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, device


def predict_quality_scores(
    config: AppConfig,
    checkpoint_path: str | Path,
    *,
    split: str,
) -> pd.DataFrame:
    metadata = load_metadata(config)
    model, device = load_iqa_checkpoint(config, checkpoint_path)

    dataset = PalmVeinDataset(
        metadata,
        split=split,
        image_size=config.data.image_size,
        target_kind="none",
        is_train=False,
        grayscale_to_rgb=config.data.grayscale_to_rgb,
    )
    loader = create_dataloader(
        dataset,
        batch_size=config.data.eval_batch_size,
        num_workers=config.runtime.num_workers,
        shuffle=False,
        pin_memory=config.data.pin_memory,
    )

    records: list[dict[str, float | int | str]] = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="predict-quality", leave=False):
            batch = move_batch_to_device(batch, device)
            outputs = model(batch["image"])
            for sample_id, class_id, score in zip(
                batch["sample_id"],
                batch["class_id"],
                outputs["score"].cpu().tolist(),
                strict=True,
            ):
                records.append({
                    "sample_id": sample_id,
                    "class_id": int(class_id),
                    "predicted_quality": float(score),
                    "split": split,
                })

    frame = pd.DataFrame(records)
    save_frame(
        frame,
        ensure_dir(config.experiment_dir / "evaluation")
        / f"{split}_quality_predictions.csv",
    )
    return frame


def evaluate_erc(config: AppConfig, quality_frame: pd.DataFrame) -> Path:
    feature_dir = config.experiment_dir / "recognizer"
    tensors = load_file(str(feature_dir / "features.safetensors"))
    feature_metadata = pd.read_csv(feature_dir / "feature_metadata.csv")

    merged = feature_metadata.merge(
        quality_frame[["sample_id", "predicted_quality"]],
        on="sample_id",
        how="inner",
    ).sort_values("predicted_quality", ascending=False)

    embeddings = tensors["embeddings"].cpu().numpy()
    sample_to_index = {
        sample_id: index
        for index, sample_id in enumerate(feature_metadata["sample_id"])
    }

    records: list[dict[str, float]] = []
    evaluator = VerificationEvaluator(
        far_targets=config.evaluation.far_targets,
        max_impostor_pairs=config.evaluation.max_impostor_pairs,
        seed=config.runtime.seed,
    )
    for reject_fraction in config.evaluation.reject_steps:
        keep_count = max(2, int(len(merged) * (1.0 - reject_fraction)))
        kept = merged.head(keep_count)
        kept_indices = [sample_to_index[sample_id] for sample_id in kept["sample_id"]]

        metrics = evaluator.evaluate(
            embeddings[kept_indices], kept["class_id"].to_numpy()
        )
        metrics["reject_fraction"] = float(reject_fraction)
        records.append(metrics)

    output_dir = ensure_dir(config.experiment_dir / "evaluation")
    metrics_path = output_dir / "erc_metrics.csv"
    frame = pd.DataFrame(records)
    save_frame(frame, metrics_path)

    if records:
        save_json(output_dir / "latest_metrics.json", records[-1])
    return metrics_path
