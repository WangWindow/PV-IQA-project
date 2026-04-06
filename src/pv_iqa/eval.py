from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from safetensors.torch import load_file
from tqdm.auto import tqdm

from pv_iqa.config import AppConfig
from pv_iqa.models.iqa import LightweightIQARegressor
from pv_iqa.train import resolve_device, save_metrics
from pv_iqa.utils.data import PalmVeinDataset, create_dataloader, load_metadata
from pv_iqa.utils.io import ensure_dir, save_frame
from pv_iqa.utils.metrics import verification_metrics


def predict_quality_scores(
    config: AppConfig,
    checkpoint_path: str | Path,
    *,
    split: str,
) -> pd.DataFrame:
    metadata = load_metadata(config)
    device = resolve_device(config)

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

    model = LightweightIQARegressor(
        backbone_name=config.iqa.backbone,
        pretrained=False,
    ).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    records: list[dict[str, float | int | str]] = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="predict-quality", leave=False):
            batch = {
                key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
                for key, value in batch.items()
            }
            outputs = model(batch["image"])
            for sample_id, class_id, score in zip(
                batch["sample_id"],
                batch["class_id"],
                outputs["score"].cpu().tolist(),
                strict=True,
            ):
                records.append(
                    {
                        "sample_id": sample_id,
                        "class_id": int(class_id),
                        "predicted_quality": float(score),
                        "split": split,
                    }
                )

    frame = pd.DataFrame(records)
    save_frame(
        frame,
        ensure_dir(config.experiment_dir / "evaluation") / f"{split}_quality_predictions.csv",
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
        sample_id: index for index, sample_id in enumerate(feature_metadata["sample_id"])
    }

    records: list[dict[str, float]] = []
    for reject_fraction in config.evaluation.reject_steps:
        keep_count = max(2, int(len(merged) * (1.0 - reject_fraction)))
        kept = merged.head(keep_count)
        kept_indices = [sample_to_index[sample_id] for sample_id in kept["sample_id"]]
        metrics = verification_metrics(
            embeddings[kept_indices],
            kept["class_id"].to_numpy(),
            far_targets=config.evaluation.far_targets,
            max_impostor_pairs=config.evaluation.max_impostor_pairs,
            seed=config.runtime.seed,
        )
        metrics["reject_fraction"] = float(reject_fraction)
        records.append(metrics)

    output_dir = ensure_dir(config.experiment_dir / "evaluation")
    metrics_path = output_dir / "erc_metrics.csv"
    frame = pd.DataFrame(records)
    save_frame(frame, metrics_path)
    if records:
        save_metrics(records[-1], output_dir / "latest_metrics.json")
    return metrics_path
