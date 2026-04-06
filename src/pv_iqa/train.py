from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from safetensors.torch import load_file, save_file
from torch import nn
from torch.optim import AdamW
from tqdm.auto import tqdm

from pv_iqa.config import AppConfig
from pv_iqa.models.iqa import LightweightIQARegressor
from pv_iqa.models.recognizer import PalmVeinRecognizer
from pv_iqa.utils.data import PalmVeinDataset, create_dataloader, load_metadata
from pv_iqa.utils.io import ensure_dir, save_frame, save_json
from pv_iqa.utils.logging import ExperimentLogger
from pv_iqa.utils.losses import IQALoss
from pv_iqa.utils.metrics import classification_accuracy, regression_summary
from pv_iqa.utils.pseudo_labels import compute_dual_branch_labels


def resolve_device(config: AppConfig) -> torch.device:
    if config.runtime.device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(config.runtime.device)


def create_autocast(device: torch.device, enabled: bool) -> Any:
    if device.type == "cuda" and enabled:
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def maybe_compile(model: nn.Module, config: AppConfig) -> nn.Module:
    if config.runtime.compile_model and hasattr(torch, "compile"):
        return torch.compile(model)
    return model


def _move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def _build_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    total_epochs: int,
    warmup_epochs: int,
) -> torch.optim.lr_scheduler.LRScheduler:
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.2,
        end_factor=1.0,
        total_iters=max(1, warmup_epochs),
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, total_epochs - warmup_epochs),
    )
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[warmup_epochs],
    )


def prepare_recognition_loaders(config: AppConfig) -> tuple[Any, Any, pd.DataFrame]:
    metadata = load_metadata(config)
    train_dataset = PalmVeinDataset(
        metadata,
        split=config.recognizer.train_split,
        image_size=config.data.image_size,
        target_kind="class_id",
        is_train=True,
        grayscale_to_rgb=config.data.grayscale_to_rgb,
    )
    val_dataset = PalmVeinDataset(
        metadata,
        split=config.recognizer.val_split,
        image_size=config.data.image_size,
        target_kind="class_id",
        is_train=False,
        grayscale_to_rgb=config.data.grayscale_to_rgb,
    )
    train_loader = create_dataloader(
        train_dataset,
        batch_size=config.data.batch_size,
        num_workers=config.runtime.num_workers,
        shuffle=True,
        pin_memory=config.data.pin_memory,
    )
    val_loader = create_dataloader(
        val_dataset,
        batch_size=config.data.eval_batch_size,
        num_workers=config.runtime.num_workers,
        shuffle=False,
        pin_memory=config.data.pin_memory,
    )
    return train_loader, val_loader, metadata


def train_recognizer(config: AppConfig) -> Path:
    train_loader, val_loader, metadata = prepare_recognition_loaders(config)
    device = resolve_device(config)
    output_dir = ensure_dir(config.experiment_dir / "recognizer")
    logger = ExperimentLogger(config, "recognizer", output_dir)

    model = PalmVeinRecognizer(
        backbone_name=config.recognizer.backbone,
        num_classes=int(metadata["class_id"].nunique()),
        embedding_dim=config.recognizer.embedding_dim,
        dropout=config.recognizer.dropout,
        margin=config.recognizer.margin,
        scale=config.recognizer.scale,
        pretrained=config.recognizer.pretrained,
    ).to(device)
    model = maybe_compile(model, config)

    optimizer = AdamW(
        model.parameters(),
        lr=config.recognizer.learning_rate,
        weight_decay=config.recognizer.weight_decay,
    )
    scheduler = _build_scheduler(
        optimizer,
        total_epochs=config.recognizer.epochs,
        warmup_epochs=config.recognizer.warmup_epochs,
    )
    criterion = nn.CrossEntropyLoss()
    scaler = torch.GradScaler(enabled=device.type == "cuda" and config.runtime.amp)

    best_accuracy = float("-inf")
    best_checkpoint = output_dir / "best.pt"

    for epoch in range(1, config.recognizer.epochs + 1):
        model.train()
        epoch_losses: list[float] = []
        for batch in tqdm(train_loader, desc=f"recognizer-train-{epoch}", leave=False):
            batch = _move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with create_autocast(device, config.runtime.amp):
                outputs = model(batch["image"], batch["target"])
                loss = criterion(outputs.logits, batch["target"])
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_losses.append(float(loss.detach().item()))

        model.eval()
        predictions: list[int] = []
        targets: list[int] = []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"recognizer-val-{epoch}", leave=False):
                batch = _move_batch(batch, device)
                outputs = model(batch["image"])
                predictions.extend(outputs.logits.argmax(dim=1).cpu().tolist())
                targets.extend(batch["target"].cpu().tolist())

        val_accuracy = classification_accuracy(targets, predictions)
        logger.log_metrics(
            {
                "recognizer/train_loss": float(sum(epoch_losses) / max(1, len(epoch_losses))),
                "recognizer/val_accuracy": val_accuracy,
                "recognizer/lr": float(optimizer.param_groups[0]["lr"]),
            },
            step=epoch,
        )
        scheduler.step()

        if val_accuracy >= best_accuracy:
            best_accuracy = val_accuracy
            torch.save({"model_state": model.state_dict(), "best_accuracy": best_accuracy}, best_checkpoint)

    logger.info("recognizer training finished", best_accuracy=best_accuracy)
    logger.finish()
    return best_checkpoint


def export_recognition_artifacts(config: AppConfig, checkpoint_path: str | Path) -> Path:
    metadata = load_metadata(config)
    output_dir = ensure_dir(config.experiment_dir / "recognizer")
    device = resolve_device(config)

    model = PalmVeinRecognizer(
        backbone_name=config.recognizer.backbone,
        num_classes=int(metadata["class_id"].nunique()),
        embedding_dim=config.recognizer.embedding_dim,
        dropout=config.recognizer.dropout,
        margin=config.recognizer.margin,
        scale=config.recognizer.scale,
        pretrained=False,
    ).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    embeddings: list[torch.Tensor] = []
    class_ids: list[int] = []
    sample_ids: list[str] = []
    with torch.no_grad():
        for split in metadata["split"].drop_duplicates().tolist():
            dataset = PalmVeinDataset(
                metadata,
                split=split,
                image_size=config.data.image_size,
                target_kind="class_id",
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
            for batch in tqdm(loader, desc=f"export-features-{split}", leave=False):
                batch = _move_batch(batch, device)
                outputs = model(batch["image"])
                embeddings.append(outputs.embeddings.cpu())
                class_ids.extend(int(item) for item in batch["class_id"].cpu().tolist())
                sample_ids.extend(batch["sample_id"])

    feature_path = output_dir / "features.safetensors"
    metadata_path = output_dir / "feature_metadata.csv"
    save_file(
        {
            "embeddings": torch.cat(embeddings, dim=0),
            "classifier_weight": model.head.weight.detach().cpu(),
            "class_ids": torch.tensor(class_ids, dtype=torch.int64),
        },
        str(feature_path),
    )
    ordered_metadata = metadata.set_index("sample_id").loc[sample_ids].reset_index()
    save_frame(ordered_metadata, metadata_path)
    return feature_path


def attach_quality_scores(config: AppConfig, pseudo_frame: pd.DataFrame) -> Path:
    metadata = load_metadata(config).set_index("sample_id")
    metadata.loc[pseudo_frame["sample_id"], "quality_score"] = pseudo_frame.set_index("sample_id")[
        "quality_score"
    ]
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
        selected_mask = feature_metadata["split"].eq(config.pseudo_labels.split).to_numpy()
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

    output_path = ensure_dir(config.experiment_dir / "pseudo_labels") / "pseudo_labels.csv"
    save_frame(pseudo_frame, output_path)
    attach_quality_scores(config, pseudo_frame[["sample_id", "quality_score"]])
    return output_path


def train_iqa(config: AppConfig) -> Path:
    metadata = load_metadata(config)
    if "quality_score" not in metadata.columns:
        raise ValueError("quality_score column not found. Run `generate-pseudo-labels` first.")

    device = resolve_device(config)
    output_dir = ensure_dir(config.experiment_dir / "iqa")
    logger = ExperimentLogger(config, "iqa", output_dir)

    train_dataset = PalmVeinDataset(
        metadata,
        split=config.iqa.train_split,
        image_size=config.data.image_size,
        target_kind="quality_score",
        is_train=True,
        grayscale_to_rgb=config.data.grayscale_to_rgb,
    )
    val_dataset = PalmVeinDataset(
        metadata,
        split=config.iqa.val_split,
        image_size=config.data.image_size,
        target_kind="quality_score",
        is_train=False,
        grayscale_to_rgb=config.data.grayscale_to_rgb,
    )
    train_loader = create_dataloader(
        train_dataset,
        batch_size=config.data.batch_size,
        num_workers=config.runtime.num_workers,
        shuffle=True,
        pin_memory=config.data.pin_memory,
    )
    val_loader = create_dataloader(
        val_dataset,
        batch_size=config.data.eval_batch_size,
        num_workers=config.runtime.num_workers,
        shuffle=False,
        pin_memory=config.data.pin_memory,
    )

    model = LightweightIQARegressor(
        backbone_name=config.iqa.backbone,
        pretrained=config.iqa.pretrained,
    ).to(device)
    model = maybe_compile(model, config)

    optimizer = AdamW(
        model.parameters(),
        lr=config.iqa.learning_rate,
        weight_decay=config.iqa.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, config.iqa.epochs),
    )
    criterion = IQALoss(
        delta=config.iqa.huber_delta,
        ranking_margin=config.iqa.ranking_margin,
        ranking_weight=config.iqa.ranking_weight,
        min_ranking_gap=config.iqa.min_ranking_gap,
    )
    scaler = torch.GradScaler(enabled=device.type == "cuda" and config.runtime.amp)

    best_mae = float("inf")
    best_checkpoint = output_dir / "best.pt"
    for epoch in range(1, config.iqa.epochs + 1):
        model.train()
        losses: list[float] = []
        for batch in tqdm(train_loader, desc=f"iqa-train-{epoch}", leave=False):
            batch = _move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with create_autocast(device, config.runtime.amp):
                outputs = model(batch["image"])
                loss_dict = criterion(outputs["score"], batch["target"].float())
            scaler.scale(loss_dict["loss"]).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss_dict["loss"].detach().item()))

        model.eval()
        predictions: list[float] = []
        targets: list[float] = []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"iqa-val-{epoch}", leave=False):
                batch = _move_batch(batch, device)
                outputs = model(batch["image"])
                predictions.extend(outputs["score"].cpu().tolist())
                targets.extend(batch["target"].cpu().tolist())

        summary = regression_summary(targets, predictions)
        logger.log_metrics(
            {
                "iqa/train_loss": float(sum(losses) / max(1, len(losses))),
                "iqa/val_mae": summary["mae"],
                "iqa/val_rmse": summary["rmse"],
                "iqa/lr": float(optimizer.param_groups[0]["lr"]),
            },
            step=epoch,
        )
        scheduler.step()
        if summary["mae"] <= best_mae:
            best_mae = summary["mae"]
            torch.save({"model_state": model.state_dict(), "best_mae": best_mae}, best_checkpoint)

    logger.info("iqa training finished", best_mae=best_mae)
    logger.finish()
    return best_checkpoint


def save_metrics(metrics: dict[str, float], path: str | Path) -> None:
    save_json(path, metrics)
