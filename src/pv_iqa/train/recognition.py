from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import torch
from safetensors.torch import save_file
from torch.optim import AdamW
from tqdm.auto import tqdm

from pv_iqa.config import AppConfig
from pv_iqa.models.recognition import PalmVeinRecognizer
from pv_iqa.utils.common import (
    build_warmup_cosine_scheduler,
    create_autocast,
    ensure_dir,
    maybe_compile,
    move_batch_to_device,
    resolve_device,
    save_frame,
)
from pv_iqa.utils.datasets import PalmVeinDataset, create_dataloader, load_metadata
from pv_iqa.utils.logging import ExperimentLogger
from pv_iqa.utils.losses import build_recognition_objective
from pv_iqa.utils.metrics import evaluate_classification


def export_recognition_artifacts(
    config: AppConfig, checkpoint_path: str | Path
) -> Path:
    metadata = load_metadata(config)
    output_dir = ensure_dir(config.experiment_dir / "recognizer")
    device = resolve_device(config)

    model = build_recognizer(
        config,
        int(metadata["class_id"].nunique()),
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
                batch = move_batch_to_device(batch, device)
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


def build_recognizer(
    config: AppConfig,
    num_classes: int,
    *,
    pretrained: bool | None = None,
) -> PalmVeinRecognizer:
    return PalmVeinRecognizer(
        backbone_name=config.recognizer.backbone,
        num_classes=num_classes,
        embedding_dim=config.recognizer.embedding_dim,
        dropout=config.recognizer.dropout,
        margin=config.recognizer.margin,
        scale=config.recognizer.scale,
        pretrained=config.recognizer.pretrained if pretrained is None else pretrained,
        image_size=config.data.image_size,
    )


def train_recognizer(config: AppConfig) -> Path:
    train_loader, val_loader, metadata = prepare_recognition_loaders(config)
    device = resolve_device(config)
    output_dir = ensure_dir(config.experiment_dir / "recognizer")
    logger = ExperimentLogger(config, "recognizer", output_dir)

    model = build_recognizer(config, int(metadata["class_id"].nunique())).to(device)
    model = maybe_compile(model, config)

    optimizer = AdamW(
        model.parameters(),
        lr=config.recognizer.learning_rate,
        weight_decay=config.recognizer.weight_decay,
    )
    scheduler = build_warmup_cosine_scheduler(
        optimizer,
        total_epochs=config.recognizer.epochs,
        warmup_epochs=config.recognizer.warmup_epochs,
    )
    criterion = build_recognition_objective()
    scaler = torch.GradScaler(enabled=device.type == "cuda" and config.runtime.amp)

    best_accuracy = float("-inf")
    best_checkpoint = output_dir / "best.pt"

    for epoch in range(1, config.recognizer.epochs + 1):
        model.train()
        train_losses: list[float] = []

        # 训练阶段只关注分类目标，让 backbone 先学到稳定身份表征。
        for batch in tqdm(train_loader, desc=f"recognizer-train-{epoch}", leave=False):
            batch = move_batch_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)

            with create_autocast(device, config.runtime.amp):
                outputs = model(batch["image"], batch["target"])
                loss_output = criterion(outputs.logits, batch["target"])

            scaler.scale(loss_output.total).backward()
            scaler.step(optimizer)
            scaler.update()
            train_losses.append(float(loss_output.total.detach().item()))

        model.eval()
        predictions: list[int] = []
        targets: list[int] = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"recognizer-val-{epoch}", leave=False):
                batch = move_batch_to_device(batch, device)
                outputs = model(batch["image"])
                predictions.extend(outputs.logits.argmax(dim=1).cpu().tolist())
                targets.extend(batch["target"].cpu().tolist())

        classification_report = evaluate_classification(targets, predictions)
        val_accuracy = classification_report.accuracy
        logger.log_metrics(
            {
                "recognizer/train_loss": float(
                    sum(train_losses) / max(1, len(train_losses))
                ),
                "recognizer/val_accuracy": val_accuracy,
                "recognizer/lr": float(optimizer.param_groups[0]["lr"]),
            },
            step=epoch,
        )
        scheduler.step()

        if val_accuracy >= best_accuracy:
            best_accuracy = val_accuracy
            torch.save(
                {"model_state": model.state_dict(), "best_accuracy": best_accuracy},
                best_checkpoint,
            )

    logger.info("recognizer training finished", best_accuracy=best_accuracy)
    logger.finish()
    return best_checkpoint
