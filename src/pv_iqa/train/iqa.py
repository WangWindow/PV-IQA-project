from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.optim import AdamW
from tqdm.auto import tqdm

from pv_iqa.config import AppConfig
from pv_iqa.models.iqa import LightweightIQARegressor
from pv_iqa.utils.common import (
    create_autocast,
    ensure_dir,
    maybe_compile,
    move_batch_to_device,
    resolve_device,
)
from pv_iqa.utils.datasets import PalmVeinDataset, create_dataloader, load_metadata
from pv_iqa.utils.logging import ExperimentLogger
from pv_iqa.utils.losses import build_iqa_objective
from pv_iqa.utils.metrics import evaluate_regression


def prepare_iqa_loaders(config: AppConfig) -> tuple[Any, Any, pd.DataFrame]:
    metadata = load_metadata(config)
    if "quality_score" not in metadata.columns:
        raise ValueError(
            "quality_score column not found. Run `generate-pseudo-labels` first."
        )

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
    return train_loader, val_loader, metadata


def build_iqa_model(config: AppConfig) -> LightweightIQARegressor:
    return LightweightIQARegressor(
        backbone_name=config.iqa.backbone,
        pretrained=config.iqa.pretrained,
    )


def train_iqa(config: AppConfig) -> Path:
    train_loader, val_loader, _ = prepare_iqa_loaders(config)
    device = resolve_device(config)
    output_dir = ensure_dir(config.experiment_dir / "iqa")
    logger = ExperimentLogger(config, "iqa", output_dir)

    model = build_iqa_model(config).to(device)
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
    criterion = build_iqa_objective(
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
        train_losses: list[float] = []

        # IQA 训练阶段同时优化回归损失和排序损失。
        for batch in tqdm(train_loader, desc=f"iqa-train-{epoch}", leave=False):
            batch = move_batch_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)

            with create_autocast(device, config.runtime.amp):
                outputs = model(batch["image"])
                loss_output = criterion(outputs["score"], batch["target"].float())

            scaler.scale(loss_output.total).backward()
            scaler.step(optimizer)
            scaler.update()
            train_losses.append(float(loss_output.total.detach().item()))

        model.eval()
        predictions: list[float] = []
        targets: list[float] = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"iqa-val-{epoch}", leave=False):
                batch = move_batch_to_device(batch, device)
                outputs = model(batch["image"])
                predictions.extend(outputs["score"].cpu().tolist())
                targets.extend(batch["target"].cpu().tolist())

        summary = evaluate_regression(targets, predictions)
        logger.log_metrics(
            {
                "iqa/train_loss": float(sum(train_losses) / max(1, len(train_losses))),
                "iqa/val_mae": summary.mae,
                "iqa/val_rmse": summary.rmse,
                "iqa/lr": float(optimizer.param_groups[0]["lr"]),
            },
            step=epoch,
        )
        scheduler.step()

        if summary.mae <= best_mae:
            best_mae = summary.mae
            torch.save(
                {"model_state": model.state_dict(), "best_mae": best_mae},
                best_checkpoint,
            )

    logger.info("iqa training finished", best_mae=best_mae)
    logger.finish()
    return best_checkpoint
