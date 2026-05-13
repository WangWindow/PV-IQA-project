"""基于伪标签监督的 IQA 回归模型训练。

Loss = huber_w × Huber(pred, label)
     + rank_w   × LabelRank(pred, label)
     + drank_w  × DegradeRank(pred_mild, pred_severe)
"""

from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm.auto import tqdm


from pv_iqa.config import Config
from pv_iqa.models import PalmVeinIQARegressor
from pv_iqa.utils.common import autocast, ensure_dir, resolve_device, to_device
from pv_iqa.utils.datasets import PalmVeinDataset, create_dataloader, load_metadata
from pv_iqa.utils.degradation import generate_ranking_pair
from pv_iqa.utils.logging import ExperimentLogger
from pv_iqa.utils.metrics import evaluate_regression


def train_iqa(config: Config) -> Path:
    metadata = load_metadata(config)
    if "quality_score" not in metadata.columns:
        raise ValueError("quality_score not found. Run pseudo-label generation first.")

    device = resolve_device(config.device)
    output_dir = ensure_dir(config.experiment_dir / "iqa")
    logger = ExperimentLogger(config, output_dir)

    train_set = PalmVeinDataset(
        metadata,
        split="train",
        image_size=config.image_size,
        target_kind="quality_score",
        is_train=True,
        grayscale_to_rgb=config.grayscale_to_rgb,
    )
    val_set = PalmVeinDataset(
        metadata,
        split="val",
        image_size=config.image_size,
        target_kind="quality_score",
        is_train=False,
        grayscale_to_rgb=config.grayscale_to_rgb,
    )
    train_loader = create_dataloader(
        train_set,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        shuffle=True,
    )
    val_loader = create_dataloader(
        val_set,
        batch_size=config.eval_batch_size,
        num_workers=config.num_workers,
        shuffle=False,
    )

    model = PalmVeinIQARegressor(
        config.iqa_backbone,
        pretrained=config.iqa_pretrained,
    ).to(device)

    opt = AdamW(model.parameters(), lr=config.iqa_lr, weight_decay=config.iqa_wd)

    warmup = torch.optim.lr_scheduler.LinearLR(
        opt,
        start_factor=0.01,
        total_iters=config.iqa_warmup_epochs * len(train_loader),
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt,
        T_max=config.iqa_epochs - config.iqa_warmup_epochs,
    )
    sched = torch.optim.lr_scheduler.SequentialLR(
        opt,
        schedulers=[warmup, cosine],
        milestones=[config.iqa_warmup_epochs * len(train_loader)],
    )
    scaler = torch.GradScaler(enabled=device.type == "cuda" and config.amp)

    best_mae = float("inf")
    best_path = output_dir / "best.pt"
    global_step = 0

    for epoch in range(1, config.iqa_epochs + 1):
        model.train()
        epoch_huber = epoch_rank = epoch_drank = 0.0
        for batch in tqdm(train_loader, desc=f"iqa-train-{epoch}", leave=False):
            batch = to_device(batch, device)
            opt.zero_grad(set_to_none=True)

            with autocast(device, config.amp):
                pred = model(batch["image"])
                target = batch["target"].float()

                l_huber = F.huber_loss(pred, target, delta=config.iqa_huber_delta)
                epoch_huber += l_huber.item()
                l_rank = torch.zeros((), device=device)
                l_drank = torch.zeros((), device=device)

                if config.iqa_rank_weight > 0:
                    t_i = target[:, None]
                    t_j = target[None, :]
                    pair_gap = t_i - t_j
                    valid = pair_gap > config.iqa_min_rank_gap
                    if valid.any():
                        p_norm = torch.sigmoid(pred / config.iqa_sigmoid_tau)
                        pn_i = p_norm.unsqueeze(1).expand(-1, len(pred))
                        pn_j = p_norm.unsqueeze(0).expand(len(pred), -1)
                        l_rank = torch.relu(pn_j[valid] - pn_i[valid]).mean()
                        epoch_rank += l_rank.item()
                    loss = l_huber + config.iqa_rank_weight * l_rank
                else:
                    loss = l_huber

                if config.iqa_degrade_rank_weight > 0:
                    low_mask = target < target.median()
                    if low_mask.any():
                        deg_mild, deg_severe = generate_ranking_pair(
                            batch["image"][low_mask]
                        )
                        pred_mild = model(deg_mild)
                        pred_severe = model(deg_severe)
                        l_drank = F.relu(
                            pred_severe - pred_mild + config.iqa_degrade_margin
                        ).mean()
                        epoch_drank += l_drank.item()
                        loss = loss + config.iqa_degrade_rank_weight * l_drank

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=config.iqa_grad_clip,
            )
            scaler.step(opt)
            scaler.update()
            global_step += 1

        sched.step()

        n_batches = len(train_loader)
        logger.log_metrics({
            "train/huber_raw": epoch_huber / n_batches,
            "train/label_rank": epoch_rank / n_batches,
            "train/degrade_rank": epoch_drank / n_batches,
        }, step=epoch)

        model.eval()
        preds, targets = [], []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"iqa-val-{epoch}", leave=False):
                batch = to_device(batch, device)
                pred = model(batch["image"])
                preds.extend(pred.cpu().tolist())
                targets.extend(batch["target"].cpu().tolist())

        report = evaluate_regression(targets, preds)
        logger.info(
            f"Epoch {epoch:3d} | MAE={report.mae:.4f} "
            f"RMSE={report.rmse:.4f} ρ={report.spearman:.3f}"
        )
        logger.log_metrics({
            "val/mae": report.mae,
            "val/rmse": report.rmse,
            "val/pearson": report.pearson,
            "val/spearman": report.spearman,
            "val/rank_acc": report.ranking_accuracy,
            "epoch": epoch,
        }, step=epoch)

        if report.mae < best_mae:
            best_mae = report.mae
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "best_mae": best_mae,
                    "backbone": config.iqa_backbone,
                },
                best_path,
            )

    logger.info(f"Best MAE={best_mae:.4f}")
    logger.finish()
    return best_path
