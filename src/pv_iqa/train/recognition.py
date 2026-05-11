"""Palm vein recognizer training and feature export.

Train an ArcFace-based recognizer, then export L2-normalized embeddings
and classifier weights for downstream pseudo-label generation.
"""

import time
from pathlib import Path

import torch
from safetensors.torch import save_file
from torch.optim import AdamW
from tqdm.auto import tqdm

from pv_iqa.config import Config
from pv_iqa.models import PalmVeinRecognizer
from pv_iqa.utils.common import (
    autocast,
    ensure_dir,
    resolve_device,
    set_seed,
    to_device,
)
from pv_iqa.utils.datasets import PalmVeinDataset, create_dataloader, load_metadata
from pv_iqa.utils.logging import ExperimentLogger


def _get_recog_splits(config: Config) -> tuple[str, str]:
    """Return correct split names based on class recognition ratio."""
    if config.class_recognition_ratio > 0:
        return "recognition_train", "recognition_val"
    return "train", "val"


def train_recognizer(config: Config) -> Path:
    """Train ArcFace recognizer. Saves best checkpoint by validation accuracy."""
    set_seed(config.seed)
    meta = load_metadata(config)
    dev = resolve_device(config.device)
    out = ensure_dir(config.experiment_dir / "recognizer")
    logger = ExperimentLogger(config, out)

    train_split, val_split = _get_recog_splits(config)
    num_classes = int(meta["class_id"].nunique())
    model = PalmVeinRecognizer(
        config.recog_backbone,
        num_classes,
        config.recog_embedding_dim,
        config.recog_dropout,
        config.recog_margin,
        config.recog_scale,
        pretrained=True,
    ).to(dev)

    opt = AdamW(model.parameters(), lr=config.recog_lr, weight_decay=config.recog_wd)
    warmup = torch.optim.lr_scheduler.LinearLR(
        opt,
        start_factor=0.2,
        end_factor=1.0,
        total_iters=config.recog_warmup_epochs,
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt,
        T_max=max(1, config.recog_epochs - config.recog_warmup_epochs),
    )
    sched = torch.optim.lr_scheduler.SequentialLR(
        opt,
        schedulers=[warmup, cosine],
        milestones=[config.recog_warmup_epochs],
    )
    scaler = torch.GradScaler(enabled=dev.type == "cuda" and config.amp)

    train_ds = PalmVeinDataset(
        meta,
        split=train_split,
        image_size=config.image_size,
        target_kind="class_id",
        is_train=True,
        grayscale_to_rgb=config.grayscale_to_rgb,
    )
    val_ds = PalmVeinDataset(
        meta,
        split=val_split,
        image_size=config.image_size,
        target_kind="class_id",
        is_train=False,
        grayscale_to_rgb=config.grayscale_to_rgb,
    )
    train_loader = create_dataloader(
        train_ds,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        shuffle=True,
    )
    val_loader = create_dataloader(
        val_ds,
        batch_size=config.eval_batch_size,
        num_workers=config.num_workers,
        shuffle=False,
    )
    loss_fn = torch.nn.CrossEntropyLoss()

    best_acc, best_epoch, best_path = 0.0, 0, out / "best.pt"
    for epoch in range(1, config.recog_epochs + 1):
        t0 = time.perf_counter()
        model.train()
        for b in tqdm(train_loader, desc=f"recog-train-{epoch}", leave=False):
            b = to_device(b, dev)
            opt.zero_grad(set_to_none=True)
            with autocast(dev, config.amp):
                _, logits = model(b["image"], b["target"])
                loss = loss_fn(logits, b["target"])
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        sched.step()

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for b in tqdm(val_loader, desc=f"recog-val-{epoch}", leave=False):
                b = to_device(b, dev)
                _, logits = model(b["image"])
                preds = logits.argmax(1)
                correct += (preds == b["target"]).sum().item()
                total += len(preds)
        acc = correct / total

        if acc > best_acc:
            best_acc, best_epoch = acc, epoch
            torch.save(
                {"model_state": model.state_dict(), "best_accuracy": best_acc},
                best_path,
            )

        logger.info(
            f"Epoch {epoch:3d} | acc={acc:.4f} best={best_acc:.4f} ({time.perf_counter() - t0:.0f}s)"
        )

    logger.info(f"Best acc={best_acc:.4f} at epoch {best_epoch}")
    logger.finish()
    return best_path


def export_features(config: Config, ckpt_path: str | Path) -> Path:
    """Extract L2-normalized embeddings and ArcFace classifier weights."""
    meta = load_metadata(config)
    out = ensure_dir(config.experiment_dir / "recognizer")
    dev = resolve_device(config.device)

    m = PalmVeinRecognizer(
        config.recog_backbone,
        int(meta["class_id"].nunique()),
        config.recog_embedding_dim,
        config.recog_dropout,
        config.recog_margin,
        config.recog_scale,
        pretrained=False,
    ).to(dev)
    m.load_state_dict(torch.load(ckpt_path, map_location=dev)["model_state"])
    m.eval()

    embs, cids, sids = [], [], []
    with torch.no_grad():
        for sp in meta["split"].drop_duplicates().tolist():
            ds = PalmVeinDataset(
                meta,
                split=sp,
                image_size=config.image_size,
                target_kind="class_id",
                is_train=False,
                grayscale_to_rgb=config.grayscale_to_rgb,
            )
            dl = create_dataloader(
                ds,
                batch_size=config.eval_batch_size,
                num_workers=config.num_workers,
                shuffle=False,
            )
            for b in tqdm(dl, desc=f"export-{sp}", leave=False):
                b = to_device(b, dev)
                emb, _ = m(b["image"])
                embs.append(emb.cpu())
                cids.extend(b["class_id"].cpu().tolist())
                sids.extend(b["sample_id"])

    all_emb = torch.cat(embs, dim=0)
    save_file(
        {
            "embeddings": all_emb,
            "classifier_weight": m.head.weight.data,
            "class_ids": torch.tensor(cids),
        },
        str(out / "features.safetensors"),
    )

    import pandas as pd

    pd.DataFrame({"sample_id": sids, "class_id": cids}).to_csv(
        out / "feature_metadata.csv",
        index=False,
    )
    return out / "features.safetensors"
