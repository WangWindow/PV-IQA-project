"""PV-IQA 评估工具：推理、指标计算与质量分析。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score
from tqdm.auto import tqdm

from pv_iqa.config import Config
from pv_iqa.models import PalmVeinIQARegressor, PalmVeinRecognizer
from pv_iqa.utils.common import ensure_dir, resolve_device, save_csv, to_device
from pv_iqa.utils.datasets import PalmVeinDataset, create_dataloader, load_metadata
from pv_iqa.utils.logging import ExperimentLogger
from pv_iqa.utils.metrics import compute_eer_from_embeddings, compute_rejection_accuracy
from pv_iqa.utils.transforms import build_transforms

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def _log(logger: ExperimentLogger | None, msg: str) -> None:
    """写入 logger（如果可用），否则回退到 print。"""
    if logger is not None:
        logger.info(msg)
    else:
        print(msg)


# ---------------------------------------------------------------------------
# 推理
# ---------------------------------------------------------------------------


def load_checkpoint(
    config: Config,
    path: str | Path,
) -> tuple[PalmVeinIQARegressor, torch.device]:
    dev = resolve_device(config.device)
    ckpt = torch.load(path, map_location=dev, weights_only=False)
    m = PalmVeinIQARegressor(
        ckpt.get("backbone", config.iqa_backbone),
        pretrained=False,
    ).to(dev)
    m.load_state_dict(ckpt["model_state"])
    m.eval()
    return m, dev


def score_image(config: Config, ckpt: str | Path, img_path: str | Path) -> dict:
    m, dev = load_checkpoint(config, ckpt)
    t = build_transforms(image_size=config.image_size, is_train=False)
    img = Image.open(img_path).convert("L")
    if config.grayscale_to_rgb:
        img = img.convert("RGB")
    x = t(img).unsqueeze(0).to(dev)
    with torch.no_grad():
        s = float(m(x).item())
    return {"image_path": str(img_path), "quality_score": s}


def score_folder(config: Config, ckpt: str | Path, folder: str | Path) -> list[dict]:
    m, dev = load_checkpoint(config, ckpt)
    t = build_transforms(image_size=config.image_size, is_train=False)
    res = []
    for p in sorted(Path(folder).rglob("*")):
        if p.suffix.lower() not in EXTS:
            continue
        img = Image.open(p).convert("L")
        if config.grayscale_to_rgb:
            img = img.convert("RGB")
        x = t(img).unsqueeze(0).to(dev)
        with torch.no_grad():
            res.append({"image_path": str(p), "quality_score": float(m(x).item())})
    return res


def predict_quality_scores(
    config: Config,
    ckpt: str | Path,
    split: str,
) -> pd.DataFrame:
    meta = load_metadata(config)
    model, dev = load_checkpoint(config, ckpt)
    ds = PalmVeinDataset(
        meta,
        split=split,
        image_size=config.image_size,
        target_kind="none",
        is_train=False,
        grayscale_to_rgb=config.grayscale_to_rgb,
    )
    loader = create_dataloader(
        ds,
        batch_size=config.eval_batch_size,
        num_workers=config.num_workers,
        shuffle=False,
    )
    recs = []
    with torch.no_grad():
        for b in tqdm(loader, desc="predict", leave=False):
            b = to_device(b, dev)
            scores = model(b["image"])
            for sid, cid, s in zip(
                b["sample_id"], b["class_id"], scores.cpu().tolist()
            ):
                recs.append(
                    {
                        "sample_id": sid,
                        "class_id": int(cid),
                        "predicted_quality": float(s),
                        "split": split,
                    }
                )
    df = pd.DataFrame(recs)
    save_csv(
        df,
        ensure_dir(config.experiment_dir / "evaluation") / f"{split}_predictions.csv",
    )
    return df


# ---------------------------------------------------------------------------
# err_roi 评估
# ---------------------------------------------------------------------------


def parse_err_roi_labels(
    labels_csv: str = "datasets/err_roi/labels.csv"
) -> dict[str, int]:
    """从 labels.csv 读取标签。返回 {文件名: 1/0}，高质量为 1。"""
    df = pd.read_csv(labels_csv, dtype={"image": str, "quality": int})
    labels: dict[str, int] = {}
    for _, row in df.iterrows():
        labels[str(row["image"]).strip()] = int(row["quality"])
    return labels


def evaluate_err_roi(
    config: Config,
    ckpt_path: str | Path,
    logger: ExperimentLogger | None = None,
) -> dict:
    """对 err_roi 图像评分，计算 AUC / ScoreGap / Overlap。"""
    labels_map = parse_err_roi_labels()
    model, dev = load_checkpoint(config, ckpt_path)

    transform = build_transforms(image_size=config.image_size, is_train=False)
    scores, gts = [], []

    for p in sorted(Path("datasets/err_roi").iterdir()):
        name = p.name
        if name not in labels_map:
            continue
        img = Image.open(p).convert("L")
        if config.grayscale_to_rgb:
            img = img.convert("RGB")
        x = transform(img).unsqueeze(0).to(dev)
        with torch.no_grad():
            s = float(model(x).item())
        scores.append(s)
        gts.append(labels_map[name])

    scores_arr = np.array(scores)
    gts_arr = np.array(gts)

    high_scores = scores_arr[gts_arr == 1]
    low_scores = scores_arr[gts_arr == 0]

    score_gap = float(high_scores.mean() - low_scores.mean())
    overlap = float(
        sum(1 for s in high_scores if s < low_scores.max())
        + sum(1 for s in low_scores if s > high_scores.min())
    ) / len(scores_arr)

    try:
        auc = float(roc_auc_score(gts_arr, scores_arr))
    except ValueError:
        auc = 0.0

    msg = (
        f"  err_roi | AUC={auc:.4f}  Gap={score_gap:.2f}  "
        f"Overlap={overlap:.3f}  High={high_scores.mean():.1f}  "
        f"Low={low_scores.mean():.1f}"
    )
    _log(logger, msg)

    return {
        "err_roi_auc": auc,
        "err_roi_score_gap": score_gap,
        "err_roi_overlap": overlap,
        "err_roi_high_mean": float(high_scores.mean()),
        "err_roi_low_mean": float(low_scores.mean()),
    }


# ---------------------------------------------------------------------------
# EER / AOC 评估
# ---------------------------------------------------------------------------
def evaluate_eer_aoc(
    config: Config,
    iqa_ckpt: str | Path,
    recog_run: str = "auto",
    logger: ExperimentLogger | None = None,
) -> dict:
    """在 class-disjoint 测试集上，计算拒绝率 0%, 5%, …, 30% 时的 EER。

    AOC (Area of Curve) = 1 − ∫₀^0.95 EER(r) dr  (PGRG Eq.13-14)。
    """
    meta = load_metadata(config)
    test_meta = meta[meta["split"] == "test"]
    if len(test_meta) == 0:
        _log(logger, "  ⚠ No test split found")
        return {}

    _log(
        logger,
        f"\n  --- EER/AOC (test: {test_meta['class_id'].nunique()} unseen classes, "
        f"{len(test_meta)} images) ---",
    )

    iqa_model, dev = load_checkpoint(config, iqa_ckpt)

    # 加载识别器用于 embedding 提取
    recog_dir = Path(config.output_root)
    if config.recog_checkpoint:
        recog_ckpt_path = Path(config.recog_checkpoint)
    elif recog_run == "auto":
        recog_ckpt_path = recog_dir / config.name / "recognizer" / "best.pt"
    else:
        recog_ckpt_path = recog_dir / recog_run / "recognizer" / "best.pt"
    recog_ckpt = torch.load(
        str(recog_ckpt_path), map_location=dev, weights_only=False
    )
    recog = PalmVeinRecognizer(
        config.recog_backbone,
        int(meta["class_id"].nunique()),
        config.recog_embedding_dim,
        config.recog_dropout,
        config.recog_margin,
        config.recog_scale,
        pretrained=False,
    ).to(dev)
    recog.load_state_dict(recog_ckpt["model_state"])
    recog.eval()

    ds = PalmVeinDataset(
        meta,
        split="test",
        image_size=config.image_size,
        target_kind="none",
        is_train=False,
        grayscale_to_rgb=config.grayscale_to_rgb,
    )
    loader = create_dataloader(
        ds,
        batch_size=config.eval_batch_size,
        num_workers=config.num_workers,
        shuffle=False,
    )

    recs = []
    with torch.no_grad():
        for b in loader:
            b = to_device(b, dev)
            quality = iqa_model(b["image"])
            emb, _ = recog(b["image"])
            for i in range(len(b["sample_id"])):
                recs.append(
                    {
                        "sample_id": b["sample_id"][i],
                        "class_id": int(b["class_id"][i]),
                        "quality": float(quality[i]),
                        "embedding": emb[i].cpu().numpy(),
                    }
                )

    df = pd.DataFrame(recs)
    if config.eval_sample_size > 0 and len(df) > config.eval_sample_size:
        rng = np.random.default_rng(42)
        df = df.iloc[rng.choice(len(df), config.eval_sample_size, replace=False)]
    df = df.sort_values("quality", ascending=True)
    total = len(df)

    all_embs = np.stack(df["embedding"].values).astype(np.float32)
    norms = np.linalg.norm(all_embs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    all_embs /= norms
    all_ids = df["class_id"].values

    rejection_rates = [0.0, 0.1, 0.2, 0.3]
    eer_values, aoc = compute_eer_from_embeddings(all_embs, all_ids, rejection_rates)

    _log(logger, "  reject% | EER%")
    for rr, eer in zip(rejection_rates, eer_values):
        _log(logger, f"  {rr:6.0%} | {eer * 100:.4f}")
    _log(logger, f"  AOC = {aoc:.4f}")

    results = {"eer_aoc": float(aoc)}
    for rr, eer in zip(rejection_rates, eer_values):
        if not np.isnan(eer):
            results[f"eer@{1 - rr:.0%}"] = float(eer)
    return results


# ---------------------------------------------------------------------------
# 拒绝准确率
# ---------------------------------------------------------------------------
def evaluate_rejection_accuracy(
    config: Config,
    iqa_ckpt: str | Path,
    recog_run: str = "auto",
    logger: ExperimentLogger | None = None,
) -> dict:
    """按质量拒掉最差的 N% 后，计算 Rank-1 识别准确率。

    使用 embedding 余弦相似度而非分类器 logits，
    因此适用于类别未见过的开放集测试划分。
    """
    meta = load_metadata(config)
    iqa_model, dev = load_checkpoint(config, iqa_ckpt)

    recog_dir = Path(config.output_root)
    if config.recog_checkpoint:
        recog_ckpt_path = Path(config.recog_checkpoint)
    elif recog_run == "auto":
        recog_ckpt_path = recog_dir / config.name / "recognizer" / "best.pt"
    else:
        recog_ckpt_path = recog_dir / recog_run / "recognizer" / "best.pt"
    recog_ckpt = torch.load(
        str(recog_ckpt_path), map_location=dev, weights_only=False
    )
    recog = PalmVeinRecognizer(
        config.recog_backbone,
        int(meta["class_id"].nunique()),
        config.recog_embedding_dim,
        config.recog_dropout,
        config.recog_margin,
        config.recog_scale,
        pretrained=False,
    ).to(dev)
    recog.load_state_dict(recog_ckpt["model_state"])
    recog.eval()

    ds = PalmVeinDataset(
        meta,
        split="test",
        image_size=config.image_size,
        target_kind="none",
        is_train=False,
        grayscale_to_rgb=config.grayscale_to_rgb,
    )
    loader = create_dataloader(
        ds,
        batch_size=config.eval_batch_size,
        num_workers=config.num_workers,
        shuffle=False,
    )

    all_embs, all_ids, all_scores = [], [], []
    with torch.no_grad():
        for b in loader:
            b = to_device(b, dev)
            scores = iqa_model(b["image"])
            emb, _ = recog(b["image"])
            all_embs.append(emb.cpu().numpy())
            all_ids.extend(b["class_id"].cpu().tolist())
            all_scores.extend(scores.cpu().tolist())

    embeddings = np.concatenate(all_embs, axis=0).astype(np.float32)
    class_ids = np.array(all_ids)
    qualities = np.array(all_scores)

    if config.eval_sample_size > 0 and len(embeddings) > config.eval_sample_size:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(embeddings), config.eval_sample_size, replace=False)
        embeddings = embeddings[idx]
        class_ids = class_ids[idx]
        qualities = qualities[idx]

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    embeddings /= norms

    rejection_rates = [0.0, 0.1, 0.2, 0.3]
    results = compute_rejection_accuracy(
        embeddings, class_ids, qualities, rejection_rates
    )

    _log(logger, "\n  --- Rejection Accuracy (rank-1) ---")
    total = len(qualities)
    for reject_rate in rejection_rates:
        keep_n = max(1, int(total * (1 - reject_rate)))
        acc = results[f"acc@{1 - reject_rate:.0%}"]
        _log(
            logger,
            f"  reject {reject_rate:.0%} → keep {keep_n}/{total} → acc={acc:.4f}",
        )

    return results


# ---------------------------------------------------------------------------
# 完整评估流程
# ---------------------------------------------------------------------------
def run_evaluation(
    config: Config,
    iqa_ckpt: str | Path,
    logger: ExperimentLogger | None = None,
) -> dict:
    """运行所有评估指标，返回汇总结果。"""
    results = {}

    _log(logger, "\n========== Evaluation ==========")

    r = evaluate_err_roi(config, iqa_ckpt, logger=logger)
    results.update(r)

    r = evaluate_eer_aoc(config, iqa_ckpt, logger=logger)
    results.update(r)

    r = evaluate_rejection_accuracy(config, iqa_ckpt, logger=logger)
    results.update(r)

    _log(logger, f"\nResults: {results}")
    return results
