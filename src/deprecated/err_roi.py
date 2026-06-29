"""已弃用的 err_roi 评估函数。

原位于 src/pv_iqa/eval.py，因指标不再需要而移至此处。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score

from pv_iqa.config import Config
from pv_iqa.eval import load_checkpoint
from pv_iqa.utils.logging import ExperimentLogger
from pv_iqa.utils.transforms import build_transforms

# 数据目录默认值（可按实际位置修改）
DATA_DIR = Path("datasets/err_roi")


def parse_err_roi_labels(
    labels_csv: str | None = None,
) -> dict[str, int]:
    """从 labels.csv 读取标签。返回 {文件名: 1/0}，高质量为 1。"""
    import pandas as pd

    path = labels_csv or str(DATA_DIR / "labels.csv")
    df = pd.read_csv(path, dtype={"image": str, "quality": int})
    labels: dict[str, int] = {}
    for _, row in df.iterrows():
        labels[str(row["image"]).strip()] = int(row["quality"])
    return labels


def evaluate_err_roi(
    config: Config,
    ckpt_path: str | Path,
    data_dir: str | Path | None = None,
    logger: ExperimentLogger | None = None,
) -> dict:
    """对 err_roi 图像评分，计算 AUC / ScoreGap / Overlap。"""
    labels_map = parse_err_roi_labels()
    model, dev = load_checkpoint(config, ckpt_path)

    img_dir = Path(data_dir or DATA_DIR)
    transform = build_transforms(image_size=config.image_size, is_train=False)
    scores, gts = [], []

    for p in sorted(img_dir.iterdir()):
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
    if logger:
        logger.info(msg)
    else:
        print(msg)

    return {
        "err_roi_auc": auc,
        "err_roi_score_gap": score_gap,
        "err_roi_overlap": overlap,
        "err_roi_high_mean": float(high_scores.mean()),
        "err_roi_low_mean": float(low_scores.mean()),
    }
