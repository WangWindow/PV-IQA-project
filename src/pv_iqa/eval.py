"""PV-IQA 评估工具：推理、指标计算与质量分析。

可作为模块导入，也可直接运行：
    uv run python src/pv_iqa/eval.py checkpoints/xxx/iqa/best.pt datasets/my_images/
    uv run python src/pv_iqa/eval.py checkpoints/xxx/iqa/best.pt img.png -o report.md
"""

from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from pv_iqa.config import Config
from pv_iqa.models import PalmVeinIQARegressor, PalmVeinRecognizer
from pv_iqa.utils.common import resolve_device, to_device
from pv_iqa.utils.datasets import PalmVeinDataset, create_dataloader, load_metadata
from pv_iqa.utils.logging import ExperimentLogger
from pv_iqa.utils.metrics import compute_eer_from_embeddings, compute_rejection_accuracy
from pv_iqa.utils.transforms import build_transforms

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def _log(logger: ExperimentLogger | None, msg: str) -> None:
    if logger is not None:
        logger.info(msg)
    else:
        print(msg)


# ---------------------------------------------------------------------------
# 推理
# ---------------------------------------------------------------------------


def score_image(config: Config, ckpt: str | Path, img_path: str | Path) -> dict:
    model, device = _load_checkpoint(config, ckpt)
    t = build_transforms(image_size=config.image_size, is_train=False)
    img = Image.open(img_path).convert("L")
    if config.grayscale_to_rgb:
        img = img.convert("RGB")
    x = t(img).unsqueeze(0).to(device)
    with torch.no_grad():
        s = float(model(x).item())
    return {"image_path": str(img_path), "quality_score": s}


def score_folder(config: Config, ckpt: str | Path, folder: str | Path) -> list[dict]:
    model, device = _load_checkpoint(config, ckpt)
    t = build_transforms(image_size=config.image_size, is_train=False)
    res = []
    for p in sorted(Path(folder).rglob("*")):
        if p.suffix.lower() not in EXTS:
            continue
        img = Image.open(p).convert("L")
        if config.grayscale_to_rgb:
            img = img.convert("RGB")
        x = t(img).unsqueeze(0).to(device)
        with torch.no_grad():
            res.append({"image_path": str(p), "quality_score": float(model(x).item())})
    return res


# ---------------------------------------------------------------------------
# 测试集评估
# ---------------------------------------------------------------------------


def run_evaluation(
    config: Config,
    iqa_ckpt: str | Path,
    recog_run: str = "auto",
    logger: ExperimentLogger | None = None,
) -> dict:
    meta = load_metadata(config)
    test_meta = meta[meta["split"] == "test"]
    if len(test_meta) == 0:
        _log(logger, "  ⚠ No test split found")
        return {}

    _log(logger, "\n========== Evaluation ==========")
    _log(
        logger,
        f"  Test set: {test_meta['class_id'].nunique()} unseen classes, {len(test_meta)} images",
    )

    iqa_model, device = _load_checkpoint(config, iqa_ckpt)
    recog = _load_recognizer(config, meta, recog_run)
    loader = _make_test_loader(config, meta)

    recs = []
    with torch.no_grad():
        for b in loader:
            b = to_device(b, device)
            quality = iqa_model(b["image"])
            emb, _ = recog(b["image"])
            for i in range(len(b["sample_id"])):
                recs.append(
                    {
                        "class_id": int(b["class_id"][i]),
                        "quality": float(quality[i]),
                        "embedding": emb[i].cpu().numpy(),
                    }
                )

    df = pd.DataFrame(recs)
    if config.eval_sample_size > 0 and len(df) > config.eval_sample_size:
        df = df.iloc[
            np.random.default_rng(42).choice(
                len(df), config.eval_sample_size, replace=False
            )
        ]
    df = df.sort_values("quality", ascending=True)

    embs = np.stack(df["embedding"].values).astype(np.float32)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    embs /= norms
    ids = df["class_id"].values

    rejection_rates = [0.0, 0.1, 0.2, 0.3]
    results: dict[str, float] = {}

    eer_values, aoc = compute_eer_from_embeddings(embs, ids, rejection_rates)
    results["eer_aoc"] = float(aoc)
    for rr, eer in zip(rejection_rates, eer_values):
        if not np.isnan(eer):
            results[f"eer@{1 - rr:.0%}"] = float(eer)

    _log(logger, "  reject% | EER%")
    for rr, eer in zip(rejection_rates, eer_values):
        _log(logger, f"  {rr:6.0%} | {eer * 100:.4f}")
    _log(logger, f"  AOC = {aoc:.4f}")

    acc = compute_rejection_accuracy(embs, ids, df["quality"].values, rejection_rates)
    results.update(acc)

    _log(logger, "\n  --- Rejection Accuracy (rank-1) ---")
    total = len(df)
    for reject_rate in rejection_rates:
        keep_n = max(1, int(total * (1 - reject_rate)))
        _log(
            logger,
            f"  reject {reject_rate:.0%} → keep {keep_n}/{total} → acc={acc[f'acc@{1 - reject_rate:.0%}']:.4f}",
        )

    _log(logger, f"\nResults: {results}")
    return results


# ---------------------------------------------------------------------------
# 内部
# ---------------------------------------------------------------------------


def _load_recognizer(
    config: Config,
    meta: pd.DataFrame,
    recog_run: str,
) -> PalmVeinRecognizer:
    device = resolve_device(config.device)
    recog_dir = Path(config.output_root)
    if config.recog_checkpoint:
        path = Path(config.recog_checkpoint)
    elif recog_run == "auto":
        path = recog_dir / config.name / "recognizer" / "best.pt"
    else:
        path = recog_dir / recog_run / "recognizer" / "best.pt"
    ckpt = torch.load(str(path), map_location=device, weights_only=False)
    recog = PalmVeinRecognizer(
        config.recog_backbone,
        int(meta["class_id"].nunique()),
        config.recog_embedding_dim,
        config.recog_dropout,
        config.recog_margin,
        config.recog_scale,
        pretrained=False,
    ).to(device)
    recog.load_state_dict(ckpt["model_state"])
    recog.eval()
    return recog


def _make_test_loader(config: Config, meta: pd.DataFrame):
    ds = PalmVeinDataset(
        meta,
        split="test",
        image_size=config.image_size,
        target_kind="none",
        is_train=False,
        grayscale_to_rgb=config.grayscale_to_rgb,
    )
    return create_dataloader(
        ds,
        batch_size=config.eval_batch_size,
        num_workers=config.num_workers,
        shuffle=False,
    )


def _load_checkpoint(
    config: Config,
    path: str | Path,
) -> tuple[PalmVeinIQARegressor, torch.device]:
    device = resolve_device(config.device)
    ckpt = torch.load(path, map_location=device, weights_only=False)
    use_sa = ckpt.get("use_structure_aware", False)
    model = PalmVeinIQARegressor(
        ckpt.get("backbone", config.iqa_backbone),
        pretrained=False,
        use_structure_aware=use_sa,
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, device


# ---------------------------------------------------------------------------
# 输出
# ---------------------------------------------------------------------------


def _save_csv(results: list[dict], path: Path) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image_path", "quality_score"])
        writer.writeheader()
        writer.writerows(results)


def _save_markdown(results: list[dict], path: Path) -> None:
    lines = [
        "# 掌静脉图像质量评分",
        "",
        f"共 {len(results)} 张图像，按质量降序排列。",
        "",
        "| # | Image | Score |",
        "|---|-------|-------|",
    ]
    results = sorted(results, key=lambda r: r["quality_score"], reverse=True)
    for i, r in enumerate(results, 1):
        img_path = r["image_path"]
        score = r["quality_score"]
        lines.append(f"| {i} | ![]({img_path}) `{Path(img_path).name}` | {score:.2f} |")
    with open(path, "w") as f:
        f.write("\n".join(lines))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


# fmt: off
def main() -> None:
    parser = argparse.ArgumentParser(description="掌静脉图像质量评分")
    parser.add_argument("model", type=str, help="IQA 模型检查点路径 (.pt)")
    parser.add_argument("input", type=str, help="输入图像路径或文件夹")
    parser.add_argument("-o", "--output", type=str, default=None, help="输出文件路径（.csv 或 .md，默认: scores_<时间戳>.csv）")
    parser.add_argument("-d", "--device", type=str, default="cpu", help="推理设备 (default: cpu)")
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.is_file():
        sys.exit(f"模型文件不存在: {model_path}")

    input_path = Path(args.input)
    if not input_path.exists():
        sys.exit(f"输入路径不存在: {input_path}")

    output_path = Path(args.output or f"scores_{datetime.now():%Y%m%d_%H%M%S}.csv")

    config = Config()
    config.grayscale_to_rgb = True
    config.device = args.device

    if input_path.is_file():
        results = [score_image(config, str(model_path), str(input_path))]
    elif input_path.is_dir():
        print(f"正在扫描 {input_path} ...")
        results = score_folder(config, str(model_path), str(input_path))
        print(f"完成，共 {len(results)} 张图像。")
    else:
        sys.exit(f"输入必须为文件或文件夹: {input_path}")

    if output_path.suffix.lower() == ".md":
        _save_markdown(results, output_path)
    else:
        _save_csv(results, output_path)

    print(f"结果已保存至 {output_path}")
# fmt: on

if __name__ == "__main__":
    main()
