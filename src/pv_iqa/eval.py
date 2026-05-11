"""Evaluation utilities for PV-IQA: inference, metrics, and quality analysis."""

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
from pv_iqa.utils.transforms import build_transforms

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


def load_checkpoint(
    config: Config, path: str | Path,
) -> tuple[PalmVeinIQARegressor, torch.device]:
    dev = resolve_device(config.device)
    ckpt = torch.load(path, map_location=dev, weights_only=False)
    m = PalmVeinIQARegressor(
        ckpt.get("backbone", config.iqa_backbone), pretrained=False,
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
    config: Config, ckpt: str | Path, split: str,
) -> pd.DataFrame:
    meta = load_metadata(config)
    model, dev = load_checkpoint(config, ckpt)
    ds = PalmVeinDataset(
        meta, split=split, image_size=config.image_size,
        target_kind="none", is_train=False,
        grayscale_to_rgb=config.grayscale_to_rgb,
    )
    loader = create_dataloader(
        ds, batch_size=config.eval_batch_size,
        num_workers=config.num_workers, shuffle=False,
    )
    recs = []
    with torch.no_grad():
        for b in tqdm(loader, desc="predict", leave=False):
            b = to_device(b, dev)
            scores = model(b["image"])
            for sid, cid, s in zip(b["sample_id"], b["class_id"], scores.cpu().tolist()):
                recs.append({
                    "sample_id": sid, "class_id": int(cid),
                    "predicted_quality": float(s), "split": split,
                })
    df = pd.DataFrame(recs)
    save_csv(df, ensure_dir(config.experiment_dir / "evaluation") / f"{split}_predictions.csv")
    return df


# ---------------------------------------------------------------------------
# err_roi evaluation
# ---------------------------------------------------------------------------

def parse_err_roi_labels(err_roi_dir: str = "datasets/err_roi",
                          desc_path: str = "datasets/err_roi描述.txt") -> dict[str, int]:
    """Returns {filename_stem: 1} for high-quality, 0 for low-quality images."""
    text = Path(desc_path).read_text(encoding="utf-8")
    line2 = text.strip().split("\n")[1]
    high_quality = set()
    for token in line2.replace("、", ",").split(","):
        token = token.strip()
        if token.isdigit():
            high_quality.add(int(token))

    labels = {}
    for p in sorted(Path(err_roi_dir).iterdir()):
        if p.suffix.lower() not in EXTS:
            continue
        stem = p.stem
        try:
            num = int("".join(c for c in stem if c.isdigit()))
        except ValueError:
            continue
        labels[stem] = 1 if num in high_quality else 0
    return labels


def evaluate_err_roi(config: Config, ckpt_path: str | Path) -> dict:
    """Score err_roi images and compute AUC / ScoreGap / Overlap."""
    labels_map = parse_err_roi_labels()
    model, dev = load_checkpoint(config, ckpt_path)

    transform = build_transforms(image_size=config.image_size, is_train=False)
    scores, gts = [], []

    for p in sorted(Path("datasets/err_roi").iterdir()):
        stem = p.stem
        if stem not in labels_map:
            continue
        img = Image.open(p).convert("L")
        if config.grayscale_to_rgb:
            img = img.convert("RGB")
        x = transform(img).unsqueeze(0).to(dev)
        with torch.no_grad():
            s = float(model(x).item())
        scores.append(s)
        gts.append(labels_map[stem])

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

    print(
        f"  err_roi | AUC={auc:.4f}  Gap={score_gap:.2f}  "
        f"Overlap={overlap:.3f}  High={high_scores.mean():.1f}  "
        f"Low={low_scores.mean():.1f}"
    )

    return {
        "err_roi_auc": auc,
        "err_roi_score_gap": score_gap,
        "err_roi_overlap": overlap,
        "err_roi_high_mean": float(high_scores.mean()),
        "err_roi_low_mean": float(low_scores.mean()),
    }


# ---------------------------------------------------------------------------
# EER / AOC evaluation (PGRG Sec.IV-C)
# ---------------------------------------------------------------------------

def evaluate_eer_aoc(
    config: Config,
    iqa_ckpt: str | Path,
    recog_run: str = "auto",
) -> dict:
    """EER at rejection rates 0%, 5%, …, 30% on class-disjoint test split.

    AOC (Area of Curve) = 1 − ∫₀^0.95 EER(r) dr  (PGRG Eq.13-14).
    """
    meta = load_metadata(config)
    test_meta = meta[meta["split"] == "test"]
    if len(test_meta) == 0:
        print("  ⚠ No test split found")
        return {}

    print(
        f"\n  --- EER/AOC (test: {test_meta['class_id'].nunique()} unseen classes, "
        f"{len(test_meta)} images) ---"
    )

    dev = resolve_device(config.device)
    iqa_model, _ = load_checkpoint(config, iqa_ckpt)

    # Load recognizer for embedding extraction
    recog_dir = Path(config.output_root)
    if recog_run == "auto":
        recog_dir = recog_dir / config.name / "recognizer"
    else:
        recog_dir = recog_dir / recog_run / "recognizer"
    recog_ckpt = torch.load(str(recog_dir / "best.pt"), map_location=dev, weights_only=False)
    recog = PalmVeinRecognizer(
        config.recog_backbone, int(meta["class_id"].nunique()),
        config.recog_embedding_dim, config.recog_dropout,
        config.recog_margin, config.recog_scale, pretrained=False,
    ).to(dev)
    recog.load_state_dict(recog_ckpt["model_state"])
    recog.eval()

    ds = PalmVeinDataset(
        meta, split="test", image_size=config.image_size,
        target_kind="none", is_train=False,
        grayscale_to_rgb=config.grayscale_to_rgb,
    )
    loader = create_dataloader(
        ds, batch_size=config.eval_batch_size,
        num_workers=config.num_workers, shuffle=False,
    )

    recs = []
    with torch.no_grad():
        for b in loader:
            b = to_device(b, dev)
            quality = iqa_model(b["image"])
            emb, _ = recog(b["image"])
            for i in range(len(b["sample_id"])):
                recs.append({
                    "sample_id": b["sample_id"][i],
                    "class_id": int(b["class_id"][i]),
                    "quality": float(quality[i]),
                    "embedding": emb[i].cpu().numpy(),
                })

    df = pd.DataFrame(recs).sort_values("quality", ascending=True)
    total = len(df)

    rejection_rates = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
    eer_values = []

    print("  reject% | EER%")
    for rr in rejection_rates:
        keep_n = max(3, int(total * (1 - rr)))
        keep_df = df.tail(keep_n)

        embs = np.stack(keep_df["embedding"].values)
        class_ids = keep_df["class_id"].values
        N = len(keep_df)

        genuine, impostor = [], []
        for i in range(N):
            for j in range(i + 1, N):
                sim = float(np.dot(embs[i], embs[j]))
                if class_ids[i] == class_ids[j]:
                    genuine.append(sim)
                else:
                    impostor.append(sim)

        if len(genuine) == 0 or len(impostor) == 0:
            eer_values.append(float("nan"))
            continue

        genuine_arr = np.array(genuine)
        impostor_arr = np.array(impostor)
        thresholds = np.linspace(
            min(genuine_arr.min(), impostor_arr.min()),
            max(genuine_arr.max(), impostor_arr.max()), 1000,
        )

        best_eer = 1.0
        for t in thresholds:
            far = (impostor_arr >= t).mean()
            frr = (genuine_arr < t).mean()
            best_eer = min(best_eer, (far + frr) / 2.0)
        eer_values.append(best_eer)
        print(f"  {rr:6.0%} | {best_eer * 100:.4f}")

    aoc = 0.0
    for i in range(len(rejection_rates) - 1):
        if not np.isnan(eer_values[i]) and not np.isnan(eer_values[i + 1]):
            aoc += (eer_values[i] + eer_values[i + 1]) / 2.0 * (
                rejection_rates[i + 1] - rejection_rates[i]
            )
    print(f"  AOC = {aoc:.4f}")

    results = {"eer_aoc": float(aoc)}
    for rr, eer in zip(rejection_rates, eer_values):
        if not np.isnan(eer):
            results[f"eer@{1 - rr:.0%}"] = float(eer)
    return results


# ---------------------------------------------------------------------------
# Rejection accuracy
# ---------------------------------------------------------------------------

def evaluate_rejection_accuracy(
    config: Config,
    iqa_ckpt: str | Path,
    recog_run: str = "auto",
) -> dict:
    """Reject worst N% → measure recognition accuracy on remaining.

    Standard biometric IQA metric (PGRG / CR-FIQA / SDD-FIQA).
    """
    meta = load_metadata(config)
    dev = resolve_device(config.device)
    iqa_model, _ = load_checkpoint(config, iqa_ckpt)

    recog_dir = Path(config.output_root)
    if recog_run == "auto":
        recog_dir = recog_dir / config.name / "recognizer"
    else:
        recog_dir = recog_dir / recog_run / "recognizer"
    recog_ckpt = torch.load(str(recog_dir / "best.pt"), map_location=dev, weights_only=False)
    recog = PalmVeinRecognizer(
        config.recog_backbone, int(meta["class_id"].nunique()),
        config.recog_embedding_dim, config.recog_dropout,
        config.recog_margin, config.recog_scale, pretrained=False,
    ).to(dev)
    recog.load_state_dict(recog_ckpt["model_state"])
    recog.eval()

    ds = PalmVeinDataset(
        meta, split="test", image_size=config.image_size,
        target_kind="none", is_train=False,
        grayscale_to_rgb=config.grayscale_to_rgb,
    )
    loader = create_dataloader(
        ds, batch_size=config.eval_batch_size,
        num_workers=config.num_workers, shuffle=False,
    )

    recs = []
    with torch.no_grad():
        for b in loader:
            b = to_device(b, dev)
            scores = iqa_model(b["image"])
            _, logits = recog(b["image"])
            preds = logits.argmax(dim=1)
            for sid, cid, s, p in zip(
                b["sample_id"], b["class_id"],
                scores.cpu().tolist(), preds.cpu().tolist(),
            ):
                recs.append({
                    "sample_id": sid, "class_id": int(cid),
                    "quality": float(s), "predicted_class": p,
                })

    df = pd.DataFrame(recs).sort_values("quality", ascending=True)
    total = len(df)

    results = {}
    print("\n  --- Rejection Accuracy ---")
    for reject_rate in [0.0, 0.1, 0.2, 0.3]:
        keep_n = max(1, int(total * (1 - reject_rate)))
        keep_df = df.tail(keep_n)
        acc = (keep_df["class_id"] == keep_df["predicted_class"]).mean()
        results[f"acc@{1 - reject_rate:.0%}"] = float(acc)
        print(f"  reject {reject_rate:.0%} → keep {keep_n}/{total} → acc={acc:.4f}")

    return results


# ---------------------------------------------------------------------------
# Full evaluation pipeline
# ---------------------------------------------------------------------------

def run_evaluation(config: Config, iqa_ckpt: str | Path) -> dict:
    """Run all evaluation metrics and return aggregated results."""
    results = {}

    r = evaluate_err_roi(config, iqa_ckpt)
    results.update(r)

    r = evaluate_eer_aoc(config, iqa_ckpt)
    results.update(r)

    r = evaluate_rejection_accuracy(config, iqa_ckpt)
    results.update(r)

    return results
