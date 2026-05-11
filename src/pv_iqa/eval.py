from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from tqdm.auto import tqdm

from pv_iqa.config import Config
from pv_iqa.models import PalmVeinIQARegressor
from pv_iqa.utils.common import ensure_dir, resolve_device, save_csv, to_device
from pv_iqa.utils.datasets import PalmVeinDataset, create_dataloader, load_metadata
from pv_iqa.utils.transforms import build_transforms

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def load_checkpoint(
    config: Config, path: str | Path
) -> tuple[PalmVeinIQARegressor, torch.device]:
    dev = resolve_device(config.device)
    ckpt = torch.load(path, map_location=dev, weights_only=False)
    m = PalmVeinIQARegressor(
        ckpt.get("backbone", config.iqa_backbone), pretrained=False
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
    config: Config, ckpt: str | Path, split: str
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
