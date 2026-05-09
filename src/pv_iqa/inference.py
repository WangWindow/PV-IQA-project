from pathlib import Path

import torch
from PIL import Image

from pv_iqa.config import Config
from pv_iqa.eval import load_checkpoint
from pv_iqa.utils.transforms import build_transforms

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


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
