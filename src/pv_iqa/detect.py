from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from PIL import Image

from pv_iqa.config import AppConfig
from pv_iqa.models.iqa import LightweightIQARegressor
from pv_iqa.train import resolve_device
from pv_iqa.utils.data import IMAGE_EXTENSIONS
from pv_iqa.utils.io import ensure_dir, save_frame, save_json
from pv_iqa.utils.transforms import build_transforms


def _load_model(config: AppConfig, checkpoint_path: str | Path) -> tuple[LightweightIQARegressor, torch.device]:
    device = resolve_device(config)
    model = LightweightIQARegressor(
        backbone_name=config.iqa.backbone,
        pretrained=False,
    ).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, device


def score_image(
    config: AppConfig,
    checkpoint_path: str | Path,
    image_path: str | Path,
) -> dict[str, float | str]:
    model, device = _load_model(config, checkpoint_path)
    transform = build_transforms(image_size=config.data.image_size, is_train=False)
    resolved = Path(image_path)
    image = Image.open(resolved).convert("L")
    if config.data.grayscale_to_rgb:
        image = image.convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        score = float(model(tensor)["score"].item())
    return {"image_path": str(resolved), "quality_score": score}


def score_folder(
    config: AppConfig,
    checkpoint_path: str | Path,
    image_root: str | Path,
) -> list[dict[str, float | str]]:
    model, device = _load_model(config, checkpoint_path)
    transform = build_transforms(image_size=config.data.image_size, is_train=False)
    records: list[dict[str, float | str]] = []

    for image_path in sorted(Path(image_root).rglob("*")):
        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        image = Image.open(image_path).convert("L")
        if config.data.grayscale_to_rgb:
            image = image.convert("RGB")
        tensor = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            score = float(model(tensor)["score"].item())
        records.append({"image_path": str(image_path), "quality_score": score})
    return records


def predict_folder(
    config: AppConfig,
    checkpoint_path: str | Path,
    image_root: str | Path,
) -> Path:
    records = score_folder(config, checkpoint_path, image_root)

    output_path = ensure_dir(config.experiment_dir / "detect") / "folder_predictions.csv"
    save_frame(pd.DataFrame(records), output_path)
    return output_path


def predict_image(
    config: AppConfig,
    checkpoint_path: str | Path,
    image_path: str | Path,
) -> Path:
    record = score_image(config, checkpoint_path, image_path)
    output_path = ensure_dir(config.experiment_dir / "detect") / "single_prediction.json"
    save_json(output_path, record)
    return output_path
