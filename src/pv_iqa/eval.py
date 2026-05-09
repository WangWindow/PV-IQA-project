from pathlib import Path

import pandas as pd
import torch
from tqdm.auto import tqdm

from pv_iqa.config import Config
from pv_iqa.models.iqa import IQARegressor
from pv_iqa.utils.common import ensure_dir, resolve_device, save_csv, to_device
from pv_iqa.utils.datasets import PalmVeinDataset, create_dataloader, load_metadata


def load_checkpoint(config: Config, path: str | Path) -> tuple[IQARegressor, torch.device]:
    dev = resolve_device(config.device)
    ckpt = torch.load(path, map_location=dev, weights_only=False)
    m = IQARegressor(ckpt.get("backbone", config.iqa_backbone), pretrained=False).to(dev)
    m.load_state_dict(ckpt["model_state"])
    m.eval()
    return m, dev

def predict_quality_scores(config: Config, ckpt: str | Path, split: str) -> pd.DataFrame:
    meta = load_metadata(config)
    model, dev = load_checkpoint(config, ckpt)
    ds = PalmVeinDataset(meta, split=split, image_size=config.image_size, target_kind="none", is_train=False, grayscale_to_rgb=config.grayscale_to_rgb)
    loader = create_dataloader(ds, batch_size=config.eval_batch_size, num_workers=config.num_workers, shuffle=False)
    recs = []
    with torch.no_grad():
        for b in tqdm(loader, desc="predict", leave=False):
            b = to_device(b, dev)
            scores = model(b["image"])
            for sid, cid, s in zip(b["sample_id"], b["class_id"], scores.cpu().tolist()):
                recs.append({"sample_id": sid, "class_id": int(cid), "predicted_quality": float(s), "split": split})
    df = pd.DataFrame(recs)
    save_csv(df, ensure_dir(config.experiment_dir / "evaluation") / f"{split}_predictions.csv")
    return df
