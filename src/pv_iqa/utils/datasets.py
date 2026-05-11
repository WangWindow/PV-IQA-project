"""Dataset utilities: metadata builder, PalmVeinDataset, and DataLoader factory."""

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.utils import check_random_state
from torch.utils.data import DataLoader, Dataset

from pv_iqa.config import Config
from pv_iqa.utils.common import save_csv
from pv_iqa.utils.transforms import build_transforms

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass(slots=True)
class ImageRecord:
    image_path: str
    sample_id: str
    source_identity: str
    merged_identity: str
    person_id: str
    hand_side: str
    class_name: str
    class_id: int
    split: str


def _resolve_class_name(folder: str, mode: str) -> str:
    return folder.rsplit("_", maxsplit=1)[0] if mode == "merge_person" else folder


def build_metadata(config: Config) -> pd.DataFrame:
    """Scan dataset directory and build class-disjoint metadata (PGRG Sec.IV-B).

    Three non-overlapping splits:
      recognition dataset → recognition_train / recognition_val
      IQA dataset         → iqa_train / iqa_val
      test dataset        → test
    """
    root = Path(config.data_root)
    if not root.exists():
        raise FileNotFoundError(f"Dataset not found: {root}")

    grouped: dict[str, list[Path]] = defaultdict(list)
    for p in sorted(root.glob("*/*")):
        if p.suffix.lower() in IMAGE_EXTENSIONS:
            grouped[p.parent.name].append(p)

    classes = sorted({_resolve_class_name(f, config.identity_mode) for f in grouped})
    c2id = {c: i for i, c in enumerate(classes)}
    rng = check_random_state(config.seed)

    class_ids = list(range(len(classes)))
    rng.shuffle(class_ids)

    n_recognition = int(len(classes) * config.class_recognition_ratio)
    n_iqa = int(len(classes) * config.class_iqa_ratio)
    recognition_classes = set(class_ids[:n_recognition])
    test_classes = set(class_ids[n_recognition + n_iqa :])

    # Split labels: recognition → "recognition_*", IQA → "train"/"val", test → "test"
    SPLIT_MAP = {
        "recognition": ("recognition_train", "recognition_val"),
        "iqa": ("train", "val"),
    }

    recs = []
    for folder, paths in sorted(grouped.items()):
        cn = _resolve_class_name(folder, config.identity_mode)
        cid = c2id[cn]
        pid, hs = folder.rsplit("_", maxsplit=1)

        if cid in test_classes:
            for img in paths:
                recs.append(
                    ImageRecord(
                        str(img),
                        f"{folder}/{img.stem}",
                        folder,
                        pid,
                        pid,
                        hs,
                        cn,
                        cid,
                        "test",
                    )
                )
        else:
            group = "recognition" if cid in recognition_classes else "iqa"
            tr_label, val_label = SPLIT_MAP[group]

            shuffled = [str(p) for p in paths]
            rng.shuffle(shuffled)
            n_train = max(1, int(len(shuffled) * 0.8))
            for i, img_path in enumerate(shuffled):
                sp = tr_label if i < n_train else val_label
                img = Path(img_path)
                recs.append(
                    ImageRecord(
                        str(img),
                        f"{folder}/{img.stem}",
                        folder,
                        pid,
                        pid,
                        hs,
                        cn,
                        cid,
                        sp,
                    )
                )

    df = pd.DataFrame(recs)
    save_csv(df, config.metadata_path)
    return df


def load_metadata(config: Config) -> pd.DataFrame:
    return pd.read_csv(config.metadata_path)


class PalmVeinDataset(Dataset):
    def __init__(
        self,
        meta: pd.DataFrame,
        split: str,
        image_size: int,
        target_kind: Literal["class_id", "quality_score", "none"],
        is_train: bool,
        grayscale_to_rgb: bool,
    ):
        self.frame = meta[meta["split"] == split].reset_index(drop=True)
        self.target_kind = target_kind
        self.transform = build_transforms(image_size=image_size, is_train=is_train)
        self._grayscale_to_rgb = grayscale_to_rgb

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict:
        row = self.frame.iloc[index]
        img = Image.open(row["image_path"]).convert("L")
        if self._grayscale_to_rgb:
            img = img.convert("RGB")

        item: dict = {
            "image": self.transform(img),
            "sample_id": row["sample_id"],
            "class_id": int(row["class_id"]),
        }
        if self.target_kind == "class_id":
            item["target"] = int(row["class_id"])
        elif self.target_kind == "quality_score":
            item["target"] = float(row["quality_score"])
        return item


def create_dataloader(
    ds: Dataset,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
) -> DataLoader:
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=shuffle,
    )
