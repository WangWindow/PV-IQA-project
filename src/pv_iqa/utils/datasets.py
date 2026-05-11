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


def _assign_split(items: list[Path], ratios: tuple[float, float, float]) -> list[str]:
    tr, vr, _ = ratios
    n = len(items)
    if n < 3:
        return ["train"] * n
    tc, vc = max(1, int(n * tr)), max(1, int(n * vr))
    rest = n - tc - vc
    splits = ["train"] * tc + ["val"] * vc + ["test"] * max(0, rest)
    return splits[:n]


def build_metadata(config: Config) -> pd.DataFrame:
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
    recs = []

    if config.split_mode == "class":
        return _build_metadata_class_split(grouped, classes, c2id, rng, config)

    for folder, paths in sorted(grouped.items()):
        rng.shuffle(paths)
        splits = _assign_split(
            paths, (config.train_ratio, config.val_ratio, config.test_ratio)
        )
        cn = _resolve_class_name(folder, config.identity_mode)
        pid, hs = folder.rsplit("_", maxsplit=1)
        for img, sp in zip(paths, splits):
            recs.append(
                ImageRecord(
                    str(img),
                    f"{folder}/{img.stem}",
                    folder,
                    pid,
                    pid,
                    hs,
                    cn,
                    c2id[cn],
                    sp,
                )
            )
    df = pd.DataFrame(recs)
    save_csv(df, config.metadata_path)
    return df


def _build_metadata_class_split(
    grouped: dict[str, list[Path]],
    classes: list[str],
    c2id: dict[str, int],
    rng: np.random.RandomState,  # type: ignore[type-arg]
    config: Config,
) -> pd.DataFrame:
    """Class-disjoint split: recog / quality / test (PGRG Sec.IV-B).

    Classes (identities) are partitioned into up to three groups:
      - recog classes: for ArcFace recognizer training (if ratio > 0)
      - quality classes: split into IQA train/val by sample
      - test classes: held out entirely for final EER/AOC evaluation
    No class appears in more than one group.
    """
    class_ids = list(range(len(classes)))
    rng.shuffle(class_ids)

    n_recog = int(len(classes) * config.class_split_recog_ratio)
    n_quality = int(len(classes) * config.class_split_quality_ratio)

    recog_classes = set(class_ids[:n_recog]) if n_recog > 0 else set()
    test_classes = set(class_ids[n_recog + n_quality :])

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
        elif cid in recog_classes:
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
                        "recog",
                    )
                )
        else:
            shuffled = list(paths)
            rng.shuffle(shuffled)
            n_train = max(1, int(len(shuffled) * 0.8))
            for i, img in enumerate(shuffled):
                sp = "train" if i < n_train else "val"
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
        self.frame = meta.query("split == @split").reset_index(drop=True)
        self.target_kind = target_kind
        self.grayscale_to_rgb = grayscale_to_rgb
        self.transform = build_transforms(image_size=image_size, is_train=is_train)

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        img = Image.open(row["image_path"]).convert("L")
        if self.grayscale_to_rgb:
            img = img.convert("RGB")
        s = {
            "image": self.transform(img),
            "sample_id": row["sample_id"],
            "class_id": int(row["class_id"]),
            "image_path": row["image_path"],
        }
        if self.target_kind == "class_id":
            s["target"] = int(row["class_id"])
        elif self.target_kind == "quality_score":
            s["target"] = float(row["quality_score"])
        return s


def create_dataloader(ds: Dataset, batch_size: int, num_workers: int, shuffle: bool):
    return DataLoader(
        dataset=ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=shuffle,
    )
