from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd
import torch
from PIL import Image
from sklearn.utils import check_random_state
from torch.utils.data import DataLoader, Dataset

from pv_iqa.config import AppConfig
from pv_iqa.utils.io import ensure_dir, save_frame
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


def _resolve_class_name(folder_name: str, identity_mode: str) -> str:
    if identity_mode == "merge_person":
        return folder_name.rsplit("_", maxsplit=1)[0]
    return folder_name


def _assign_split(items: list[Path], ratios: tuple[float, float, float]) -> list[str]:
    train_ratio, val_ratio, _ = ratios
    total = len(items)
    if total < 3:
        return ["train"] * total

    train_count = max(1, int(total * train_ratio))
    val_count = max(1, int(total * val_ratio))
    test_count = max(1, total - train_count - val_count)

    while train_count + val_count + test_count > total:
        if train_count > val_count and train_count > 1:
            train_count -= 1
        elif val_count > 1:
            val_count -= 1
        else:
            test_count -= 1

    splits = (
        ["train"] * train_count
        + ["val"] * val_count
        + ["test"] * max(0, total - train_count - val_count)
    )
    return splits[:total]


def build_metadata(config: AppConfig) -> pd.DataFrame:
    dataset_root = Path(config.data.root)
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")

    grouped_paths: dict[str, list[Path]] = defaultdict(list)
    for image_path in sorted(dataset_root.glob("*/*")):
        if image_path.suffix.lower() in IMAGE_EXTENSIONS:
            grouped_paths[image_path.parent.name].append(image_path)

    class_names = sorted(
        {_resolve_class_name(folder_name, config.data.identity_mode) for folder_name in grouped_paths}
    )
    class_to_id = {class_name: index for index, class_name in enumerate(class_names)}
    rng = check_random_state(config.runtime.seed)

    records: list[ImageRecord] = []
    for folder_name, image_paths in sorted(grouped_paths.items()):
        paths = list(image_paths)
        rng.shuffle(paths)
        splits = _assign_split(
            paths,
            (config.data.train_ratio, config.data.val_ratio, config.data.test_ratio),
        )
        class_name = _resolve_class_name(folder_name, config.data.identity_mode)
        person_id, hand_side = folder_name.rsplit("_", maxsplit=1)
        for image_path, split in zip(paths, splits, strict=True):
            records.append(
                ImageRecord(
                    image_path=str(image_path),
                    sample_id=f"{folder_name}/{image_path.stem}",
                    source_identity=folder_name,
                    merged_identity=person_id,
                    person_id=person_id,
                    hand_side=hand_side,
                    class_name=class_name,
                    class_id=class_to_id[class_name],
                    split=split,
                )
            )

    frame = pd.DataFrame(records)
    save_frame(frame, config.data.metadata_path)
    return frame


def load_metadata(config: AppConfig) -> pd.DataFrame:
    metadata_path = Path(config.data.metadata_path)
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Metadata not found: {metadata_path}. Run `pv-iqa prepare-data` first."
        )
    return pd.read_csv(metadata_path)


class PalmVeinDataset(Dataset[dict[str, torch.Tensor | int | float | str]]):
    def __init__(
        self,
        metadata: pd.DataFrame,
        *,
        split: str,
        image_size: int,
        target_kind: Literal["class_id", "quality_score", "none"],
        is_train: bool,
        grayscale_to_rgb: bool,
    ) -> None:
        self.frame = metadata.query("split == @split").reset_index(drop=True)
        self.target_kind = target_kind
        self.grayscale_to_rgb = grayscale_to_rgb
        self.transform = build_transforms(image_size=image_size, is_train=is_train)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | int | float | str]:
        row = self.frame.iloc[index]
        image = Image.open(Path(row["image_path"])).convert("L")
        if self.grayscale_to_rgb:
            image = image.convert("RGB")

        sample: dict[str, torch.Tensor | int | float | str] = {
            "image": self.transform(image),
            "sample_id": row["sample_id"],
            "class_id": int(row["class_id"]),
            "image_path": row["image_path"],
        }
        if self.target_kind == "class_id":
            sample["target"] = int(row["class_id"])
        elif self.target_kind == "quality_score":
            sample["target"] = float(row["quality_score"])
        return sample


def create_dataloader(
    dataset: Dataset[dict[str, torch.Tensor | int | float | str]],
    *,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    pin_memory: bool,
) -> DataLoader[dict[str, torch.Tensor | int | float | str]]:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=shuffle,
    )
