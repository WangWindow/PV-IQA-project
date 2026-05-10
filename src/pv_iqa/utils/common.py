import json
import random
import warnings
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        warnings.warn("CUDA not available when setting seed; continuing on CPU")


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(path: str | Path, data: dict[str, Any]) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    p.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )


def save_csv(df: pd.DataFrame, path: str | Path) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    df.to_csv(p, index=False)


def resolve_device(device: str) -> torch.device:
    try:
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # If user explicitly requests CUDA but it's not available, fall back to CPU
        if isinstance(device, str) and device.startswith("cuda"):
            if torch.cuda.is_available():
                return torch.device(device)
            warnings.warn(
                f"Requested device '{device}' not available; falling back to CPU"
            )
            return torch.device("cpu")

        return torch.device(device)
    except Exception as e:
        warnings.warn(f"Failed to resolve device '{device}': {e}. Falling back to CPU")
        return torch.device("cpu")


def autocast(device: torch.device, enabled: bool = True):
    if device.type == "cuda" and enabled:
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
        for k, v in batch.items()
    }
