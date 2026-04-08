from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def _build_tensor(
    image_path: str,
    image_size: int,
    grayscale_to_rgb: bool,
    normalize_mean: list[float],
    normalize_std: list[float],
) -> np.ndarray:
    with Image.open(Path(image_path)) as image:
        image = image.convert("L")
        if grayscale_to_rgb:
            image = image.convert("RGB")
        image = image.resize((image_size, image_size), resample=Image.Resampling.BILINEAR)
        array = np.asarray(image, dtype=np.float32) / 255.0

    if array.ndim == 2:
        array = array[np.newaxis, :, :]
        mean = np.asarray(normalize_mean[:1] or [0.0], dtype=np.float32)[:, None, None]
        std = np.asarray(normalize_std[:1] or [1.0], dtype=np.float32)[:, None, None]
    else:
        array = np.transpose(array, (2, 0, 1))
        mean = np.asarray(normalize_mean, dtype=np.float32)[:, None, None]
        std = np.asarray(normalize_std, dtype=np.float32)[:, None, None]

    return ((array - mean) / std).astype(np.float32, copy=False)


def main() -> None:
    stdin = sys.stdin.buffer
    stdout = sys.stdout.buffer

    while True:
        raw_line = stdin.readline()
        if not raw_line:
            break

        try:
            request = json.loads(raw_line)
            tensors = [
                _build_tensor(
                    image_path=image_path,
                    image_size=int(request["image_size"]),
                    grayscale_to_rgb=bool(request["grayscale_to_rgb"]),
                    normalize_mean=list(request["normalize_mean"]),
                    normalize_std=list(request["normalize_std"]),
                )
                for image_path in request["image_paths"]
            ]
            payload = np.stack(tensors, axis=0).astype(np.float32, copy=False).tobytes(order="C")
            stdout.write(b"\x00")
            stdout.write(struct.pack("<Q", len(payload)))
            stdout.write(payload)
            stdout.flush()
        except Exception as exc:  # noqa: BLE001
            message = str(exc).encode("utf-8", errors="replace")
            stdout.write(b"\x01")
            stdout.write(struct.pack("<Q", len(message)))
            stdout.write(message)
            stdout.flush()


if __name__ == "__main__":
    main()
