from pathlib import Path

import torch

from pv_iqa.config import Config
from pv_iqa.models.iqa import IQARegressor
from pv_iqa.utils.common import save_json

M = [0.485, 0.456, 0.406]
S = [0.229, 0.224, 0.225]


def export_onnx(config: Config, ckpt: str | Path) -> Path:
    c = torch.load(ckpt, map_location="cpu", weights_only=False)
    m = IQARegressor(c.get("backbone", config.iqa_backbone), pretrained=False)
    m.load_state_dict(c["model_state"])
    m.eval()
    onnx_p = Path(ckpt).with_suffix(".onnx")
    torch.onnx.export(
        m,
        torch.zeros(1, 3, config.image_size, config.image_size),  # ty:ignore[invalid-argument-type]
        str(onnx_p),
        input_names=["image"],
        output_names=["score"],
        opset_version=14,
        do_constant_folding=True,
        dynamic_axes={"image": {0: "batch"}, "score": {0: "batch"}},
    )
    save_json(
        Path(ckpt).with_suffix(".onnx.json"),
        {
            "input_name": "image",
            "output_name": "score",
            "image_size": config.image_size,
            "grayscale_to_rgb": config.grayscale_to_rgb,
            "normalize_mean": M,
            "normalize_std": S,
        },
    )
    return onnx_p
