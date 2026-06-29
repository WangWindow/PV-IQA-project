"""排序预训练的图像退化生成器。

退化类型：
  - gaussian_blur:    高斯模糊（torchvision）
  - overexpose:       过曝（逐像素 × level → clamp [-1,1]）
  - underexpose:      过暗（逐像素 × level → clamp [-1,1]）
  - corner_cut:       角落楔形切割（随机顶点出发，level=切割角度°）
  - low_resolution:   低分辨率（bilinear 缩至 level 再放大回原尺寸）
  - block_occlusion:  方块遮挡（随机位置黑色方块，level=边长占图像比）

所有操作在 [-1, 1] 范围的 torch 张量上执行。
"""

import math

import torch
import torch.nn.functional as F
from torchvision.transforms.functional import gaussian_blur

# fmt: off
DEGRADE_TYPES: dict[str, list[float]] = {
    "gaussian_blur":    [3,    5,    7,    9   ],
    "overexpose":       [1.5,  2,    2.5,  3   ],
    "underexpose":      [0.8,  0.6,  0.4,  0.2 ],
    "corner_cut":       [10,   20,   30,   40  ],
    "low_resolution":   [0.8,  0.6,  0.4,  0.2 ],
    "block_occlusion":  [0.05, 0.10, 0.15, 0.20],
}
# fmt: on


def apply_degradation(
    images: torch.Tensor,
    degrade_type: str,
    level: float,
) -> torch.Tensor:
    if degrade_type == "gaussian_blur":
        ks = int(level) if int(level) % 2 == 1 else int(level) + 1
        return gaussian_blur(images, kernel_size=ks)

    elif degrade_type == "overexpose":
        return torch.clamp(images * level, -1.0, 1.0)

    elif degrade_type == "underexpose":
        return torch.clamp(images * level, -1.0, 1.0)

    elif degrade_type == "corner_cut":
        B, C, H, W = images.shape
        angle_rad = math.radians(min(level, 90.0))

        Y, X = torch.meshgrid(
            torch.arange(H, device=images.device, dtype=torch.float32),
            torch.arange(W, device=images.device, dtype=torch.float32),
            indexing="ij",
        )

        dy0, dx0 = Y, X
        dy1, dx1 = Y, W - 1 - X
        dy2, dx2 = H - 1 - Y, X
        dy3, dx3 = H - 1 - Y, W - 1 - X

        wedge0 = torch.atan2(dy0, dx0) < angle_rad  # top-left
        wedge1 = torch.atan2(dy1, dx1) < angle_rad  # top-right
        wedge2 = torch.atan2(dy2, dx2) < angle_rad  # bottom-left
        wedge3 = torch.atan2(dy3, dx3) < angle_rad  # bottom-right

        masks = torch.stack([wedge0, wedge1, wedge2, wedge3])

        out = images.clone()
        corners = torch.randint(0, 4, (B,), device=images.device)
        for i in range(B):
            out[i, :, masks[corners[i]]] = -1.0
        return out

    elif degrade_type == "block_occlusion":
        B, C, H, W = images.shape
        s = max(8, int(min(H, W) * level))
        out = images.clone()
        for i in range(B):
            top = torch.randint(0, H - s + 1, (1,), device=images.device).item()
            left = torch.randint(0, W - s + 1, (1,), device=images.device).item()
            out[i, :, top : top + s, left : left + s] = -1.0
        return out

    elif degrade_type == "low_resolution":
        _, _, H, W = images.shape
        small_H = max(4, int(H * level))
        small_W = max(4, int(W * level))
        down = F.interpolate(
            images, size=(small_H, small_W), mode="bilinear", align_corners=False
        )
        return F.interpolate(down, size=(H, W), mode="bilinear", align_corners=False)

    raise ValueError(f"Unknown degrade_type: {degrade_type}")


def get_degrade_type_idx() -> dict[str, int]:
    """返回退化类型名称到整数索引的映射。"""
    return {name: i for i, name in enumerate(DEGRADE_TYPES.keys())}


def generate_ranking_pair(
    images: torch.Tensor,
    rng: torch.Generator | None = None,
):
    names = list(DEGRADE_TYPES.keys())
    d_idx = torch.randint(0, len(names), (1,), generator=rng).item()
    d_type = names[d_idx]
    levels = DEGRADE_TYPES[d_type]
    degraded = tuple(apply_degradation(images, d_type, lv) for lv in levels)
    return degraded, d_idx
