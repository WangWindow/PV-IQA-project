"""PGRG 风格排序预训练的图像退化生成器。

退化类型：
  - gaussian_blur:  高斯模糊 (模糊)
  - overexpose:     过曝
  - underexpose:    过暗
  - occlude:        不完整（角落遮挡，黑色填充）

所有操作均在 [-1, 1] 范围的 torch 张量（归一化后）上执行。
"""

import math

import torch
import torch.nn.functional as F
from torchvision.transforms.functional import gaussian_blur

DEGRADE_TYPES = [
    "gaussian_blur",
    "overexpose",
    "underexpose",
    "occlude",
]

DEFAULT_LEVELS = {
    "gaussian_blur": [3, 7],
    "overexpose":    [1.5, 3.0],
    "underexpose":   [0.6, 0.2],
    "occlude":       [0.15, 0.30],
}


def _make_motion_kernel(kernel_size: int, angle_deg: float) -> torch.Tensor:
    ks2 = kernel_size // 2
    angle_rad = math.radians(angle_deg)
    cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
    kernel = torch.zeros(kernel_size, kernel_size, dtype=torch.float32)
    for i in range(kernel_size):
        y = i - ks2
        for j in range(kernel_size):
            x = j - ks2
            if abs(x * cos_a + y * sin_a) <= 0.5:
                kernel[i, j] = 1.0
    return kernel / kernel.sum()


def _apply_motion_blur(images: torch.Tensor, kernel_size: int = 7) -> torch.Tensor:
    B, C, H, W = images.shape
    out = images.clone()
    for i in range(B):
        angle = float(torch.randint(0, 180, (1,)).item())
        kernel = _make_motion_kernel(kernel_size, angle).to(images.device)
        kernel = kernel.view(1, 1, kernel_size, kernel_size).repeat(C, 1, 1, 1)
        out[i : i + 1] = F.conv2d(images[i : i + 1], kernel, padding=kernel_size // 2, groups=C)
    return out


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

    elif degrade_type == "occlude":
        B, C, H, W = images.shape
        out = images.clone()
        frac = min(level, 0.5)
        oh, ow = int(H * frac), int(W * frac)
        for i in range(B):
            corner = torch.randint(0, 4, (1,)).item()
            if corner == 0:
                out[i, :, :oh, :ow] = -1.0
            elif corner == 1:
                out[i, :, :oh, W - ow:] = -1.0
            elif corner == 2:
                out[i, :, H - oh:, :ow] = -1.0
            else:
                out[i, :, H - oh:, W - ow:] = -1.0
        return out

    raise ValueError(f"Unknown degrade_type: {degrade_type}")


def generate_ranking_pair(
    images: torch.Tensor,
    rng: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    d_idx = torch.randint(0, len(DEGRADE_TYPES), (1,), generator=rng).item()
    d_type = DEGRADE_TYPES[d_idx]
    levels = DEFAULT_LEVELS[d_type]
    mild = apply_degradation(images, d_type, levels[0])
    severe = apply_degradation(images, d_type, levels[1])
    return mild, severe
