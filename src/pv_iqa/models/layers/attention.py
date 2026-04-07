from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class TransposedAttentionBlock(nn.Module):
    """在通道维上做转置注意力，突出判别性更强的深层响应。"""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = nn.BatchNorm2d(channels)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        qkv = self.qkv(self.norm(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = F.normalize(q.flatten(2), dim=-1)
        k = F.normalize(k.flatten(2), dim=-1)
        v = v.flatten(2)

        attention = torch.softmax((q @ k.transpose(1, 2)) * self.temperature, dim=-1)
        fused = (attention @ v).view(batch, channels, height, width)
        return x + self.proj(fused)
