"""图片元数据服务 — 提取图片尺寸、亮度、对比度、直方图等信息。

使用 Pillow 读取图片并计算统计指标，供前端预览组件展示。
不涉及算法代码，仅读取分析图片属性。
"""

from __future__ import annotations

import math
from pathlib import Path

from PIL import Image


def _format_file_size(size_bytes: int) -> str:
    """将字节数格式化为人类可读的文件大小。"""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    if size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    return f"{size_bytes / 1024 / 1024:.1f} MB"


def extract_metadata(image_path: str) -> dict[str, object]:
    """提取图片元数据，返回结构化字典。

    计算项：
      - 基础信息：文件名、格式、大小、尺寸、色彩模式、DPI
      - 亮度：像素灰度均值（0-255）
      - 对比度：灰度标准差
      - 信噪比估计：基于信号均值与噪声方差的比值
      - 颜色直方图：R/G/B 三通道 + 亮度通道（各 256 级）
    """
    path = Path(image_path)
    file_size = path.stat().st_size

    img = Image.open(path)
    width, height = img.size

    # 确保 RGB 模式进行统计分析
    img_rgb = img.convert("RGB") if img.mode != "RGB" else img

    # 基础信息
    result: dict[str, object] = {
        "filename": path.name,
        "format": img.format,
        "size_bytes": file_size,
        "size_human": _format_file_size(file_size),
        "width": width,
        "height": height,
        "mode": img.mode,
        "dpi": None,
        "brightness": None,
        "contrast": None,
        "snr_estimate": None,
        "histogram": None,
    }

    # DPI 信息
    if img.info.get("dpi"):
        dpi = img.info["dpi"]
        result["dpi"] = (float(dpi[0]), float(dpi[1]))

    # 计算亮度、对比度、信噪比
    pixels = list(img_rgb.getdata())
    total_pixels = len(pixels)
    if total_pixels > 0:
        # 灰度值 (ITU-R BT.601)
        gray_values = [0.299 * r + 0.587 * g + 0.114 * b for r, g, b in pixels]

        mean_brightness = sum(gray_values) / total_pixels
        variance = sum((g - mean_brightness) ** 2 for g in gray_values) / total_pixels
        std_deviation = math.sqrt(variance)

        result["brightness"] = round(mean_brightness, 1)
        result["contrast"] = round(std_deviation, 1)

        # 信噪比估计：信号均值 / 噪声标准差（避免除零）
        if std_deviation > 0 and mean_brightness > 0:
            snr = mean_brightness / std_deviation
            result["snr_estimate"] = round(snr, 1)

    # 颜色直方图
    try:
        histogram_data = img_rgb.histogram()
        # histogram_data 长度 = 通道数 * 256，按通道分组
        hist_r = histogram_data[0:256]
        hist_g = histogram_data[256:512]
        hist_b = histogram_data[512:768]

        # 亮度直方图（基于灰度值的 64 档近似直方图）
        lum_bins = [0] * 64
        if total_pixels > 0:
            for gv in gray_values:
                bin_index = min(int(gv / 4), 63)
                lum_bins[bin_index] += 1

        result["histogram"] = {
            "r": hist_r,
            "g": hist_g,
            "b": hist_b,
            "luminance": lum_bins,
        }
    except Exception:
        pass

    img.close()
    return result