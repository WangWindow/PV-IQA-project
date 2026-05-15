"""图片信息路由 — 图片元数据提取。"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException, Query

from ..config import UPLOAD_ROOT
from ..services.metadata import extract_metadata

router = APIRouter(prefix="/api/images", tags=["images"])


@router.get("/metadata")
async def image_metadata(path: str = Query(..., description="图片路径（相对于 uploads 目录或绝对路径）")) -> dict[str, object]:
    """提取图片元数据（尺寸、亮度、对比度、直方图等）。

    参数 path 可以是：
      - 相对路径（如 task_abc123/image.jpg）
      - 绝对路径
      - 公开 URL 路径（如 /uploads/task_abc123/image.jpg）
    """
    # 处理公开 URL 前缀
    if path.startswith("/uploads/"):
        relative = path.removeprefix("/uploads/")
        full_path = UPLOAD_ROOT / relative
    elif Path(path).is_absolute():
        full_path = Path(path)
    else:
        full_path = UPLOAD_ROOT / path

    # 安全检查：确保路径在 uploads 目录内
    try:
        full_path = full_path.resolve()
        uploads_resolved = UPLOAD_ROOT.resolve()
        if not str(full_path).startswith(str(uploads_resolved)):
            raise HTTPException(403, "Access denied: path outside uploads directory")
    except Exception:
        raise HTTPException(400, "Invalid path")

    if not full_path.exists():
        raise HTTPException(404, f"Image not found: {path}")

    if not full_path.is_file():
        raise HTTPException(400, f"Not a file: {path}")

    try:
        metadata = extract_metadata(str(full_path))
        return {"metadata": metadata}
    except Exception as exc:
        raise HTTPException(500, f"Failed to extract metadata: {exc}")