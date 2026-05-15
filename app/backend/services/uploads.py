"""上传服务 — 文件保存与路径管理。

职责：
  - save_upload: 将上传文件保存到 data/uploads 目录
  - 自动处理文件名冲突（递增后缀）
  - 返回绝对路径、原始文件名和公开 URL
"""

from __future__ import annotations

from pathlib import Path

from fastapi import UploadFile

from ..config import UPLOAD_ROOT


def save_upload(upload: UploadFile, task_dir: str) -> tuple[str, str, str]:
    """保存上传文件到磁盘。

    返回:
        (绝对路径, 原始文件名, 公开URL路径)
    """
    directory = UPLOAD_ROOT / task_dir
    directory.mkdir(parents=True, exist_ok=True)

    name = upload.filename or "unknown"
    path = directory / name
    path.parent.mkdir(parents=True, exist_ok=True)

    # 处理文件名冲突：同名文件自动加序号后缀
    if path.exists():
        stem = Path(name).stem
        suffix = Path(name).suffix
        counter = 1
        while (directory / f"{stem}_{counter}{suffix}").exists():
            counter += 1
        name = f"{stem}_{counter}{suffix}"
        path = directory / name

    path.write_bytes(upload.file.read())
    return str(path), upload.filename or "", f"/uploads/{task_dir}/{name}"