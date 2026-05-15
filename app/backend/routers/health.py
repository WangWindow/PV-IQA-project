"""健康检查与运行列表路由。

提供系统健康状态和可用模型运行列表。
无需认证即可访问。
"""

from __future__ import annotations

from fastapi import APIRouter

from ..config import APP_DIR, BIN_CPU, BIN_CUDA, REPO_DIR, API_PORT
from ..services.scoring import binary_available, default_run_name

router = APIRouter(prefix="/api", tags=["health"])


@router.get("/health")
async def health() -> dict[str, object]:
    """系统健康检查，返回后端状态和默认模型信息。"""
    default_name = default_run_name()
    return {
        "status": "ok",
        "port": API_PORT,
        "defaultRunName": default_name,
        "backends": {
            "python": {
                "available": True,
                "label": "Python",
                "state": "ready",
                "detail": "PyTorch",
            },
            "rust": binary_available(BIN_CPU),
        },
    }


@router.get("/runs")
async def runs() -> list[str]:
    """返回所有可用的模型运行名称（有 best.onnx 的）。"""
    checkpoints = REPO_DIR / "checkpoints"
    if not checkpoints.is_dir():
        return []
    return sorted(
        run_dir.name
        for run_dir in checkpoints.iterdir()
        if (run_dir / "iqa" / "best.onnx").exists()
    )