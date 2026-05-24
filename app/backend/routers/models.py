from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter

from ..config import REPO_DIR

router = APIRouter(prefix="/api/models", tags=["models"])


@router.get("")
async def list_models() -> dict[str, object]:
    """返回所有可用模型的列表及元数据。"""
    checkpoints_dir = REPO_DIR / "checkpoints"
    models: list[dict] = []

    if not checkpoints_dir.is_dir():
        return {"models": models}

    for run_dir in sorted(checkpoints_dir.iterdir(), reverse=True):
        if not run_dir.is_dir():
            continue
        iqa_dir = run_dir / "iqa"
        if not iqa_dir.is_dir():
            continue

        has_pytorch = (iqa_dir / "best.pt").exists()
        has_onnx = (iqa_dir / "best.onnx").exists()
        if not has_pytorch and not has_onnx:
            continue

        model: dict[str, object] = {
            "run_name": run_dir.name,
            "has_pytorch": has_pytorch,
            "has_onnx": has_onnx,
        }

        # 加载元数据
        metadata_path = iqa_dir / "metadata.json"
        if metadata_path.exists():
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                model["name"] = metadata.get("name", "")
                model["date"] = metadata.get("date", "")
                model["duration"] = metadata.get("duration", "")
                model["overridden"] = metadata.get("overridden", [])
                model["params"] = metadata.get("params", {})
                model["metrics"] = metadata.get("metrics", {})
            except (json.JSONDecodeError, OSError):
                pass

        models.append(model)

    return {"models": models}
