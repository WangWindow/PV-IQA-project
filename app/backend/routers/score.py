"""评分路由 — 单张/批量图片评分的 API 端点。

接收上传图片，创建后台评分任务并返回任务信息。
任务自动绑定当前登录用户。
"""

from __future__ import annotations

import threading
import uuid

from fastapi import APIRouter, Depends, File, Form, UploadFile

from ..database import db_execute, db_fetchone, now_iso
from ..routers.auth import require_auth
from ..services.scoring import default_run_name, process_job
from ..services.uploads import save_upload

router = APIRouter(prefix="/api", tags=["score"])


@router.post("/score/image")
async def score_image_endpoint(
    file: UploadFile = File(...),
    backend: str = Form("python"),
    device: str = Form("cpu"),
    runName: str = Form(""),
    user: dict = Depends(require_auth),
) -> dict[str, object]:
    """单张图片评分 — 上传一张图片并创建评分任务。"""
    if not file.filename:
        from ..middleware.error_handler import UploadError
        raise UploadError("缺少图片文件", code="MISSING_FILE")

    run_name = runName.strip() or default_run_name()
    job_id = uuid.uuid4().hex
    task_dir = f"task_{job_id[:8]}"
    abs_path, relative_path, public_url = save_upload(file, task_dir)

    db_execute(
        """
        INSERT INTO jobs (
            id, kind, status, backend, device, run_name, input_count, user_id,
            processed_count, result_count, progress, stage, error, average_score,
            best_score, worst_score, created_at, updated_at, completed_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0, 0, 0, ?, NULL, NULL, NULL, NULL, ?, ?, NULL)
        """,
        (
            job_id,
            "image",
            "running",
            backend,
            device,
            run_name,
            1,
            user["id"],
            "scoring",
            now_iso(),
            now_iso(),
        ),
    )

    # 在后台线程中执行评分，避免阻塞 API 响应
    threading.Thread(
        target=process_job,
        args=(job_id, run_name, [(abs_path, relative_path, public_url)], backend, device),
        daemon=True,
    ).start()

    job = db_fetchone("SELECT * FROM jobs WHERE id = ?", (job_id,))
    return {"job": {**dict(job or {}), "results": []}}


@router.post("/score/folder")
async def score_folder_endpoint(
    files: list[UploadFile] = File(...),
    backend: str = Form("python"),
    device: str = Form("cpu"),
    runName: str = Form(""),
    user: dict = Depends(require_auth),
) -> dict[str, object]:
    """批量图片评分 — 上传多张图片并创建评分任务。"""
    run_name = runName.strip() or default_run_name()
    job_id = uuid.uuid4().hex
    task_dir = f"task_{job_id[:8]}"
    saved = [save_upload(upload, task_dir) for upload in files if upload.filename]

    if not saved:
        from ..middleware.error_handler import UploadError
        raise UploadError("未上传任何图片", code="NO_IMAGES")

    db_execute(
        """
        INSERT INTO jobs (
            id, kind, status, backend, device, run_name, input_count, user_id,
            processed_count, result_count, progress, stage, error, average_score,
            best_score, worst_score, created_at, updated_at, completed_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0, 0, 0, ?, NULL, NULL, NULL, NULL, ?, ?, NULL)
        """,
        (
            job_id,
            "folder",
            "running",
            backend,
            device,
            run_name,
            len(saved),
            user["id"],
            "scoring",
            now_iso(),
            now_iso(),
        ),
    )

    threading.Thread(
        target=process_job,
        args=(job_id, run_name, saved, backend, device),
        daemon=True,
    ).start()

    job = db_fetchone("SELECT * FROM jobs WHERE id = ?", (job_id,))
    return {"job": {**dict(job or {}), "results": []}}