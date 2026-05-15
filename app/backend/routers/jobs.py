"""任务管理路由 — 任务的 CRUD、停止、恢复、重跑等操作。

普通用户只能查看和操作自己的任务，管理员可以查看所有任务。
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

from fastapi import APIRouter, Depends

from ..database import db_execute, db_fetchall, db_fetchone, now_iso
from ..middleware.error_handler import ForbiddenError, JobError
from ..routers.auth import require_admin, require_auth

router = APIRouter(prefix="/api/jobs", tags=["jobs"])


@router.get("")
async def jobs(user: dict = Depends(require_auth)) -> dict[str, object]:
    """获取任务列表。普通用户仅看自己的，管理员看全部。"""
    if user.get("role") == "admin":
        rows = db_fetchall("SELECT * FROM jobs ORDER BY created_at DESC")
    else:
        rows = db_fetchall(
            "SELECT * FROM jobs WHERE user_id = ? ORDER BY created_at DESC",
            (user["id"],),
        )
    return {"jobs": [dict(row) for row in rows]}


@router.get("/{job_id}")
async def job(job_id: str, user: dict = Depends(require_auth)) -> dict[str, object]:
    """获取单个任务详情（含评分结果）。"""
    current = db_fetchone("SELECT * FROM jobs WHERE id = ?", (job_id,))
    if current is None:
        raise JobError("任务不存在", code="JOB_NOT_FOUND", detail={"job_id": job_id})

    # 普通用户只能查看自己的任务
    job_data = dict(current)
    if user.get("role") != "admin" and job_data.get("user_id") != user["id"]:
        raise ForbiddenError("无权查看此任务")

    results = db_fetchall(
        "SELECT * FROM results WHERE job_id = ? ORDER BY quality_score DESC",
        (job_id,),
    )
    return {"job": {**job_data, "results": [dict(row) for row in results]}}


@router.delete("/{job_id}")
async def delete_job(job_id: str, user: dict = Depends(require_auth)) -> dict[str, bool]:
    """删除任务及其关联的评分结果。"""
    current = db_fetchone("SELECT * FROM jobs WHERE id = ?", (job_id,))
    if current is None:
        raise JobError("任务不存在", code="JOB_NOT_FOUND", detail={"job_id": job_id})

    # 普通用户只能删除自己的任务
    job_data = dict(current)
    if user.get("role") != "admin" and job_data.get("user_id") != user["id"]:
        raise ForbiddenError("无权删除此任务")

    db_execute("DELETE FROM results WHERE job_id = ?", (job_id,))
    db_execute("DELETE FROM jobs WHERE id = ?", (job_id,))
    return {"ok": True}


@router.post("/{job_id}/stop")
async def stop_job(job_id: str, user: dict = Depends(require_auth)) -> dict[str, object]:
    """停止正在运行的任务（标记为 interrupted）。"""
    current = db_fetchone("SELECT * FROM jobs WHERE id = ?", (job_id,))
    if current is None:
        raise JobError("任务不存在", code="JOB_NOT_FOUND")
    job_data = dict(current)
    if user.get("role") != "admin" and job_data.get("user_id") != user["id"]:
        raise ForbiddenError("无权操作此任务")
    if job_data.get("status") not in ("running",):
        raise JobError("只能停止运行中的任务", code="JOB_NOT_RUNNING")

    db_execute(
        "UPDATE jobs SET status='interrupted', updated_at=?, completed_at=? WHERE id=?",
        (now_iso(), now_iso(), job_id),
    )
    job = db_fetchone("SELECT * FROM jobs WHERE id = ?", (job_id,))
    return {"job": dict(job or {})}


@router.post("/{job_id}/resume")
async def resume_job(job_id: str, user: dict = Depends(require_auth)) -> dict[str, object]:
    """恢复被中断的任务。"""
    current = db_fetchone("SELECT * FROM jobs WHERE id = ?", (job_id,))
    if current is None:
        raise JobError("任务不存在", code="JOB_NOT_FOUND")
    job_data = dict(current)
    if user.get("role") != "admin" and job_data.get("user_id") != user["id"]:
        raise ForbiddenError("无权操作此任务")
    if job_data.get("status") != "interrupted":
        raise JobError("只能恢复被中断的任务", code="JOB_NOT_INTERRUPTED")

    db_execute(
        "UPDATE jobs SET status='running', updated_at=? WHERE id=?",
        (now_iso(), job_id),
    )
    job = db_fetchone("SELECT * FROM jobs WHERE id = ?", (job_id,))
    return {"job": dict(job or {})}


@router.post("/{job_id}/rerun")
async def rerun_job(job_id: str, user: dict = Depends(require_auth)) -> dict[str, object]:
    """重新运行任务（清除旧结果，从 uploads 目录恢复并重评分）。"""
    current = db_fetchone("SELECT * FROM jobs WHERE id = ?", (job_id,))
    if current is None:
        raise JobError("任务不存在", code="JOB_NOT_FOUND")
    job_data = dict(current)
    if user.get("role") != "admin" and job_data.get("user_id") != user["id"]:
        raise ForbiddenError("无权操作此任务")

    # 清除旧结果
    db_execute("DELETE FROM results WHERE job_id = ?", (job_id,))
    db_execute(
        """
        UPDATE jobs
           SET status='running',
               processed_count=0,
               result_count=0,
               progress=0,
               stage='scoring',
               error=NULL,
               average_score=NULL,
               best_score=NULL,
               worst_score=NULL,
               completed_at=NULL,
               updated_at=?
         WHERE id=?
        """,
        (now_iso(), job_id),
    )

    # 从 uploads 目录恢复原始图片路径并重新启动评分
    import threading

    from ..config import UPLOAD_ROOT
    from ..services.scoring import process_job

    task_dir_name = f"task_{job_id[:8]}"
    task_dir = UPLOAD_ROOT / task_dir_name
    if task_dir.is_dir():
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp", ".avif"}
        paths: list[tuple[str, str, str]] = []
        for img_path in sorted(task_dir.iterdir()):
            if img_path.suffix.lower() in image_extensions:
                relative = img_path.name
                public_url = f"/uploads/{task_dir_name}/{img_path.name}"
                paths.append((str(img_path), relative, public_url))

        if paths:
            threading.Thread(
                target=process_job,
                args=(job_id, job_data.get("run_name", ""), paths, job_data.get("backend", "python"), job_data.get("device", "cpu")),
                daemon=True,
            ).start()

    job = db_fetchone("SELECT * FROM jobs WHERE id = ?", (job_id,))
    return {"job": {**dict(job or {}), "results": []}}


@router.post("/batch-delete")
async def batch_delete_jobs(body: dict[str, list[str]], user: dict = Depends(require_auth)) -> dict[str, object]:
    """批量删除任务。普通用户只能删除自己的。"""
    job_ids = body.get("job_ids", [])
    if not job_ids:
        raise JobError("未提供要删除的任务 ID", code="JOB_IDS_EMPTY")

    deleted = 0
    for job_id in job_ids:
        if user.get("role") != "admin":
            # 普通用户检查归属
            row = db_fetchone("SELECT user_id FROM jobs WHERE id = ?", (job_id,))
            if row and dict(row).get("user_id") != user["id"]:
                continue  # 跳过非自己的任务
        db_execute("DELETE FROM results WHERE job_id = ?", (job_id,))
        cursor = db_execute("DELETE FROM jobs WHERE id = ?", (job_id,))
        if cursor.rowcount > 0:
            deleted += 1
    return {"ok": True, "deleted": deleted}


@router.post("/cleanup")
async def cleanup_jobs(admin: dict = Depends(require_admin)) -> dict[str, object]:
    """清理指定天数前已完成的任务（仅管理员）。"""
    days = 30
    # 从查询参数获取天数（简化处理）
    cutoff_dt = datetime.now(tz=timezone.utc) - timedelta(days=days)
    cutoff_str = cutoff_dt.isoformat()

    count_row = db_fetchone(
        "SELECT COUNT(*) as count FROM jobs WHERE status IN ('completed', 'failed', 'interrupted') AND completed_at < ?",
        (cutoff_str,),
    )
    count = dict(count_row or {}).get("count", 0) if count_row else 0

    db_execute(
        "DELETE FROM results WHERE job_id IN (SELECT id FROM jobs WHERE status IN ('completed', 'failed', 'interrupted') AND completed_at < ?)",
        (cutoff_str,),
    )
    db_execute(
        "DELETE FROM jobs WHERE status IN ('completed', 'failed', 'interrupted') AND completed_at < ?",
        (cutoff_str,),
    )
    return {"ok": True, "cleaned": count, "cutoff_days": days}