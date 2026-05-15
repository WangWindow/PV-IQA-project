"""设置路由 — 系统配置的查询与更新。"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from ..database import db_execute, db_fetchall, db_fetchone, now_iso
from ..middleware.error_handler import ForbiddenError
from ..routers.auth import require_admin
from ..models import SettingsUpdateRequest

router = APIRouter(prefix="/api/settings", tags=["settings"])

# ── 默认设置值 ──────────────────────────────────────────
DEFAULTS: dict[str, str] = {
    "default_backend": "python",
    "default_device": "cpu",
    "max_upload_size_mb": "100",
    "allowed_image_types": "jpg,jpeg,png,bmp,tiff,webp,avif",
    "job_retention_days": "90",
    "system_name": "PV-IQA",
}


@router.get("")
async def get_settings() -> dict[str, object]:
    """获取所有系统设置（公开端点，不暴露敏感信息）。"""
    rows = db_fetchall("SELECT key, value FROM settings")
    stored = {row["key"]: row["value"] for row in rows}
    # 合并默认值
    result = {**DEFAULTS, **stored}
    return {"settings": result}


@router.put("")
async def update_settings(
    body: SettingsUpdateRequest,
    admin: dict = Depends(require_admin),
) -> dict[str, object]:
    """批量更新系统设置（仅管理员）。"""
    allowed_keys = set(DEFAULTS.keys())
    updated = []
    for item in body.settings:
        if item.key not in allowed_keys:
            continue
        db_execute(
            "INSERT OR REPLACE INTO settings (key, value, updated_at, updated_by) VALUES (?, ?, ?, ?)",
            (item.key, item.value, now_iso(), admin["id"]),
        )
        updated.append(item.key)
    return {"ok": True, "updated": updated}


@router.get("/db-stats")
async def db_stats(admin: dict = Depends(require_admin)) -> dict[str, object]:
    """获取数据库统计信息（仅管理员）。"""
    jobs_count = db_fetchone("SELECT COUNT(*) as count FROM jobs")
    results_count = db_fetchone("SELECT COUNT(*) as count FROM results")
    users_count = db_fetchone("SELECT COUNT(*) as count FROM users")
    logs_count = db_fetchone("SELECT COUNT(*) as count FROM audit_logs")

    return {
        "stats": {
            "jobs": dict(jobs_count or {})["count"] if jobs_count else 0,
            "results": dict(results_count or {})["count"] if results_count else 0,
            "users": dict(users_count or {})["count"] if users_count else 0,
            "audit_logs": dict(logs_count or {})["count"] if logs_count else 0,
        },
    }