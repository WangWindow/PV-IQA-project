"""审计日志路由 — 操作日志的分页查询与导出。"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from ..database import db_fetchall, db_fetchone
from ..middleware.error_handler import AppError
from ..routers.auth import require_admin

router = APIRouter(prefix="/api/logs", tags=["logs"])


@router.get("")
async def query_logs(
    user_id: str | None = None,
    action: str | None = None,
    target_type: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    admin: dict = Depends(require_admin),
) -> dict[str, object]:
    """分页查询审计日志（仅管理员）。

    支持按用户、操作类型、目标类型和时间范围筛选。
    """
    conditions: list[str] = []
    params: list[object] = []

    if user_id:
        conditions.append("user_id = ?")
        params.append(user_id)
    if action:
        conditions.append("action = ?")
        params.append(action)
    if target_type:
        conditions.append("target_type = ?")
        params.append(target_type)
    if start_date:
        conditions.append("created_at >= ?")
        params.append(start_date)
    if end_date:
        conditions.append("created_at <= ?")
        params.append(end_date)

    where = f" WHERE {' AND '.join(conditions)}" if conditions else ""

    # 查询总数
    count_row = db_fetchone(f"SELECT COUNT(*) as total FROM audit_logs{where}", tuple(params))
    total = dict(count_row or {})["total"] if count_row else 0

    # 分页查询
    offset = (page - 1) * page_size
    rows = db_fetchall(
        f"SELECT * FROM audit_logs{where} ORDER BY created_at DESC LIMIT ? OFFSET ?",
        tuple(params + [page_size, offset]),
    )

    return {
        "logs": [dict(row) for row in rows],
        "pagination": {
            "page": page,
            "page_size": page_size,
            "total": total,
            "total_pages": (total + page_size - 1) // page_size if total > 0 else 0,
        },
    }


@router.get("/export")
async def export_logs(
    admin: dict = Depends(require_admin),
) -> dict[str, object]:
    """导出所有审计日志（用于 CSV 下载）。"""
    rows = db_fetchall("SELECT * FROM audit_logs ORDER BY created_at DESC")
    return {"logs": [dict(row) for row in rows]}