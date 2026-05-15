"""请求日志中间件 — 记录所有 API 请求到审计日志。

使用纯 ASGI 中间件而非 BaseHTTPMiddleware，
避免 Starlette BaseHTTPMiddleware 与 FastAPI 依赖注入的已知冲突
（BaseHTTPMiddleware 会创建新的 Request scope，导致 Authorization header、
Depends 注入等出现异常）。

纯 ASGI 方式不创建新 scope，直接透传请求/响应。
"""

from __future__ import annotations

import time
from starlette.types import ASGIApp, Message, Receive, Scope, Send


class AuditLogMiddleware:
    """API 请求审计日志中间件（纯 ASGI 实现）。

    对每个 /api 请求记录：
      - 请求方法、路径
      - 响应状态码
      - 处理耗时（ms）
      - 用户 ID（若已认证）
      - 客户端 IP
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "")
        if not path.startswith("/api"):
            await self.app(scope, receive, send)
            return

        start = time.perf_counter()
        status_code = 200

        async def send_with_status(message: Message) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message.get("status", 200)
            await send(message)

        await self.app(scope, receive, send_with_status)

        elapsed_ms = round((time.perf_counter() - start) * 1000, 1)

        # 异步写入审计日志
        try:
            from ..database import db_execute, now_iso

            method = scope.get("method", "GET")
            action = _map_action(method, path)
            target_type = _map_target_type(path)
            target_id = _extract_id_from_path(path)

            # 从 ASGI scope 中读取认证用户 ID。
            # Starlette 的 request.state 底层存储在 scope["state"] 字典中，
            # 认证依赖 (get_current_user) 在处理请求时设置了 request.state.user_id，
            # 因此在响应完成后可通过 scope["state"] 读取。
            state = scope.get("state", {})
            user_id: str | None = None
            if isinstance(state, dict):
                user_id = state.get("user_id")
            # Starlette 的 State 对象也支持属性访问
            elif hasattr(state, "user_id"):
                user_id = getattr(state, "user_id", None)

            db_execute(
                """
                INSERT INTO audit_logs (user_id, action, target_type, target_id, detail, ip_address, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    user_id,
                    action,
                    target_type,
                    target_id,
                    f"{method} {path} → {status_code} ({elapsed_ms}ms)",
                    _get_client_ip(scope),
                    now_iso(),
                ),
            )
        except Exception:
            # 日志写入失败不应影响正常请求
            pass


def _map_action(method: str, path: str) -> str:
    """将 HTTP 方法和路径映射为操作名称。"""
    method_map = {"GET": "read", "POST": "create", "PUT": "update", "DELETE": "delete", "PATCH": "update"}
    if "/login" in path:
        return "login"
    if "/register" in path:
        return "register"
    if "/stop" in path:
        return "stop_job"
    if "/resume" in path:
        return "resume_job"
    if "/rerun" in path:
        return "rerun_job"
    return method_map.get(method, method.lower())


def _map_target_type(path: str) -> str:
    """从 URL 路径推断目标资源类型。"""
    if "/jobs" in path:
        return "job"
    if "/score" in path:
        return "job"
    if "/auth" in path:
        return "user"
    if "/settings" in path:
        return "setting"
    if "/logs" in path:
        return "log"
    if "/images" in path:
        return "image"
    return "other"


def _extract_id_from_path(path: str) -> str | None:
    """从 URL 路径中提取资源 ID。"""
    parts = path.rstrip("/").split("/")
    for i, part in enumerate(parts):
        if part in ("jobs", "score") and i + 1 < len(parts):
            candidate = parts[i + 1]
            if candidate and candidate not in ("image", "folder", "stop", "resume", "rerun", "batch-delete", "cleanup"):
                return candidate
    return None


def _get_client_ip(scope: Scope) -> str:
    """获取客户端真实 IP（考虑代理头）。"""
    headers = dict(scope.get("headers", []))
    forwarded = headers.get(b"x-forwarded-for")
    if forwarded:
        return forwarded.decode().split(",")[0].strip()
    client = scope.get("client")
    if client:
        return client[0]
    return "unknown"