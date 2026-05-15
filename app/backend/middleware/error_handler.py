"""异常处理中间件 — 统一的结构化错误响应。

所有 API 错误统一返回 JSON：
  { "code": "ERR_CODE", "message": "人类可读描述", "detail": null }
而非裸的 { "error": "..." }。

业务异常使用 AppError 子类，HTTP 异常保持 FastAPI 兼容。
"""

from __future__ import annotations

from typing import Any

from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

# ── 自定义异常类体系 ───────────────────────────────────────

class AppError(Exception):
    """应用业务异常基类。"""

    def __init__(
        self,
        code: str,
        message: str,
        status_code: int = 400,
        detail: Any = None,
    ) -> None:
        self.code = code
        self.message = message
        self.status_code = status_code
        self.detail = detail
        super().__init__(message)


class AuthError(AppError):
    """认证/权限相关错误。"""

    def __init__(
        self,
        message: str = "认证失败",
        code: str = "AUTH_FAILED",
        detail: Any = None,
    ) -> None:
        super().__init__(code=code, message=message, status_code=401, detail=detail)


class ForbiddenError(AppError):
    """权限不足错误。"""

    def __init__(
        self,
        message: str = "权限不足",
        code: str = "FORBIDDEN",
        detail: Any = None,
    ) -> None:
        super().__init__(code=code, message=message, status_code=403, detail=detail)


class JobError(AppError):
    """任务相关错误。"""

    def __init__(
        self,
        message: str,
        code: str = "JOB_ERROR",
        detail: Any = None,
    ) -> None:
        super().__init__(code=code, message=message, status_code=400, detail=detail)


class UploadError(AppError):
    """上传相关错误。"""

    def __init__(
        self,
        message: str,
        code: str = "UPLOAD_ERROR",
        detail: Any = None,
    ) -> None:
        super().__init__(code=code, message=message, status_code=400, detail=detail)


# ── 异常处理器注册 ─────────────────────────────────────────

def _error_payload(code: str, message: str, detail: Any = None) -> dict[str, Any]:
    """构造统一的错误响应体。"""
    payload: dict[str, Any] = {"code": code, "message": message}
    if detail is not None:
        payload["detail"] = detail
    return payload


async def app_error_handler(_request: Request, exc: AppError) -> JSONResponse:
    """处理自定义业务异常。"""
    return JSONResponse(
        status_code=exc.status_code,
        content=_error_payload(exc.code, exc.message, exc.detail),
    )


async def http_exception_handler(_request: Request, exc: Any) -> JSONResponse:
    """处理 FastAPI HTTPException，兼容旧代码。"""
    # 兼容 FastAPI 默认 HTTPException
    status_code = getattr(exc, "status_code", 500)
    detail = getattr(exc, "detail", str(exc))

    # 如果 detail 已经是结构化格式，直接透传
    if isinstance(detail, str):
        payload = _error_payload("HTTP_ERROR", detail)
    else:
        payload = detail if isinstance(detail, dict) else _error_payload("HTTP_ERROR", str(detail))

    return JSONResponse(status_code=status_code, content=payload)


async def validation_exception_handler(_request: Request, exc: RequestValidationError) -> JSONResponse:
    """处理请求验证错误（422）。"""
    errors = exc.errors()
    messages = []
    for err in errors:
        loc = " → ".join(str(l) for l in err.get("loc", []))
        messages.append(f"{loc}: {err.get('msg', '验证失败')}")
    return JSONResponse(
        status_code=422,
        content=_error_payload(
            "VALIDATION_ERROR",
            "请求参数验证失败",
            detail=messages,
        ),
    )


async def unhandled_exception_handler(_request: Request, exc: Exception) -> JSONResponse:
    """兜底处理所有未捕获的异常。"""
    return JSONResponse(
        status_code=500,
        content=_error_payload("INTERNAL_ERROR", "服务器内部错误", detail=str(exc)),
    )


def register_error_handlers(app: Any) -> None:
    """将所有异常处理器注册到 FastAPI 应用。"""
    from fastapi import HTTPException

    app.add_exception_handler(AppError, app_error_handler)
    app.add_exception_handler(HTTPException, http_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(Exception, unhandled_exception_handler)
