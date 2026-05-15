"""PV-IQA 后端应用入口。

职责仅限于：
  1. 创建 FastAPI 实例
  2. 执行数据库迁移
  3. 挂载中间件和路由
  4. 启动时打印提示信息

所有业务逻辑均在其他模块中，主文件保持精简。
启动命令：uvicorn app.backend.main:app --host 0.0.0.0 --port 6005 --reload
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from .config import UPLOAD_ROOT
from .database import run_migrations
from .middleware.error_handler import register_error_handlers
from .middleware.logging import AuditLogMiddleware
from .routers import auth, health, images, jobs, logs, score, settings

# ── 数据库迁移 ──────────────────────────────────────────
# 启动时自动执行所有未应用的迁移
run_migrations()

# ── 创建 FastAPI 应用 ──────────────────────────────────
app = FastAPI(title="PV-IQA", version="3.1.0")

# ── 静态文件服务（上传目录） ────────────────────────────
app.mount("/uploads", StaticFiles(directory=str(UPLOAD_ROOT)), name="uploads")

# ── 中间件注册 ──────────────────────────────────────────
# 注意：AuditLogMiddleware 使用纯 ASGI 实现（非 BaseHTTPMiddleware），
# 直接传给 app.add_middleware，无需额外参数
app.add_middleware(AuditLogMiddleware)
register_error_handlers(app)

# ── 路由注册 ────────────────────────────────────────────
app.include_router(health.router)
app.include_router(score.router)
app.include_router(jobs.router)
app.include_router(auth.router)
app.include_router(settings.router)
app.include_router(logs.router)
app.include_router(images.router)


@app.on_event("startup")
async def startup_event() -> None:
    """应用启动时执行的初始化逻辑。"""
    print("✅ PV-IQA 后端已启动")
    print(f"   📁 上传目录: {UPLOAD_ROOT}")
    print(f"   🔗 API 文档: http://localhost:6005/docs")