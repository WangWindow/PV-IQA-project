"""FastAPI 依赖注入 — 提供可复用的请求级依赖。

当前导出认证相关的依赖，供路由模块使用。
后续可扩展数据库会话注入、限流等。
"""

from __future__ import annotations

# 依赖注入函数定义在 routers/auth.py 中（因为与 JWT 紧密耦合）
# 此文件仅作为依赖注入模块的占位和未来扩展入口
# 导出认证依赖供路由直接引用
from .routers.auth import get_current_user, require_admin, require_auth  # noqa: F401
