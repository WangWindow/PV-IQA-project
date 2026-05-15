"""Pydantic 数据模型 — 请求/响应 schema 定义。

所有 API 的请求体和响应体统一在此定义，
保证类型安全和自动文档生成。
"""

from __future__ import annotations

from pydantic import BaseModel, Field

# ── 认证相关 ──────────────────────────────────────────────

class LoginRequest(BaseModel):
    """登录请求体。"""
    username: str = Field(..., min_length=1, max_length=64, description="用户名")
    password: str = Field(..., min_length=6, max_length=128, description="密码")


class RegisterRequest(BaseModel):
    """注册请求体。"""
    username: str = Field(..., min_length=2, max_length=64, description="用户名")
    password: str = Field(..., min_length=6, max_length=128, description="密码")
    role: str = Field(default="user", description="角色：admin 或 user")


class TokenResponse(BaseModel):
    """认证令牌响应。"""
    access_token: str
    token_type: str = "bearer"
    user: "UserInfo"


class UserInfo(BaseModel):
    """用户基本信息。"""
    id: str
    username: str
    role: str


class PasswordChangeRequest(BaseModel):
    """修改密码请求体。"""
    old_password: str = Field(..., min_length=6, description="旧密码")
    new_password: str = Field(..., min_length=6, description="新密码")


# ── 系统设置 ──────────────────────────────────────────────

class SettingItem(BaseModel):
    """单个设置项。"""
    key: str
    value: str

class SettingsUpdateRequest(BaseModel):
    """批量更新设置请求体。"""
    settings: list[SettingItem]


# ── 操作日志 ──────────────────────────────────────────────

class AuditLogQuery(BaseModel):
    """日志查询参数。"""
    user_id: str | None = None
    action: str | None = None
    target_type: str | None = None
    start_date: str | None = None      # ISO 格式
    end_date: str | None = None        # ISO 格式
    page: int = Field(1, ge=1)
    page_size: int = Field(20, ge=1, le=100)


# ── 任务管理 ──────────────────────────────────────────────

class JobNotesUpdate(BaseModel):
    """任务备注更新请求。"""
    notes: str = Field(default="", max_length=2000, description="备注内容")


class JobTagsUpdate(BaseModel):
    """任务标签更新请求。"""
    tags: list[str] = Field(default_factory=list, description="标签列表")


class BatchDeleteRequest(BaseModel):
    """批量删除任务请求。"""
    job_ids: list[str] = Field(..., min_length=1, description="要删除的任务 ID 列表")


class CleanupRequest(BaseModel):
    """清理过期任务请求。"""
    days: int = Field(30, ge=1, description="清理多少天前的已完成任务")


# ── 图片元数据 ────────────────────────────────────────────

class ImageMetadata(BaseModel):
    """图片元数据响应。"""
    filename: str
    format: str | None = None
    size_bytes: int
    size_human: str
    width: int
    height: int
    mode: str | None = None
    dpi: tuple[float, float] | None = None
    brightness: float | None = None
    contrast: float | None = None
    snr_estimate: float | None = None
    histogram: dict[str, list[int]] | None = None
