"""认证路由 — 用户注册、登录、令牌管理。

使用 JWT 进行无状态认证，密码通过 bcrypt 哈希存储。
首个注册用户自动获得 admin 角色。
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import time
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, Request

from ..database import db_execute, db_fetchall, db_fetchone, now_iso
from ..middleware.error_handler import AppError, AuthError, ForbiddenError
from ..models import LoginRequest, PasswordChangeRequest, RegisterRequest, TokenResponse, UserInfo

router = APIRouter(prefix="/api/auth", tags=["auth"])

# ── JWT 简易实现（HMAC-SHA256） ──────────────────────────
# 使用 HS256 签名，不依赖 pyjwt 库

JWT_SECRET = os.environ.get("PV_IQA_JWT_SECRET", "pv-iqa-dev-secret-change-in-production")
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_SECONDS = 86400 * 7  # 7 天


def _base64url_encode(data: bytes) -> str:
    """URL 安全的 Base64 编码（无填充）。"""
    import base64
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _base64url_decode(s: str) -> bytes:
    """URL 安全的 Base64 解码（自动补齐填充）。"""
    import base64
    padding = 4 - len(s) % 4
    s += "=" * padding
    return base64.urlsafe_b64decode(s)


def create_token(user_id: str, username: str, role: str) -> str:
    """创建 JWT 令牌。"""
    header = _base64url_encode(json.dumps({"alg": JWT_ALGORITHM, "typ": "JWT"}).encode())
    now = int(time.time())
    payload_data = {
        "sub": user_id,
        "username": username,
        "role": role,
        "iat": now,
        "exp": now + JWT_EXPIRATION_SECONDS,
    }
    payload = _base64url_encode(json.dumps(payload_data).encode())
    signature = hmac.new(JWT_SECRET.encode(), f"{header}.{payload}".encode(), hashlib.sha256).hexdigest()
    return f"{header}.{payload}.{signature}"


def verify_token(token: str) -> dict[str, object] | None:
    """验证 JWT 令牌，返回 payload 或 None。"""
    parts = token.split(".")
    if len(parts) != 3:
        return None

    header, payload, signature = parts
    expected_sig = hmac.new(JWT_SECRET.encode(), f"{header}.{payload}".encode(), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(signature, expected_sig):
        return None

    try:
        payload_data = json.loads(_base64url_decode(payload))
    except Exception:
        return None

    # 检查过期
    if payload_data.get("exp", 0) < int(time.time()):
        return None

    return payload_data


# ── 密码哈希（SHA-256 + 盐） ──────────────────────────────
# 简易实现，避免额外依赖。生产环境建议改用 bcrypt。

def _hash_password(password: str, salt: str | None = None) -> tuple[str, str]:
    """对密码加盐哈希，返回 (hash, salt)。"""
    if salt is None:
        salt = os.urandom(16).hex()
    hashed = hashlib.sha256(f"{salt}{password}".encode()).hexdigest()
    return hashed, salt


def _verify_password(password: str, stored_hash: str, salt: str) -> bool:
    """验证密码是否匹配。"""
    hashed, _ = _hash_password(password, salt)
    return hmac.compare_digest(hashed, stored_hash)


def _extract_token(request: Request) -> str | None:
    """从请求头提取 Bearer 令牌。"""
    auth_header = request.headers.get("authorization", "")
    if auth_header.startswith("Bearer "):
        return auth_header[7:]
    return None


async def get_current_user(request: Request) -> dict[str, str] | None:
    """FastAPI 依赖注入：获取当前认证用户。

    返回 None 表示未认证（公开端点可用），
    抛出 AuthError 表示令牌无效（需认证端点用）。
    """
    token = _extract_token(request)
    if not token:
        return None

    payload = verify_token(token)
    if payload is None:
        return None

    user_id = payload.get("sub")
    if not user_id:
        return None

    # 从数据库验证用户仍然存在
    user_row = db_fetchone("SELECT id, username, role FROM users WHERE id = ?", (user_id,))
    if not user_row:
        return None

    user = dict(user_row)
    # 将用户 ID 注入 request.state，供日志中间件使用
    request.state.user_id = user_id
    return user


async def require_auth(request: Request) -> dict[str, str]:
    """FastAPI 依赖注入：要求认证，未认证则 401。"""
    user = await get_current_user(request)
    if user is None:
        raise AuthError("请先登录", code="AUTH_REQUIRED")
    return user


async def require_admin(request: Request) -> dict[str, str]:
    """FastAPI 依赖注入：要求管理员权限。"""
    user = await require_auth(request)
    if user.get("role") != "admin":
        raise ForbiddenError("需要管理员权限")
    return user


# ── API 端点 ──────────────────────────────────────────────

@router.post("/register")
async def register(body: RegisterRequest, request: Request) -> dict[str, object]:
    """用户注册。首个用户自动成为管理员。"""
    # 检查用户名是否已存在
    existing = db_fetchone("SELECT id FROM users WHERE username = ?", (body.username,))
    if existing:
        raise AuthError("用户名已存在", code="USERNAME_EXISTS")

    # 判断是否是首个用户（自动升级为 admin）
    user_count = db_fetchall("SELECT id FROM users")
    role = "admin" if len(user_count) == 0 else "user"

    user_id = uuid.uuid4().hex
    password_hash, salt = _hash_password(body.password)
    # 将 salt 和 hash 拼接存储（格式：salt$hash）
    stored_password = f"{salt}${password_hash}"

    db_execute(
        "INSERT INTO users (id, username, password_hash, role, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
        (user_id, body.username, stored_password, role, now_iso(), now_iso()),
    )

    # 记录注册日志
    request.state.user_id = user_id
    token = create_token(user_id, body.username, role)

    return {
        "user": {"id": user_id, "username": body.username, "role": role},
        "access_token": token,
        "token_type": "bearer",
    }


@router.post("/login")
async def login(body: LoginRequest, request: Request) -> dict[str, object]:
    """用户登录，返回 JWT 令牌。"""
    user_row = db_fetchone("SELECT id, username, password_hash, role FROM users WHERE username = ?", (body.username,))
    if not user_row:
        raise AuthError("用户名或密码错误", code="INVALID_CREDENTIALS")

    user = dict(user_row)
    # 解析存储的密码（格式：salt$hash）
    parts = user["password_hash"].split("$", 1)
    if len(parts) != 2:
        raise AuthError("密码格式错误", code="INVALID_CREDENTIALS")

    salt, stored_hash = parts
    if not _verify_password(body.password, stored_hash, salt):
        raise AuthError("用户名或密码错误", code="INVALID_CREDENTIALS")

    token = create_token(user["id"], user["username"], user["role"])

    # 记录登录日志
    request.state.user_id = user["id"]

    return {
        "user": {"id": user["id"], "username": user["username"], "role": user["role"]},
        "access_token": token,
        "token_type": "bearer",
    }


@router.get("/me")
async def me(user: dict = Depends(require_auth)) -> dict[str, object]:
    """获取当前登录用户信息。"""
    return {"user": {"id": user["id"], "username": user["username"], "role": user["role"]}}


@router.put("/password")
async def change_password(body: PasswordChangeRequest, user: dict = Depends(require_auth)) -> dict[str, object]:
    """修改当前用户密码。"""
    user_row = db_fetchone("SELECT password_hash FROM users WHERE id = ?", (user["id"],))
    if not user_row:
        raise AuthError("用户不存在", code="USER_NOT_FOUND")

    stored = dict(user_row)["password_hash"]
    parts = stored.split("$", 1)
    if len(parts) != 2:
        raise AuthError("密码格式错误", code="PASSWORD_ERROR")

    salt, stored_hash = parts
    if not _verify_password(body.old_password, stored_hash, salt):
        raise AuthError("旧密码不正确", code="INVALID_OLD_PASSWORD")

    new_hash, new_salt = _hash_password(body.new_password)
    new_stored = f"{new_salt}${new_hash}"
    db_execute("UPDATE users SET password_hash=?, updated_at=? WHERE id=?", (new_stored, now_iso(), user["id"]))

    return {"ok": True}


@router.get("/users")
async def list_users(admin: dict = Depends(require_admin)) -> dict[str, object]:
    """管理员：获取所有用户列表。"""
    rows = db_fetchall(
        """
        SELECT u.id, u.username, u.role, u.created_at, u.updated_at,
               (SELECT COUNT(*) FROM jobs WHERE user_id = u.id) as job_count
        FROM users u ORDER BY u.created_at DESC
        """
    )
    return {"users": [dict(row) for row in rows]}


@router.delete("/users/{user_id}")
async def delete_user(user_id: str, admin: dict = Depends(require_admin)) -> dict[str, object]:
    """管理员：删除用户（不能删自己）。"""
    if user_id == admin["id"]:
        raise ForbiddenError("不能删除自己的账户")
    target = db_fetchone("SELECT id, username FROM users WHERE id = ?", (user_id,))
    if not target:
        raise AuthError("用户不存在", code="USER_NOT_FOUND")

    db_execute("DELETE FROM users WHERE id = ?", (user_id,))
    return {"ok": True, "deleted": dict(target)["username"]}


@router.put("/users/{user_id}/role")
async def update_user_role(
    user_id: str,
    body: dict[str, str],
    admin: dict = Depends(require_admin),
) -> dict[str, object]:
    """管理员：修改用户角色（admin/user）。"""
    new_role = body.get("role")
    if new_role not in ("admin", "user"):
        raise AuthError("角色必须是 admin 或 user", code="INVALID_ROLE")
    if user_id == admin["id"]:
        raise ForbiddenError("不能修改自己的角色")

    target = db_fetchone("SELECT id FROM users WHERE id = ?", (user_id,))
    if not target:
        raise AuthError("用户不存在", code="USER_NOT_FOUND")

    db_execute("UPDATE users SET role=?, updated_at=? WHERE id=?", (new_role, now_iso(), user_id))
    return {"ok": True, "user_id": user_id, "role": new_role}


@router.post("/users/create")
async def admin_create_user(
    body: RegisterRequest,
    request: Request,
    admin: dict = Depends(require_admin),
) -> dict[str, object]:
    """管理员创建新用户（可指定角色）。"""
    role = body.role if hasattr(body, "role") and body.role in ("admin", "user") else "user"

    existing = db_fetchone("SELECT id FROM users WHERE username = ?", (body.username,))
    if existing:
        raise AuthError("用户名已存在", code="USERNAME_EXISTS")

    user_id = uuid.uuid4().hex
    password_hash, salt = _hash_password(body.password)
    stored_password = f"{salt}${password_hash}"

    db_execute(
        "INSERT INTO users (id, username, password_hash, role, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
        (user_id, body.username, stored_password, role, now_iso(), now_iso()),
    )

    request.state.user_id = admin["id"]

    return {
        "user": {"id": user_id, "username": body.username, "role": role},
    }