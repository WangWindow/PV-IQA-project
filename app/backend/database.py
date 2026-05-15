"""数据库管理 — 连接、迁移、CRUD 辅助函数。

职责：
  1. 初始化 SQLite 连接，设置 WAL 模式和 Row 工厂
  2. 版本化迁移（migrations 列表）
  3. 线程安全的 db_execute / db_fetchone / db_fetchall
  4. 应用启动时自动执行未完成的迁移
"""

from __future__ import annotations

import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

from .config import DB_PATH


def now_iso() -> str:
    """返回当前 UTC 时间的 ISO 格式字符串。"""
    return datetime.now(tz=timezone.utc).isoformat()

# ── 全局连接 ──────────────────────────────────────────────
# check_same_thread=False 允许后台线程写入，
# 所有 DB 访问通过 db_lock 串行化保证线程安全
_connection: sqlite3.Connection | None = None
_db_lock = threading.Lock()


def get_connection() -> sqlite3.Connection:
    """获取或创建全局数据库连接（懒初始化）。"""
    global _connection
    if _connection is None:
        _connection = sqlite3.connect(str(DB_PATH), check_same_thread=False)
        _connection.row_factory = sqlite3.Row
        _connection.execute("PRAGMA journal_mode = WAL")
        _connection.execute("PRAGMA foreign_keys = ON")
    return _connection


def db_execute(query: str, params: tuple[object, ...] = ()) -> sqlite3.Cursor:
    """执行写操作并自动提交。"""
    con = get_connection()
    with _db_lock:
        cursor = con.execute(query, params)
        con.commit()
        return cursor


def db_fetchone(query: str, params: tuple[object, ...] = ()) -> sqlite3.Row | None:
    """查询单行记录。"""
    con = get_connection()
    with _db_lock:
        return con.execute(query, params).fetchone()


def db_fetchall(query: str, params: tuple[object, ...] = ()) -> list[sqlite3.Row]:
    """查询多行记录。"""
    con = get_connection()
    with _db_lock:
        return con.execute(query, params).fetchall()


# ── 版本化迁移 ────────────────────────────────────────────
# 每个迁移是一段 SQL（可包含多条语句）。
# 迁移执行后，会在 _migrations 表中记录版本号。
# 新增迁移只需在列表末尾追加即可。

_MIGRATIONS: list[tuple[int, str]] = [
    # v1: 初始表结构（jobs + results）
    (
        1,
        """
        CREATE TABLE IF NOT EXISTS jobs (
            id TEXT PRIMARY KEY,
            kind TEXT,
            status TEXT,
            backend TEXT,
            device TEXT,
            run_name TEXT,
            input_count INTEGER,
            processed_count INTEGER DEFAULT 0,
            result_count INTEGER DEFAULT 0,
            progress REAL DEFAULT 0,
            stage TEXT DEFAULT '',
            error TEXT,
            average_score REAL,
            best_score REAL,
            worst_score REAL,
            created_at TEXT,
            updated_at TEXT,
            completed_at TEXT
        );
        CREATE TABLE IF NOT EXISTS results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            job_id TEXT REFERENCES jobs(id),
            image_path TEXT,
            relative_path TEXT,
            public_url TEXT,
            quality_score REAL
        );
        """,
    ),
    # v2: 用户表 — 支持登录认证
    (
        2,
        """
        CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            role TEXT DEFAULT 'user',
            created_at TEXT,
            updated_at TEXT
        );
        """,
    ),
    # v3: 操作审计日志表
    (
        3,
        """
        CREATE TABLE IF NOT EXISTS audit_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
            action TEXT NOT NULL,
            target_type TEXT,
            target_id TEXT,
            detail TEXT,
            ip_address TEXT,
            created_at TEXT
        );
        """,
    ),
    # v4: 系统设置表（key-value）
    (
        4,
        """
        CREATE TABLE IF NOT EXISTS settings (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL,
            updated_at TEXT,
            updated_by TEXT
        );
        """,
    ),
    # v5: jobs 表增加 priority、tags、notes 字段
    (
        5,
        """
        ALTER TABLE jobs ADD COLUMN priority INTEGER DEFAULT 0;
        ALTER TABLE jobs ADD COLUMN tags TEXT DEFAULT '[]';
        ALTER TABLE jobs ADD COLUMN notes TEXT DEFAULT '';
        """,
    ),
    # v6: jobs 表增加 user_id 字段，关联任务归属
    (
        6,
        """
        ALTER TABLE jobs ADD COLUMN user_id TEXT;
        """,
    ),
]


def _ensure_migration_table(con: sqlite3.Connection) -> None:
    """确保迁移版本跟踪表存在。"""
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS _migrations (
            version INTEGER PRIMARY KEY
        );
        """
    )


def run_migrations() -> None:
    """执行所有尚未应用的数据库迁移。

    在应用启动时调用一次。采用版本号追踪，
    只执行 _migrations 表中未记录的迁移。
    """
    con = get_connection()
    _ensure_migration_table(con)

    # 读取已执行版本
    applied = {
        row[0]
        for row in con.execute("SELECT version FROM _migrations").fetchall()
    }

    for version, sql in _MIGRATIONS:
        if version in applied:
            continue
        # 执行迁移脚本（可能包含多条语句）
        con.executescript(sql)
        con.execute("INSERT INTO _migrations (version) VALUES (?)", (version,))
        con.commit()