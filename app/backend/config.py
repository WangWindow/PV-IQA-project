"""应用配置常量 — 路径、端口、密钥等集中管理。

所有硬编码路径和环境变量统一在此处定义，
其他模块通过 import config 引用，避免散落各处。
"""

from __future__ import annotations

import os
from pathlib import Path

# ── 环境变量 ──────────────────────────────────────────────
# 在 import 层尽早设置，确保下游依赖（如 wandb）生效
os.environ.setdefault("WANDB_MODE", "offline")

# ── 目录路径 ──────────────────────────────────────────────
APP_DIR = Path(__file__).resolve().parents[1]   # app/
REPO_DIR = APP_DIR.parent                       # PV-IQA-project/
UPLOAD_ROOT = APP_DIR / "data" / "uploads"
DB_PATH = APP_DIR / "data" / "app.db"

# ── 外部二进制路径 ────────────────────────────────────────
BIN_CPU = APP_DIR / "bin" / "pv-iqa-cpu"
BIN_CUDA = APP_DIR / "bin" / "pv-iqa-cuda"

# ── 服务端口 ──────────────────────────────────────────────
API_PORT = 6005

# ── 确保关键目录存在 ──────────────────────────────────────
UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
DB_PATH.parent.mkdir(parents=True, exist_ok=True)