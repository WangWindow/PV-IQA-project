"""评分服务 — 封装 Rust/Python 推理逻辑。

职责：
  - rust_score: 调用 Rust 编译的二进制进行评分
  - py_score: 调用 Python (PyTorch) 模型评分
  - do_score: 自动选择后端的评分入口
  - process_job: 后台线程执行的完整任务流程
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from ..config import BIN_CPU, BIN_CUDA, REPO_DIR
from ..database import db_execute, now_iso


def default_run_name() -> str:
    """返回最新的可用模型运行名称，无可用时返回空字符串。"""
    checkpoints = REPO_DIR / "checkpoints"
    if not checkpoints.is_dir():
        return ""

    candidates = sorted(
        (
            run_dir.name
            for run_dir in checkpoints.iterdir()
            if (run_dir / "iqa" / "best.onnx").exists()
        )
    )
    return candidates[-1] if candidates else ""


def binary_available(path: Path) -> dict[str, object]:
    """检查 Rust 二进制文件是否可用。"""
    available = path.exists()
    return {"available": available, "state": "ready" if available else "error"}


def rust_score(
    run: str,
    paths: list[str],
    device: str,
    job_id: str = "",
    total: int = 0,
) -> list[dict[str, object]]:
    """使用 Rust CLI 对图片进行质量评分。

    单张图片直接评分，批量图片走 batch 模式并流式上报进度。
    """
    onnx_path = REPO_DIR / "checkpoints" / run / "iqa" / "best.onnx"
    if not onnx_path.exists():
        raise RuntimeError(f"Rust model not found: {onnx_path}")

    binary = BIN_CUDA if device == "cuda" else BIN_CPU

    # 单张图片模式
    if len(paths) == 1:
        result = subprocess.run(
            [str(binary), "score", "--model", str(onnx_path), "--image", paths[0], "--device", device],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or result.stdout.strip())
        return [json.loads(result.stdout)]

    # 批量模式：从 stderr 流式读取进度
    batch_dir = str(Path(paths[0]).parent)
    proc = subprocess.Popen(
        [str(binary), "batch", "--model", str(onnx_path), "--dir", batch_dir, "--device", device],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    import select

    while proc.poll() is None:
        if proc.stderr:
            line = proc.stderr.readline()
            if line and line.startswith("PROGRESS:") and job_id:
                try:
                    parts = line.strip().split(":")[1]
                    done, total_str = parts.split("/")
                    pct = round(int(done) / int(total_str) * 100, 1)
                    db_execute(
                        "UPDATE jobs SET processed_count=?, progress=?, stage=?, updated_at=? WHERE id=?",
                        (int(done), pct, f"scoring {done}/{total_str}", now_iso(), job_id),
                    )
                except Exception:
                    pass

    stdout, stderr = proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError((stderr or stdout).strip())
    return json.loads(stdout)


def py_score(run: str, paths: list[str]) -> list[dict[str, object]]:
    """使用 Python (PyTorch) 模型对图片评分。"""
    sys.path.insert(0, str(REPO_DIR / "src"))
    from pv_iqa.config import Config
    from pv_iqa.eval import score_folder

    config = Config()
    config.name = run
    config.metadata_path = str(REPO_DIR / "checkpoints" / run / "data" / "metadata.csv")
    checkpoint = str(REPO_DIR / "checkpoints" / run / "iqa" / "best.pt")
    if len(paths) == 1:
        from pv_iqa.eval import score_image
        return [score_image(config, checkpoint, paths[0])]
    return score_folder(config, checkpoint, Path(paths[0]).parent)


def do_score(run: str, paths: list[str], backend: str, device: str) -> list[dict[str, object]]:
    """根据后端类型选择评分引擎。Rust 失败时降级为 Python。"""
    if backend == "rust":
        try:
            return rust_score(run, paths, device)
        except Exception:
            return py_score(run, paths)
    return py_score(run, paths)


def process_job(
    job_id: str,
    run_name: str,
    paths: list[tuple[str, str, str]],
    backend: str,
    device: str,
) -> None:
    """后台线程执行评分任务的完整流程。

    步骤：
      1. 选择后端并逐张图片评分
      2. 实时更新数据库进度
      3. 汇总评分结果并写入 results 表
      4. 更新任务状态为 completed / failed
    """
    from ..database import db_fetchone, get_connection, _db_lock

    sys.path.insert(0, str(REPO_DIR / "src"))
    from pv_iqa.config import Config
    from pv_iqa.eval import score_image

    config = Config()
    config.name = run_name
    config.metadata_path = str(REPO_DIR / "checkpoints" / run_name / "data" / "metadata.csv")
    checkpoint = str(REPO_DIR / "checkpoints" / run_name / "iqa" / "best.pt")
    total = len(paths)

    try:
        results: list[dict[str, object]] = []
        con = get_connection()

        if backend == "rust":
            all_paths = [p[0] for p in paths]
            db_execute(
                "UPDATE jobs SET stage=?, updated_at=? WHERE id=?",
                (f"scoring 0/{total}", now_iso(), job_id),
            )
            try:
                results = rust_score(run_name, all_paths, device, job_id, total)
            except Exception:
                # Rust 失败，降级为 Python 逐张评分
                for abs_path, relative_path, public_url in paths:
                    row = score_image(config, checkpoint, abs_path)
                    row["relative_path"] = relative_path
                    row["public_url"] = public_url
                    results.append(row)
                    db_execute(
                        "UPDATE jobs SET processed_count=?, progress=?, stage=?, updated_at=? WHERE id=?",
                        (len(results), round(len(results) / total * 100, 1), f"scoring {len(results)}/{total}", now_iso(), job_id),
                    )
        else:
            for abs_path, relative_path, public_url in paths:
                row = score_image(config, checkpoint, abs_path)
                row["relative_path"] = relative_path
                row["public_url"] = public_url
                results.append(row)
                db_execute(
                    "UPDATE jobs SET processed_count=?, progress=?, stage=?, updated_at=? WHERE id=?",
                    (len(results), round(len(results) / total * 100, 1), f"scoring {len(results)}/{total}", now_iso(), job_id),
                )

        # 批量写入结果
        with _db_lock:
            for row in results:
                con.execute(
                    "INSERT INTO results (job_id, image_path, relative_path, public_url, quality_score) VALUES (?, ?, ?, ?, ?)",
                    (job_id, str(row["image_path"]), str(row.get("relative_path", "")), str(row.get("public_url", "")), float(row["quality_score"])),
                )

        scores = [float(r["quality_score"]) for r in results]
        average_score = sum(scores) / len(scores)
        best_score = max(scores)
        worst_score = min(scores)

        db_execute(
            "UPDATE jobs SET status='completed', result_count=?, processed_count=?, progress=100, average_score=?, best_score=?, worst_score=?, completed_at=?, updated_at=?, stage='done' WHERE id=?",
            (len(results), len(results), average_score, best_score, worst_score, now_iso(), now_iso(), job_id),
        )

    except Exception as exc:
        db_execute(
            """
            UPDATE jobs
               SET status = 'failed',
                   error = ?,
                   completed_at = ?,
                   updated_at = ?
             WHERE id = ?
            """,
            (str(exc), now_iso(), now_iso(), job_id),
        )