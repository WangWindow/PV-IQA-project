from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

os.environ.setdefault("WANDB_MODE", "offline")

APP = Path(__file__).resolve().parents[1]
REPO = APP.parent
UPLOAD_ROOT = APP / "data" / "uploads"
DB_PATH = APP / "data" / "app.db"
BIN_CPU = APP / "bin" / "pv-iqa-cpu"
BIN_CUDA = APP / "bin" / "pv-iqa-cuda"

UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="PV-IQA")
app.mount("/uploads", StaticFiles(directory=str(UPLOAD_ROOT)), name="uploads")

con = sqlite3.connect(str(DB_PATH), check_same_thread=False)
con.row_factory = sqlite3.Row
con.executescript(
    """
    PRAGMA journal_mode = WAL;
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
    """
)
con.commit()

db_lock = threading.Lock()


def now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


@app.exception_handler(HTTPException)
async def handle_http_exception(_: Request, exc: HTTPException) -> JSONResponse:
    detail = exc.detail if isinstance(exc.detail, str) else str(exc.detail)
    return JSONResponse(status_code=exc.status_code, content={"error": detail})


@app.exception_handler(Exception)
async def handle_unhandled_exception(_: Request, exc: Exception) -> JSONResponse:
    return JSONResponse(status_code=500, content={"error": str(exc)})


def db_execute(query: str, params: tuple[object, ...] = ()) -> sqlite3.Cursor:
    with db_lock:
        cursor = con.execute(query, params)
        con.commit()
        return cursor


def db_fetchone(query: str, params: tuple[object, ...] = ()) -> sqlite3.Row | None:
    with db_lock:
        return con.execute(query, params).fetchone()


def db_fetchall(query: str, params: tuple[object, ...] = ()) -> list[sqlite3.Row]:
    with db_lock:
        return con.execute(query, params).fetchall()


def default_run_name() -> str:
    checkpoints = REPO / "checkpoints"
    if not checkpoints.is_dir():
        return ""

    candidates = sorted(
        (
            run_dir.name
            for run_dir in checkpoints.iterdir()
            if (run_dir / "iqa" / "best.onnx").exists()
        )
    )
    return candidates[0] if candidates else ""


def binary_available(path: Path) -> dict[str, object]:
    available = path.exists()
    return {"available": available, "state": "ready" if available else "error"}


def rust_score(run: str, paths: list[str], device: str) -> list[dict[str, object]]:
    onnx_path = REPO / "checkpoints" / run / "iqa" / "best.onnx"
    if not onnx_path.exists():
        raise RuntimeError(f"Rust model not found: {onnx_path}")

    binary = BIN_CUDA if device == "cuda" else BIN_CPU
    result = subprocess.run(
        [str(binary), "score", "--model", str(onnx_path), "--image", paths[0], "--device", device],
        capture_output=True, text=True, timeout=120,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip())
    data = json.loads(result.stdout)
    return [data]


def py_score(run: str, paths: list[str]) -> list[dict[str, object]]:
    sys.path.insert(0, str(REPO / "src"))
    from pv_iqa.config import Config
    from pv_iqa.inference import score_image

    config = Config()
    config.name = run
    config.metadata_path = str(REPO / "checkpoints" / run / "data" / "metadata.csv")
    checkpoint = str(REPO / "checkpoints" / run / "iqa" / "best.pt")
    return [score_image(config, checkpoint, path) for path in paths]


def do_score(run: str, paths: list[str], backend: str, device: str) -> list[dict[str, object]]:
    if backend == "rust":
        try:
            return rust_score(run, paths, device)
        except Exception:
            return py_score(run, paths)
    return py_score(run, paths)


def save_upload(upload: UploadFile, task_dir: str) -> tuple[str, str, str]:
    directory = UPLOAD_ROOT / task_dir
    directory.mkdir(parents=True, exist_ok=True)

    name = upload.filename or "unknown"
    path = directory / name
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        stem = Path(name).stem
        suffix = Path(name).suffix
        counter = 1
        while (directory / f"{stem}_{counter}{suffix}").exists():
            counter += 1
        name = f"{stem}_{counter}{suffix}"
        path = directory / name

    path.write_bytes(upload.file.read())
    return str(path), upload.filename or "", f"/uploads/{task_dir}/{name}"


def process_job(
    job_id: str,
    run_name: str,
    paths: list[tuple[str, str, str]],
    backend: str,
    device: str,
) -> None:
    sys.path.insert(0, str(REPO / "src"))
    from pv_iqa.config import Config
    from pv_iqa.inference import score_image

    config = Config()
    config.name = run_name
    config.metadata_path = str(REPO / "checkpoints" / run_name / "data" / "metadata.csv")
    checkpoint = str(REPO / "checkpoints" / run_name / "iqa" / "best.pt")
    total = len(paths)

    try:
        results: list[dict[str, object]] = []
        for index, (abs_path, relative_path, public_url) in enumerate(paths):
            if backend == "rust":
                try:
                    row = rust_score(run_name, [abs_path], device)[0]
                except Exception:
                    row = score_image(config, checkpoint, abs_path)
            else:
                row = score_image(config, checkpoint, abs_path)

            row["relative_path"] = relative_path
            row["public_url"] = public_url
            results.append(row)

            progress = round((index + 1) / total * 100, 1)
            db_execute(
                """
                UPDATE jobs
                   SET processed_count = ?,
                       progress = ?,
                       stage = ?,
                       updated_at = ?
                 WHERE id = ?
                """,
                (index + 1, progress, f"scoring {index + 1}/{total}", now_iso(), job_id),
            )

        scores = [float(row["quality_score"]) for row in results]
        average_score = sum(scores) / len(scores)
        best_score = max(scores)
        worst_score = min(scores)

        with db_lock:
            for row in results:
                con.execute(
                    """
                    INSERT INTO results (
                        job_id, image_path, relative_path, public_url, quality_score
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        job_id,
                        str(row["image_path"]),
                        str(row.get("relative_path", "")),
                        str(row.get("public_url", "")),
                        float(row["quality_score"]),
                    ),
                )
            con.execute(
                """
                UPDATE jobs
                   SET status = 'completed',
                       result_count = ?,
                       processed_count = ?,
                       progress = 100,
                       average_score = ?,
                       best_score = ?,
                       worst_score = ?,
                       completed_at = ?,
                       updated_at = ?,
                       stage = 'done'
                 WHERE id = ?
                """,
                (
                    len(results),
                    len(results),
                    average_score,
                    best_score,
                    worst_score,
                    now_iso(),
                    now_iso(),
                    job_id,
                ),
            )
            con.commit()
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


@app.get("/api/health")
async def health() -> dict[str, object]:
    default_name = default_run_name()
    cuda_available = BIN_CUDA.exists()
    return {
        "status": "ok",
        "port": 6005,
        "defaultRunName": default_name,
        "backends": {
            "python": {
                "available": True,
                "label": "Python",
                "state": "ready",
                "detail": "PyTorch",
                "device": "cuda",
            },
            "rust": {
                "available": True,
                "label": "Rust",
                "state": "ready",
                "detail": "Rust CLI",
                "device": "cuda" if cuda_available else "cpu",
            },
        },
    }


@app.get("/api/runs")
async def runs() -> list[str]:
    checkpoints = REPO / "checkpoints"
    if not checkpoints.is_dir():
        return []
    return sorted(
        run_dir.name
        for run_dir in checkpoints.iterdir()
        if (run_dir / "iqa" / "best.onnx").exists()
    )


@app.post("/api/score/image")
async def score_image_endpoint(
    file: UploadFile = File(...),
    backend: str = Form("python"),
    device: str = Form("cpu"),
    runName: str = Form(""),
) -> dict[str, object]:
    if not file.filename:
        raise HTTPException(400, "Missing image file.")

    run_name = runName.strip() or default_run_name()
    job_id = uuid.uuid4().hex
    task_dir = f"task_{job_id[:8]}"
    abs_path, relative_path, public_url = save_upload(file, task_dir)
    db_execute(
        """
        INSERT INTO jobs (
            id, kind, status, backend, device, run_name, input_count,
            processed_count, result_count, progress, stage, error, average_score,
            best_score, worst_score, created_at, updated_at, completed_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, 0, 0, 0, ?, NULL, NULL, NULL, NULL, ?, ?, NULL)
        """,
        (
            job_id,
            "image",
            "running",
            backend,
            device,
            run_name,
            1,
            "scoring",
            now_iso(),
            now_iso(),
        ),
    )
    threading.Thread(
        target=process_job,
        args=(job_id, run_name, [(abs_path, relative_path, public_url)], backend, device),
        daemon=True,
    ).start()
    job = db_fetchone("SELECT * FROM jobs WHERE id = ?", (job_id,))
    return {"job": {**dict(job or {}), "results": []}}


@app.post("/api/score/folder")
async def score_folder_endpoint(
    files: list[UploadFile] = File(...),
    backend: str = Form("python"),
    device: str = Form("cpu"),
    runName: str = Form(""),
) -> dict[str, object]:
    run_name = runName.strip() or default_run_name()
    job_id = uuid.uuid4().hex
    task_dir = f"task_{job_id[:8]}"
    saved = [save_upload(upload, task_dir) for upload in files if upload.filename]
    if not saved:
        raise HTTPException(400, "No images were uploaded.")

    db_execute(
        """
        INSERT INTO jobs (
            id, kind, status, backend, device, run_name, input_count,
            processed_count, result_count, progress, stage, error, average_score,
            best_score, worst_score, created_at, updated_at, completed_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, 0, 0, 0, ?, NULL, NULL, NULL, NULL, ?, ?, NULL)
        """,
        (
            job_id,
            "folder",
            "running",
            backend,
            device,
            run_name,
            len(saved),
            "scoring",
            now_iso(),
            now_iso(),
        ),
    )
    threading.Thread(
        target=process_job,
        args=(job_id, run_name, saved, backend, device),
        daemon=True,
    ).start()
    job = db_fetchone("SELECT * FROM jobs WHERE id = ?", (job_id,))
    return {"job": {**dict(job or {}), "results": []}}


@app.get("/api/jobs")
async def jobs() -> dict[str, object]:
    rows = db_fetchall("SELECT * FROM jobs ORDER BY created_at DESC")
    return {"jobs": [dict(row) for row in rows]}


@app.get("/api/jobs/{job_id}")
async def job(job_id: str) -> dict[str, object]:
    current = db_fetchone("SELECT * FROM jobs WHERE id = ?", (job_id,))
    if current is None:
        raise HTTPException(404, "Job not found.")
    results = db_fetchall(
        "SELECT * FROM results WHERE job_id = ? ORDER BY quality_score DESC",
        (job_id,),
    )
    return {"job": {**dict(current), "results": [dict(row) for row in results]}}


@app.delete("/api/jobs/{job_id}")
async def delete_job(job_id: str) -> dict[str, bool]:
    db_execute("DELETE FROM results WHERE job_id = ?", (job_id,))
    db_execute("DELETE FROM jobs WHERE id = ?", (job_id,))
    return {"ok": True}
