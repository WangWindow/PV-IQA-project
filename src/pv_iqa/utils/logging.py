from __future__ import annotations

import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any

import wandb

from pv_iqa.config import AppConfig
from pv_iqa.utils.io import ensure_dir, save_json


def setup_logging(log_dir: str | Path) -> logging.Logger:
    ensure_dir(log_dir)
    logger = logging.getLogger("pv_iqa")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(Path(log_dir) / "run.log", encoding="utf-8")
    file_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    logger.propagate = False
    return logger


class ExperimentLogger:
    def __init__(self, config: AppConfig, run_name: str, output_dir: Path) -> None:
        self.config = config
        self.run_name = run_name
        self.output_dir = output_dir
        self.logger = setup_logging(output_dir)
        self._wandb_enabled = config.logger.use_wandb
        self._wandb_run = None

        save_json(output_dir / "config.json", asdict(config))
        if self._wandb_enabled:
            self._wandb_run = wandb.init(
                project=config.logger.wandb_project,
                entity=config.logger.wandb_entity,
                name=run_name,
                mode=config.logger.wandb_mode,
                tags=config.logger.tags,
                config=asdict(config),
                dir=str(output_dir),
                reinit=True,
            )

    def info(self, message: str, **extra: Any) -> None:
        if extra:
            message = f"{message} | {extra}"
        self.logger.info(message)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        self.logger.info("metrics=%s step=%s", metrics, step)
        if self._wandb_run is not None:
            self._wandb_run.log(metrics, step=step)

    def log_artifact(self, name: str, path: str | Path) -> None:
        artifact_path = Path(path)
        self.logger.info("artifact=%s path=%s", name, artifact_path)
        if self._wandb_run is not None:
            artifact = wandb.Artifact(name=name, type="dataset")
            artifact.add_file(str(artifact_path))
            self._wandb_run.log_artifact(artifact)

    def finish(self) -> None:
        if self._wandb_run is not None:
            self._wandb_run.finish()
