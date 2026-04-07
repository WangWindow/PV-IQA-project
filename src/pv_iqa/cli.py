from __future__ import annotations

import json

import typer

from pv_iqa.config import AppConfig, load_config, resolve_run_config
from pv_iqa.eval import evaluate_erc, predict_quality_scores
from pv_iqa.inference import predict_folder, score_folder, score_image
from pv_iqa.train.iqa import train_iqa
from pv_iqa.train.pseudo_labels import generate_pseudo_labels
from pv_iqa.train.recognition import (
    export_recognition_artifacts,
    train_recognizer,
)
from pv_iqa.utils.common import ensure_dir, set_seed
from pv_iqa.utils.datasets import build_metadata

app = typer.Typer(help="PV-IQA full pipeline CLI")


def _load(config_path: str, run_name: str | None = None) -> AppConfig:
    config = resolve_run_config(load_config(config_path), run_name=run_name)
    ensure_dir(config.experiment_dir)
    set_seed(config.runtime.seed)
    return config


@app.command("prepare-data")
def prepare_data(
    config_path: str = "configs/default.yaml",
    run_name: str | None = None,
) -> None:
    config = _load(config_path, run_name=run_name)
    frame = build_metadata(config)
    typer.echo(f"run directory: {config.experiment_dir}")
    typer.echo(f"metadata saved: {config.data.metadata_path} ({len(frame)} samples)")


@app.command("train-recognizer")
def train_recognizer_command(
    config_path: str = "configs/default.yaml",
    run_name: str | None = None,
) -> None:
    config = _load(config_path, run_name=run_name)
    checkpoint = train_recognizer(config)
    features = export_recognition_artifacts(config, checkpoint)
    typer.echo(f"run directory: {config.experiment_dir}")
    typer.echo(f"recognizer checkpoint: {checkpoint}")
    typer.echo(f"features exported: {features}")


@app.command("generate-pseudo-labels")
def generate_pseudo_labels_command(
    config_path: str = "configs/default.yaml",
    run_name: str | None = None,
) -> None:
    config = _load(config_path, run_name=run_name)
    path = generate_pseudo_labels(config)
    typer.echo(f"run directory: {config.experiment_dir}")
    typer.echo(f"pseudo labels saved: {path}")


@app.command("train-iqa")
def train_iqa_command(
    config_path: str = "configs/default.yaml",
    run_name: str | None = None,
) -> None:
    config = _load(config_path, run_name=run_name)
    checkpoint = train_iqa(config)
    typer.echo(f"run directory: {config.experiment_dir}")
    typer.echo(f"iqa checkpoint: {checkpoint}")


@app.command("evaluate")
def evaluate_command(
    config_path: str = "configs/default.yaml",
    run_name: str | None = None,
) -> None:
    config = _load(config_path, run_name=run_name)
    checkpoint_path = config.experiment_dir / "iqa" / "best.pt"
    quality_frame = predict_quality_scores(
        config=config,
        checkpoint_path=checkpoint_path,
        split=config.evaluation.split,
    )
    erc_path = evaluate_erc(config, quality_frame)
    typer.echo(f"run directory: {config.experiment_dir}")
    typer.echo(f"erc metrics saved: {erc_path}")


@app.command("detect-folder")
def detect_folder_command(
    image_root: str,
    config_path: str = "configs/default.yaml",
    run_name: str | None = None,
    json_output: bool = False,
) -> None:
    config = _load(config_path, run_name=run_name)
    checkpoint_path = config.experiment_dir / "iqa" / "best.pt"
    if json_output:
        typer.echo(
            json.dumps(
                {
                    "run_directory": str(config.experiment_dir),
                    "results": score_folder(config, checkpoint_path, image_root),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    typer.echo(f"run directory: {config.experiment_dir}")
    csv_path = predict_folder(config, checkpoint_path, image_root)
    typer.echo(f"folder predictions saved: {csv_path}")


@app.command("detect-image")
def detect_image_command(
    image_path: str,
    config_path: str = "configs/default.yaml",
    run_name: str | None = None,
    json_output: bool = True,
) -> None:
    config = _load(config_path, run_name=run_name)
    checkpoint_path = config.experiment_dir / "iqa" / "best.pt"
    result = score_image(config, checkpoint_path, image_path)
    if json_output:
        typer.echo(
            json.dumps(
                {
                    "run_directory": str(config.experiment_dir),
                    "result": result,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    typer.echo(f"run directory: {config.experiment_dir}")
    typer.echo(f"image score: {result['quality_score']:.6f}")


@app.command("run-all")
def run_all(
    config_path: str = "configs/default.yaml",
    run_name: str | None = None,
) -> None:
    config = _load(config_path, run_name=run_name)
    build_metadata(config)
    recognizer_checkpoint = train_recognizer(config)
    export_recognition_artifacts(config, recognizer_checkpoint)
    generate_pseudo_labels(config)
    train_iqa(config)
    quality_frame = predict_quality_scores(
        config=config,
        checkpoint_path=config.experiment_dir / "iqa" / "best.pt",
        split=config.evaluation.split,
    )
    erc_path = evaluate_erc(config, quality_frame)
    typer.echo(f"run directory: {config.experiment_dir}")
    typer.echo(f"pipeline finished, erc metrics: {erc_path}")


def main() -> None:
    app()
