from __future__ import annotations

from dataclasses import MISSING, dataclass, field, fields, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, get_args, get_origin, get_type_hints

import yaml


@dataclass(slots=True)
class LoggerConfig:
    use_wandb: bool = True
    wandb_project: str = "pv-iqa"
    wandb_entity: str | None = None
    wandb_mode: str = "offline"
    tags: list[str] = field(default_factory=lambda: ["pv", "iqa"])


@dataclass(slots=True)
class RuntimeConfig:
    seed: int = 42
    device: str = "auto"
    amp: bool = True
    num_workers: int = 4
    compile_model: bool = False


@dataclass(slots=True)
class DataConfig:
    root: str = "datasets/ROI_Data"
    metadata_path: str = "auto"
    identity_mode: str = "separate"
    image_size: int = 224
    batch_size: int = 32
    eval_batch_size: int = 64
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    grayscale_to_rgb: bool = True
    pin_memory: bool = True


@dataclass(slots=True)
class ExperimentConfig:
    name: str = "auto"
    output_root: str = "checkpoints"


@dataclass(slots=True)
class RecognizerConfig:
    backbone: str = "mobilenetv3_large_100"
    pretrained: bool = True
    embedding_dim: int = 256
    dropout: float = 0.1
    margin: float = 0.3
    scale: float = 30.0
    epochs: int = 12
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    warmup_epochs: int = 1
    train_split: str = "train"
    val_split: str = "val"


@dataclass(slots=True)
class PseudoLabelConfig:
    split: str = "all"
    alpha: float = 0.5
    adaptive_alpha: bool = True
    negative_samples: int = 512
    gmm_components: int = 2
    min_positive_count: int = 4
    eps: float = 1e-6


@dataclass(slots=True)
class IQAConfig:
    backbone: str = "mobilenetv3_large_100"
    pretrained: bool = True
    use_waveformer_layer: bool = False
    waveformer_mlp_ratio: float = 2.0
    epochs: int = 15
    learning_rate: float = 2e-4
    weight_decay: float = 5e-5
    huber_delta: float = 0.1
    ranking_margin: float = 0.03
    ranking_weight: float = 0.3
    min_ranking_gap: float = 0.05
    train_split: str = "train"
    val_split: str = "val"


@dataclass(slots=True)
class EvaluationConfig:
    split: str = "test"
    reject_steps: list[float] = field(
        default_factory=lambda: [0.0, 0.05, 0.1, 0.2, 0.3]
    )
    far_targets: list[float] = field(default_factory=lambda: [1e-2, 1e-3, 1e-4])
    max_impostor_pairs: int = 20000


@dataclass(slots=True)
class AppConfig:
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    logger: LoggerConfig = field(default_factory=LoggerConfig)
    data: DataConfig = field(default_factory=DataConfig)
    recognizer: RecognizerConfig = field(default_factory=RecognizerConfig)
    pseudo_labels: PseudoLabelConfig = field(default_factory=PseudoLabelConfig)
    iqa: IQAConfig = field(default_factory=IQAConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    @property
    def experiment_dir(self) -> Path:
        return Path(self.experiment.output_root) / self.experiment.name


def _convert_value(field_type: Any, value: Any) -> Any:
    origin = get_origin(field_type)
    if is_dataclass(field_type):
        return _build_dataclass(field_type, value)
    if origin is list:
        inner = get_args(field_type)[0]
        return [_convert_value(inner, item) for item in value]
    if origin in {tuple, set}:
        inner = get_args(field_type)[0]
        converted = [_convert_value(inner, item) for item in value]
        return origin(converted)
    if origin is None:
        return value
    if origin is dict:
        key_type, val_type = get_args(field_type)
        return {
            _convert_value(key_type, key): _convert_value(val_type, item)
            for key, item in value.items()
        }
    args = [arg for arg in get_args(field_type) if arg is not type(None)]
    if len(args) == 1:
        return _convert_value(args[0], value)
    return value


def _build_dataclass(schema: type[Any], data: dict[str, Any] | None) -> Any:
    values: dict[str, Any] = {}
    payload = data or {}
    type_hints = get_type_hints(schema)
    for field_info in fields(schema):
        field_type = type_hints.get(field_info.name, field_info.type)
        if field_info.name in payload:
            values[field_info.name] = _convert_value(
                field_type, payload[field_info.name]
            )
        elif field_info.default is not MISSING:
            values[field_info.name] = field_info.default
        elif field_info.default_factory is not MISSING:  # type: ignore[attr-defined]
            values[field_info.name] = field_info.default_factory()
    return schema(**values)


def load_config(path: str | Path) -> AppConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    return _build_dataclass(AppConfig, payload)


def resolve_run_config(config: AppConfig, run_name: str | None = None) -> AppConfig:
    resolved_name = run_name or config.experiment.name
    if resolved_name in {"", "auto", None}:
        resolved_name = f"{datetime.now():%Y%m%d-%H%M%S}"

    config.experiment.name = resolved_name
    if config.data.metadata_path in {"", "auto"}:
        config.data.metadata_path = str(config.experiment_dir / "data" / "metadata.csv")
    return config
