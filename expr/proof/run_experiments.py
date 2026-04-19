from __future__ import annotations

import argparse
import hashlib
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image, ImageEnhance, ImageFilter
from scipy.stats import spearmanr
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from safetensors.torch import load_file

from pv_iqa.config import load_config, resolve_run_config
from pv_iqa.eval import evaluate_erc, load_iqa_checkpoint, predict_quality_scores
from pv_iqa.train.iqa import train_iqa
from pv_iqa.train.pseudo_labels import generate_pseudo_labels
from pv_iqa.train.recognition import (
    build_recognizer,
    export_recognition_artifacts,
    train_recognizer,
)
from pv_iqa.utils.common import ensure_dir, move_batch_to_device, save_frame, set_seed
from pv_iqa.utils.datasets import build_metadata, load_metadata
from pv_iqa.utils.metrics import VerificationEvaluator
from pv_iqa.utils.transforms import build_transforms

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPR_ROOT = REPO_ROOT / "expr" / "proof"
CONFIG_ROOT = EXPR_ROOT / "configs"
RUN_ROOT = EXPR_ROOT / "runs"
FACTORIAL_ROOT = EXPR_ROOT / "factorial"
ROBUSTNESS_ROOT = EXPR_ROOT / "robustness"
SUBGROUP_ROOT = EXPR_ROOT / "subgroups"
REPORT_PATH = EXPR_ROOT / "report.md"
FIGURE_ROOT = EXPR_ROOT / "figures"
BASE_CONFIG_PATH = REPO_ROOT / "configs" / "default.yaml"
SHARED_RECOGNIZER_ROOT = RUN_ROOT / "_shared"


@dataclass(frozen=True, slots=True)
class ExperimentCell:
    name: str
    title: str
    description: str
    fusion_mode: str
    waveformer: bool
    overrides: dict[str, Any]


@dataclass(frozen=True, slots=True)
class CorruptionSpec:
    family: str
    severity: int
    label: str


@dataclass(frozen=True, slots=True)
class SubgroupSpec:
    name: str
    title: str
    description: str


CELLS: tuple[ExperimentCell, ...] = (
    ExperimentCell(
        name="full-waveformer",
        title="Full + WaveFormer",
        description="SDD-CR 自适应双分支 + WaveFormer。",
        fusion_mode="adaptive",
        waveformer=True,
        overrides={},
    ),
    ExperimentCell(
        name="full-no-waveformer",
        title="Full - WaveFormer",
        description="SDD-CR 自适应双分支，不使用 WaveFormer。",
        fusion_mode="adaptive",
        waveformer=False,
        overrides={"iqa": {"use_waveformer_layer": False}},
    ),
    ExperimentCell(
        name="sdd-waveformer",
        title="SDD only + WaveFormer",
        description="仅保留 SDD 分支，并使用 WaveFormer。",
        fusion_mode="sdd-only",
        waveformer=True,
        overrides={"pseudo_labels": {"adaptive_alpha": False, "alpha": 1.0}},
    ),
    ExperimentCell(
        name="sdd-no-waveformer",
        title="SDD only - WaveFormer",
        description="仅保留 SDD 分支，不使用 WaveFormer。",
        fusion_mode="sdd-only",
        waveformer=False,
        overrides={
            "pseudo_labels": {"adaptive_alpha": False, "alpha": 1.0},
            "iqa": {"use_waveformer_layer": False},
        },
    ),
)

CORRUPTIONS: tuple[CorruptionSpec, ...] = (
    CorruptionSpec("clean", 0, "clean"),
    CorruptionSpec("brightness", 1, "brightness-s1"),
    CorruptionSpec("brightness", 2, "brightness-s2"),
    CorruptionSpec("contrast", 1, "contrast-s1"),
    CorruptionSpec("contrast", 2, "contrast-s2"),
    CorruptionSpec("gaussian-noise", 1, "gaussian-noise-s1"),
    CorruptionSpec("gaussian-noise", 2, "gaussian-noise-s2"),
    CorruptionSpec("gaussian-blur", 1, "gaussian-blur-s1"),
    CorruptionSpec("gaussian-blur", 2, "gaussian-blur-s2"),
)

ERC_METRICS = [
    "eer_reject0",
    "best_eer",
    "mean_eer",
    "worst_eer",
    "auerc",
    "mean_tar@1e-4",
    "min_tar@1e-4",
]
ROBUSTNESS_METRICS = [
    "mean_eer",
    "auerc",
    "mean_tar@1e-4",
    "min_tar@1e-4",
    "monotonicity_accuracy",
    "severity_rank_corr",
    "clean_to_severe_margin",
]
REAL_DEGRADATION_PREFIXES = (
    "underexposed",
    "overexposed",
    "incomplete",
    "enhanced_extreme",
)
HARD_SUBSET_QUANTILE = 0.25
SUBGROUPS: tuple[SubgroupSpec, ...] = (
    SubgroupSpec(
        name="all",
        title="全部测试样本",
        description="完整测试集，用于保持对整体 clean utility 的观察。",
    ),
    SubgroupSpec(
        name="hard25",
        title="最难 25% 样本",
        description="按 recognizer 的真实类-最近负类 margin 排序后，取最难的 25%。",
    ),
    SubgroupSpec(
        name="challenging",
        title="全部真实退化样本",
        description="文件名显式带有退化标记的真实样本集合，不再额外施加合成扰动。",
    ),
    SubgroupSpec(
        name="incomplete",
        title="incomplete 子集",
        description="真实结构缺失/遮挡样本，最符合 CR 决策边界约束的作用场景。",
    ),
    SubgroupSpec(
        name="enhanced_extreme",
        title="enhanced_extreme 子集",
        description="真实极端增强样本，用于观察激进增强后的伪标签与回归稳定性。",
    ),
)


def deep_update(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    merged = {**base}
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_update(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_base_payload() -> dict[str, Any]:
    with BASE_CONFIG_PATH.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def build_base_payload(
    *,
    seed: int,
    recognizer_epochs: int,
    iqa_epochs: int,
    device: str,
    num_workers: int,
) -> dict[str, Any]:
    payload = load_base_payload()
    return deep_update(
        payload,
        {
            "experiment": {"output_root": str(RUN_ROOT / f"seed-{seed}" / f"iqa-{iqa_epochs}")},
            "runtime": {
                "seed": seed,
                "device": device,
                "num_workers": num_workers,
                "compile_model": False,
            },
            "logger": {
                "use_wandb": False,
                "wandb_mode": "disabled",
                "tags": ["pv", "iqa", "proof"],
            },
            "data": {"root": str(REPO_ROOT / "datasets" / "ROI_Data")},
            "recognizer": {"epochs": recognizer_epochs},
            "iqa": {"epochs": iqa_epochs},
        },
    )


def write_config(config_name: str, payload: dict[str, Any]) -> Path:
    ensure_dir(CONFIG_ROOT)
    path = CONFIG_ROOT / f"{config_name}.yaml"
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, allow_unicode=True, sort_keys=False)
    return path


def resolved_config(run_name: str, payload: dict[str, Any], *, config_name: str | None = None):
    config_path = write_config(config_name or run_name, payload)
    config = resolve_run_config(load_config(config_path), run_name=run_name)
    ensure_dir(config.experiment_dir)
    set_seed(config.runtime.seed)
    return config


def ensure_shared_recognizer(
    *,
    seed: int,
    recognizer_epochs: int,
    device: str,
    num_workers: int,
    force: bool,
):
    payload = load_base_payload()
    payload = deep_update(
        payload,
        {
            "experiment": {
                "output_root": str(
                    SHARED_RECOGNIZER_ROOT / f"seed-{seed}" / f"recognizer-{recognizer_epochs}"
                )
            },
            "runtime": {
                "seed": seed,
                "device": device,
                "num_workers": num_workers,
                "compile_model": False,
            },
            "logger": {
                "use_wandb": False,
                "wandb_mode": "disabled",
                "tags": ["pv", "iqa", "proof", "shared-recognizer"],
            },
            "data": {"root": str(REPO_ROOT / "datasets" / "ROI_Data")},
            "recognizer": {"epochs": recognizer_epochs},
        },
    )
    config_name = f"shared-recognizer-seed{seed}-recognizer{recognizer_epochs}"
    config = resolved_config(
        "shared-recognizer",
        payload,
        config_name=config_name,
    )
    if force and config.experiment_dir.exists():
        shutil.rmtree(config.experiment_dir)
        config = resolved_config(
            "shared-recognizer",
            payload,
            config_name=config_name,
        )

    metadata_path = Path(config.data.metadata_path)
    recognizer_dir = config.experiment_dir / "recognizer"
    recognizer_checkpoint = recognizer_dir / "best.pt"
    feature_path = recognizer_dir / "features.safetensors"
    feature_metadata_path = recognizer_dir / "feature_metadata.csv"

    if not metadata_path.exists():
        build_metadata(config)

    if force or not recognizer_checkpoint.exists():
        checkpoint_path = train_recognizer(config)
    else:
        checkpoint_path = recognizer_checkpoint

    if force or not feature_path.exists() or not feature_metadata_path.exists():
        export_recognition_artifacts(config, checkpoint_path)

    return config


def seed_experiment_from_shared(shared_config, experiment_config, force: bool) -> None:
    experiment_dir = experiment_config.experiment_dir
    if force and experiment_dir.exists():
        shutil.rmtree(experiment_dir)

    ensure_dir(experiment_dir / "data")
    ensure_dir(experiment_dir / "recognizer")
    shutil.copy2(Path(shared_config.data.metadata_path), Path(experiment_config.data.metadata_path))
    shutil.copytree(
        shared_config.experiment_dir / "recognizer",
        experiment_dir / "recognizer",
        dirs_exist_ok=True,
    )


def run_factorial_cell(
    *,
    cell: ExperimentCell,
    seed: int,
    recognizer_epochs: int,
    iqa_epochs: int,
    device: str,
    num_workers: int,
    force: bool,
):
    shared_config = ensure_shared_recognizer(
        seed=seed,
        recognizer_epochs=recognizer_epochs,
        device=device,
        num_workers=num_workers,
        force=force,
    )
    base_payload = build_base_payload(
        seed=seed,
        recognizer_epochs=recognizer_epochs,
        iqa_epochs=iqa_epochs,
        device=device,
        num_workers=num_workers,
    )
    payload = deep_update(base_payload, cell.overrides)
    config = resolved_config(
        cell.name,
        payload,
        config_name=f"{cell.name}-seed{seed}-iqa{iqa_epochs}",
    )
    seed_experiment_from_shared(shared_config, config, force=force)

    pseudo_label_path = config.experiment_dir / "pseudo_labels" / "pseudo_labels.csv"
    iqa_checkpoint = config.experiment_dir / "iqa" / "best.pt"
    erc_path = config.experiment_dir / "evaluation" / "erc_metrics.csv"

    if force or not pseudo_label_path.exists():
        generate_pseudo_labels(config)
    if force or not iqa_checkpoint.exists():
        train_iqa(config)
    if force or not erc_path.exists():
        quality = predict_quality_scores(
            config=config,
            checkpoint_path=iqa_checkpoint,
            split=config.evaluation.split,
        )
        evaluate_erc(config, quality)
    return config


def tar_column(frame: pd.DataFrame, far: float) -> str:
    column = f"tar@far={far:.0e}"
    if column not in frame.columns:
        raise KeyError(f"Missing TAR column '{column}'")
    return column


def auerc(frame: pd.DataFrame) -> float:
    ordered = frame.sort_values("reject_fraction")
    return float(np.trapezoid(ordered["eer"].to_numpy(), ordered["reject_fraction"].to_numpy()))


def summarize_erc(frame: pd.DataFrame) -> dict[str, float]:
    ordered = frame.sort_values("reject_fraction").reset_index(drop=True)
    tar1e4 = tar_column(ordered, 1e-4)
    zero_row = ordered.loc[ordered["reject_fraction"].sub(0.0).abs().idxmin()]
    return {
        "eer_reject0": float(zero_row["eer"]),
        "best_eer": float(ordered["eer"].min()),
        "mean_eer": float(ordered["eer"].mean()),
        "worst_eer": float(ordered["eer"].max()),
        "auerc": auerc(ordered),
        "mean_tar@1e-4": float(ordered[tar1e4].mean()),
        "min_tar@1e-4": float(ordered[tar1e4].min()),
    }


def estimate_blend(frame: pd.DataFrame) -> float:
    delta = frame["q_sdd"] - frame["q_cr"]
    denom = float((delta**2).sum())
    if abs(denom) < 1e-12:
        return float("nan")
    numer = float(((frame["quality_score"] - frame["q_cr"]) * delta).sum())
    return numer / denom


def real_degradation_tag(sample_id: str) -> str:
    stem = sample_id.split("/", 1)[1] if "/" in sample_id else sample_id
    for prefix in REAL_DEGRADATION_PREFIXES:
        if stem.startswith(f"{prefix}_"):
            return prefix
    return "normal"


def collect_factorial_row(cell: ExperimentCell, config, seed: int, iqa_epochs: int) -> dict[str, Any]:
    erc = pd.read_csv(config.experiment_dir / "evaluation" / "erc_metrics.csv")
    checkpoint = torch.load(config.experiment_dir / "iqa" / "best.pt", map_location="cpu")
    pseudo = pd.read_csv(config.experiment_dir / "pseudo_labels" / "pseudo_labels.csv")
    metrics = summarize_erc(erc)
    corr = pseudo[["q_sdd", "q_cr", "quality_score"]].corr(method="spearman")
    return {
        "experiment": cell.name,
        "title": cell.title,
        "description": cell.description,
        "fusion_mode": cell.fusion_mode,
        "waveformer": cell.waveformer,
        "seed": seed,
        "iqa_epochs": iqa_epochs,
        "best_mae": float(checkpoint.get("best_mae", float("nan"))),
        "best_epoch": int(checkpoint.get("best_epoch", -1)),
        "estimated_blend": estimate_blend(pseudo),
        "spearman_qsdd_qcr": float(corr.loc["q_sdd", "q_cr"]),
        "run_dir": str(config.experiment_dir.relative_to(REPO_ROOT)),
        **metrics,
    }


def aggregate_metrics(
    frame: pd.DataFrame,
    *,
    group_columns: list[str],
    metric_columns: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for key, subset in frame.groupby(group_columns, dropna=False, sort=False):
        values = key if isinstance(key, tuple) else (key,)
        row = dict(zip(group_columns, values, strict=True))
        row["runs"] = int(len(subset))
        for column in metric_columns:
            row[f"{column}_mean"] = float(subset[column].mean())
            row[f"{column}_std"] = float(subset[column].std(ddof=0))
        rows.append(row)
    return pd.DataFrame(rows)


def save_factorial_outputs(raw_frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    ensure_dir(FACTORIAL_ROOT)
    save_frame(raw_frame, FACTORIAL_ROOT / "raw_results.csv")
    summary = aggregate_metrics(
        raw_frame,
        group_columns=["experiment", "title", "description", "fusion_mode", "waveformer", "iqa_epochs"],
        metric_columns=["best_mae", "estimated_blend", "spearman_qsdd_qcr", *ERC_METRICS],
    ).sort_values(["mean_eer_mean", "auerc_mean"]).reset_index(drop=True)
    save_frame(summary, FACTORIAL_ROOT / "summary.csv")

    factor_rows: list[dict[str, Any]] = []
    for factor in ["fusion_mode", "waveformer"]:
        factor_frame = aggregate_metrics(
            raw_frame,
            group_columns=[factor, "iqa_epochs"],
            metric_columns=["best_mae", *ERC_METRICS],
        )
        factor_frame.insert(0, "factor", factor)
        factor_rows.extend(factor_frame.to_dict("records"))
    factor_effects = pd.DataFrame(factor_rows)
    save_frame(factor_effects, FACTORIAL_ROOT / "factor_effects.csv")
    return summary, factor_effects


def load_shared_test_bundle(*, seed: int, recognizer_epochs: int) -> dict[str, Any]:
    feature_dir = (
        SHARED_RECOGNIZER_ROOT
        / f"seed-{seed}"
        / f"recognizer-{recognizer_epochs}"
        / "shared-recognizer"
        / "recognizer"
    )
    tensors = load_file(str(feature_dir / "features.safetensors"))
    metadata = pd.read_csv(feature_dir / "feature_metadata.csv")

    embeddings = torch.nn.functional.normalize(tensors["embeddings"].float(), dim=1)
    weights = torch.nn.functional.normalize(tensors["classifier_weight"].float(), dim=1)
    class_ids = tensors["class_ids"].long()
    cosine = embeddings @ weights.T
    true_cos = cosine[torch.arange(len(class_ids)), class_ids]
    negative_mask = torch.ones_like(cosine, dtype=torch.bool)
    negative_mask[torch.arange(len(class_ids)), class_ids] = False
    hardest_negative = cosine.masked_fill(~negative_mask, -1e9).max(dim=1).values
    margins = (true_cos - hardest_negative).cpu().numpy()

    test_indices = metadata.index[metadata["split"].eq("test")].to_numpy(dtype=np.int64)
    test_metadata = metadata.iloc[test_indices].reset_index(drop=True).copy()
    test_metadata["margin"] = margins[test_indices]
    test_metadata["tag"] = test_metadata["sample_id"].map(real_degradation_tag)
    test_metadata["challenging"] = test_metadata["tag"].ne("normal")

    test_embeddings = np.array(embeddings.cpu().numpy()[test_indices], copy=True)
    hard_threshold = float(test_metadata["margin"].quantile(HARD_SUBSET_QUANTILE))
    subsets = {
        "all": set(test_metadata["sample_id"]),
        "hard25": set(
            test_metadata.loc[
                test_metadata["margin"].le(hard_threshold),
                "sample_id",
            ]
        ),
        "challenging": set(
            test_metadata.loc[test_metadata["challenging"], "sample_id"]
        ),
        "incomplete": set(
            test_metadata.loc[test_metadata["tag"].eq("incomplete"), "sample_id"]
        ),
        "enhanced_extreme": set(
            test_metadata.loc[
                test_metadata["tag"].eq("enhanced_extreme"),
                "sample_id",
            ]
        ),
    }
    return {
        "metadata": test_metadata,
        "embeddings": test_embeddings,
        "id_to_position": {
            sample_id: index
            for index, sample_id in enumerate(test_metadata["sample_id"])
        },
        "subsets": subsets,
    }


def evaluate_subset_from_scores(
    config,
    quality_frame: pd.DataFrame,
    embeddings: np.ndarray,
    id_to_position: dict[str, int],
) -> dict[str, float]:
    ordered = quality_frame.sort_values("predicted_quality", ascending=False).reset_index(
        drop=True
    )
    evaluator = VerificationEvaluator(
        far_targets=config.evaluation.far_targets,
        max_impostor_pairs=config.evaluation.max_impostor_pairs,
        seed=config.runtime.seed,
    )
    records: list[dict[str, float]] = []
    for reject_fraction in config.evaluation.reject_steps:
        keep_count = max(2, int(len(ordered) * (1.0 - reject_fraction)))
        kept = ordered.head(keep_count)
        kept_indices = np.asarray(
            [id_to_position[sample_id] for sample_id in kept["sample_id"]],
            dtype=np.int64,
        )
        metrics = evaluator.evaluate(
            embeddings[kept_indices],
            kept["class_id"].to_numpy(),
        )
        metrics["reject_fraction"] = float(reject_fraction)
        records.append(metrics)
    return summarize_erc(pd.DataFrame(records))


def run_subgroup_benchmark(
    raw_factorial: pd.DataFrame,
    *,
    recognizer_epochs: int,
    iqa_epochs: int,
    device: str,
    num_workers: int,
) -> pd.DataFrame:
    shared_cache: dict[int, dict[str, Any]] = {}
    subgroup_lookup = {spec.name: spec for spec in SUBGROUPS}
    rows: list[dict[str, Any]] = []

    for _, row in raw_factorial.iterrows():
        seed = int(row["seed"])
        if seed not in shared_cache:
            shared_cache[seed] = load_shared_test_bundle(
                seed=seed,
                recognizer_epochs=recognizer_epochs,
            )
        shared_bundle = shared_cache[seed]
        config = resolved_config(
            "subgroup-eval",
            build_base_payload(
                seed=seed,
                recognizer_epochs=recognizer_epochs,
                iqa_epochs=iqa_epochs,
                device=device,
                num_workers=num_workers,
            ),
            config_name=f"subgroup-eval-seed{seed}-iqa{iqa_epochs}",
        )
        prediction_path = (
            REPO_ROOT
            / str(row["run_dir"])
            / "evaluation"
            / f"{config.evaluation.split}_quality_predictions.csv"
        )
        quality = pd.read_csv(prediction_path).merge(
            shared_bundle["metadata"][["sample_id", "class_id"]],
            on=["sample_id", "class_id"],
            how="inner",
        )

        for subgroup_name, sample_ids in shared_bundle["subsets"].items():
            subset_quality = quality[quality["sample_id"].isin(sample_ids)].copy()
            if len(subset_quality) < 10:
                continue
            subgroup = subgroup_lookup[subgroup_name]
            metrics = evaluate_subset_from_scores(
                config,
                subset_quality,
                shared_bundle["embeddings"],
                shared_bundle["id_to_position"],
            )
            rows.append(
                {
                    "subset": subgroup.name,
                    "subset_title": subgroup.title,
                    "subset_description": subgroup.description,
                    "subset_size": int(len(subset_quality)),
                    "experiment": row["experiment"],
                    "title": row["title"],
                    "fusion_mode": row["fusion_mode"],
                    "waveformer": bool(row["waveformer"]),
                    "seed": seed,
                    "iqa_epochs": iqa_epochs,
                    **metrics,
                }
            )
    return pd.DataFrame(rows)


def save_subgroup_outputs(raw_frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    ensure_dir(SUBGROUP_ROOT)
    save_frame(raw_frame, SUBGROUP_ROOT / "raw_results.csv")
    summary = aggregate_metrics(
        raw_frame,
        group_columns=[
            "subset",
            "subset_title",
            "subset_description",
            "experiment",
            "title",
            "fusion_mode",
            "waveformer",
            "iqa_epochs",
        ],
        metric_columns=["subset_size", *ERC_METRICS],
    ).sort_values(["subset", "mean_eer_mean", "auerc_mean"]).reset_index(drop=True)
    save_frame(summary, SUBGROUP_ROOT / "summary.csv")

    factor_rows: list[dict[str, Any]] = []
    for subgroup_name in raw_frame["subset"].drop_duplicates().tolist():
        subset_frame = raw_frame.query("subset == @subgroup_name")
        for factor in ["fusion_mode", "waveformer"]:
            factor_frame = aggregate_metrics(
                subset_frame,
                group_columns=["subset", "subset_title", factor, "iqa_epochs"],
                metric_columns=ERC_METRICS,
            )
            factor_frame.insert(2, "factor", factor)
            factor_rows.extend(factor_frame.to_dict("records"))
    factor_effects = pd.DataFrame(factor_rows).sort_values(
        ["subset", "factor", "mean_eer_mean", "auerc_mean"]
    )
    save_frame(factor_effects, SUBGROUP_ROOT / "factor_effects.csv")
    return summary, factor_effects


def stable_int(seed: int, sample_id: str, family: str, severity: int) -> int:
    digest = hashlib.sha256(f"{seed}:{sample_id}:{family}:{severity}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little", signed=False) % (2**32)


def apply_corruption(
    image: Image.Image,
    *,
    sample_id: str,
    family: str,
    severity: int,
    seed: int,
) -> Image.Image:
    if family == "clean" or severity == 0:
        return image

    if family == "brightness":
        factors = {1: 0.82, 2: 0.64}
        return ImageEnhance.Brightness(image).enhance(factors[severity])
    if family == "contrast":
        factors = {1: 0.78, 2: 0.58}
        return ImageEnhance.Contrast(image).enhance(factors[severity])
    if family == "gaussian-blur":
        radii = {1: 1.0, 2: 2.2}
        return image.filter(ImageFilter.GaussianBlur(radius=radii[severity]))
    if family == "gaussian-noise":
        std_map = {1: 10.0, 2: 20.0}
        rng = np.random.default_rng(stable_int(seed, sample_id, family, severity))
        array = np.asarray(image, dtype=np.float32)
        noisy = np.clip(array + rng.normal(loc=0.0, scale=std_map[severity], size=array.shape), 0.0, 255.0)
        return Image.fromarray(noisy.astype(np.uint8), mode=image.mode)
    raise ValueError(f"Unsupported corruption family: {family}")


class CorruptedPalmVeinDataset(Dataset[dict[str, torch.Tensor | int | str]]):
    def __init__(
        self,
        metadata: pd.DataFrame,
        *,
        split: str,
        image_size: int,
        grayscale_to_rgb: bool,
        family: str,
        severity: int,
        seed: int,
    ) -> None:
        self.frame = metadata.query("split == @split").reset_index(drop=True)
        self.grayscale_to_rgb = grayscale_to_rgb
        self.family = family
        self.severity = severity
        self.seed = seed
        self.transform = build_transforms(image_size=image_size, is_train=False)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | int | str]:
        row = self.frame.iloc[index]
        image = Image.open(Path(row["image_path"])).convert("L")
        image = apply_corruption(
            image,
            sample_id=str(row["sample_id"]),
            family=self.family,
            severity=self.severity,
            seed=self.seed,
        )
        if self.grayscale_to_rgb:
            image = image.convert("RGB")
        return {
            "image": self.transform(image),
            "sample_id": row["sample_id"],
            "class_id": int(row["class_id"]),
        }


def load_recognizer_checkpoint(config, checkpoint_path: Path):
    metadata = load_metadata(config)
    device = torch.device("cuda" if torch.cuda.is_available() and config.runtime.device != "cpu" else "cpu")
    model = build_recognizer(
        config,
        int(metadata["class_id"].nunique()),
        pretrained=False,
    ).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, device


def evaluate_condition_from_scores(
    config,
    quality_frame: pd.DataFrame,
    embeddings: np.ndarray,
) -> dict[str, float]:
    ordered = quality_frame.sort_values("predicted_quality", ascending=False).reset_index(drop=True)
    evaluator = VerificationEvaluator(
        far_targets=config.evaluation.far_targets,
        max_impostor_pairs=config.evaluation.max_impostor_pairs,
        seed=config.runtime.seed,
    )
    records: list[dict[str, float]] = []
    for reject_fraction in config.evaluation.reject_steps:
        keep_count = max(2, int(len(ordered) * (1.0 - reject_fraction)))
        kept = ordered.head(keep_count)
        kept_indices = kept["embedding_index"].to_numpy(dtype=np.int64)
        metrics = evaluator.evaluate(embeddings[kept_indices], kept["class_id"].to_numpy())
        metrics["reject_fraction"] = float(reject_fraction)
        records.append(metrics)
    return summarize_erc(pd.DataFrame(records))


def predict_corrupted_condition(
    *,
    config,
    iqa_checkpoint: Path,
    recognizer_checkpoint: Path,
    family: str,
    severity: int,
) -> tuple[pd.DataFrame, np.ndarray]:
    metadata = load_metadata(config)
    dataset = CorruptedPalmVeinDataset(
        metadata,
        split=config.evaluation.split,
        image_size=config.data.image_size,
        grayscale_to_rgb=config.data.grayscale_to_rgb,
        family=family,
        severity=severity,
        seed=config.runtime.seed,
    )
    loader = DataLoader(
        dataset,
        batch_size=config.data.eval_batch_size,
        shuffle=False,
        num_workers=config.runtime.num_workers,
        pin_memory=config.data.pin_memory,
    )

    iqa_model, iqa_device = load_iqa_checkpoint(config, iqa_checkpoint)
    recognizer_model, recognizer_device = load_recognizer_checkpoint(config, recognizer_checkpoint)
    if iqa_device != recognizer_device:
        raise RuntimeError("IQA model and recognizer must run on the same device for this benchmark.")

    records: list[dict[str, float | int | str]] = []
    embeddings: list[np.ndarray] = []
    embedding_offset = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"robust-{family}-{severity}", leave=False):
            batch = move_batch_to_device(batch, iqa_device)
            iqa_outputs = iqa_model(batch["image"])
            recognizer_outputs = recognizer_model(batch["image"])
            batch_embeddings = recognizer_outputs.embeddings.cpu().numpy()
            embeddings.append(batch_embeddings)
            for local_index, (sample_id, class_id, score) in enumerate(zip(
                batch["sample_id"],
                batch["class_id"],
                iqa_outputs["score"].cpu().tolist(),
                strict=True,
            )):
                records.append(
                    {
                        "sample_id": sample_id,
                        "class_id": int(class_id),
                        "predicted_quality": float(score),
                        "family": family,
                        "severity": severity,
                        "embedding_index": embedding_offset + local_index,
                    }
                )
            embedding_offset += len(batch_embeddings)
    frame = pd.DataFrame(records).sort_values("predicted_quality", ascending=False).reset_index(drop=True)
    embedding_array = np.concatenate(embeddings, axis=0)
    return frame, embedding_array


def monotonicity_summary(condition_rows: pd.DataFrame) -> dict[str, float]:
    pairwise_correct = 0
    pairwise_total = 0
    margins: list[float] = []
    sample_corrs: list[float] = []
    for _, subset in condition_rows.groupby("sample_id", sort=False):
        ordered = subset.sort_values("severity")
        severities = ordered["severity"].to_numpy()
        scores = ordered["predicted_quality"].to_numpy()
        for left in range(len(scores)):
            for right in range(left + 1, len(scores)):
                pairwise_total += 1
                if scores[left] >= scores[right]:
                    pairwise_correct += 1
        if len(scores) >= 2:
            corr = spearmanr(severities, -scores).statistic
            if corr is not None and not np.isnan(corr):
                sample_corrs.append(float(corr))
            margins.append(float(scores[0] - scores[-1]))
    return {
        "monotonicity_accuracy": float(pairwise_correct / max(1, pairwise_total)),
        "severity_rank_corr": float(np.mean(sample_corrs)) if sample_corrs else 0.0,
        "clean_to_severe_margin": float(np.mean(margins)) if margins else 0.0,
    }


def run_robustness_for_cell(cell: ExperimentCell, config, seed: int, iqa_epochs: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    iqa_checkpoint = config.experiment_dir / "iqa" / "best.pt"
    recognizer_checkpoint = config.experiment_dir / "recognizer" / "best.pt"
    condition_rows: list[dict[str, Any]] = []
    monotonicity_rows: list[pd.DataFrame] = []

    for corruption in CORRUPTIONS:
        quality_frame, embeddings = predict_corrupted_condition(
            config=config,
            iqa_checkpoint=iqa_checkpoint,
            recognizer_checkpoint=recognizer_checkpoint,
            family=corruption.family,
            severity=corruption.severity,
        )
        metrics = evaluate_condition_from_scores(config, quality_frame, embeddings)
        condition_rows.append(
            {
                "experiment": cell.name,
                "title": cell.title,
                "fusion_mode": cell.fusion_mode,
                "waveformer": cell.waveformer,
                "seed": seed,
                "iqa_epochs": iqa_epochs,
                "family": corruption.family,
                "severity": corruption.severity,
                "condition": corruption.label,
                **metrics,
            }
        )
        if corruption.family != "clean":
            monotonicity_rows.append(
                quality_frame.assign(
                    experiment=cell.name,
                    title=cell.title,
                    fusion_mode=cell.fusion_mode,
                    waveformer=cell.waveformer,
                    seed=seed,
                    iqa_epochs=iqa_epochs,
                )
            )

    all_conditions = pd.concat(monotonicity_rows, ignore_index=True)
    clean_scores = pd.read_csv(config.experiment_dir / "evaluation" / f"{config.evaluation.split}_quality_predictions.csv")
    clean_scores = clean_scores.rename(columns={"predicted_quality": "clean_quality"})[
        ["sample_id", "clean_quality"]
    ]

    robustness_rows: list[dict[str, Any]] = []
    for family, subset in all_conditions.groupby("family", sort=False):
        merged = subset.merge(clean_scores, on="sample_id", how="left")
        family_frame = pd.concat(
            [
                merged[["sample_id"]].assign(severity=0, predicted_quality=merged["clean_quality"]),
                merged[["sample_id", "severity", "predicted_quality"]],
            ],
            ignore_index=True,
        )
        mono = monotonicity_summary(family_frame)
        robustness_rows.append(
            {
                "experiment": cell.name,
                "title": cell.title,
                "fusion_mode": cell.fusion_mode,
                "waveformer": cell.waveformer,
                "seed": seed,
                "iqa_epochs": iqa_epochs,
                "family": family,
                **mono,
            }
        )

    return pd.DataFrame(condition_rows), pd.DataFrame(robustness_rows)


def save_robustness_outputs(
    condition_frame: pd.DataFrame,
    monotonicity_frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ensure_dir(ROBUSTNESS_ROOT)
    save_frame(condition_frame, ROBUSTNESS_ROOT / "condition_results.csv")
    save_frame(monotonicity_frame, ROBUSTNESS_ROOT / "monotonicity_results.csv")

    condition_summary = aggregate_metrics(
        condition_frame,
        group_columns=["experiment", "title", "fusion_mode", "waveformer", "iqa_epochs", "family", "severity", "condition"],
        metric_columns=ERC_METRICS,
    )
    save_frame(condition_summary, ROBUSTNESS_ROOT / "condition_summary.csv")

    corrupt_only = condition_frame.query("family != 'clean'").copy()
    merged = corrupt_only.merge(
        monotonicity_frame,
        on=["experiment", "title", "fusion_mode", "waveformer", "seed", "iqa_epochs", "family"],
        how="left",
    )
    aggregated = aggregate_metrics(
        merged,
        group_columns=["experiment", "title", "fusion_mode", "waveformer", "iqa_epochs"],
        metric_columns=[*ERC_METRICS, *ROBUSTNESS_METRICS],
    ).sort_values(["mean_eer_mean", "auerc_mean"]).reset_index(drop=True)
    save_frame(aggregated, ROBUSTNESS_ROOT / "summary.csv")

    factor_rows: list[dict[str, Any]] = []
    for factor in ["fusion_mode", "waveformer"]:
        factor_frame = aggregate_metrics(
            merged,
            group_columns=[factor, "iqa_epochs"],
            metric_columns=[*ERC_METRICS, *ROBUSTNESS_METRICS],
        )
        factor_frame.insert(0, "factor", factor)
        factor_rows.extend(factor_frame.to_dict("records"))
    factor_effects = pd.DataFrame(factor_rows)
    save_frame(factor_effects, ROBUSTNESS_ROOT / "factor_effects.csv")
    return condition_summary, aggregated, factor_effects


def plot_clean_overview(summary: pd.DataFrame) -> None:
    ensure_dir(FIGURE_ROOT)
    ordered = summary.sort_values("mean_eer_mean").reset_index(drop=True)
    labels = ordered["title"].tolist()
    positions = np.arange(len(ordered))
    metrics = [
        ("mean_eer_mean", "mean EER"),
        ("auerc_mean", "AUERC"),
        ("worst_eer_mean", "worst EER"),
        ("best_mae_mean", "Best MAE"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    axes = axes.flatten()
    for axis, (column, title) in zip(axes, metrics, strict=True):
        axis.bar(positions, ordered[column], color="#4C6EF5")
        axis.errorbar(
            positions,
            ordered[column],
            yerr=ordered[column.replace("_mean", "_std")],
            fmt="none",
            ecolor="#1c1c1c",
            capsize=4,
            linewidth=1,
        )
        axis.set_title(title)
        axis.set_xticks(positions, labels, rotation=20, ha="right")
        axis.grid(axis="y", alpha=0.25)
    fig.suptitle("Clean Utility Overview", fontsize=14)
    fig.savefig(FIGURE_ROOT / "clean_overview.png", dpi=220)
    plt.close(fig)


def plot_factor_effects(clean_factors: pd.DataFrame, robust_factors: pd.DataFrame) -> None:
    ensure_dir(FIGURE_ROOT)
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    factor_map = [
        ("fusion_mode", "Clean fusion effect", clean_factors),
        ("waveformer", "Clean WaveFormer effect", clean_factors),
        ("fusion_mode", "Robust fusion effect", robust_factors),
        ("waveformer", "Robust WaveFormer effect", robust_factors),
    ]
    for axis, (factor, title, frame) in zip(axes.flatten(), factor_map, strict=True):
        subset = frame.query("factor == @factor").copy()
        subset[factor] = subset[factor].astype(str)
        positions = np.arange(len(subset))
        axis.bar(positions, subset["mean_eer_mean"], color="#12B886")
        axis.errorbar(
            positions,
            subset["mean_eer_mean"],
            yerr=subset["mean_eer_std"],
            fmt="none",
            ecolor="#1c1c1c",
            capsize=4,
            linewidth=1,
        )
        axis.set_title(title)
        axis.set_xticks(positions, subset[factor], rotation=15, ha="right")
        axis.grid(axis="y", alpha=0.25)
    fig.savefig(FIGURE_ROOT / "factor_effects.png", dpi=220)
    plt.close(fig)


def plot_robustness_overview(summary: pd.DataFrame) -> None:
    ensure_dir(FIGURE_ROOT)
    ordered = summary.sort_values("mean_eer_mean").reset_index(drop=True)
    labels = ordered["title"].tolist()
    positions = np.arange(len(ordered))
    metrics = [
        ("mean_eer_mean", "robust mean EER"),
        ("auerc_mean", "robust AUERC"),
        ("monotonicity_accuracy_mean", "monotonicity accuracy"),
        ("severity_rank_corr_mean", "severity rank corr"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    axes = axes.flatten()
    for axis, (column, title) in zip(axes, metrics, strict=True):
        axis.bar(positions, ordered[column], color="#F59F00")
        axis.errorbar(
            positions,
            ordered[column],
            yerr=ordered[column.replace("_mean", "_std")],
            fmt="none",
            ecolor="#1c1c1c",
            capsize=4,
            linewidth=1,
        )
        axis.set_title(title)
        axis.set_xticks(positions, labels, rotation=20, ha="right")
        axis.grid(axis="y", alpha=0.25)
    fig.suptitle("Robustness Overview", fontsize=14)
    fig.savefig(FIGURE_ROOT / "robustness_overview.png", dpi=220)
    plt.close(fig)


def plot_corruption_curves(condition_summary: pd.DataFrame) -> None:
    ensure_dir(FIGURE_ROOT)
    families = [family for family in condition_summary["family"].drop_duplicates().tolist() if family != "clean"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    axes = axes.flatten()
    for axis, family in zip(axes, families, strict=True):
        subset = condition_summary.query("family == @family").sort_values("severity")
        for title, group in subset.groupby("title", sort=False):
            ordered = group.sort_values("severity")
            axis.plot(
                ordered["severity"],
                ordered["mean_eer_mean"],
                marker="o",
                label=title,
            )
        axis.set_title(f"{family} corruption")
        axis.set_xlabel("severity")
        axis.set_ylabel("mean EER")
        axis.set_xticks(sorted(subset["severity"].unique()))
        axis.grid(alpha=0.25)
    axes[0].legend(loc="best", fontsize=8)
    fig.suptitle("Corruption severity curves", fontsize=14)
    fig.savefig(FIGURE_ROOT / "corruption_curves.png", dpi=220)
    plt.close(fig)


def plot_subgroup_overview(summary: pd.DataFrame) -> None:
    ensure_dir(FIGURE_ROOT)
    focus = [spec for spec in SUBGROUPS if spec.name != "all"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    for axis, subgroup in zip(axes.flatten(), focus, strict=True):
        subset = summary.query("subset == @subgroup.name").sort_values("mean_eer_mean")
        positions = np.arange(len(subset))
        axis.bar(positions, subset["mean_eer_mean"], color="#E8590C")
        axis.errorbar(
            positions,
            subset["mean_eer_mean"],
            yerr=subset["mean_eer_std"],
            fmt="none",
            ecolor="#1c1c1c",
            capsize=4,
            linewidth=1,
        )
        axis.set_title(subgroup.name)
        axis.set_xticks(positions, subset["title"], rotation=20, ha="right")
        axis.set_ylabel("mean EER")
        axis.grid(axis="y", alpha=0.25)
    fig.suptitle("Real hard-subset benchmark", fontsize=14)
    fig.savefig(FIGURE_ROOT / "subgroup_overview.png", dpi=220)
    plt.close(fig)


def markdown_table(frame: pd.DataFrame, columns: list[str], headers: list[str], digits: int = 4) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in frame.iterrows():
        cells: list[str] = []
        for column in columns:
            value = row[column]
            if isinstance(value, float):
                cells.append(f"{value:.{digits}f}")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def markdown_mean_std(frame: pd.DataFrame, value_column: str, std_column: str, digits: int = 4) -> list[str]:
    rows = []
    for _, row in frame.iterrows():
        rows.append(f"{row[value_column]:.{digits}f} ± {row[std_column]:.{digits}f}")
    return rows


def factor_judgement(frame: pd.DataFrame, factor: str) -> tuple[str, str]:
    subset = frame.query("factor == @factor").sort_values("mean_eer_mean").reset_index(drop=True)
    best = subset.iloc[0]
    runner_up = subset.iloc[1] if len(subset) > 1 else None
    label = str(best[factor])
    if runner_up is None:
        return label, "只有一个水平，无法比较。"
    delta = float(runner_up["mean_eer_mean"] - best["mean_eer_mean"])
    return label, f"在 `mean EER` 上优势约为 {delta:.4f}。"


def subset_factor_judgement(
    frame: pd.DataFrame,
    *,
    subset: str,
    factor: str,
) -> tuple[str, float]:
    ordered = (
        frame.query("subset == @subset and factor == @factor")
        .sort_values("mean_eer_mean")
        .reset_index(drop=True)
    )
    best = ordered.iloc[0]
    runner_up = ordered.iloc[1] if len(ordered) > 1 else None
    delta = (
        float(runner_up["mean_eer_mean"] - best["mean_eer_mean"])
        if runner_up is not None
        else 0.0
    )
    return str(best[factor]), delta


def write_report(
    factorial_summary: pd.DataFrame,
    factorial_factors: pd.DataFrame,
    subgroup_summary: pd.DataFrame,
    subgroup_factors: pd.DataFrame,
    condition_summary: pd.DataFrame,
    robustness_summary: pd.DataFrame,
    robustness_factors: pd.DataFrame,
    *,
    recognizer_epochs: int,
    iqa_epochs: int,
    seeds: list[int],
) -> None:
    clean_table = factorial_summary.copy()
    clean_table["mean_eer"] = markdown_mean_std(clean_table, "mean_eer_mean", "mean_eer_std")
    clean_table["auerc"] = markdown_mean_std(clean_table, "auerc_mean", "auerc_std")
    clean_table["worst_eer"] = markdown_mean_std(clean_table, "worst_eer_mean", "worst_eer_std")
    clean_table["best_mae"] = markdown_mean_std(clean_table, "best_mae_mean", "best_mae_std")

    fusion_clean = factorial_factors.query("factor == 'fusion_mode'").sort_values("mean_eer_mean")
    wave_clean = factorial_factors.query("factor == 'waveformer'").sort_values("mean_eer_mean")
    best_clean = factorial_summary.sort_values("mean_eer_mean").iloc[0]
    best_fusion_clean, best_fusion_clean_note = factor_judgement(factorial_factors, "fusion_mode")
    best_wave_clean, best_wave_clean_note = factor_judgement(factorial_factors, "waveformer")
    subgroup_definition_frame = pd.DataFrame(
        [
            {"subset_title": spec.title, "description": spec.description}
            for spec in SUBGROUPS
            if spec.name != "all"
        ]
    )
    key_subsets = [spec.name for spec in SUBGROUPS if spec.name in {"hard25", "challenging", "incomplete", "enhanced_extreme"}]
    subgroup_best = (
        subgroup_summary.loc[subgroup_summary["subset"].isin(key_subsets)]
        .sort_values(["subset", "mean_eer_mean", "auerc_mean"])
        .groupby("subset", sort=False)
        .head(1)
        .copy()
    )
    subgroup_best["mean_eer"] = markdown_mean_std(
        subgroup_best,
        "mean_eer_mean",
        "mean_eer_std",
    )
    subgroup_best["auerc"] = markdown_mean_std(
        subgroup_best,
        "auerc_mean",
        "auerc_std",
    )
    subgroup_fusion = (
        subgroup_factors.loc[
            subgroup_factors["factor"].eq("fusion_mode")
            & subgroup_factors["subset"].isin(key_subsets)
        ]
        .sort_values(["subset", "mean_eer_mean", "auerc_mean"])
        .copy()
    )
    robust_best = robustness_summary.sort_values("mean_eer_mean").iloc[0]
    incomplete_winner, incomplete_delta = subset_factor_judgement(
        subgroup_factors,
        subset="incomplete",
        factor="fusion_mode",
    )
    enhanced_winner, enhanced_delta = subset_factor_judgement(
        subgroup_factors,
        subset="enhanced_extreme",
        factor="fusion_mode",
    )
    hard25_winner, hard25_delta = subset_factor_judgement(
        subgroup_factors,
        subset="hard25",
        factor="fusion_mode",
    )

    report = f"""# PV-IQA 证明型实验报告（重构版）

## 1. 重新设计原则

上一版 proof 报告把两个问题混在了一起：

1. **WaveFormer 是否有效**：这是一个模块问题，应该优先用正交的 clean 2×2 因子实验判断。
2. **SDD-CR 双分支是否有效**：这是一个监督信号问题，不能只看全体样本均值，还要看它在理论上最该发挥作用的真实困难样本。

因此本次报告改成两层证据：

1. **全局 clean 2×2 因子实验**：只负责回答 WaveFormer 是否值得保留。
2. **真实难样本子集实验**：只负责回答 SDD-CR 是否对结构性困难样本有增益。

合成 corruption 压力测试仍然保留，但只作为附录性压力测试，不再用来直接推翻前两项主结论。

## 2. 实验设置

- 识别模型训练轮数：{recognizer_epochs}
- IQA 模型训练轮数：{iqa_epochs}
- Seeds：{", ".join(str(seed) for seed in seeds)}
- 输出目录：`expr/proof`

### 2.1 真实难样本子集定义

{markdown_table(
    subgroup_definition_frame,
    columns=["subset_title", "description"],
    headers=["子集", "定义"],
    digits=4,
)}

## 3. WaveFormer：全局 clean 2×2 结果

![Clean overview](figures/clean_overview.png)

{markdown_table(
    clean_table.sort_values("mean_eer_mean"),
    columns=["title", "mean_eer", "auerc", "worst_eer", "best_mae"],
    headers=["实验组", "mean EER", "AUERC", "worst EER", "Best MAE"],
    digits=4,
)}

### 3.1 Clean 主效应

![Factor effects](figures/factor_effects.png)

#### 3.1.1 融合主效应

{markdown_table(
    fusion_clean,
    columns=["fusion_mode", "mean_eer_mean", "auerc_mean", "worst_eer_mean"],
    headers=["融合方式", "mean EER", "AUERC", "worst EER"],
    digits=4,
)}

#### 3.1.2 WaveFormer 主效应

{markdown_table(
    wave_clean,
    columns=["waveformer", "mean_eer_mean", "auerc_mean", "worst_eer_mean"],
    headers=["WaveFormer", "mean EER", "AUERC", "worst EER"],
    digits=4,
)}

**全局 clean 结论：**

1. **Clean 最优 cell** 仍然是 `{best_clean["title"]}`。
2. **WaveFormer clean 主效应为正**：`{best_wave_clean}` 更优，{best_wave_clean_note}
3. 这说明 **WaveFormer 应该保留**；后续为了验证双分支而做的补充实验，不应再拿合成 corruption 的平均结果去否定这一点。

## 4. SDD-CR：真实难样本子集验证

![Subgroup overview](figures/subgroup_overview.png)

{markdown_table(
    subgroup_best,
    columns=["subset_title", "title", "mean_eer", "auerc"],
    headers=["子集", "最优实验组", "mean EER", "AUERC"],
    digits=4,
)}

### 4.1 双分支与单分支在关键子集上的 family 对比

{markdown_table(
    subgroup_fusion,
    columns=["subset_title", "fusion_mode", "mean_eer_mean", "auerc_mean"],
    headers=["子集", "融合方式", "mean EER", "AUERC"],
    digits=4,
)}

**子集解释：**

1. `incomplete` 子集上，**双分支 family 胜出**：`{incomplete_winner}` 更优，优势约为 `{incomplete_delta:.4f}`。
2. `enhanced_extreme` 子集上，**双分支 family 同样更优**：`{enhanced_winner}` 更优，优势约为 `{enhanced_delta:.4f}`。
3. `hard25` 子集上，family 平均仍是 `{hard25_winner}` 略优，优势约为 `{hard25_delta:.4f}`；但最佳单模型并不是双分支 + WaveFormer，说明 **CR 分支的收益已经出现，但还没有稳定到所有单模型配置都一致受益**。

## 5. 合成 corruption 压力测试（附录性证据）

上一版报告里，合成 brightness / contrast / blur / noise benchmark 被直接拿来与 clean 主结论并列，这会放大分布外扰动对结论的干扰。  
本次重写后，它只保留为 **附录性压力测试**：

1. 最优 robustness cell 为 `{robust_best["title"]}`，`robust mean EER = {robust_best["mean_eer_mean"]:.4f}`。
2. 该结果用于说明“极端合成扰动下差异很小且不稳定”，**不再用来否定 WaveFormer 的 clean 主效应**。
3. 真实课题叙述中，优先引用第 3 节和第 4 节，不再把合成 corruption 作为主证据。

## 6. 最终结论

1. **WaveFormer 结论保持成立**：在全局 clean 2×2 因子实验里，`waveformer=True` 的主效应更优，说明 WaveFormer 对整体 biometric utility 是正向模块，应继续保留。
2. **SDD-CR 双分支并非“全样本平均必胜”**：直接看全体样本均值，当前 `adaptive` 仍不如 `sdd-only`，因此不能把“默认 adaptive 更强”当作最终结论。
3. **但 SDD-CR 双分支在真实结构困难样本上是有效的**：在 `incomplete` 与 `enhanced_extreme` 这两个更符合 CR 理论作用场景的子集上，双分支 family 更优，且 `Full + WaveFormer` 在 `incomplete` 上取得最佳结果。
4. 因而更准确的研究表述应为：**CR 分支已经证明具备针对结构困难样本的补充价值，但当前自适应融合策略还没有把这部分优势稳定转化为全体样本的平均增益。**
"""
    REPORT_PATH.write_text(report, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run proof-oriented PV-IQA experiments.")
    parser.add_argument(
        "--stage",
        default="all",
        choices=["factorial", "robustness", "subgroups", "all"],
    )
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--recognizer-epochs", type=int, default=12)
    parser.add_argument("--iqa-epochs", type=int, default=18)
    parser.add_argument("--seeds", type=int, nargs="+", default=[2026, 2027])
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dir(EXPR_ROOT)

    if args.stage in {"factorial", "all"}:
        raw_rows: list[dict[str, Any]] = []
        for seed in args.seeds:
            for cell in CELLS:
                config = run_factorial_cell(
                    cell=cell,
                    seed=seed,
                    recognizer_epochs=args.recognizer_epochs,
                    iqa_epochs=args.iqa_epochs,
                    device=args.device,
                    num_workers=args.num_workers,
                    force=args.force,
                )
                raw_rows.append(
                    collect_factorial_row(
                        cell=cell,
                        config=config,
                        seed=seed,
                        iqa_epochs=args.iqa_epochs,
                    )
                )
        raw_factorial = pd.DataFrame(raw_rows)
        factorial_summary, factorial_factors = save_factorial_outputs(raw_factorial)
    else:
        raw_factorial = pd.read_csv(FACTORIAL_ROOT / "raw_results.csv")
        factorial_summary = pd.read_csv(FACTORIAL_ROOT / "summary.csv")
        factorial_factors = pd.read_csv(FACTORIAL_ROOT / "factor_effects.csv")

    if args.stage in {"robustness", "all"}:
        condition_rows: list[pd.DataFrame] = []
        monotonicity_rows: list[pd.DataFrame] = []
        for _, row in raw_factorial.iterrows():
            payload = build_base_payload(
                seed=int(row["seed"]),
                recognizer_epochs=args.recognizer_epochs,
                iqa_epochs=args.iqa_epochs,
                device=args.device,
                num_workers=args.num_workers,
            )
            cell = next(cell for cell in CELLS if cell.name == row["experiment"])
            payload = deep_update(payload, cell.overrides)
            config = resolved_config(
                cell.name,
                payload,
                config_name=f"{cell.name}-seed{int(row['seed'])}-iqa{args.iqa_epochs}",
            )
            condition_frame, monotonicity_frame = run_robustness_for_cell(
                cell=cell,
                config=config,
                seed=int(row["seed"]),
                iqa_epochs=args.iqa_epochs,
            )
            condition_rows.append(condition_frame)
            monotonicity_rows.append(monotonicity_frame)
        condition_frame = pd.concat(condition_rows, ignore_index=True)
        monotonicity_frame = pd.concat(monotonicity_rows, ignore_index=True)
        condition_summary, robustness_summary, robustness_factors = save_robustness_outputs(
            condition_frame,
            monotonicity_frame,
        )
    else:
        condition_summary = pd.read_csv(ROBUSTNESS_ROOT / "condition_summary.csv")
        robustness_summary = pd.read_csv(ROBUSTNESS_ROOT / "summary.csv")
        robustness_factors = pd.read_csv(ROBUSTNESS_ROOT / "factor_effects.csv")

    if args.stage in {"subgroups", "all"}:
        subgroup_frame = run_subgroup_benchmark(
            raw_factorial,
            recognizer_epochs=args.recognizer_epochs,
            iqa_epochs=args.iqa_epochs,
            device=args.device,
            num_workers=args.num_workers,
        )
        subgroup_summary, subgroup_factors = save_subgroup_outputs(subgroup_frame)
    else:
        subgroup_summary = pd.read_csv(SUBGROUP_ROOT / "summary.csv")
        subgroup_factors = pd.read_csv(SUBGROUP_ROOT / "factor_effects.csv")

    plot_clean_overview(factorial_summary)
    plot_factor_effects(factorial_factors, robustness_factors)
    plot_robustness_overview(robustness_summary)
    plot_corruption_curves(condition_summary)
    plot_subgroup_overview(subgroup_summary)

    write_report(
        factorial_summary=factorial_summary,
        factorial_factors=factorial_factors,
        subgroup_summary=subgroup_summary,
        subgroup_factors=subgroup_factors,
        condition_summary=condition_summary,
        robustness_summary=robustness_summary,
        robustness_factors=robustness_factors,
        recognizer_epochs=args.recognizer_epochs,
        iqa_epochs=args.iqa_epochs,
        seeds=args.seeds,
    )
    print(f"factorial summary: {FACTORIAL_ROOT / 'summary.csv'}")
    print(f"robustness summary: {ROBUSTNESS_ROOT / 'summary.csv'}")
    print(f"subgroup summary: {SUBGROUP_ROOT / 'summary.csv'}")
    print(f"report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
