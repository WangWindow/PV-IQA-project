from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


@dataclass
class Config:
    name: str = "auto"
    output_root: str = "checkpoints"
    seed: int = 2026
    device: str = "auto"
    amp: bool = True
    num_workers: int = 4

    data_root: str = "datasets/ROI_Data"
    metadata_path: str = "auto"
    identity_mode: str = "separate"
    image_size: int = 224
    batch_size: int = 32
    eval_batch_size: int = 64
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    grayscale_to_rgb: bool = True

    recog_backbone: str = "mobilenetv3_large_100"
    recog_pretrained: bool = True
    recog_embedding_dim: int = 256
    recog_dropout: float = 0.1
    recog_margin: float = 0.3
    recog_scale: float = 30.0
    recog_epochs: int = 20
    recog_lr: float = 3e-4
    recog_wd: float = 1e-4
    recog_warmup_epochs: int = 1

    pseudo_split: str = "all"
    pseudo_alpha: float = 0.5
    pseudo_beta: float = 0.0
    pseudo_negative_samples: int = 512
    pseudo_gmm_components: int = 2
    pseudo_min_positive: int = 4
    pseudo_eps: float = 1e-6

    iqa_backbone: str = "mobilenetv3_large_100"
    iqa_pretrained: bool = True
    iqa_epochs: int = 20
    iqa_lr: float = 2e-4
    iqa_wd: float = 5e-5
    iqa_warmup_epochs: int = 2
    iqa_grad_clip: float = 1.0
    iqa_huber_delta: float = 0.1
    iqa_rank_margin: float = 0.03
    iqa_rank_weight: float = 0.3
    iqa_min_rank_gap: float = 0.05

    eval_split: str = "test"
    eval_reject_steps: list[float] = field(
        default_factory=lambda: [0.0, 0.05, 0.1, 0.2, 0.3]
    )
    eval_far_targets: list[float] = field(default_factory=lambda: [1e-2, 1e-3, 1e-4])
    eval_max_impostor_pairs: int = 20000

    wandb_enabled: bool = False

    @property
    def experiment_dir(self) -> Path:
        return Path(self.output_root) / self.name

    def resolve(self) -> "Config":
        if self.name in ("", "auto"):
            self.name = f"{datetime.now():%Y%m%d-%H%M%S}"
        if self.metadata_path in ("", "auto"):
            self.metadata_path = str(self.experiment_dir / "data" / "metadata.csv")
        return self
