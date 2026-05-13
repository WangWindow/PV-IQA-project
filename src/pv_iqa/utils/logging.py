import logging
from pathlib import Path

from pv_iqa.config import Config
from pv_iqa.utils.common import ensure_dir


class ExperimentLogger:
    def __init__(self, config: Config, output_dir: Path) -> None:
        ensure_dir(output_dir)
        self.logger = logging.getLogger("pv_iqa")
        self.logger.setLevel(logging.INFO)
        self.logger.handlers.clear()

        fmt = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S"
        )

        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        self.logger.addHandler(sh)

        fh = logging.FileHandler(output_dir / "run.log", encoding="utf-8")
        fh.setFormatter(fmt)
        self.logger.addHandler(fh)

        self.logger.propagate = False

        self.wandb = None
        if config.wandb_enabled:
            try:
                import wandb
                self.wandb = wandb
                self.wandb.init(
                    project=config.wandb_project,
                    name=config.wandb_run_name or config.name,
                    config={
                        k: getattr(config, k)
                        for k in [
                            "pseudo_beta", "pseudo_mode",
                            "iqa_epochs", "iqa_lr", "iqa_wd",
                            "iqa_huber_delta",
                            "iqa_rank_weight", "iqa_min_rank_gap",
                            "iqa_degrade_rank_weight", "iqa_degrade_margin",
                            "iqa_sigmoid_tau",
                        ]
                    },
                    reinit=True,
                )
            except Exception as e:
                self.info(f"wandb init failed: {e}")

    def log_metrics(self, metrics: dict, step: int | None = None) -> None:
        if self.wandb:
            self.wandb.log(metrics, step=step)

    def info(self, msg: str) -> None:
        self.logger.info(msg)

    def finish(self) -> None:
        for h in self.logger.handlers:
            h.close()
        if self.wandb:
            self.wandb.finish()
