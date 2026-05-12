import os
import sys

os.environ.setdefault("WANDB_MODE", "offline")
sys.path.insert(0, "src")

from pv_iqa.config import Config
from pv_iqa.eval import run_evaluation
from pv_iqa.train.iqa import train_iqa
from pv_iqa.train.pseudo_labels import generate_pseudo_labels
from pv_iqa.train.recognition import export_features, train_recognizer
from pv_iqa.utils.common import ensure_dir, set_seed
from pv_iqa.utils.datasets import build_metadata
from pv_iqa.utils.export_onnx import export_onnx
from pv_iqa.utils.logging import ExperimentLogger


def main():
    config = Config().resolve()
    set_seed(config.seed)

    print(f"Run: {config.name}")
    print("1/5 prepare data")
    build_metadata(config)
    print("2/5 train recognizer")
    ckpt = train_recognizer(config)
    export_features(config, ckpt)
    print("3/5 pseudo-labels")
    generate_pseudo_labels(config)
    print("4/6 train iqa")
    ckpt = train_iqa(config)
    print("5/6 evaluate")
    eval_logger = ExperimentLogger(config, ensure_dir(config.experiment_dir / "iqa"))
    results = run_evaluation(config, ckpt, logger=eval_logger)
    eval_logger.finish()
    print("6/6 export onnx")
    export_onnx(config, ckpt)
    print(f"\nDone: checkpoints/{config.name}")
    return results


if __name__ == "__main__":
    main()
