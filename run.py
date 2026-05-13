import os
import sys

os.environ.setdefault("WANDB_MODE", "offline")
sys.path.insert(0, "src")

from pv_iqa.config import Config
from pv_iqa.train.iqa import train_iqa
from pv_iqa.train.pseudo_labels import generate_pseudo_labels
from pv_iqa.train.recognition import export_features, train_recognizer
from pv_iqa.utils.common import set_seed
from pv_iqa.utils.datasets import build_metadata
from pv_iqa.utils.export_onnx import export_onnx


def main():
    config = Config().resolve()
    set_seed(config.seed)

    print(f"Run: {config.name}")
    print("1 prepare data")
    build_metadata(config)

    print("2 train recognizer")
    ckpt = train_recognizer(config)
    export_features(config, ckpt)

    print("3 pseudo-labels")
    generate_pseudo_labels(config)

    print("4 train iqa")
    ckpt = train_iqa(config)

    print("== export onnx ==")
    export_onnx(config, ckpt)

    print(f"\nDone: checkpoints/{config.name}")


if __name__ == "__main__":
    main()
