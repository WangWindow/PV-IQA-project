# import os
import sys

# os.environ.setdefault("WANDB_MODE", "offline")
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
    config.wandb_enabled = False
    set_seed(config.seed)

    print(f"Run: {config.name}")
    print("1 prepare data")
    build_metadata(config)

    if config.recog_checkpoint:
        print("2 recognizer — 复用 checkpoint")
        recog_ckpt = config.recog_checkpoint
    else:
        print("2 train recognizer")
        recog_ckpt = train_recognizer(config)
    export_features(config, recog_ckpt)

    print("3 pseudo-labels")
    generate_pseudo_labels(config)

    print("4 train iqa")
    iqa_ckpt = train_iqa(config)

    print("== export onnx ==")
    export_onnx(config, iqa_ckpt)

    print(f"\nDone: checkpoints/{config.name}")


if __name__ == "__main__":
    main()
