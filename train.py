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
    config.device = "cuda"
    config.recog_epochs = 20
    config.iqa_epochs = 20
    config.iqa_lr = 1e-3
    set_seed(config.seed)

    print(f"Run: {config.name}")
    print("1/5 prepare data")
    build_metadata(config)
    print("2/5 train recognizer")
    ckpt = train_recognizer(config)
    export_features(config, ckpt)
    print("3/5 pseudo-labels")
    generate_pseudo_labels(config)
    print("4/5 train iqa")
    ckpt = train_iqa(config)
    print("5/5 export onnx")
    export_onnx(config, ckpt)
    print(f"Done: checkpoints/{config.name}")


if __name__ == "__main__":
    main()
