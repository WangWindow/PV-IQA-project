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

    def info(self, msg: str) -> None:
        self.logger.info(msg)

    def finish(self) -> None:
        for h in self.logger.handlers:
            h.close()
