"""Hydra config loader via Compose API.

The Compose API avoids the @hydra.main decorator which fights with
torchrun's process spawning (output dir collisions across ranks, cwd
changes, etc.). Every training script calls ``load_config()`` instead.

Override any parameter from the CLI:
    torchrun ... train_ddp.py train.batch_size=32 model.n_layer=12
"""

import sys
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

_CONF_DIR = str((Path(__file__).resolve().parent.parent / "conf").absolute())


def load_config(overrides: list[str] | None = None) -> DictConfig:
    if overrides is None:
        overrides = [a for a in sys.argv[1:] if "=" in a]
    with initialize_config_dir(config_dir=_CONF_DIR, version_base="1.3"):
        cfg = compose(config_name="config", overrides=overrides)
    return cfg
