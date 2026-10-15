"""Console entry point for Lightning training."""

import hydra
from omegaconf import DictConfig

from cityscapes_tools.training.lightning.runner import run


@hydra.main(version_base=None, config_path="../conf", config_name="lightning")
def main(config: DictConfig) -> None:
    """Compose the experiment configuration and start the Lightning runner."""
    run(config)


if __name__ == "__main__":
    main()
