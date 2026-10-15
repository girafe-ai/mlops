"""Console entry point for raw PyTorch training."""

import hydra
from omegaconf import DictConfig

from cityscapes_tools.training.torch.runner import run


@hydra.main(version_base=None, config_path="../conf", config_name="torch")
def main(config: DictConfig) -> None:
    """Compose the experiment configuration and start the PyTorch runner."""
    run(config)


if __name__ == "__main__":
    main()
