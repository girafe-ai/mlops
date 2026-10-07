#!/usr/bin/env python3
"""Build tabular Cityscapes metadata according to the Hydra configuration."""

from pathlib import Path

import hydra
from cityscapes_tools.metadata import build_metadata
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="../conf", config_name="conf")
def main(config: DictConfig) -> None:
    """Extract image and object metadata from fine annotations."""
    counts = build_metadata(
        annotations_root=Path(config.data.annotations_root),
        images_root=Path(config.data.images_root),
        output_dir=Path(config.data.metadata_dir),
    )
    print(
        f"Built metadata for {counts['images']} images and {counts['objects']} objects."
    )


if __name__ == "__main__":
    main()
