#!/usr/bin/env python3
"""Run the Cityscapes downloader command-line interface.

Examples:
    uv run python scripts/download_cityscapes.py download \
        --packages=leftImg8bit_trainvaltest.zip,gtFine_trainvaltest.zip \
        --destination=data/cityscapes
    uv run python scripts/download_cityscapes.py download --dry-run
"""

import fire
from cityscapes_tools.downloader import download

if __name__ == "__main__":
    fire.Fire({"download": download})
