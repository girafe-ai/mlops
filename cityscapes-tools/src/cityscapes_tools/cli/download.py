"""Console entry point for the Cityscapes archive downloader."""

import argparse

from cityscapes_tools.pipelines.download import download


def main() -> None:
    """Parse download options and call the archive downloader."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--packages",
        help="Comma-separated archive names; defaults to left images and fine labels.",
    )
    parser.add_argument("--destination", default="data/cityscapes")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    arguments = vars(parser.parse_args())
    if arguments["packages"] is None:
        del arguments["packages"]
    download(**arguments)


if __name__ == "__main__":
    main()
