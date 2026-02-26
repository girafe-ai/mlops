"""Download Cityscapes dataset packages.

Requires a Cityscapes account: https://www.cityscapes-dataset.com/register/

Usage:
    uv run python scripts/download.py list
    uv run python scripts/download.py download --destination_path ./data
    uv run python scripts/download.py download --package_names '["gtFine_trainvaltest.zip"]' --destination_path ./data
"""

from __future__ import annotations

import glob
import os
import zipfile

import fire
from cityscapesscripts.download.downloader import (
    download_packages,
    list_available_packages,
    login,
)

DEFAULT_PACKAGES = [
    "gtFine_trainvaltest.zip",
]


class CityscapesDownloader:
    """CLI for downloading Cityscapes dataset."""

    def __init__(self):
        self._session = None

    @property
    def session(self):
        if self._session is None:
            self._session = login()
        return self._session

    def list(self):
        """List all available packages for download."""
        list_available_packages(session=self.session)

    def download(
        self,
        destination_path: str = "./data",
        package_names: list[str] | None = None,
        resume: bool = False,
    ):
        """Download Cityscapes dataset packages.

        Args:
            destination_path: Directory to save downloaded files.
            package_names: List of package names to download.
                           Defaults to gtFine_trainvaltest.zip and leftImg8bit_trainvaltest.zip.
            resume: Resume a previously interrupted download.
        """
        if package_names is None:
            package_names = DEFAULT_PACKAGES

        os.makedirs(destination_path, exist_ok=True)

        download_packages(
            session=self.session,
            package_names=package_names,
            destination_path=destination_path,
            resume=resume,
        )
        print(f"Download complete. Files saved to '{destination_path}'.")

        for archive in glob.glob(os.path.join(destination_path, "*.zip")):
            print(f"Extracting '{archive}'...")
            with zipfile.ZipFile(archive, "r") as zf:
                zf.extractall(destination_path)
            os.remove(archive)
            print(f"Extracted and removed '{archive}'.")


if __name__ == "__main__":
    fire.Fire(CityscapesDownloader)
