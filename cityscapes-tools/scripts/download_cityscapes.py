#!/usr/bin/env python3
"""Download packages from the Cityscapes Dataset website.

Examples:
    uv run python scripts/download_cityscapes.py download \
        --packages=leftImg8bit_trainvaltest.zip,gtFine_trainvaltest.zip \
        --destination=data/cityscapes
    uv run python scripts/download_cityscapes.py download --dry-run
"""

import hashlib
import os
import re
from pathlib import Path

import fire
import requests


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = PROJECT_ROOT / ".env"
LOGIN_URL = "https://www.cityscapes-dataset.com/login"
PACKAGES_URL = "https://www.cityscapes-dataset.com/downloads/?list"
MD5_URL = "https://www.cityscapes-dataset.com/md5-sum/?packageID={}"
DOWNLOAD_URL = "https://www.cityscapes-dataset.com/file-handling/?packageID={}"
REQUEST_TIMEOUT_SECONDS = 30
CHUNK_SIZE = 1024 * 1024


def load_dotenv(path: Path = ENV_FILE) -> None:
    """Load simple KEY=VALUE entries without overwriting exported environment variables."""
    if not path.is_file():
        return

    assignment = re.compile(r"(?:export )?([A-Za-z_][A-Za-z0-9_]*)=(.*)")
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        match = assignment.fullmatch(line)
        if match is None:
            raise ValueError("Invalid .env entry: {}".format(line))
        key, value = match.groups()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        os.environ.setdefault(key, value)


def get_credentials() -> dict[str, str]:
    """Read credentials from the environment or the project .env file."""
    load_dotenv()
    username = os.getenv("CITYSCAPES_USERNAME", "").strip()
    password = os.getenv("CITYSCAPES_PASSWORD", "")
    if not username or not password:
        raise ValueError(
            "Set CITYSCAPES_USERNAME and CITYSCAPES_PASSWORD in .env or the environment."
        )
    return {"username": username, "password": password}


def parse_packages(packages: str) -> list[str]:
    """Convert a comma-separated package list into safe file names."""
    names = [name.strip() for name in packages.split(",") if name.strip()]
    if not names:
        raise ValueError("Provide at least one package name with --packages.")
    if any(Path(name).name != name for name in names):
        raise ValueError("Package names must not contain directory components.")
    return names


def login() -> requests.Session:
    """Authenticate against the Cityscapes website and return an authenticated session."""
    credentials = get_credentials()
    session = requests.Session()
    response = session.get(LOGIN_URL, allow_redirects=False, timeout=REQUEST_TIMEOUT_SECONDS)
    response.raise_for_status()
    response = session.post(
        LOGIN_URL,
        data={**credentials, "submit": "Login"},
        allow_redirects=False,
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    if response.status_code != 302:
        raise RuntimeError("Cityscapes rejected the supplied credentials.")
    return session


def available_packages(session: requests.Session) -> dict[str, str]:
    """Return a mapping from visible package name to its Cityscapes package ID."""
    response = session.get(PACKAGES_URL, allow_redirects=False, timeout=REQUEST_TIMEOUT_SECONDS)
    response.raise_for_status()
    return {package["name"]: package["packageID"] for package in response.json()}


def file_md5(path: Path) -> str:
    """Calculate the MD5 checksum of a completed download."""
    digest = hashlib.md5()
    with path.open("rb") as downloaded_file:
        for chunk in iter(lambda: downloaded_file.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(
    packages: str = "leftImg8bit_trainvaltest.zip,gtFine_trainvaltest.zip",
    destination: str = "data/cityscapes",
    resume: bool = False,
    dry_run: bool = False,
) -> None:
    """Download comma-separated Cityscapes package names into a destination directory.

    Set dry_run=True to validate the command locally without authenticating,
    contacting Cityscapes, creating directories, or downloading files.
    """
    package_names = parse_packages(packages)
    destination_path = Path(destination).expanduser()

    if dry_run:
        load_dotenv()
        print("Dry run: no network or files will be changed.")
        print("Would download: {}".format(", ".join(package_names)))
        print("Destination: {}".format(destination_path))
        print("Resume enabled: {}".format(resume))
        if not ENV_FILE.is_file():
            print("Credential file not found: {}".format(ENV_FILE))
        elif not os.getenv("CITYSCAPES_USERNAME") or not os.getenv("CITYSCAPES_PASSWORD"):
            print("Credentials are not configured in {} yet.".format(ENV_FILE))
        else:
            print("Credentials are configured for a real download.")
        return

    destination_path.mkdir(parents=True, exist_ok=True)
    session = login()
    packages_by_name = available_packages(session)
    unavailable = [name for name in package_names if name not in packages_by_name]
    if unavailable:
        raise ValueError(
            "These packages do not exist or are not available to this account: {}".format(
                ", ".join(unavailable)
            )
        )

    for package_name in package_names:
        package_id = packages_by_name[package_name]
        target = destination_path / package_name
        if target.exists() and not resume:
            raise FileExistsError(
                "{} already exists. Re-run with --resume or choose another destination.".format(
                    target
                )
            )

        checksum_response = session.get(
            MD5_URL.format(package_id),
            allow_redirects=False,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        checksum_response.raise_for_status()
        expected_md5 = checksum_response.text.split()[0]
        mode = "ab" if resume else "wb"
        offset = target.stat().st_size if resume and target.exists() else 0
        headers = {"Range": "bytes={}-".format(offset)} if resume else {}

        print("Downloading {} to {}".format(package_name, target))
        with session.get(
            DOWNLOAD_URL.format(package_id),
            allow_redirects=False,
            stream=True,
            headers=headers,
            timeout=REQUEST_TIMEOUT_SECONDS,
        ) as response:
            response.raise_for_status()
            if response.status_code not in (200, 206):
                raise RuntimeError("Unexpected download response: {}".format(response.status_code))
            with target.open(mode) as downloaded_file:
                for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                    if chunk:
                        downloaded_file.write(chunk)

        actual_md5 = file_md5(target)
        if actual_md5 != expected_md5:
            raise RuntimeError(
                "Checksum mismatch for {}. Expected {}, received {}.".format(
                    target, expected_md5, actual_md5
                )
            )
        print("Verified {}".format(target))


if __name__ == "__main__":
    fire.Fire({"download": download})
