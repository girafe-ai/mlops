"""Download and validate archives from the Cityscapes Dataset website.

The public entry point is `download`. It authenticates with the Cityscapes
portal, downloads requested archives, and verifies their MD5 checksums.
"""

import hashlib
import os
import re
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[2]
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
            raise ValueError(f"Invalid .env entry: {line}")
        key, value = match.groups()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        os.environ.setdefault(key, value)


def get_credentials() -> dict[str, str]:
    """Read Cityscapes credentials from the environment or project `.env` file.

    Exported environment variables take precedence over values in `.env`.

    Returns:
        A mapping with ``username`` and ``password`` values.

    Raises:
        ValueError: If either Cityscapes credential is not configured.
    """
    load_dotenv()
    username = os.getenv("CITYSCAPES_USERNAME", "").strip()
    password = os.getenv("CITYSCAPES_PASSWORD", "")
    if not username or not password:
        raise ValueError(
            "Set CITYSCAPES_USERNAME and CITYSCAPES_PASSWORD in .env or the environment."
        )
    return {"username": username, "password": password}


def parse_packages(packages: str) -> list[str]:
    """Convert a comma-separated package list into safe archive names.

    Args:
        packages: Comma-separated archive names from the Cityscapes portal.

    Returns:
        Package names with surrounding whitespace removed.

    Raises:
        ValueError: If no package is supplied or a name contains a directory
            component.
    """
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
    response = session.get(
        LOGIN_URL, allow_redirects=False, timeout=REQUEST_TIMEOUT_SECONDS
    )
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
    response = session.get(
        PACKAGES_URL, allow_redirects=False, timeout=REQUEST_TIMEOUT_SECONDS
    )
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
    """Download Cityscapes archives and verify their MD5 checksums.

    Args:
        packages: Comma-separated archive names from the Cityscapes portal.
        destination: Directory where downloaded archives are stored.
        resume: Continue partially downloaded archives when possible.
        dry_run: Validate arguments without using the network or creating files.

    Raises:
        ValueError: If package names or credentials are invalid.
        FileExistsError: If an archive already exists and resume is disabled.
        RuntimeError: If authentication, download, or checksum verification fails.
    """
    package_names = parse_packages(packages)
    destination_path = Path(destination).expanduser()

    if dry_run:
        load_dotenv()
        print("Dry run: no network or files will be changed.")
        print("Would download: {}".format(", ".join(package_names)))
        print(f"Destination: {destination_path}")
        print(f"Resume enabled: {resume}")
        if not ENV_FILE.is_file():
            print(f"Credential file not found: {ENV_FILE}")
        elif not os.getenv("CITYSCAPES_USERNAME") or not os.getenv(
            "CITYSCAPES_PASSWORD"
        ):
            print(f"Credentials are not configured in {ENV_FILE} yet.")
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
                f"{target} already exists. Re-run with --resume or choose another destination."
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
        headers = {"Range": f"bytes={offset}-"} if resume else {}

        print(f"Downloading {package_name} to {target}")
        with session.get(
            DOWNLOAD_URL.format(package_id),
            allow_redirects=False,
            stream=True,
            headers=headers,
            timeout=REQUEST_TIMEOUT_SECONDS,
        ) as response:
            response.raise_for_status()
            if response.status_code not in (200, 206):
                raise RuntimeError(
                    f"Unexpected download response: {response.status_code}"
                )
            with target.open(mode) as downloaded_file:
                for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                    if chunk:
                        downloaded_file.write(chunk)

        actual_md5 = file_md5(target)
        if actual_md5 != expected_md5:
            raise RuntimeError(
                f"Checksum mismatch for {target}. Expected {expected_md5}, received {actual_md5}."
            )
        print(f"Verified {target}")
