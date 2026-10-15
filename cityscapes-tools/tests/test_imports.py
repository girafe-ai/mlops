import importlib
import subprocess
import sys

import requests


def test_downloader_import_has_no_network_side_effects(monkeypatch):
    def fail_if_session_is_created(*args, **kwargs):
        raise AssertionError("Importing downloader must not create an HTTP session")

    monkeypatch.setattr(requests, "Session", fail_if_session_is_created)
    monkeypatch.delitem(
        sys.modules, "cityscapes_tools.pipelines.download", raising=False
    )

    module = importlib.import_module("cityscapes_tools.pipelines.download")

    assert callable(module.download)


def test_raw_torch_entry_point_does_not_import_lightning():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import cityscapes_tools.cli.train_torch; "
                "assert not any(name.split('.')[0] in ('lightning', 'pytorch_lightning') "
                "for name in sys.modules)"
            ),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
