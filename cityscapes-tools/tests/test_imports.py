import importlib
import sys

import requests


def test_downloader_import_has_no_network_side_effects(monkeypatch):
    def fail_if_session_is_created(*args, **kwargs):
        raise AssertionError("Importing downloader must not create an HTTP session")

    monkeypatch.setattr(requests, "Session", fail_if_session_is_created)
    monkeypatch.delitem(sys.modules, "cityscapes_tools.downloader", raising=False)

    module = importlib.import_module("cityscapes_tools.downloader")

    assert callable(module.download)
