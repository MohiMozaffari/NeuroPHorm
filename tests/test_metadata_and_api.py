import importlib
import re
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
README_PATH = ROOT / "README.md"
SETUP_PATH = ROOT / "setup.py"
NODE_REMOVAL_PATH = ROOT / "neurophorm" / "node_removal.py"


def test_version_consistency():
    import neurophorm

    assert neurophorm.__version__ == "1.0.0"

    setup_text = SETUP_PATH.read_text(encoding="utf-8")
    match = re.search(r'version="([^"]+)"', setup_text)
    assert match is not None
    assert match.group(1) == "1.0.0"


def test_dependency_import_smoke():
    importlib.import_module("statsmodels.stats.multitest")
    importlib.import_module("neurophorm")


def test_deprecated_load_removal_data_alias(monkeypatch):
    import neurophorm.node_removal as node_removal

    sentinel = object()
    called = {}

    def fake_loader(*args, **kwargs):
        called["args"] = args
        called["kwargs"] = kwargs
        return sentinel

    monkeypatch.setattr(node_removal, "load_node_removal_data", fake_loader)

    with pytest.deprecated_call(DeprecationWarning):
        out = node_removal.load_removal_data("dummy_out", atlas=[1, 2, 3], per_subject=False)

    assert out is sentinel
    assert called["args"] == ("dummy_out",)
    assert called["kwargs"]["atlas"] == [1, 2, 3]
    assert called["kwargs"]["per_subject"] is False


def test_node_removal_regex_separator_is_raw_string():
    node_removal_text = NODE_REMOVAL_PATH.read_text(encoding="utf-8")
    assert 'sep=r"\\s+"' in node_removal_text


def test_readme_has_citation_anchor_and_zenodo_note():
    readme_text = README_PATH.read_text(encoding="utf-8")

    assert re.search(r"^##\s+.*Citation", readme_text, flags=re.MULTILINE) is not None
    assert "Recent updates on `main` may not yet be archived on Zenodo." in readme_text
