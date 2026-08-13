"""Smoke test for the notebooks that ship as the site.

Notebooks are discovered from the filesystem rather than named here, so adding or
renaming a chapter needs no change to this file. The contract is narrow on purpose:
a notebook that fails to import also fails to export to WebAssembly, which breaks
the deploy.
"""

import importlib.util
import sys
from pathlib import Path

import marimo
import pytest

NOTEBOOK_DIR = Path(__file__).resolve().parents[1] / "notebooks"
NOTEBOOKS = sorted(NOTEBOOK_DIR.glob("*.py"))


def test_notebooks_are_discovered():
    assert NOTEBOOKS, f"no notebooks found in {NOTEBOOK_DIR}"


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.stem)
def test_notebook_imports(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = module
    spec.loader.exec_module(module)
    assert isinstance(module.app, marimo.App)
