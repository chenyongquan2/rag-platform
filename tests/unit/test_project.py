from __future__ import annotations

import shlex
import tomllib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_pytest_markers_and_default_selection(request: pytest.FixtureRequest) -> None:
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text("utf-8"))
    ini = pyproject["tool"]["pytest"]["ini_options"]

    registered = {m.split(":", 1)[0].strip() for m in ini["markers"]}
    assert registered == {"unit", "integration", "llm"}

    addopts = ini["addopts"]
    args = addopts if isinstance(addopts, list) else shlex.split(addopts)
    assert "--strict-markers" in args
    assert args[args.index("-m") + 1] == "not integration and not llm"

    # tests/conftest.py marks unmarked tests as unit.
    assert request.node.get_closest_marker("unit") is not None


def test_env_file_is_git_ignored() -> None:
    lines = (REPO_ROOT / ".gitignore").read_text("utf-8").splitlines()
    assert ".env" in lines
    assert ".env.*" in lines
