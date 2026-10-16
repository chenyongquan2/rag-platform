"""Paths of files committed to the repo (manifests, KB configs), independent of the working dir."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
"""The checkout this package runs from: `src/rag/config/paths.py` is three levels down."""


def repo_path(path: str | Path) -> Path:
    """A relative path is taken relative to the repo root, not the current working directory."""
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path
