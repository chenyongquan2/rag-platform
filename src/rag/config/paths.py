"""Paths of files committed to the repo (manifests, KB configs), independent of the working dir."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
"""The checkout this package runs from: `src/rag/config/paths.py` is three levels down."""


def repo_path(path: str | Path) -> Path:
    """A relative path is taken relative to the repo root, not the current working directory."""
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def main_checkout(repo_root: Path = REPO_ROOT) -> Path:
    """The main checkout of this git repository, also when running from a linked worktree.

    In a linked worktree `.git` is a file `gitdir: <main>/.git/worktrees/<name>`, wherever the
    worktree directory lives. Without git metadata (an installed copy) this is `repo_root`.
    """
    dot_git = repo_root / ".git"
    if dot_git.is_file():
        text = dot_git.read_text(encoding="utf-8").strip()
        if text.startswith("gitdir:"):
            gitdir = Path(text.removeprefix("gitdir:").strip())
            if not gitdir.is_absolute():
                gitdir = (repo_root / gitdir).resolve()
            if gitdir.parent.name == "worktrees" and gitdir.parent.parent.name == ".git":
                return gitdir.parent.parent.parent
    return repo_root


def shared_data_dir(repo_root: Path = REPO_ROOT) -> Path:
    """`<parent of the main checkout>/rag-platform-data`: runtime data that the main checkout and
    every worktree share (the cost ledger), outside any checkout and never committed."""
    return main_checkout(repo_root).parent / "rag-platform-data"
