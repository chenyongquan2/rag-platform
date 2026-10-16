from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]

_SAMPLE_TESTS = """
import pytest

def test_plain():
    pass

@pytest.mark.integration
def test_needs_docker():
    pass

@pytest.mark.llm
def test_costs_money():
    pass
"""


def _pytest_ini_section() -> str:
    text = (REPO_ROOT / "pyproject.toml").read_text("utf-8")
    start = text.index("[tool.pytest.ini_options]")
    end = text.find("\n[", start + 1)
    return text[start:] if end == -1 else text[start:end]


def _clean_git_env(**extra: str) -> dict[str, str]:
    # When this runs inside a git hook, GIT_DIR / GIT_INDEX_FILE point at the outer repository.
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    env.update(extra)
    return env


def test_pytest_markers_and_default_selection(pytester: pytest.Pytester) -> None:
    pytester.makepyprojecttoml(_pytest_ini_section())
    tests_dir = pytester.mkdir("tests")
    (tests_dir / "conftest.py").write_text(
        (REPO_ROOT / "tests" / "conftest.py").read_text("utf-8"), "utf-8"
    )
    (tests_dir / "test_sample.py").write_text(_SAMPLE_TESTS, "utf-8")

    # Default run: only the unmarked (unit) test.
    pytester.runpytest().assert_outcomes(passed=1, deselected=2)
    # Unmarked tests get the unit marker.
    pytester.runpytest("-m", "unit").assert_outcomes(passed=1, deselected=2)
    # -m on the command line replaces the default selection.
    pytester.runpytest("-m", "integration").assert_outcomes(passed=1, deselected=2)
    pytester.runpytest("-m", "llm").assert_outcomes(passed=1, deselected=2)
    # Unregistered markers are errors (--strict-markers).
    (tests_dir / "test_typo.py").write_text(
        "import pytest\n\n@pytest.mark.integraton\ndef test_x():\n    pass\n", "utf-8"
    )
    typo = pytester.runpytest()
    assert typo.ret != 0
    typo.stdout.fnmatch_lines(["*integraton*not found in*markers*"])


def _git(*args: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args], cwd=cwd, env=_clean_git_env(), capture_output=True, text=True
    )


@pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")
def test_env_file_is_git_ignored() -> None:
    def ignored(path: str) -> bool:
        return _git("check-ignore", "-q", "--no-index", path, cwd=REPO_ROOT).returncode == 0

    assert ignored(".env")
    assert ignored(".env.local")
    assert ignored("services/api/.env")
    assert not ignored(".env.example")


@pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")
def test_forbid_env_hook_rejects_staged_env(tmp_path: Path) -> None:
    config = yaml.safe_load((REPO_ROOT / ".pre-commit-config.yaml").read_text("utf-8"))
    hook = next(
        h
        for repo in config["repos"]
        if repo["repo"] == "local"
        for h in repo["hooks"]
        if h["id"] == "forbid-env-files"
    )
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".pre-commit-config.yaml").write_text(
        yaml.safe_dump({"repos": [{"repo": "local", "hooks": [hook]}]}), "utf-8"
    )
    (repo / ".env").write_text("DUMMY_SETTING=1\n", "utf-8")
    (repo / ".env.example").write_text("DUMMY_SETTING=\n", "utf-8")
    assert _git("init", "-q", cwd=repo).returncode == 0
    assert _git("add", ".pre-commit-config.yaml", ".env", ".env.example", cwd=repo).returncode == 0

    rejected = subprocess.run(
        [sys.executable, "-m", "pre_commit", "run", "forbid-env-files"],
        cwd=repo,
        env=_clean_git_env(PRE_COMMIT_HOME=str(tmp_path / "pre-commit-home")),
        capture_output=True,
        text=True,
    )

    assert rejected.returncode == 1, rejected.stdout + rejected.stderr
    assert "Failed" in rejected.stdout
    # The hook lists the offending files: .env is rejected, .env.example is allowed.
    reported = {line.strip() for line in rejected.stdout.splitlines()}
    assert ".env" in reported
    assert ".env.example" not in reported
