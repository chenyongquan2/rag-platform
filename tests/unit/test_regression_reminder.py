"""Spec EVAL-4 AC14: the pre-commit reminder for the regression command."""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path
from types import ModuleType

import pytest
import yaml

from rag.config import repo_path


def _script() -> ModuleType:
    path = repo_path("scripts/regression_reminder.py")
    spec = importlib.util.spec_from_file_location("regression_reminder", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _hook() -> dict[str, object]:
    config = yaml.safe_load(repo_path(".pre-commit-config.yaml").read_text(encoding="utf-8"))
    hooks = [
        h for repo in config["repos"] for h in repo["hooks"] if h["id"] == "regression-reminder"
    ]
    assert len(hooks) == 1
    return hooks[0]


def test_reminder_text_and_exit_code(capsys: pytest.CaptureFixture[str]) -> None:
    module = _script()
    for argv in ([], ["src/rag/agent/graph.py"], ["a", "b", "c", "d", "e", "f", "g"], [""]):
        assert module.main(argv) == 0  # never blocks a commit
        out = capsys.readouterr().out
        assert "please run the regression command" in out
        assert module.COMMAND in out
    assert "src/rag/agent/graph.py" in module.reminder(["src/rag/agent/graph.py"])


def test_hook_definition_matches_guarded_paths() -> None:
    hook = _hook()
    pattern = re.compile(str(hook["files"]))
    guarded = [
        "src/rag/agent/nodes.py",
        "src/rag/answer/generate.py",
        "src/rag/retrieve/hybrid.py",
        "src/rag/config/kb.py",
        "kb/public/hf_docs.yaml",
        "kb/private/notes.yml",
    ]
    for path in guarded:
        assert pattern.search(path), path
    for path in [
        "tests/unit/test_agent.py",
        "docs/progress.md",
        "src/rag/eval/agentic/cli.py",
        "src/rag/ingest/pipeline.py",
        "specs/EVAL-4-regression-set.md",
        "src/rag/agentish/x.py",
    ]:
        assert not pattern.search(path), path
    assert hook["verbose"] is True  # the message shows although the hook passes
    assert hook["pass_filenames"] is False
    assert "regression_reminder.py" in str(hook["entry"])
    assert hook.get("always_run") in (None, False)
    assert Path(repo_path("scripts/regression_reminder.py")).is_file()
