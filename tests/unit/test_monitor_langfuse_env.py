"""Spec P5-02: `python -m rag.monitor langfuse-env` writes the Langfuse secrets to .env."""

from __future__ import annotations

import base64
from pathlib import Path

import pytest

from rag.monitor import cli
from rag.monitor.langfuse_env import KEYS, ensure_env


def _read(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text("utf-8").splitlines():
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            values[key] = value
    return values


def test_missing_secrets_are_generated_and_the_basic_auth_matches_the_keys(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    env.write_text("LLM_API_KEY=keep-me\n", "utf-8", newline="")
    added = ensure_env(env)
    values = _read(env)
    assert values["LLM_API_KEY"] == "keep-me"
    assert set(added) == set(KEYS) and all(values[k] for k in KEYS)
    assert values["LANGFUSE_PUBLIC_KEY"].startswith("pk-lf-")
    assert values["LANGFUSE_SECRET_KEY"].startswith("sk-lf-")
    assert len(values["LANGFUSE_ENCRYPTION_KEY"]) == 64
    decoded = base64.b64decode(values["LANGFUSE_BASIC_AUTH"]).decode()
    assert decoded == f"{values['LANGFUSE_PUBLIC_KEY']}:{values['LANGFUSE_SECRET_KEY']}"


def test_existing_values_are_never_overwritten_and_a_second_run_adds_nothing(
    tmp_path: Path,
) -> None:
    env = tmp_path / ".env"
    env.write_text("LANGFUSE_SALT=mine\n", "utf-8", newline="")
    first = ensure_env(env)
    assert "LANGFUSE_SALT" not in first
    snapshot = env.read_bytes()
    assert ensure_env(env) == []
    assert env.read_bytes() == snapshot
    assert _read(env)["LANGFUSE_SALT"] == "mine"


def test_a_file_without_a_final_newline_and_a_missing_file_both_work(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    env.write_text("A=1", "utf-8", newline="")
    ensure_env(env)
    assert _read(env)["A"] == "1"
    fresh = tmp_path / "new.env"
    ensure_env(fresh)
    assert set(KEYS) <= set(_read(fresh))


def test_the_command_names_the_keys_it_wrote_but_never_a_value(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    env = tmp_path / ".env"
    from rag.testing import observe_settings

    assert cli.main(["langfuse-env", "--env-file", str(env)], settings=observe_settings()) == 0
    out = capsys.readouterr()
    values = _read(env)
    for key in KEYS:
        assert key in out.out
        assert values[key] not in out.out + out.err
