"""Spec P5-01 review round 1 (item 2): an integration test that starts a real server process
must not let it use the shared cache: its entries (1 h) would be served to the next run of the
test. Every such test builds the environment with `server_env()` (cache off)."""

from __future__ import annotations

import re
from pathlib import Path

INTEGRATION = Path(__file__).resolve().parents[1] / "integration"


def test_tests_that_start_a_server_process_use_server_env() -> None:
    spawning = []
    for path in sorted(INTEGRATION.glob("test_*.py")):
        text = path.read_text("utf-8")
        if re.search(r"rag\.serve", text) and re.search(
            r"StdioServerParameters|subprocess|Popen", text
        ):
            spawning.append(path)
    assert spawning, "the check must find the stdio test"
    for path in spawning:
        text = path.read_text("utf-8")
        assert "server_env(" in text, f"{path.name} starts a server without server_env()"


def test_server_env_turns_the_cache_off_and_keeps_what_the_test_adds() -> None:
    from rag.testing.server import server_env

    env = server_env(EMBED_BACKEND="fake")
    assert env["CACHE_MODE"] == "off" and env["EMBED_BACKEND"] == "fake"
    assert server_env(CACHE_MODE="l1")["CACHE_MODE"] == "l1"  # a test about the cache may ask
