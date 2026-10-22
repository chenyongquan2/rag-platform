"""Shared pytest setup.

- Tests without an explicit `integration` or `llm` marker are marked `unit`.
- Every test starts with no shared chat gateway breaker (spec P8-01), so failures that one test
  sends to a fake gateway never open the breaker of the next.
- Every test runs with LlamaIndex `Settings.llm` set to a stub that raises when called and
  `Settings.embed_model` set to a deterministic fake, so nothing silently reaches OpenAI or MockLLM.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from rag.llm.breaker import reset_shared_breakers
from rag.testing import use_safe_llama_settings

pytest_plugins = ["pytester"]

_NON_UNIT_MARKERS = {"integration", "llm"}


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    for item in items:
        if not _NON_UNIT_MARKERS & {m.name for m in item.iter_markers()}:
            item.add_marker(pytest.mark.unit)


@pytest.fixture(autouse=True)
def _safe_llama_settings() -> Iterator[None]:
    with use_safe_llama_settings():
        yield


@pytest.fixture(autouse=True)
def _fresh_gateway_breakers() -> Iterator[None]:
    reset_shared_breakers()
    yield
    reset_shared_breakers()


@pytest.fixture(autouse=True)
def _pinned_budget(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    """Unit tests see the code's default budget (30 CNY, stop at 28), whatever the
    developer's .env says (the owner raised the stop line to 33 on 2026-10-22). Environment
    variables win over .env, so pinning them here keeps the tests independent of it."""
    if request.node.get_closest_marker("unit") is None:
        return
    monkeypatch.setenv("BUDGET_CNY", "30")
    monkeypatch.setenv("BUDGET_STOP_CNY", "28")
