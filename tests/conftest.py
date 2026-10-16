"""Shared pytest setup.

- Tests without an explicit `integration` or `llm` marker are marked `unit`.
- Every test runs with LlamaIndex `Settings.llm` set to a stub that raises when called and
  `Settings.embed_model` set to a deterministic fake, so nothing silently reaches OpenAI or MockLLM.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

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
