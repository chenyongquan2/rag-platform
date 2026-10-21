"""Spec P5-02 AC1: the app-side gauge of Elasticsearch and Redis reachability."""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from rag.observe import shutdown_observability
from rag.observe.health import DependencyChecker, register_dependency_gauge
from rag.testing import memory_observability, metric_points


@pytest.fixture(autouse=True)
def _no_active_observability() -> Iterator[None]:
    shutdown_observability()
    yield
    shutdown_observability()


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def _values(obs: object) -> dict[str, float]:
    return {
        attrs["rag.dependency"]: point.value
        for attrs, point in metric_points(obs, "rag.dependency.up")  # type: ignore[arg-type]
    }


def test_gauge_reports_each_dependency() -> None:
    def broken() -> None:
        raise ConnectionRefusedError("closed port")

    checker = DependencyChecker({"elasticsearch": lambda: None, "redis": broken})
    with memory_observability() as obs:
        register_dependency_gauge(obs.meter, checker)
        assert _values(obs) == {"elasticsearch": 1, "redis": 0}


def test_check_failure_is_zero_and_cached() -> None:
    calls: list[str] = []
    clock = Clock()

    def es() -> None:
        calls.append("es")
        raise TimeoutError("slow")

    checker = DependencyChecker({"elasticsearch": es}, ttl_s=10.0, clock=clock)
    with memory_observability() as obs:
        register_dependency_gauge(obs.meter, checker)
        assert _values(obs) == {"elasticsearch": 0}
        assert _values(obs) == {"elasticsearch": 0}
        assert calls == ["es"]  # the second export inside the TTL used the cached verdict
        clock.now += 11
        assert _values(obs) == {"elasticsearch": 0}
        assert calls == ["es", "es"]


def test_a_dependency_that_comes_back_turns_to_one() -> None:
    state = {"up": False}
    clock = Clock()

    def es() -> None:
        if not state["up"]:
            raise OSError("down")

    checker = DependencyChecker({"elasticsearch": es}, ttl_s=5.0, clock=clock)
    assert checker.status() == {"elasticsearch": False}
    state["up"] = True
    clock.now += 6
    assert checker.status() == {"elasticsearch": True}


def test_from_settings_names_the_two_dependencies() -> None:
    from rag.testing import observe_settings

    checker = DependencyChecker.from_settings(
        observe_settings(es_url="http://127.0.0.1:1", redis_url="redis://127.0.0.1:1/0")
    )
    assert checker.names == ("elasticsearch", "redis")
    # Nothing listens on port 1: both checks fail fast and return False (never raise).
    assert checker.status() == {"elasticsearch": False, "redis": False}


def test_the_server_command_registers_the_gauge(monkeypatch: pytest.MonkeyPatch) -> None:
    """`python -m rag.serve` reports the gauge with the app's own connection strings."""
    from rag.config import AccessConfig, KnowledgeBaseConfig
    from rag.observe import get_observability, init_observability
    from rag.serve import cli
    from rag.serve.service import SearchService
    from rag.testing import observe_settings
    from rag.testing.serve import FakePipelineBuilder, FakeSearchPipeline

    settings = observe_settings(es_url="http://127.0.0.1:1", redis_url="redis://127.0.0.1:1/0")
    seen: dict[str, float] = {}

    def run_stdio(mcp: object) -> None:
        seen.update(_values(get_observability()))

    def factory(s: object) -> SearchService:
        configs = {"hf_docs": KnowledgeBaseConfig(kb="hf_docs", access=AccessConfig(callers=["*"]))}
        builder = FakePipelineBuilder({"hf_docs": FakeSearchPipeline("hf_docs")})
        return SearchService(settings, configs=configs, builder=builder)

    monkeypatch.setattr(
        cli,
        "init_observability",
        lambda s, name: init_observability(s, name, configure_logging=False),
    )
    assert cli.main(["stdio"], settings=settings, service_factory=factory, run_stdio=run_stdio) == 0
    assert seen == {"elasticsearch": 0, "redis": 0}
