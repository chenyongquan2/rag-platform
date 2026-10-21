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


def test_from_settings_uses_the_factories_so_credentials_in_the_url_work(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Review P5-02 round 1, issue 5: `urllib` on an ES URL with `user:password@` never
    authenticates, so the gauge said "down" forever. The checks go through the factories (which
    turn the user info into basic auth), and no URL is ever logged."""
    import logging

    from rag import factories
    from rag.testing import observe_settings

    secret_es = "http://elastic:s3cret-pw@127.0.0.1:9200"
    secret_redis = "redis://:r3dis-pw@127.0.0.1:6379/0"
    seen: dict[str, str] = {}

    class FakeEs:
        def __init__(self, up: bool) -> None:
            self.up = up

        def ping(self) -> bool:
            return self.up

        def close(self) -> None:
            pass

    class FakeRedis:
        def ping(self) -> bool:
            raise ConnectionError(f"Error connecting to {secret_redis}")

        def close(self) -> None:
            pass

    state = {"es_up": True}

    def fake_es(url: str, **kwargs: object) -> FakeEs:
        seen["es"] = url
        return FakeEs(state["es_up"])

    def fake_redis(url: str, **kwargs: object) -> FakeRedis:
        seen["redis"] = url
        return FakeRedis()

    monkeypatch.setattr(factories, "make_es_client", fake_es)
    monkeypatch.setattr(factories, "make_redis_client", fake_redis)
    clock = Clock()
    checker = DependencyChecker.from_settings(
        observe_settings(es_url=secret_es, redis_url=secret_redis), clock=clock
    )
    with caplog.at_level(logging.DEBUG, logger="rag.observe.health"):
        assert checker.status() == {"elasticsearch": True, "redis": False}
        state["es_up"] = False  # ping() answers False instead of raising
        clock.now += 60
        assert checker.status() == {"elasticsearch": False, "redis": False}
    assert seen == {"es": secret_es, "redis": secret_redis}
    assert "s3cret-pw" not in caplog.text and "r3dis-pw" not in caplog.text
