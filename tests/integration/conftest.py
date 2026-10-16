"""Fixtures for integration tests against the docker-compose services (`scripts/dev_up.sh`).

- Each service fixture skips its tests with a clear message when the service is not reachable,
  so `uv run pytest -m integration` without Docker reports skips, not errors.
- Everything a test creates carries a prefix unique to this pytest run (`test<run>_` for
  indices, `test:<run>:` for Redis keys) and is deleted at the end; real indices and keys of
  the same services are never touched.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable, Iterator

import pytest
from elastic_transport import TransportError
from elasticsearch import ApiError, Elasticsearch
from redis import Redis
from redis.exceptions import RedisError

from rag.config import ServiceSettings
from rag.config.settings import mask_url
from rag.factories import make_es_client, make_redis_client
from rag.index.generation import DEFAULT_KEY_PREFIX
from rag.index.names import KbIndexNames

RUN_ID = uuid.uuid4().hex[:8]
INDEX_PREFIX = f"test{RUN_ID}_"
REDIS_KEY_PREFIX = f"test:{RUN_ID}:"
# The CLI puts its --index-prefix into the data generation key as well.
CLI_REDIS_KEY_PREFIX = f"{DEFAULT_KEY_PREFIX}{INDEX_PREFIX}"
START_HINT = "start the services with scripts/dev_up.sh (docker compose up -d)"


@pytest.fixture(scope="session")
def service_settings() -> ServiceSettings:
    return ServiceSettings()


@pytest.fixture(scope="session")
def index_prefix() -> str:
    """Prefix of every index created by this test run."""
    return INDEX_PREFIX


@pytest.fixture(scope="session")
def redis_key_prefix() -> str:
    """Prefix of every Redis key created by this test run."""
    return REDIS_KEY_PREFIX


@pytest.fixture(scope="session")
def cli_redis_key_prefix() -> str:
    """Data generation key prefix the CLI uses with `--index-prefix <index_prefix>`."""
    return CLI_REDIS_KEY_PREFIX


@pytest.fixture(scope="session")
def es(service_settings: ServiceSettings) -> Iterator[Elasticsearch]:
    url = service_settings.es_url
    client = make_es_client(url, request_timeout=30.0, max_retries=0)
    try:
        client.info()
    except (TransportError, ApiError) as exc:
        pytest.skip(f"Elasticsearch not reachable at {mask_url(url)} ({exc}); {START_HINT}")
    yield client
    delete_test_indices(client, f"{INDEX_PREFIX}*")
    client.close()


def delete_test_indices(client: Elasticsearch, pattern: str) -> None:
    """Delete indices matching `pattern` one by one (ES refuses wildcard deletes by default)."""
    if not pattern.startswith(INDEX_PREFIX):
        raise ValueError(f"refusing to delete indices outside this test run: {pattern!r}")
    resolved = client.indices.resolve_index(name=pattern, expand_wildcards="all")
    for item in resolved["indices"]:
        client.indices.delete(index=item["name"])


@pytest.fixture
def make_names(es: Elasticsearch) -> Iterator[Callable[[], KbIndexNames]]:
    """Factory of index names for fresh test knowledge bases; their indices go after the test."""
    created: list[KbIndexNames] = []

    def factory() -> KbIndexNames:
        names = KbIndexNames(f"k{uuid.uuid4().hex[:8]}", prefix=INDEX_PREFIX)
        created.append(names)
        return names

    yield factory
    for names in created:
        delete_test_indices(es, names.pattern)


@pytest.fixture(scope="session")
def redis_client(service_settings: ServiceSettings) -> Iterator[Redis]:
    url = service_settings.redis_url
    client = make_redis_client(url, socket_timeout=5.0)
    try:
        client.ping()
    except (RedisError, OSError) as exc:
        pytest.skip(f"Redis not reachable at {mask_url(url)} ({exc}); {START_HINT}")
    yield client
    for prefix in (REDIS_KEY_PREFIX, CLI_REDIS_KEY_PREFIX):
        for key in client.scan_iter(match=f"{prefix}*", count=100):
            client.delete(key)
    client.close()
