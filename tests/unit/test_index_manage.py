from __future__ import annotations

from typing import Any

import pytest
from elastic_transport import ApiResponseMeta, HttpHeaders, NodeConfig
from elasticsearch import NotFoundError

from rag.index.manage import AliasConflictError, IndexStateError, switch_alias
from rag.index.names import KbIndexNames

NAMES = KbIndexNames("hf_docs")
V1, V2 = NAMES.index(1), NAMES.index(2)


def _not_found(error_type: str, index: str | None = None) -> NotFoundError:
    error: dict[str, Any] = {"type": error_type, "reason": f"{error_type} for test"}
    if index is not None:
        error["index"] = index
    meta = ApiResponseMeta(
        status=404,
        http_version="1.1",
        headers=HttpHeaders(),
        duration=0.0,
        node=NodeConfig("http", "127.0.0.1", 29200),
    )
    return NotFoundError(message=error_type, meta=meta, body={"error": error, "status": 404})


class _FakeIndices:
    def __init__(self, error: NotFoundError) -> None:
        self.error = error
        self.actions: list[dict[str, Any]] = []

    def exists(self, index: str) -> bool:
        return True

    def update_aliases(self, actions: list[dict[str, Any]]) -> None:
        self.actions = actions
        raise self.error


class _FakeEs:
    def __init__(self, error: NotFoundError) -> None:
        self.indices = _FakeIndices(error)


def test_switch_alias_classifies_not_found_errors() -> None:
    # The alias is no longer on the expected index: a conflict.
    es = _FakeEs(_not_found("aliases_not_found_exception"))
    with pytest.raises(AliasConflictError, match=V1):
        switch_alias(es, NAMES, V2, expected_current=V1)  # type: ignore[arg-type]
    assert es.indices.actions == [
        {"remove": {"index": V1, "alias": NAMES.alias, "must_exist": True}},
        {"add": {"index": V2, "alias": NAMES.alias}},
    ]

    # The expected index is gone, so the alias cannot be on it: a conflict.
    es = _FakeEs(_not_found("index_not_found_exception", index=V1))
    with pytest.raises(AliasConflictError, match="does not exist"):
        switch_alias(es, NAMES, V2, expected_current=V1)  # type: ignore[arg-type]

    # The target was deleted after the existence check: not an alias conflict.
    es = _FakeEs(_not_found("index_not_found_exception", index=V2))
    with pytest.raises(IndexStateError, match=V2):
        switch_alias(es, NAMES, V2, expected_current=V1)  # type: ignore[arg-type]

    # Anything else is passed on unchanged.
    es = _FakeEs(_not_found("resource_not_found_exception"))
    with pytest.raises(NotFoundError):
        switch_alias(es, NAMES, V2, expected_current=V1)  # type: ignore[arg-type]

    # Expecting the alias already on the target still sends the checked remove first.
    es = _FakeEs(_not_found("aliases_not_found_exception"))
    with pytest.raises(AliasConflictError):
        switch_alias(es, NAMES, V2, expected_current=V2)  # type: ignore[arg-type]
    assert es.indices.actions == [
        {"remove": {"index": V2, "alias": NAMES.alias, "must_exist": True}},
        {"add": {"index": V2, "alias": NAMES.alias}},
    ]
