"""In-memory stand-ins for Elasticsearch and Redis in unit tests (no Docker, no network).

Only the calls our code makes are implemented, with the response shapes of the real clients.
Writes are visible at once (no refresh interval); `refresh` calls are only counted. Tests can
inject failures: `fail_ids` makes bulk items fail like a mapping error, `delete_failures` makes
`delete_by_query` report failures.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from elasticsearch import BadRequestError, NotFoundError


@dataclass
class _Meta:
    status: int


@dataclass
class _Response:
    body: Any
    meta: _Meta = field(default_factory=lambda: _Meta(200))

    def __getitem__(self, key: str) -> Any:
        return self.body[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.body.get(key, default)


def _values(source: Mapping[str, Any], name: str) -> list[Any]:
    value = source.get(name)
    if value is None:
        return []
    return list(value) if isinstance(value, list) else [value]


def matches(source: Mapping[str, Any], doc_id: str, query: Mapping[str, Any] | None) -> bool:
    """The subset of the query DSL our code sends: match_all, term, terms, ids, bool."""
    if not query:
        return True
    ((kind, body),) = query.items()
    if kind == "match_all":
        return True
    if kind == "term":
        ((name, value),) = body.items()
        if isinstance(value, Mapping):
            value = value["value"]
        return value in _values(source, name)
    if kind == "terms":
        ((name, wanted),) = body.items()
        return any(v in wanted for v in _values(source, name))
    if kind == "ids":
        return doc_id in body["values"]
    if kind == "bool":

        def clauses(key: str) -> list[Mapping[str, Any]]:
            items = body.get(key, [])
            return [items] if isinstance(items, Mapping) else list(items)

        if not all(matches(source, doc_id, q) for q in clauses("filter") + clauses("must")):
            return False
        if any(matches(source, doc_id, q) for q in clauses("must_not")):
            return False
        should = clauses("should")
        return not should or any(matches(source, doc_id, q) for q in should)
    raise NotImplementedError(f"FakeElasticsearch does not support query {kind!r}")


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0


class _FakeIndices:
    def __init__(self, es: FakeElasticsearch) -> None:
        self._es = es

    def create(
        self, *, index: str, settings: Any = None, mappings: Mapping[str, Any] | None = None
    ) -> dict[str, Any]:
        if index in self._es.mappings:
            raise BadRequestError(
                "resource_already_exists_exception",
                _Meta(400),  # type: ignore[arg-type]
                {"error": {"type": "resource_already_exists_exception"}},
            )
        self._es.mappings[index] = copy.deepcopy(dict(mappings or {}))
        self._es.docs[index] = {}
        return {"acknowledged": True, "index": index}

    def exists(self, *, index: str) -> bool:
        return index in self._es.mappings

    def get_mapping(self, *, index: str) -> dict[str, Any]:
        self._es.require(index)
        return {index: {"mappings": copy.deepcopy(self._es.mappings[index])}}

    def put_mapping(
        self,
        *,
        index: str,
        properties: Mapping[str, Any] | None = None,
        meta: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        self._es.require(index)
        mapping = self._es.mappings[index]
        props = mapping.setdefault("properties", {})
        for name, spec in (properties or {}).items():
            if name in props and props[name] != spec:
                raise BadRequestError(
                    "illegal_argument_exception",
                    _Meta(400),  # type: ignore[arg-type]
                    {"error": {"type": "illegal_argument_exception", "reason": f"{name} differs"}},
                )
            props[name] = copy.deepcopy(dict(spec))
        if meta is not None:
            mapping["_meta"] = copy.deepcopy(dict(meta))
        return {"acknowledged": True}

    def refresh(self, *, index: str) -> dict[str, Any]:
        self._es.require(index)
        self._es.refreshes.append(index)
        return {"_shards": {"failed": 0}}


class FakeElasticsearch:
    """`docs[index][_id] = _source`; `mappings[index]` as created. See the module doc."""

    def __init__(self) -> None:
        self.docs: dict[str, dict[str, dict[str, Any]]] = {}
        self.mappings: dict[str, dict[str, Any]] = {}
        self.indices = _FakeIndices(self)
        self.bulk_calls: list[int] = []
        """Number of operations (documents) in each bulk request."""
        self.refreshes: list[str] = []
        self.fail_ids: set[str] = set()
        self.delete_failures: list[dict[str, Any]] = []
        self.delete_queries: list[dict[str, Any]] = []

    def require(self, index: str) -> None:
        if index not in self.mappings:
            raise NotFoundError(
                "index_not_found_exception",
                _Meta(404),  # type: ignore[arg-type]
                {"error": {"type": "index_not_found_exception", "index": index}},
            )

    def options(self, **_: Any) -> FakeElasticsearch:
        return self

    def close(self) -> None:
        pass

    # ---- writes -----------------------------------------------------------------------------

    def _check_source(self, index: str, source: Mapping[str, Any]) -> dict[str, Any] | None:
        mapping = self.mappings[index]
        props = mapping.get("properties", {})
        if mapping.get("dynamic") == "strict":
            unknown = sorted(set(source) - set(props))
            if unknown:
                return {
                    "type": "strict_dynamic_mapping_exception",
                    "reason": f"mapping set to strict, dynamic introduction of [{unknown[0]}]",
                }
        for name, spec in props.items():
            if (
                spec.get("type") == "dense_vector"
                and name in source
                and len(source[name]) != spec["dims"]
            ):
                return {
                    "type": "document_parsing_exception",
                    "reason": f"[{name}] has {len(source[name])} dims, expected {spec['dims']}",
                }
        return None

    def bulk(self, *, operations: Sequence[Mapping[str, Any]], **_: Any) -> dict[str, Any]:
        items: list[dict[str, Any]] = []
        errors = False
        pairs = list(zip(operations[0::2], operations[1::2], strict=True))
        self.bulk_calls.append(len(pairs))
        for action, source in pairs:
            ((op, meta),) = action.items()
            if op != "index":
                raise NotImplementedError(f"FakeElasticsearch bulk supports index only, got {op}")
            index, doc_id = meta["_index"], meta["_id"]
            self.require(index)
            error = self._check_source(index, source)
            if doc_id in self.fail_ids:
                error = {"type": "injected_failure", "reason": f"injected failure for {doc_id}"}
            if error is not None:
                errors = True
                items.append(
                    {"index": {"_index": index, "_id": doc_id, "status": 400, "error": error}}
                )
                continue
            created = doc_id not in self.docs[index]
            self.docs[index][doc_id] = copy.deepcopy(dict(source))
            items.append(
                {
                    "index": {
                        "_index": index,
                        "_id": doc_id,
                        "status": 201 if created else 200,
                        "result": "created" if created else "updated",
                    }
                }
            )
        return {"took": 1, "errors": errors, "items": items}

    def delete_by_query(
        self, *, index: str, query: Mapping[str, Any], **kwargs: Any
    ) -> dict[str, Any]:
        self.require(index)
        self.delete_queries.append({"query": copy.deepcopy(dict(query)), **kwargs})
        if self.delete_failures:
            return {"deleted": 0, "failures": list(self.delete_failures)}
        doomed = [i for i, s in self.docs[index].items() if matches(s, i, query)]
        for doc_id in doomed:
            del self.docs[index][doc_id]
        if kwargs.get("refresh"):
            self.refreshes.append(index)
        return {"deleted": len(doomed), "failures": []}

    # ---- reads ------------------------------------------------------------------------------

    def _hits(
        self, index: str, query: Mapping[str, Any] | None
    ) -> list[tuple[str, dict[str, Any]]]:
        self.require(index)
        return [(i, s) for i, s in self.docs[index].items() if matches(s, i, query)]

    def count(self, *, index: str, query: Mapping[str, Any] | None = None) -> dict[str, Any]:
        return {"count": len(self._hits(index, query))}

    def search(
        self,
        *,
        index: str,
        query: Mapping[str, Any] | None = None,
        knn: Mapping[str, Any] | None = None,
        size: int = 10,
        **_: Any,
    ) -> dict[str, Any]:
        if knn is not None:
            hits = self._hits(index, knn.get("filter"))
            scored = [
                (i, s, (1.0 + _cosine(knn["query_vector"], s[knn["field"]])) / 2.0)
                for i, s in hits
                if knn["field"] in s
            ]
            scored.sort(key=lambda t: -t[2])
            scored = scored[: knn["k"]]
        else:
            scored = [(i, s, 1.0) for i, s in self._hits(index, query)]
        return {
            "hits": {
                "total": {"value": len(scored)},
                "hits": [
                    {"_index": index, "_id": i, "_score": score, "_source": copy.deepcopy(s)}
                    for i, s, score in scored[:size]
                ],
            }
        }


class FakeRedis:
    """Enough of redis-py for `rag.index.generation`: `register_script` of its Lua script.

    The generation script's semantics: a missing counter starts at `start`, then `get`/`incr`.
    """

    def __init__(self, *, start: int = 1_000) -> None:
        self.values: dict[str, int] = {}
        self.start = start
        self.fail = False

    def register_script(self, _source: str) -> Callable[..., int]:
        def run(*, keys: Iterable[str], args: Iterable[str]) -> int:
            if self.fail:
                from redis.exceptions import ConnectionError as RedisConnectionError

                raise RedisConnectionError("injected Redis failure")
            (key,) = keys
            (op,) = args
            value = self.values.setdefault(key, self.start)
            if op == "incr":
                value += 1
                self.values[key] = value
            return value

        return run
