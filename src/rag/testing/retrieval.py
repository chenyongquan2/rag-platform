"""Scripted stand-ins for retrieval tests.

- `FakeRetriever`: a scripted `rag.eval.retrieval.Retriever` for eval and tool tests (P0-10).
- `FakeMsearchClient`: a scripted Elasticsearch `_msearch` for `EsHybridRetriever` (P0-07), with
  `es_hit`, `search_response` and `search_error` to build the per-search answers.
"""

from __future__ import annotations

import copy
import time
from collections.abc import Mapping, Sequence
from typing import Any

from rag.types import RetrievalResult


class FakeRetriever:
    """Returns the result scripted for each question text; an exception is raised instead.

    The final list is cut to `top_k` like a real retriever's; route lists are returned as
    scripted. Every call is recorded in `calls` as `(question, kb, top_k)`.
    """

    def __init__(self, results: Mapping[str, RetrievalResult | Exception]) -> None:
        self.results = dict(results)
        self.calls: list[tuple[str, str, int]] = []

    def retrieve(self, question: str, kb: str, top_k: int) -> RetrievalResult:
        self.calls.append((question, kb, top_k))
        if question not in self.results:
            raise KeyError(f"FakeRetriever has no result for question {question!r}")
        result = self.results[question]
        if isinstance(result, Exception):
            raise result
        return result.model_copy(update={"chunks": result.chunks[:top_k]})


def es_hit(
    chunk_id: str, score: float | None, *, index: str = "kb_hf_docs_v1", **source: Any
) -> dict[str, Any]:
    """One search hit; `_source` gets `chunk_id`, `doc_id` (the part before `#`), `kb`
    (`hf_docs`) and `content` unless given."""
    body: dict[str, Any] = {
        "chunk_id": chunk_id,
        "doc_id": chunk_id.split("#")[0],
        "kb": "hf_docs",
        "content": f"text of {chunk_id}",
        **source,
    }
    return {"_index": index, "_id": chunk_id, "_score": score, "_source": body}


def search_response(
    hits: Sequence[Mapping[str, Any]], *, timed_out: bool = False
) -> dict[str, Any]:
    """One successful search answer inside an `_msearch` response."""
    return {
        "took": 3,
        "timed_out": timed_out,
        "_shards": {"total": 1, "successful": 1, "skipped": 0, "failed": 0},
        "hits": {"max_score": None, "hits": [copy.deepcopy(dict(h)) for h in hits]},
        "status": 200,
    }


def search_error(status: int, error_type: str, reason: str) -> dict[str, Any]:
    """One failed search inside an `_msearch` response (the request itself got HTTP 200)."""
    return {
        "error": {
            "root_cause": [{"type": error_type, "reason": reason}],
            "type": error_type,
            "reason": reason,
        },
        "status": status,
    }


class _Meta:
    def __init__(self, status: int) -> None:
        self.status = status


class _AliasResponse:
    def __init__(self, body: dict[str, Any], status: int) -> None:
        self.body = body
        self.meta = _Meta(status)


class _FakeIndices:
    def __init__(self, client: FakeMsearchClient) -> None:
        self._client = client

    def get_alias(self, *, name: str, **_: Any) -> _AliasResponse:
        """Like `es.options(ignore_status=404).indices.get_alias(name=...)`."""
        self._client.alias_calls.append(name)
        index = self._client.aliases.get(name)
        if index is None:
            return _AliasResponse({"error": "alias missing", "status": 404}, 404)
        return _AliasResponse({index: {"aliases": {name: {}}}}, 200)


class _Options:
    def __init__(self, client: FakeMsearchClient, options: dict[str, Any]) -> None:
        self._client = client
        self._options = options
        self.indices = client.indices

    def msearch(self, **kwargs: Any) -> dict[str, Any]:
        return self._client._msearch(self._options, **kwargs)


class FakeMsearchClient:
    """Answers `es.options(...).msearch(index=..., searches=[header, body, ...])`.

    A body with `knn` is the `vector` route, any other the `bm25` route; each gets the answer
    scripted in `responses[route]` (default: no hits). `raises` makes the whole request fail
    instead; `delay_s` sleeps before answering. Every call is recorded in `calls` as
    `{"index", "searches", "options"}` (deep copies). `aliases` (alias -> index) answers
    `indices.get_alias`; each lookup is recorded in `alias_calls`. With `search_ms`, each search
    is taken to need that long inside Elasticsearch: a search whose `timeout` is shorter answers
    `timed_out: true` without hits, as Elasticsearch does.
    """

    def __init__(
        self,
        *,
        bm25: Mapping[str, Any] | None = None,
        vector: Mapping[str, Any] | None = None,
        raises: BaseException | None = None,
        delay_s: float = 0.0,
        aliases: Mapping[str, str] | None = None,
        search_ms: float = 0.0,
    ) -> None:
        self.responses: dict[str, Mapping[str, Any]] = {}
        if bm25 is not None:
            self.responses["bm25"] = bm25
        if vector is not None:
            self.responses["vector"] = vector
        self.raises = raises
        self.delay_s = delay_s
        self.calls: list[dict[str, Any]] = []
        self.search_ms = search_ms
        self.aliases = dict(aliases or {})
        self.alias_calls: list[str] = []
        self.indices = _FakeIndices(self)

    def options(self, **options: Any) -> _Options:
        return _Options(self, dict(options))

    def msearch(self, **kwargs: Any) -> dict[str, Any]:
        return self._msearch({}, **kwargs)

    def _msearch(
        self,
        options: dict[str, Any],
        *,
        searches: Sequence[Mapping[str, Any]],
        index: str | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        self.calls.append(
            {"index": index, "searches": copy.deepcopy(list(searches)), "options": options}
        )
        if self.delay_s:
            time.sleep(self.delay_s)
        if self.raises is not None:
            raise self.raises
        answers = []
        for body in searches[1::2]:
            route = "vector" if "knn" in body else "bm25"
            timeout = str(body.get("timeout", ""))
            if self.search_ms and timeout.endswith("ms") and float(timeout[:-2]) < self.search_ms:
                answers.append(search_response([], timed_out=True))
                continue
            answers.append(copy.deepcopy(dict(self.responses.get(route, search_response([])))))
        return {"took": 5, "responses": answers}
