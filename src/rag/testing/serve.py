"""Stand-ins for the retrieval chain in search-service tests (spec P0-09).

`FakeSearchPipeline` answers like `SearchPipeline` (scripted `RetrievalResult` and
`ContextResult`) and records every call with the ACL groups it got; `FakePipelineBuilder` is
the service's `PipelineBuilder` over such fakes, with an optional gate that holds the "model
load" until a test releases it.
"""

from __future__ import annotations

import threading
import uuid
from collections.abc import Mapping, Sequence
from typing import Any

from rag.config import KnowledgeBaseConfig
from rag.types import Chunk, ContextResult, RetrievalResult, RouteHit, ThresholdDecision


def fake_chunk(kb: str, n: int, *, index: int = 0, rerank_score: float | None = None) -> Chunk:
    return Chunk(
        chunk_id=f"{kb}:p{n}.md#{index}",
        doc_id=f"{kb}:p{n}.md",
        kb=kb,
        text=f"body of page {n} part {index}",
        title_path=[f"Page {n}"],
        source_rel=f"p{n}.md",
        source_url=f"https://example.org/p{n}.md",
        source_version="fake@1",
        score=1.0 / (60 + n),
        rerank_score=rerank_score,
        metadata={"doc_type": "md", "language": "en", "chunk_index": index},
    )


def retrieval_result(
    kb: str,
    *,
    n: int = 3,
    request_id: str = "",
    timings_ms: Mapping[str, float] | None = None,
    threshold: ThresholdDecision | None = None,
    not_found: bool = False,
) -> RetrievalResult:
    """A plausible pipeline result: `n` chunks reranked best first, both routes, a threshold
    decision and step timings."""
    chunks = [fake_chunk(kb, i, rerank_score=round(0.95 - i / 10, 2)) for i in range(1, n + 1)]
    route = [
        RouteHit(chunk_id=c.chunk_id, doc_id=c.doc_id, rank=r, score=10.0 - r)
        for r, c in enumerate(chunks, 1)
    ]
    return RetrievalResult(
        request_id=request_id or uuid.uuid4().hex,
        kb=kb,
        index_version=f"kb_{kb}_v1",
        data_generation=1,
        timings_ms=dict(timings_ms or {"search": 1.0, "rerank": 2.0, "total": 3.0}),
        chunks=[] if not_found else chunks,
        below_threshold=chunks[:3] if not_found else [],
        not_found=not_found,
        routes={"bm25": route, "vector": list(reversed(route))},
        rerank=[
            RouteHit(chunk_id=c.chunk_id, doc_id=c.doc_id, rank=r, score=c.rerank_score)
            for r, c in enumerate(chunks, 1)
        ],
        threshold=threshold
        or ThresholdDecision(
            threshold=0.3, top_score=chunks[0].rerank_score if chunks else None, kept=n
        ),
        config_version="cfg-fake",
    )


class FakeSearchPipeline:
    """Scripted `SearchPipeline`. `raises` / `context_raises` make the next calls fail."""

    def __init__(
        self,
        kb: str,
        *,
        result: RetrievalResult | None = None,
        config_version: str = "cfg-fake",
        raises: BaseException | None = None,
        context_raises: BaseException | None = None,
    ) -> None:
        self.kb = kb
        self.result = result
        self.config_version = config_version
        self.raises = raises
        self.context_raises = context_raises
        self.calls: list[dict[str, Any]] = []
        self.context_calls: list[dict[str, Any]] = []
        self.closed = False

    def retrieve(
        self,
        question: str,
        kb: str,
        top_k: int,
        *,
        filters: Mapping[str, Any] | None = None,
        deadline_ms: int | None = None,
        request_id: str | None = None,
        acl_groups: Sequence[str] | None = None,
        rerank_reserve_ms: int = 0,
    ) -> RetrievalResult:
        self.calls.append(
            {
                "question": question,
                "kb": kb,
                "top_k": top_k,
                "filters": dict(filters or {}),
                "deadline_ms": deadline_ms,
                "request_id": request_id,
                "acl_groups": tuple(acl_groups) if acl_groups is not None else None,
                "rerank_reserve_ms": rerank_reserve_ms,
            }
        )
        if self.raises is not None:
            raise self.raises
        result = self.result or retrieval_result(self.kb)
        return result.model_copy(
            update={
                "request_id": request_id or uuid.uuid4().hex,
                "kb": kb,
                "chunks": result.chunks[:top_k],
                "config_version": self.config_version,
            }
        )

    def get_context(
        self,
        chunk_id: str,
        mode: str | None = None,
        *,
        deadline_ms: int | None = None,
        acl_groups: Sequence[str] | None = None,
    ) -> ContextResult:
        self.context_calls.append(
            {
                "chunk_id": chunk_id,
                "mode": mode,
                "deadline_ms": deadline_ms,
                "acl_groups": tuple(acl_groups) if acl_groups is not None else None,
            }
        )
        if self.context_raises is not None:
            raise self.context_raises
        doc_id = chunk_id.rpartition("#")[0]
        chunk = fake_chunk(self.kb, 1).model_copy(update={"chunk_id": chunk_id, "doc_id": doc_id})
        return ContextResult(chunk_id=chunk_id, doc_id=doc_id, mode=mode or "chunk", chunks=[chunk])

    def close(self) -> None:
        self.closed = True


class FakePipelineBuilder:
    """`PipelineBuilder` over fakes. `prepare` (the model load) waits for `prepare_gate` when
    given and raises `prepare_error`; `build` raises `failures[kb]` for broken KBs."""

    def __init__(
        self,
        pipelines: Mapping[str, FakeSearchPipeline],
        *,
        failures: Mapping[str, BaseException] | None = None,
        prepare_gate: threading.Event | None = None,
        prepare_error: BaseException | None = None,
    ) -> None:
        self.pipelines = dict(pipelines)
        self.failures = dict(failures or {})
        self.prepare_gate = prepare_gate
        self.prepare_error = prepare_error
        self.prepared = 0
        self.closed = False

    def prepare(self) -> None:
        if self.prepare_gate is not None:
            self.prepare_gate.wait(timeout=30.0)
        if self.prepare_error is not None:
            raise self.prepare_error
        self.prepared += 1

    def build(self, cfg: KnowledgeBaseConfig) -> FakeSearchPipeline:
        if cfg.kb in self.failures:
            raise self.failures[cfg.kb]
        return self.pipelines[cfg.kb]

    def close(self) -> None:
        self.closed = True
