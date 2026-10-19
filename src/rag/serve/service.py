"""The search service behind the MCP tools and the HTTP API (design s16-1, s16-4, s16-8, s15-1).

    service = SearchService(settings)      # reads the KB configs
    service.start()                        # loads the reranker and the chains in a thread
    service.search_docs(credentials, kb="hf_docs", query="...")   # -> SearchResponse

One `SearchPipeline` per knowledge base, all sharing the process's reranker, one
Elasticsearch client, one query embedding model and one Redis client (`DefaultPipelineBuilder`).
Loading the reranker takes a minute or two, so it runs in a background thread at start:
`/healthz` answers at once, `/readyz` and `ready` turn true when the load ends (its time is
logged), and a search that arrives earlier waits for the load within its own deadline, then
answers `not_ready`. A KB whose chain cannot be built (no index behind its alias, another
embedding model) is logged and answers `retrieval_failed`; the others work.

Every call:
1. resolves the caller (`AclResolver`; P0: the configured local user) and checks that it may
   use the knowledge base - an unknown and a hidden KB both answer `kb_not_found`;
2. validates the arguments (`invalid_request`) and the caller's filters against the whitelist
   (`invalid_filter`) before anything runs;
3. runs inside a `request_span` (request id, caller, kb, config version; P0-11) with the step
   under a `layer_span`; the retrieval layer reads with the caller's ACL groups;
4. writes one business record line (also for failures: category in `flags`), records step
   durations from the result's timings and the quality signals (not found, top rerank score);
5. returns a public response model, or raises `ServiceError` with only the category and the
   request id; the exception and its text go to the log with the request id.

The service is synchronous (Elasticsearch and torch calls block); the MCP and HTTP layers run
each call in a worker thread.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Protocol

from rag.agent import AgentRuntime, valid_session_id
from rag.config import KnowledgeBaseConfig, ServiceSettings, load_kb_configs, repo_path
from rag.observe import (
    BusinessRecord,
    RankedId,
    business_record_from_result,
    current_trace_id,
    layer_span,
    new_request_id,
    request_span,
    write_business_record,
)
from rag.observe import attributes as attr
from rag.observe.spans import RequestScope
from rag.retrieve.context import CONTEXT_MODES
from rag.retrieve.deadline import Deadline
from rag.retrieve.filters import FilterError, build_es_filter
from rag.retrieve.observe import observe_retrieval
from rag.security.access import (
    AclResolver,
    Caller,
    ConfigAclResolver,
    Credentials,
    kb_visible,
    visible_kbs,
)
from rag.serve.errors import ServiceError, error_category
from rag.serve.schema import (
    AskResponse,
    ContextResponse,
    KnowledgeBaseInfo,
    KnowledgeBaseList,
    SearchResponse,
)
from rag.types import ContextResult, ErrorCategory, RetrievalResult

logger = logging.getLogger(__name__)

# Longest chunk id accepted by get_context (ids are `<kb>:<path>#<n>`) and the most of a
# caller-chosen string kept in a business record.
MAX_CHUNK_ID_CHARS = 1000
RECORD_TEXT_CHARS = 64
RECORD_FILTERS_CHARS = 1000
# Errors caused by the request itself: logged without a stack trace.
_CALLER_ERRORS = {
    ErrorCategory.INVALID_REQUEST,
    ErrorCategory.INVALID_FILTER,
    ErrorCategory.KB_NOT_FOUND,
}


class SearchBackend(Protocol):
    """What the service needs of a knowledge base's chain (`SearchPipeline`)."""

    config_version: str

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
    ) -> RetrievalResult: ...

    def get_context(
        self,
        chunk_id: str,
        mode: Any = None,
        *,
        deadline_ms: int | None = None,
        acl_groups: Sequence[str] | None = None,
    ) -> ContextResult: ...

    def close(self) -> None: ...


class PipelineBuilder(Protocol):
    """Builds the chains: `prepare` loads what every chain shares (the reranker; slow), then
    `build` is called once per knowledge base."""

    def prepare(self) -> None: ...

    def build(self, cfg: KnowledgeBaseConfig) -> SearchBackend: ...

    def close(self) -> None: ...


class DefaultPipelineBuilder:
    """`SearchPipeline`s over the real services: the process's reranker (`get_rerank_runner`,
    loaded and warmed up once), one ES client without retries, the query embedding profile
    and a Redis client for the data generation (s16-7). The chains are built with the
    configured caller's groups, but every request passes its own caller's groups."""

    def __init__(self, settings: ServiceSettings) -> None:
        self.settings = settings
        self._runner: Any = None
        self._es: Any = None
        self._embed: Any = None
        self._redis: Any = None

    def prepare(self) -> None:
        from rag.embed import make_embedding
        from rag.factories import make_es_client, make_redis_client
        from rag.retrieve.rerank import get_rerank_runner

        s = self.settings
        self._runner = get_rerank_runner(s)
        self._es = make_es_client(
            s.es_url, request_timeout=s.search_max_deadline_ms / 1000.0, max_retries=0
        )
        self._embed = make_embedding("query", s)
        self._redis = make_redis_client(s.redis_url, socket_timeout=s.embed_cache_redis_timeout_s)

    def build(self, cfg: KnowledgeBaseConfig) -> SearchBackend:
        from rag.retrieve.pipeline import make_search_pipeline

        if self._runner is None:
            raise RuntimeError("prepare() must run before build()")
        return make_search_pipeline(
            cfg,
            self.settings,
            acl_groups=self.settings.service_acl_groups,
            index_prefix=self.settings.search_index_prefix,
            es=self._es,
            embed_model=self._embed,
            redis=self._redis,
            runner=self._runner,
        )

    def close(self) -> None:
        for client in (self._es, self._redis):
            if client is not None:
                try:
                    client.close()
                except Exception:
                    logger.warning("closing a client failed", exc_info=True)


@dataclass
class _Call:
    """One request being served: what its span, log lines and business record need."""

    tool: str
    request_id: str
    caller: Caller
    kb: str
    config_version: str
    extra: dict[str, Any]
    began: float = field(default_factory=time.perf_counter)
    scope: RequestScope | None = None
    record: BusinessRecord | None = None
    """The success path's record; failures get one built from the fields above."""

    def elapsed_ms(self) -> float:
        return (time.perf_counter() - self.began) * 1000.0


def _clip(value: Any, limit: int) -> str:
    return str(value)[:limit]


def _jsonable(value: Any, limit: int) -> Any:
    """`value` as plain JSON for a business record, or a clipped string when it is not JSON or
    too long (a record must never fail to serialize)."""
    try:
        text = json.dumps(value, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return _clip(value, limit)
    return json.loads(text) if len(text) <= limit else text[:limit]


def _kb_of_chunk_id(chunk_id: Any) -> str | None:
    """`hf_docs` of `hf_docs:path/page.md#3`; None when the id does not look like one."""
    if not isinstance(chunk_id, str) or not chunk_id or len(chunk_id) > MAX_CHUNK_ID_CHARS:
        return None
    doc_id, sep, unit = chunk_id.rpartition("#")
    kb, colon, path = doc_id.partition(":")
    if not sep or not unit or not colon or not kb or not path:
        return None
    return kb


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


class SearchService:
    """See the module doc. `configs` default to every KB config under `settings.kb_dir`;
    `resolver` to the configured caller; `builder` to the real chains."""

    def __init__(
        self,
        settings: ServiceSettings,
        *,
        configs: Mapping[str, KnowledgeBaseConfig] | None = None,
        resolver: AclResolver | None = None,
        builder: PipelineBuilder | None = None,
        agent: AgentRuntime | None = None,
        answer_model: Any = None,
    ) -> None:
        self.settings = settings
        self.configs: dict[str, KnowledgeBaseConfig] = (
            dict(configs) if configs is not None else load_kb_configs(repo_path(settings.kb_dir))
        )
        self.resolver: AclResolver = resolver or ConfigAclResolver.from_settings(settings)
        self._builder: PipelineBuilder = builder or DefaultPipelineBuilder(settings)
        self._pipelines: dict[str, SearchBackend] = {}
        self._unavailable: set[str] = set()
        self._state = "starting"
        self._lock = threading.Lock()
        self._done = threading.Event()
        self._thread: threading.Thread | None = None
        self._closed = False
        self.load_s: float | None = None
        self._agent = agent
        self._answer_model = answer_model  # None: the runtime's `online` chat model

    # ---- lifecycle --------------------------------------------------------------------------

    @property
    def agent(self) -> AgentRuntime:
        """The ask_docs runtime (session locks, history, checkpoints), made on first use from the
        connection strings in the settings."""
        if self._agent is None:
            self._agent = AgentRuntime.from_settings(self.settings)
        return self._agent

    @property
    def ready(self) -> bool:
        return self._state == "ready"

    def status(self) -> dict[str, Any]:
        """`status`: starting, loading, ready, failed or closed; per KB: loading, ready,
        unavailable (or failed). For logs and tests: `/readyz` shows only counts."""
        state = self._state
        kbs: dict[str, str] = {}
        for kb in sorted(self.configs):
            if kb in self._unavailable:
                kbs[kb] = "unavailable"
            elif kb in self._pipelines:
                kbs[kb] = "ready"
            else:
                kbs[kb] = "loading" if state in ("starting", "loading") else "failed"
        return {"status": state, "load_s": self.load_s, "knowledge_bases": kbs}

    def start(self) -> None:
        """Load in a background thread (once per service; later calls do nothing)."""
        with self._lock:
            if self._state != "starting":
                return
            self._state = "loading"
            self._thread = threading.Thread(
                target=self._load, name="search-service-load", daemon=True
            )
            self._thread.start()

    def load(self) -> None:
        """Load in this thread (once per service, like `start`)."""
        with self._lock:
            if self._state != "starting":
                return
            self._state = "loading"
        self._load()

    def _load(self) -> None:
        """Load the reranker, then build each knowledge base's chain. When `close` came during
        the load, stop before the next step and close what was built (this thread owns the
        clients until it ends)."""
        began = time.perf_counter()
        logger.info(
            "search service loading: knowledge bases %s, settings %s",
            sorted(self.configs),
            self.settings.redacted(),
        )
        try:
            self._builder.prepare()
        except Exception:
            logger.exception("search service could not load its models; answering not_ready")
            self._finish("failed")
            return
        for kb, cfg in sorted(self.configs.items()):
            if self._closed:
                break
            try:
                pipeline = self._builder.build(cfg)
            except Exception:
                logger.exception("knowledge base %s is unavailable", kb)
                self._unavailable.add(kb)
                continue
            with self._lock:
                self._pipelines[kb] = pipeline
        self.load_s = time.perf_counter() - began
        if self._finish("ready"):
            logger.info(
                "search service ready in %.1f s (reranker and %d knowledge bases; unavailable: %s)",
                self.load_s,
                len(self._pipelines),
                sorted(self._unavailable) or "none",
            )

    def _finish(self, state: str) -> bool:
        """End the load in `state`, or in `closed` (closing everything) when `close` came
        meanwhile; True when the service took `state`."""
        with self._lock:
            closed = self._closed
            self._state = "closed" if closed else state
        if closed:
            logger.info("search service closed during its load")
            self._close_all()
        self._done.set()
        return not closed

    def wait_ready(self, timeout_s: float | None = None) -> bool:
        self._done.wait(timeout_s)
        return self.ready

    def pipeline(self, kb: str) -> SearchBackend | None:
        return self._pipelines.get(kb)

    def close(self) -> None:
        """Close the chains and clients (once). During a load this returns at once and the
        load thread closes everything when its current step ends."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            loading = self._state == "loading"
        if not loading:
            self._state = "closed"
            self._close_all()

    def _close_all(self) -> None:
        with self._lock:
            pipelines = list(self._pipelines.values())
            self._pipelines.clear()
        for pipeline in pipelines:
            try:
                pipeline.close()
            except Exception:
                logger.warning("closing a search pipeline failed", exc_info=True)
        try:
            self._builder.close()
        except Exception:
            logger.warning("closing the pipeline builder failed", exc_info=True)

    def reject(
        self, tool: str, credentials: Credentials, *, request_id: str, category: str, detail: str
    ) -> None:
        """Answer a request the transport rejected before calling a tool (a body that is not
        JSON or does not fit): same span, metric, log line and record as any failed call;
        raises `ServiceError`."""
        caller, _ = self._resolve(credentials, "", request_id)
        with self._call(
            tool,
            request_id=request_id,
            caller=caller,
            cfg=None,
            config_version="",
            extra={"rejected": _clip(detail, RECORD_FILTERS_CHARS)},
        ):
            raise ServiceError(category, request_id, detail=detail)

    # ---- tools ------------------------------------------------------------------------------

    def search_docs(
        self,
        credentials: Credentials,
        *,
        kb: str,
        query: str,
        filters: Mapping[str, Any] | None = None,
        top_k: int | None = None,
        deadline_ms: int | None = None,
        request_id: str | None = None,
    ) -> SearchResponse:
        """Search one knowledge base (s16-1); raises `ServiceError`."""
        request_id = request_id or new_request_id()
        caller, cfg = self._resolve(credentials, kb, request_id)
        pipeline = self._pipelines.get(cfg.kb) if cfg is not None else None
        extra: dict[str, Any] = {
            "query": _clip(query, self.settings.search_max_query_chars),
            "filters": _jsonable(filters, RECORD_FILTERS_CHARS) if filters else None,
            "top_k": top_k if _is_int(top_k) else None,
            "deadline_ms": deadline_ms if _is_int(deadline_ms) else None,
        }
        if cfg is None:
            extra["requested_kb"] = _clip(kb, RECORD_TEXT_CHARS)
        with self._call(
            "search_docs",
            request_id=request_id,
            caller=caller,
            cfg=cfg,
            config_version=self._config_version(cfg, pipeline),
            extra=extra,
        ) as call:
            if cfg is None:
                raise ServiceError(ErrorCategory.KB_NOT_FOUND, request_id, detail=f"kb {kb!r}")
            retrieval = cfg.retrieval
            top_k = retrieval.final_k if top_k is None else top_k
            deadline_ms = self.settings.search_deadline_ms if deadline_ms is None else deadline_ms
            self._check_search_args(query, top_k, deadline_ms, retrieval.rerank_n, request_id)
            deadline = Deadline.after_ms(deadline_ms)
            if filters is not None and not isinstance(filters, Mapping):
                raise FilterError("filters must be an object of filter keys")
            build_es_filter(kb=cfg.kb, acl_groups=caller.acl_groups, filters=filters)
            pipeline = self._ready_pipeline(cfg.kb, deadline, request_id)
            remaining = int(deadline.remaining_ms())
            if remaining <= 0:
                raise ServiceError(ErrorCategory.RETRIEVAL_TIMEOUT, request_id, detail="no time")
            with layer_span(
                "retrieve",
                attributes={
                    attr.TOP_K: top_k,
                    attr.RERANK_N: retrieval.rerank_n,
                    attr.FINAL_K: retrieval.final_k,
                    attr.THRESHOLD: retrieval.threshold,
                },
            ) as span:
                result = pipeline.retrieve(
                    query,
                    cfg.kb,
                    top_k,
                    filters=dict(filters or {}),
                    deadline_ms=remaining,
                    request_id=request_id,
                    acl_groups=caller.acl_groups,
                )
                span.set_attribute(attr.HITS, len(result.chunks))
            assert call.scope is not None
            call.scope.set_result(result)
            self._observe_result(result, cfg.kb)
            response = SearchResponse.from_result(result, timings_ms={"request": call.elapsed_ms()})
            call.record = business_record_from_result(
                result.model_copy(update={"timings_ms": response.timings_ms}),
                tool=call.tool,
                config_version=result.config_version,
                caller=caller.caller_id,
                rerank=[RankedId(chunk_id=h.chunk_id, score=h.score) for h in result.rerank],
                threshold=result.threshold,
                extra={**extra, "top_k": top_k, "not_found": result.not_found},
            )
            return response

    async def ask_docs(
        self,
        credentials: Credentials,
        *,
        kb: str,
        question: str,
        session_id: str | None = None,
        deadline_ms: int | None = None,
        request_id: str | None = None,
    ) -> AskResponse:
        """Answer a question from one knowledge base with the ask_docs graph (s16-1, s13-3);
        raises `ServiceError`. Runs on the caller's event loop: the graph is async and the
        session locks belong to it (the blocking searches run in worker threads)."""
        request_id = request_id or new_request_id()
        caller, cfg = self._resolve(credentials, kb, request_id)
        pipeline = self._pipelines.get(cfg.kb) if cfg is not None else None
        extra: dict[str, Any] = {
            "question": _clip(question, self.settings.search_max_query_chars),
            "session_id": _clip(session_id, RECORD_TEXT_CHARS) if session_id is not None else None,
            "deadline_ms": deadline_ms if _is_int(deadline_ms) else None,
        }
        if cfg is None:
            extra["requested_kb"] = _clip(kb, RECORD_TEXT_CHARS)
        with self._call(
            "ask_docs",
            request_id=request_id,
            caller=caller,
            cfg=cfg,
            config_version=self._config_version(cfg, pipeline),
            extra=extra,
        ) as call:
            if cfg is None:
                raise ServiceError(ErrorCategory.KB_NOT_FOUND, request_id, detail=f"kb {kb!r}")
            deadline_ms = self.settings.agent_deadline_ms if deadline_ms is None else deadline_ms
            self._check_ask_args(question, session_id, deadline_ms, request_id)
            deadline = Deadline.after_ms(deadline_ms)
            pipeline = await asyncio.to_thread(self._ready_pipeline, cfg.kb, deadline, request_id)
            result = await self.agent.ask(
                question,
                cfg=cfg,
                retriever=pipeline,
                session_id=session_id,
                request_id=request_id,
                caller_id=caller.caller_id,
                acl_groups=caller.acl_groups,
                # What is left of the call's time after waiting for the load.
                deadline_ms=max(1, int(deadline.remaining_ms())),
                model=self._answer_model,
            )
            assert call.scope is not None
            call.scope.set_result(result)
            response = AskResponse.from_result(result, timings_ms={"request": call.elapsed_ms()})
            cited = {c.chunk_id for s in result.sentences for c in s.citations}
            record = self._record(
                call,
                final=[
                    RankedId(chunk_id=c.chunk_id) for c in result.sources if c.chunk_id in cited
                ],
                flags=[str(c) for c in result.error_categories],
                timings_ms=response.timings_ms,
                extra={
                    **extra,
                    "deadline_ms": deadline_ms,
                    "thread_id": result.thread_id,
                    "status": str(result.status),
                    "found": result.found,
                    "rounds": result.rounds,
                    "sources": len(result.sources),
                    "tokens_used": result.tokens_used,
                    "used_history": result.used_history,
                    "cache_allowed": result.cache_allowed,
                    "coverage_unconfirmed": result.coverage_unconfirmed,
                    "stop_reason": result.stop_reason,
                },
            )
            call.record = record.model_copy(
                update={
                    "index_version": result.index_version,
                    "data_generation": result.data_generation,
                }
            )
            return response

    def get_context(
        self,
        credentials: Credentials,
        *,
        chunk_id: str,
        mode: str | None = None,
        request_id: str | None = None,
    ) -> ContextResponse:
        """A chunk and its context (s7-6), from the knowledge base named in the chunk id."""
        request_id = request_id or new_request_id()
        kb = _kb_of_chunk_id(chunk_id)
        caller, cfg = self._resolve(credentials, kb or "", request_id)
        pipeline = self._pipelines.get(cfg.kb) if cfg is not None else None
        extra: dict[str, Any] = {
            "chunk_id": _clip(chunk_id, MAX_CHUNK_ID_CHARS),
            "mode": _clip(mode, RECORD_TEXT_CHARS) if mode is not None else None,
        }
        with self._call(
            "get_context",
            request_id=request_id,
            caller=caller,
            cfg=cfg,
            config_version=self._config_version(cfg, pipeline),
            extra=extra,
        ) as call:
            if kb is None:
                raise ServiceError(
                    ErrorCategory.INVALID_REQUEST, request_id, detail="malformed chunk id"
                )
            if cfg is None:
                raise ServiceError(ErrorCategory.KB_NOT_FOUND, request_id, detail=f"kb {kb!r}")
            if mode is not None and mode not in CONTEXT_MODES:
                raise ServiceError(
                    ErrorCategory.INVALID_REQUEST, request_id, detail=f"mode {mode!r}"
                )
            deadline = Deadline.after_ms(self.settings.search_deadline_ms)
            pipeline = self._ready_pipeline(cfg.kb, deadline, request_id)
            with layer_span("context") as span:
                result = pipeline.get_context(
                    chunk_id,
                    mode,  # None: the KB's expansion mode (`chunk` without expansion)
                    deadline_ms=max(1, int(deadline.remaining_ms())),
                    acl_groups=caller.acl_groups,
                )
                span.set_attribute(attr.HITS, len(result.chunks))
            response = ContextResponse.from_result(
                result, request_id=request_id, kb=cfg.kb, timings_ms={"request": call.elapsed_ms()}
            )
            call.record = self._record(
                call,
                final=[RankedId(chunk_id=c.chunk_id) for c in result.chunks],
                timings_ms=response.timings_ms,
                extra={
                    **extra,
                    "mode": result.mode,
                    "found": result.found,
                    "updated": result.updated,
                    "truncated": result.truncated,
                },
            )
            return response

    def list_knowledge_bases(
        self, credentials: Credentials, *, request_id: str | None = None
    ) -> KnowledgeBaseList:
        """The knowledge bases this caller may use (config only; works while loading)."""
        request_id = request_id or new_request_id()
        caller, _ = self._resolve(credentials, "", request_id)
        with self._call(
            "list_knowledge_bases",
            request_id=request_id,
            caller=caller,
            cfg=None,
            config_version="",
            extra={},
        ) as call:
            infos = [
                KnowledgeBaseInfo(
                    kb=cfg.kb,
                    description=cfg.description,
                    language=cfg.language,
                    available=cfg.kb not in self._unavailable,
                    default_top_k=cfg.retrieval.final_k,
                    max_top_k=cfg.retrieval.rerank_n,
                )
                for cfg in visible_kbs(caller, self.configs)
            ]
            call.record = self._record(
                call,
                timings_ms={"request": call.elapsed_ms()},
                extra={"knowledge_bases": [info.kb for info in infos]},
            )
            return KnowledgeBaseList(request_id=request_id, knowledge_bases=infos)

    def visible_configs(self, credentials: Credentials) -> list[KnowledgeBaseConfig]:
        """The KB configs these credentials may use (for tool descriptions)."""
        return visible_kbs(self.resolver.resolve(credentials), self.configs)

    # ---- helpers ----------------------------------------------------------------------------

    def _resolve(
        self, credentials: Credentials, kb: str, request_id: str
    ) -> tuple[Caller, KnowledgeBaseConfig | None]:
        """The caller and the KB config if the caller may use it (else None: unknown and
        hidden KBs look the same)."""
        try:
            caller = self.resolver.resolve(credentials)
        except Exception as exc:
            logger.warning("request %s: caller not resolved: %r", request_id, exc)
            raise ServiceError(ErrorCategory.INTERNAL_ERROR, request_id) from exc
        cfg = self.configs.get(kb) if isinstance(kb, str) else None
        if cfg is not None and not kb_visible(caller, cfg):
            cfg = None
        return caller, cfg

    def _config_version(
        self, cfg: KnowledgeBaseConfig | None, pipeline: SearchBackend | None
    ) -> str:
        """The chain's version; before the chain exists (load running, or the KB unavailable) a
        version of the KB config and reranker alone, marked `config:` because the index's
        embedding model is not known yet; empty for an unknown or hidden KB."""
        if pipeline is not None:
            return pipeline.config_version
        if cfg is None:
            return ""
        from rag.retrieve.pipeline import search_config_version

        return "config:" + search_config_version(cfg, self.settings, "")

    def _check_search_args(
        self, query: Any, top_k: Any, deadline_ms: Any, max_top_k: int, request_id: str
    ) -> None:
        problem = None
        if not isinstance(query, str) or not query.strip():
            problem = "query must be non-empty text"
        elif len(query) > self.settings.search_max_query_chars:
            problem = f"query longer than {self.settings.search_max_query_chars} characters"
        elif not _is_int(top_k) or not 1 <= top_k <= max_top_k:
            problem = f"top_k {top_k!r} outside 1..{max_top_k}"
        elif (
            not _is_int(deadline_ms) or not 1 <= deadline_ms <= self.settings.search_max_deadline_ms
        ):
            problem = (
                f"deadline_ms {deadline_ms!r} outside 1..{self.settings.search_max_deadline_ms}"
            )
        if problem is not None:
            raise ServiceError(ErrorCategory.INVALID_REQUEST, request_id, detail=problem)

    def _check_ask_args(
        self, question: Any, session_id: Any, deadline_ms: Any, request_id: str
    ) -> None:
        problem = None
        if not isinstance(question, str) or not question.strip():
            problem = "question must be non-empty text"
        elif len(question) > self.settings.search_max_query_chars:
            problem = f"question longer than {self.settings.search_max_query_chars} characters"
        elif session_id is not None and not valid_session_id(session_id):
            problem = "session_id must be 1 to 64 letters, digits, dot, dash or underscore"
        elif (
            not _is_int(deadline_ms) or not 1 <= deadline_ms <= self.settings.search_max_deadline_ms
        ):
            problem = (
                f"deadline_ms {deadline_ms!r} outside 1..{self.settings.search_max_deadline_ms}"
            )
        if problem is not None:
            raise ServiceError(ErrorCategory.INVALID_REQUEST, request_id, detail=problem)

    def _ready_pipeline(self, kb: str, deadline: Deadline, request_id: str) -> SearchBackend:
        """The KB's chain once the load has ended (waiting at most until the deadline)."""
        if not self._done.wait(deadline.remaining_s()) or self._state != "ready":
            raise ServiceError(ErrorCategory.NOT_READY, request_id, detail=f"service {self._state}")
        pipeline = self._pipelines.get(kb)
        if pipeline is None:
            raise ServiceError(
                ErrorCategory.RETRIEVAL_FAILED, request_id, detail=f"kb {kb} unavailable"
            )
        return pipeline

    @staticmethod
    def _observe_result(result: RetrievalResult, kb: str) -> None:
        observe_retrieval(result, kb, "search_docs")

    @staticmethod
    def _record(
        call: _Call,
        *,
        final: list[RankedId] | None = None,
        flags: list[str] | None = None,
        timings_ms: Mapping[str, float] | None = None,
        extra: Mapping[str, Any] | None = None,
    ) -> BusinessRecord:
        return BusinessRecord(
            request_id=call.request_id,
            trace_id=current_trace_id(),
            tool=call.tool,
            caller=call.caller.caller_id,
            kb=call.kb,
            config_version=call.config_version,
            final=list(final or []),
            flags=list(flags or []),
            timings_ms=dict(timings_ms or {}),
            extra=dict(extra if extra is not None else call.extra),
        )

    @staticmethod
    def _write(record: BusinessRecord) -> None:
        try:
            write_business_record(record)
        except Exception:  # a lost record must never fail the request
            logger.warning("business record %s not written", record.request_id, exc_info=True)

    @contextmanager
    def _call(
        self,
        tool: str,
        *,
        request_id: str,
        caller: Caller,
        cfg: KnowledgeBaseConfig | None,
        config_version: str,
        extra: dict[str, Any],
    ) -> Iterator[_Call]:
        """Request span, failure log line, business record; any exception leaves as
        `ServiceError` (category and request id only)."""
        call = _Call(
            tool=tool,
            request_id=request_id,
            caller=caller,
            kb=cfg.kb if cfg is not None else "",
            config_version=config_version,
            extra=extra,
        )
        try:
            with request_span(
                tool,
                # Only configured KBs the caller may use become span attributes and metric
                # labels: a caller-chosen string would add a metric series per value.
                kb=cfg.kb if cfg is not None else None,
                request_id=request_id,
                caller=caller.caller_id,
                config_version=config_version or None,
                classify=error_category,
            ) as scope:
                call.scope = scope
                try:
                    yield call
                except Exception as exc:
                    category = error_category(exc)
                    self._log_failure(call, category, exc)
                    self._write(
                        self._record(
                            call,
                            flags=[category],
                            timings_ms={"request": call.elapsed_ms()},
                            extra={**call.extra, "error": category},
                        )
                    )
                    raise
                if call.record is not None:
                    self._write(call.record)
        except ServiceError:
            raise
        except Exception as exc:
            raise ServiceError(error_category(exc), request_id) from exc

    @staticmethod
    def _log_failure(call: _Call, category: str, exc: Exception) -> None:
        detail = exc.detail if isinstance(exc, ServiceError) else f"{type(exc).__name__}: {exc}"
        logger.warning(
            "%s request %s failed: %s (%s)",
            call.tool,
            call.request_id,
            category,
            detail,
            exc_info=category not in _CALLER_ERRORS and not isinstance(exc, ServiceError),
        )
