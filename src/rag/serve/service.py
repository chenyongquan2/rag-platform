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
import re
import threading
import time
from collections.abc import AsyncIterator, Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

from rag.agent import AgentRuntime, valid_session_id
from rag.config import KnowledgeBaseConfig, ServiceSettings, load_kb_configs, repo_path
from rag.observe import (
    BusinessRecord,
    RankedId,
    business_record_from_result,
    current_trace_id,
    get_observability,
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
    FEEDBACK_COMMENT_CHARS,
    AskResponse,
    ContextResponse,
    ErrorResponse,
    FeedbackResponse,
    KnowledgeBaseInfo,
    KnowledgeBaseList,
    SearchResponse,
)
from rag.serve.stream import HEARTBEAT, Heartbeat, StreamEvent
from rag.types import AnswerStatus, AskResult, ContextResult, ErrorCategory, RetrievalResult

if TYPE_CHECKING:
    from rag.cache.layer import ResultCache

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
    ErrorCategory.INVALID_RESUME_TOKEN,
}
# The request id a feedback verdict names: what `X-Request-Id` and `new_request_id` allow.
_REQUEST_ID = re.compile(r"^[A-Za-z0-9._:-]{1,64}$")
_END = object()  # the producer of a stream has finished
# What a resume token looks like (`rag.agent.resume.new_token`): anything else cannot be one.
_RESUME_TOKEN = re.compile(r"^[A-Za-z0-9_-]{16,128}$")


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
        rerank_reserve_ms: int = 0,
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
    stage: str | None = None
    """The graph node a streamed request is in (the record of a cancelled one names it)."""
    cancel_note: dict[str, str] = field(default_factory=dict)
    """Why a streamed request is cancelled, when the stream knows (`by`: slow_client)."""

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
        helper_model: Any = None,
        cache: ResultCache | None = None,
    ) -> None:
        self.settings = settings
        self._cache = cache  # None: no result cache (a bare service; `from_settings` makes one)
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
        self._streams: set[asyncio.Task[None]] = set()  # the runs of open streams
        self._shutting_down = False
        self._answer_model = answer_model  # None: the runtime's `online` chat model
        self._helper_model = helper_model  # None: the runtime's `agent_rewrite_profile` model

    @classmethod
    def from_settings(cls, settings: ServiceSettings, **kwargs: Any) -> SearchService:
        """The service with the result cache the settings ask for (`cache_mode`; spec P5-01)."""
        from rag.cache.factory import make_result_cache

        return cls(settings, cache=make_result_cache(settings), **kwargs)

    @property
    def cache(self) -> ResultCache | None:
        return self._cache

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
        if self._cache is not None:
            self._cache.close()

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

            def compute() -> RetrievalResult:
                remaining = int(deadline.remaining_ms())
                if remaining <= 0:
                    raise ServiceError(
                        ErrorCategory.RETRIEVAL_TIMEOUT, request_id, detail="no time"
                    )
                with layer_span(
                    "retrieve",
                    attributes={
                        attr.TOP_K: top_k,
                        attr.RERANK_N: retrieval.rerank_n,
                        attr.FINAL_K: retrieval.final_k,
                        attr.THRESHOLD: retrieval.threshold,
                    },
                ) as span:
                    fresh = pipeline.retrieve(
                        query,
                        cfg.kb,
                        top_k,
                        filters=dict(filters or {}),
                        deadline_ms=remaining,
                        request_id=request_id,
                        acl_groups=caller.acl_groups,
                    )
                    span.set_attribute(attr.HITS, len(fresh.chunks))
                return fresh

            run = None
            if self._cache is None or not cfg.cache:
                result = compute()
            else:
                run = self._cache.search(
                    cfg,
                    config_version=pipeline.config_version,
                    question=query,
                    filters=filters,
                    acl_groups=caller.acl_groups,
                    top_k=top_k,
                    request_id=request_id,
                    compute=compute,
                    wait_s=deadline.remaining_s(),
                    vector_of=getattr(pipeline, "query_vector", None),
                )
                result = run.result
            assert call.scope is not None
            call.scope.set_result(result)
            if run is not None:
                self._cache.observe(call.scope.span, kb=cfg.kb, tool=call.tool, run=run)
            if run is not None and run.outcome in ("hit", "shared"):
                self._observe_cached(result, cfg.kb)  # no step timings: nothing ran
            else:
                self._observe_result(result, cfg.kb)
            response = SearchResponse.from_result(
                result,
                timings_ms={"request": call.elapsed_ms()},
                cache=self._cache.info(run) if run is not None else None,
            )
            call.record = business_record_from_result(
                result.model_copy(update={"timings_ms": response.timings_ms}),
                tool=call.tool,
                config_version=result.config_version,
                caller=caller.caller_id,
                rerank=[RankedId(chunk_id=h.chunk_id, score=h.score) for h in result.rerank],
                threshold=result.threshold,
                extra={
                    **extra,
                    "top_k": top_k,
                    "not_found": result.not_found,
                    **({"cache": self._cache.record(run)} if run is not None else {}),
                },
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
        resume_token: str | None = None,
        answer: str | None = None,
    ) -> AskResponse:
        """Answer a question from one knowledge base with the ask_docs graph (s16-1, s13-3);
        raises `ServiceError`. Runs on the caller's event loop: the graph is async and the
        session locks belong to it (the blocking searches run in worker threads).

        A result with status `need_clarification` carries a `resume_token`; the caller answers
        with `resume_token` and `answer` (and the same `kb` and `session_id`; `question` is
        required as always but not used). A token that is unknown, used, expired or not the
        caller's is `invalid_resume_token`."""
        return await self._ask_docs(
            credentials,
            kb=kb,
            question=question,
            session_id=session_id,
            deadline_ms=deadline_ms,
            request_id=request_id,
            resume_token=resume_token,
            answer=answer,
        )

    async def _ask_docs(
        self,
        credentials: Credentials,
        *,
        kb: str,
        question: str,
        session_id: str | None = None,
        deadline_ms: int | None = None,
        request_id: str | None = None,
        resume_token: str | None = None,
        answer: str | None = None,
        emit: Callable[[str, dict[str, Any]], None] | None = None,
        cancel_note: dict[str, str] | None = None,
    ) -> AskResponse:
        """The one implementation of `ask_docs` and of the streamed call (`emit` is the stream's
        listener: the run's events go to it as they happen)."""
        request_id = request_id or new_request_id()
        caller, cfg = self._resolve(credentials, kb, request_id)
        pipeline = self._pipelines.get(cfg.kb) if cfg is not None else None
        extra: dict[str, Any] = {
            # A resumed call carries only the caller's reply (and a question it need not repeat
            # right): the record gets the original question from the result.
            "question": None
            if resume_token is not None
            else _clip(question, self.settings.search_max_query_chars),
            "session_id": _clip(session_id, RECORD_TEXT_CHARS) if session_id is not None else None,
            "deadline_ms": deadline_ms if _is_int(deadline_ms) else None,
            "resumed": resume_token is not None,
        }
        if emit is not None:
            extra["stream"] = True
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
            if cancel_note is not None:
                call.cancel_note = cancel_note
            if cfg is None:
                raise ServiceError(ErrorCategory.KB_NOT_FOUND, request_id, detail=f"kb {kb!r}")
            deadline_ms = self.settings.agent_deadline_ms if deadline_ms is None else deadline_ms
            self._check_ask_args(
                question, session_id, deadline_ms, request_id, resume_token, answer
            )
            deadline = Deadline.after_ms(deadline_ms)
            pipeline = await asyncio.to_thread(self._ready_pipeline, cfg.kb, deadline, request_id)
            listener = None
            if emit is not None:

                def listener(name: str, data: dict[str, Any]) -> None:
                    if name == "progress" and data.get("phase") in ("start", "waiting"):
                        call.stage = data.get("node")  # where a cancelled request was
                    emit(name, data)

            async def run_graph() -> AskResult:
                return await self.agent.ask(
                    question,
                    cfg=cfg,
                    retriever=pipeline,
                    session_id=session_id,
                    request_id=request_id,
                    caller_id=caller.caller_id,
                    acl_groups=caller.acl_groups,
                    # What is left of the call's time after waiting for the load and the cache.
                    deadline_ms=max(1, int(deadline.remaining_ms())),
                    model=self._answer_model,
                    helper_model=self._helper_model,
                    resume_token=resume_token,
                    answer=answer,
                    emit=listener,
                )

            run = None
            if (
                self._cache is None
                or not cfg.cache
                or not self.settings.cache_ask_answers
                or resume_token is not None
            ):
                # A call that continues a question asked back is never cached (its result
                # depends on the caller's reply).
                result = await run_graph()
            else:
                run = await self._cache.ask(
                    cfg,
                    config_version=pipeline.config_version,
                    question=question,
                    acl_groups=caller.acl_groups,
                    request_id=request_id,
                    compute=run_graph,
                    # A call with a session must write its turn to the history: it does not read
                    # the cache (a complete answer to it may still fill it).
                    read=session_id is None,
                    wait_s=deadline.remaining_s(),
                )
                result = run.result
                if listener is not None and run.outcome in ("hit", "shared"):
                    listener(
                        "progress",
                        {
                            "node": "cache",
                            "phase": run.outcome,
                            "layer": "l1",
                            "ms": round(run.elapsed_ms, 3),
                            "age_s": round(run.age_s, 3) if run.age_s is not None else None,
                        },
                    )
            assert call.scope is not None
            call.scope.set_result(result)
            if run is not None:
                self._cache.observe(call.scope.span, kb=cfg.kb, tool=call.tool, run=run)
            response = AskResponse.from_result(
                result,
                timings_ms={"request": call.elapsed_ms()},
                cache=self._cache.info(run) if run is not None else None,
            )
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
                    "question": _clip(result.question, self.settings.search_max_query_chars)
                    if resume_token is not None
                    else extra["question"],
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
                    "clarified": result.clarified,
                    "standalone_question": _clip(result.standalone_question, RECORD_TEXT_CHARS)
                    if result.standalone_question
                    else None,
                    "search_queries": [_clip(q, RECORD_TEXT_CHARS) for q in result.search_queries],
                    "fallbacks": result.fallbacks,
                    "sufficiency": result.sufficiency,
                    "missing": [_clip(m, RECORD_TEXT_CHARS) for m in result.missing],
                    **({"cache": self._cache.record(run)} if run is not None else {}),
                },
            )
            call.record = record.model_copy(
                update={
                    "index_version": result.index_version,
                    "data_generation": result.data_generation,
                }
            )
            return response

    async def ask_docs_stream(
        self,
        credentials: Credentials,
        *,
        kb: str,
        question: str,
        session_id: str | None = None,
        deadline_ms: int | None = None,
        request_id: str | None = None,
        resume_token: str | None = None,
        answer: str | None = None,
        heartbeat_s: float | None = None,
    ) -> AsyncIterator[StreamEvent | Heartbeat]:
        """`ask_docs` as a stream of events (s16-9; the events are in `rag.serve.stream` and the
        spec). The run is the same one `ask_docs` makes (`_ask_docs`: the caller's access, the
        argument checks, the session lock, the deadline, the token cap, the business record,
        the spans); it runs in a task of its own that puts the events in a queue, and this
        generator hands them out, with a heartbeat after `heartbeat_s` seconds of silence.

        A failure before anything was produced (a bad argument, a hidden KB, a refused resume
        token, a service that is not ready) raises `ServiceError` from the first `anext`, so the
        HTTP layer can still answer with a status code. After that, a failure is an `error`
        event and the stream ends. Closing the generator (the client left) cancels the run:
        the graph stops where it is, the session lock is released, the request is recorded as
        `client_cancelled` and its span ends `cancelled`. The task, not the generator, does
        that work, because the generator is being cancelled itself when this happens."""
        request_id = request_id or new_request_id()
        if self._shutting_down:
            raise ServiceError(ErrorCategory.SERVER_SHUTDOWN, request_id, detail="closing")
        beat = self.settings.sse_heartbeat_s if heartbeat_s is None else heartbeat_s
        queue: asyncio.Queue[Any] = asyncio.Queue()
        began = time.perf_counter()
        task = asyncio.create_task(
            self._produce_stream(
                queue,
                began,
                credentials,
                request_id=request_id,
                kb=kb,
                question=question,
                session_id=session_id,
                deadline_ms=deadline_ms,
                resume_token=resume_token,
                answer=answer,
            ),
            name=f"ask-stream-{request_id}",
        )
        self._streams.add(task)
        task.add_done_callback(self._streams.discard)
        started = False
        ended = False  # an event that ends a stream (answer, question, error) went out
        try:
            while True:
                try:
                    # No heartbeat before the first event: until then the HTTP layer may still
                    # answer with a plain error, and nothing has been sent.
                    item = await asyncio.wait_for(queue.get(), beat if started else None)
                except TimeoutError:
                    yield HEARTBEAT
                    continue
                if item is _END:
                    if self._shutting_down and started and not ended:
                        # Cancelled by the shutdown: say so, the stream would end silently.
                        failure = ErrorResponse.of(ErrorCategory.SERVER_SHUTDOWN, request_id)
                        yield StreamEvent("error", failure.model_dump())
                    return
                if isinstance(item, ServiceError):
                    if not started:
                        raise item
                    failure = ErrorResponse.of(item.category, item.request_id)
                    item = StreamEvent("error", failure.model_dump())
                started = True
                ended = ended or item.name in ("answer", "need_clarification", "error")
                yield item
        finally:
            if not task.done():
                task.cancel()

    async def _produce_stream(
        self,
        queue: asyncio.Queue[Any],
        began: float,
        credentials: Credentials,
        *,
        request_id: str,
        **ask: Any,
    ) -> None:
        """Run the request and put its events in `queue`; a `ServiceError` for a failure; `_END`
        last, however it ends."""
        first: dict[str, float] = {}
        note: dict[str, str] = {}
        run = asyncio.current_task()

        def emit(name: str, data: dict[str, Any]) -> None:
            if name not in first and name in ("progress", "token"):
                first[name] = round((time.perf_counter() - began) * 1000.0, 3)
            if queue.qsize() >= self.settings.sse_queue_max_events and run is not None:
                # The client is not reading: stop the run instead of keeping its events.
                note["by"] = "slow_client"
                run.cancel()
                return
            queue.put_nowait(StreamEvent(name, data))

        try:
            response = await self._ask_docs(
                credentials, request_id=request_id, emit=emit, cancel_note=note, **ask
            )
            for event in self._closing_events(response, first):
                queue.put_nowait(event)
        except ServiceError as err:
            queue.put_nowait(err)
        except asyncio.CancelledError:
            raise
        except Exception:  # `_call` turns everything into a ServiceError; this is a bug
            logger.exception("stream %s failed outside the call", request_id)
            queue.put_nowait(ServiceError(ErrorCategory.INTERNAL_ERROR, request_id))
        finally:
            queue.put_nowait(_END)

    @staticmethod
    def _closing_events(response: AskResponse, first: Mapping[str, float]) -> list[StreamEvent]:
        """What ends a stream that did not fail: the checked answer (or the question to ask the
        caller back), then `done` with the numbers of the run."""
        if response.status == AnswerStatus.NEED_CLARIFICATION:
            last = StreamEvent(
                "need_clarification",
                {
                    "request_id": response.request_id,
                    "question": response.answer,
                    "resume_token": response.resume_token,
                    "session_id": response.session_id,
                },
            )
        else:
            last = StreamEvent("answer", response.model_dump(mode="json"))
        cache = (
            {
                "cache": {
                    "layer": response.cache.layer,
                    "hit": response.cache.hit,
                    "outcome": response.cache.outcome,
                }
            }
            if response.cache is not None
            else {}
        )
        done = StreamEvent(
            "done",
            {
                **cache,
                "request_id": response.request_id,
                "status": response.status,
                "rounds": response.rounds,
                "degraded": response.degraded,
                "rerank_timeout": response.rerank_timeout,
                "coverage_unconfirmed": response.coverage_unconfirmed,
                "stop_reason": response.stop_reason,
                "tokens_used": response.tokens_used,
                "timings_ms": {
                    **response.timings_ms,
                    **{f"first_{name}": ms for name, ms in first.items()},
                },
            },
        )
        return [last, done]

    def begin_shutdown(self) -> None:
        """The server is going away: no new stream starts, and the runs of the open ones
        are cancelled now. Each generator then ends its stream with an `error` event
        (`server_shutdown`) and each run is recorded as cancelled by the shutdown. Called by
        the server before it waits for open responses (`rag.serve.server`): uvicorn runs the
        app's own shutdown code only after they have ended, which for a stream is too late.
        Call it on the event loop of the streams."""
        self._shutting_down = True
        for task in list(self._streams):
            task.cancel()

    async def cancel_streams(self) -> None:
        """Cancel the runs of the open streams and wait until they have cleaned up (at shutdown;
        a stream whose client is gone has been cancelled already)."""
        loop = asyncio.get_running_loop()
        tasks = [t for t in self._streams if t.get_loop() is loop]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    def feedback(
        self,
        credentials: Credentials,
        *,
        rated_request_id: Any,
        useful: Any,
        comment: Any = None,
        request_id: str | None = None,
    ) -> FeedbackResponse:
        """The user's verdict on an answer (s16-1): one business record line with the rated
        request's id, `useful` and a comment of at most `FEEDBACK_COMMENT_CHARS` characters
        (typed by the user; nothing else about the user is stored). P5 turns it into a score
        on the trace; until then it is only this line. The id is checked for its shape, not
        for existence."""
        request_id = request_id or new_request_id()
        caller, _ = self._resolve(credentials, "", request_id)
        with self._call(
            "feedback",
            request_id=request_id,
            caller=caller,
            cfg=None,
            config_version="",
            extra={},
        ) as call:
            problem = None
            if not isinstance(rated_request_id, str) or not _REQUEST_ID.fullmatch(rated_request_id):
                problem = "request_id must be the id of a request"
            elif not isinstance(useful, bool):
                problem = "useful must be true or false"
            elif comment is not None and not isinstance(comment, str):
                problem = "comment must be text"
            elif comment is not None and len(comment) > FEEDBACK_COMMENT_CHARS:
                problem = f"comment longer than {FEEDBACK_COMMENT_CHARS} characters"
            if problem is not None:
                raise ServiceError(ErrorCategory.INVALID_REQUEST, request_id, detail=problem)
            text = comment.strip() if isinstance(comment, str) else ""
            record = self._record(
                call,
                timings_ms={"request": call.elapsed_ms()},
                extra={
                    "useful": useful,
                    "comment": text or None,
                    "feedback_request_id": request_id,
                },
            )
            # Keyed by the rated request: a join on `request_id` finds the answer and the verdict.
            call.record = record.model_copy(update={"request_id": rated_request_id})
            return FeedbackResponse(request_id=rated_request_id, recorded=True)

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
        self,
        question: Any,
        session_id: Any,
        deadline_ms: Any,
        request_id: str,
        resume_token: Any = None,
        answer: Any = None,
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
        elif resume_token is None and answer is not None:
            problem = "answer is only used together with resume_token"
        elif resume_token is not None:
            if not isinstance(resume_token, str):
                problem = "resume_token must be text"
            elif not isinstance(answer, str) or not answer.strip():
                problem = "answer must be non-empty text when resume_token is given"
            elif len(answer) > self.settings.search_max_query_chars:
                problem = f"answer longer than {self.settings.search_max_query_chars} characters"
        if problem is not None:
            raise ServiceError(ErrorCategory.INVALID_REQUEST, request_id, detail=problem)
        if resume_token is not None and not _RESUME_TOKEN.fullmatch(resume_token):
            # Cannot be a token we issued: the same answer as for one that is unknown.
            raise ServiceError(
                ErrorCategory.INVALID_RESUME_TOKEN, request_id, detail="malformed resume_token"
            )

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
    def _observe_cached(result: RetrievalResult, kb: str) -> None:
        """The quality signals of a search served from the cache: the same callers got the same
        not-found answer and top score; no step ran, so no step time is recorded."""
        scores = [c.rerank_score for c in result.chunks if c.rerank_score is not None]
        get_observability().metrics.record_retrieval(
            kb=kb,
            tool="search_docs",
            not_found=result.not_found,
            top_rerank_score=max(scores) if scores else None,
        )

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
                except asyncio.CancelledError:
                    # The caller went away (a closed stream) or the server is shutting down:
                    # not a failure, but a request that ended without an answer. One record, so
                    # the day's numbers add up; the span ends `cancelled` (in `request_span`).
                    logger.info("%s request %s cancelled in %s", tool, request_id, call.stage)
                    self._write(
                        self._record(
                            call,
                            flags=[ErrorCategory.CLIENT_CANCELLED],
                            timings_ms={"request": call.elapsed_ms()},
                            extra={
                                **call.extra,
                                "cancelled": True,
                                "cancelled_at": call.stage,
                                "cancelled_by": call.cancel_note.get("by")
                                or ("shutdown" if self._shutting_down else "client"),
                            },
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
