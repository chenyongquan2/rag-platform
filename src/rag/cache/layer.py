"""The result cache as `SearchService` uses it (design s12-1, s15-1; spec P5-01).

    cache = ResultCache(ExactCache(backend, config), settings, shadow=recorder)
    run = cache.search(cfg, config_version=..., question=..., filters=..., acl_groups=...,
                       top_k=..., request_id=..., compute=lambda: pipeline.retrieve(...))
    run = await cache.ask(cfg, ..., compute=lambda: runtime.ask(...))

It makes the key fields of a request, runs the request through the exact cache (L1) and gives
the answer back as our own result classes. `observe` puts the outcome on the request span and in
the metrics. The L2 semantic cache (shadow mode, `rag.cache.semantic`) is called after a miss:
it only records, it never changes `run.result`.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any

from opentelemetry.trace import Span

from rag.cache.exact import CacheRun, Codec, ExactCache
from rag.cache.info import CacheInfo
from rag.cache.keys import CacheKeyFields, answer_config_version, make_fields
from rag.cache.rules import (
    Entry,
    ask_cacheable,
    ask_from_payload,
    ask_payload,
    is_not_found,
    search_cacheable,
    search_from_payload,
    search_payload,
)
from rag.cache.store import CircuitBreaker
from rag.config import KnowledgeBaseConfig, ServiceSettings
from rag.observe import attributes as attr
from rag.observe import get_observability
from rag.observe.context import is_probe_request
from rag.types import AskResult, RetrievalResult

if TYPE_CHECKING:
    from rag.cache.semantic import ShadowRecorder

logger = logging.getLogger(__name__)

LAYER = "l1"

SEARCH_CODEC: Codec[RetrievalResult] = Codec(
    to_payload=search_payload,
    verdict=search_cacheable,
    not_found=is_not_found,
)
ASK_CODEC: Codec[AskResult] = Codec(
    to_payload=ask_payload,
    verdict=ask_cacheable,
    not_found=is_not_found,
)


PROBE_REASON = "probe"
"""`rag.cache.reason` of a probe's request: it bypassed the cache on purpose."""
SHADOW_MAX_PENDING = 16
"""Shadow lookups that may wait for the one worker; the next one is dropped (and counted)."""
SHADOW_BREAKER_FAILURES = 5
SHADOW_BREAKER_OPEN_S = 30.0
SHADOW_LOG_EVERY_S = 60.0


class _ShadowRunner:
    """One worker thread for the shadow lookups, with a bounded queue and a breaker: a slow or
    broken semantic store costs the requests nothing, and fills neither memory nor the log."""

    def __init__(self, max_pending: int) -> None:
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="cache-shadow")
        self._pending = 0
        self._lock = threading.Lock()
        self._idle = threading.Condition(self._lock)
        self._max_pending = max_pending
        self._breaker = CircuitBreaker(
            failures=SHADOW_BREAKER_FAILURES, open_s=SHADOW_BREAKER_OPEN_S
        )
        self._last_logged = float("-inf")

    def submit(self, work: Callable[[], None], kb: str) -> bool:
        metrics = get_observability().metrics
        if not self._breaker.allow():
            metrics.record_cache_shadow(kb=kb, tool="search_docs", verdict="skipped")
            return False
        with self._lock:
            if self._pending >= self._max_pending:
                full = True
            else:
                full = False
                self._pending += 1
        if full:
            metrics.record_cache_shadow(kb=kb, tool="search_docs", verdict="dropped")
            return False
        try:
            self._executor.submit(self._run, work, kb)
        except RuntimeError:  # closed
            self._done()
            return False
        return True

    def _run(self, work: Callable[[], None], kb: str) -> None:
        try:
            work()
            self._breaker.success()
        except Exception as exc:
            self._breaker.failure()
            get_observability().metrics.record_cache_shadow(
                kb=kb, tool="search_docs", verdict="failed"
            )
            now = time.monotonic()
            if now - self._last_logged >= SHADOW_LOG_EVERY_S:
                self._last_logged = now
                logger.warning(
                    "shadow semantic cache failed: %s (more failures are counted, not logged)",
                    type(exc).__name__,
                )
            logger.debug("shadow semantic cache failure", exc_info=True)
        finally:
            self._done()

    def _done(self) -> None:
        with self._idle:
            self._pending -= 1
            self._idle.notify_all()

    def wait(self, timeout_s: float) -> bool:
        deadline = time.monotonic() + timeout_s
        with self._idle:
            while self._pending > 0:
                left = deadline - time.monotonic()
                if left <= 0:
                    return False
                self._idle.wait(left)
        return True

    def close(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)


class ResultCache:
    """L1 for `search_docs` and `ask_docs`, plus the shadow recorder of L2 when there is one."""

    def __init__(
        self,
        exact: ExactCache,
        settings: ServiceSettings,
        *,
        shadow: ShadowRecorder | None = None,
        shadow_max_pending: int = SHADOW_MAX_PENDING,
    ) -> None:
        self.exact = exact
        self.settings = settings
        self.shadow = shadow
        self._runner = _ShadowRunner(shadow_max_pending) if shadow is not None else None
        # kb -> (the config object the version was made from, version). The object is kept so
        # its identity cannot be reused by another config after garbage collection.
        self._answer_versions: dict[str, tuple[KnowledgeBaseConfig, str]] = {}

    # ---- keys --------------------------------------------------------------------------------

    def answer_version(self, cfg: KnowledgeBaseConfig) -> str:
        known = self._answer_versions.get(cfg.kb)
        if known is not None and known[0] is cfg:
            return known[1]
        version = answer_config_version(cfg, self.settings)
        self._answer_versions[cfg.kb] = (cfg, version)
        return version

    def _fields(
        self,
        kind: str,
        cfg: KnowledgeBaseConfig,
        *,
        config_version: str,
        question: str,
        filters: Mapping[str, Any] | None,
        acl_groups: Sequence[str],
        top_k: int | None,
    ) -> Callable[[int], CacheKeyFields]:
        answer_version = self.answer_version(cfg) if kind == "ask" else ""

        def build(generation: int) -> CacheKeyFields:
            return make_fields(
                kind,  # type: ignore[arg-type]
                kb=cfg.kb,
                question=question,
                filters=filters,
                acl_groups=acl_groups,
                top_k=top_k,
                data_generation=generation,
                config_version=config_version,
                answer_version=answer_version,
                index_prefix=self.settings.search_index_prefix,
                language_setting=cfg.answer.language,
            )

        return build

    # ---- the two tools -----------------------------------------------------------------------

    def search(
        self,
        cfg: KnowledgeBaseConfig,
        *,
        config_version: str,
        question: str,
        filters: Mapping[str, Any] | None,
        acl_groups: Sequence[str],
        top_k: int,
        request_id: str,
        compute: Callable[[], RetrievalResult],
        wait_s: float | None = None,
        vector_of: Callable[[str], list[float] | None] | None = None,
    ) -> CacheRun[RetrievalResult]:
        if is_probe_request(request_id):
            # The probe must reach Elasticsearch and the gateway, and its made-up questions must
            # not fill L1 or L2 (spec P5-02, review round 1 issue 4).
            return CacheRun(compute(), "bypass", reason=PROBE_REASON)
        build = self._fields(
            "search",
            cfg,
            config_version=config_version,
            question=question,
            filters=filters,
            acl_groups=acl_groups,
            top_k=top_k,
        )

        def decode(entry: Entry) -> RetrievalResult:
            return search_from_payload(
                entry.payload,
                request_id=request_id,
                data_generation=int(entry.key_fields.get("data_generation", 0)),
                timings_ms={},
            )

        def adapt(result: RetrievalResult) -> RetrievalResult:
            return result.model_copy(update={"request_id": request_id})

        run = self.exact.get_or_compute(
            build,
            compute,
            codec=SEARCH_CODEC,
            decode=decode,
            adapt=adapt,
            wait_s=wait_s,
            kb=cfg.kb,
        )
        if run.outcome in ("hit", "shared"):
            run.result = run.result.model_copy(
                update={"timings_ms": {"cache": round(run.elapsed_ms, 3)}}
            )
        elif run.outcome == "miss" and run.stored and self.shadow is not None:
            self._shadow(cfg, config_version, question, filters, acl_groups, top_k, run, vector_of)
        return run

    def _shadow(
        self,
        cfg: KnowledgeBaseConfig,
        config_version: str,
        question: str,
        filters: Mapping[str, Any] | None,
        acl_groups: Sequence[str],
        top_k: int,
        run: CacheRun[RetrievalResult],
        vector_of: Callable[[str], list[float] | None] | None,
    ) -> None:
        """L2 in shadow mode: hand the work to the background runner and go on. It records what
        the semantic cache would have done; it never changes the result and the request never
        waits for it."""
        assert self.shadow is not None and self._runner is not None
        recorder = self.shadow
        fresh = run.result
        generation = run.generation if run.generation is not None else 0
        kb = cfg.kb

        def work() -> None:
            recorder.observe(
                kb=kb,
                question=question,
                filters=filters,
                acl_groups=acl_groups,
                top_k=top_k,
                config_version=config_version,
                data_generation=generation,
                fresh=fresh,
                vector=vector_of(question) if vector_of is not None else None,
            )

        self._runner.submit(work, kb)

    def flush_shadow(self, timeout_s: float = 10.0) -> bool:
        """Wait until the queued shadow work has run (tests, shutdown); False on time-out."""
        return self._runner.wait(timeout_s) if self._runner is not None else True

    async def ask(
        self,
        cfg: KnowledgeBaseConfig,
        *,
        config_version: str,
        question: str,
        acl_groups: Sequence[str],
        request_id: str,
        compute: Callable[[], Awaitable[AskResult]],
        read: bool = True,
        wait_s: float | None = None,
    ) -> CacheRun[AskResult]:
        """`read=False` for a call that must not be served from the cache (it has a session:
        a hit would skip writing the turn to the history); a complete answer is still written.
        A probe's call neither reads nor writes."""
        if is_probe_request(request_id):
            return CacheRun(await compute(), "bypass", reason=PROBE_REASON)
        build = self._fields(
            "ask",
            cfg,
            config_version=config_version,
            question=question,
            filters=None,
            acl_groups=acl_groups,
            top_k=None,
        )

        def decode(entry: Entry) -> AskResult:
            return ask_from_payload(
                entry.payload,
                request_id=request_id,
                question=question,
                timings_ms={},
                data_generation=int(entry.key_fields.get("data_generation", 0)),
            )

        def adapt(result: AskResult) -> AskResult:
            return result.model_copy(
                update={
                    "request_id": request_id,
                    "session_id": None,
                    "thread_id": "",
                    "question": question,
                    "tokens_used": 0,
                    "resume_token": None,
                }
            )

        run = await self.exact.aget_or_compute(
            build,
            compute,
            codec=ASK_CODEC,
            decode=decode,
            adapt=adapt,
            wait_s=wait_s,
            read=read,
            kb=cfg.kb,
        )
        if run.outcome in ("hit", "shared"):
            run.result = run.result.model_copy(
                update={"timings_ms": {"cache": round(run.elapsed_ms, 3)}}
            )
        return run

    # ---- what the caller reports -------------------------------------------------------------

    @staticmethod
    def info(run: CacheRun[Any]) -> CacheInfo:
        hit = run.outcome in ("hit", "shared")
        return CacheInfo(
            layer=LAYER,
            hit=hit,
            outcome=run.outcome,
            age_s=round(run.age_s, 3) if hit and run.age_s is not None else None,
        )

    @staticmethod
    def record(run: CacheRun[Any]) -> dict[str, Any]:
        """The `cache` entry of the business record."""
        return {
            "layer": LAYER,
            "outcome": run.outcome,
            "reason": run.reason or None,
            "stored": run.stored,
            "not_stored_reason": run.not_stored_reason or None,
            "lookup_ms": round(run.lookup_ms, 3),
            "store_ms": round(run.store_ms, 3),
            "age_s": round(run.age_s, 3) if run.age_s is not None else None,
            "error": run.outcome == "error",
        }

    @staticmethod
    def observe(span: Span | None, *, kb: str | None, tool: str, run: CacheRun[Any]) -> None:
        """Outcome on the request span, in the metrics, and the lookup time as a step."""
        if span is not None:
            span.set_attributes(
                {
                    attr.CACHE_LAYER: LAYER,
                    attr.CACHE_OUTCOME: run.outcome,
                    attr.CACHE_STORED: run.stored,
                    attr.CACHE_LOOKUP_MS: round(run.lookup_ms, 3),
                    **(
                        {attr.CACHE_REASON: run.reason or run.not_stored_reason}
                        if (run.reason or run.not_stored_reason)
                        else {}
                    ),
                }
            )
        metrics = get_observability().metrics
        metrics.record_cache(
            kb=kb,
            tool=tool,
            layer=LAYER,
            outcome=run.outcome,
            not_stored_reason=run.not_stored_reason if run.outcome == "miss" else "",
            reason=run.reason if run.outcome in ("bypass", "error") else "",
        )
        metrics.record_step("cache_lookup", run.lookup_ms / 1000.0, outcome=attr.OUTCOME_OK, kb=kb)

    # ---- admin -------------------------------------------------------------------------------

    def drop_kb(self, kb: str) -> int:
        """Remove every entry of one knowledge base (L1 keys and flight locks)."""
        return self.exact.drop_prefix(f"{self.settings.search_index_prefix}{kb}:")

    def close(self) -> None:
        if self._runner is not None:
            self._runner.close()
        backend = self.exact.backend
        for name in ("client", "generation_client"):
            client = getattr(backend, name, None)
            if client is not None:
                try:
                    client.close()
                except Exception:
                    logger.warning("closing the cache client failed", exc_info=True)
