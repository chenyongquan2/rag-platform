"""Rerank with a local cross-encoder, in batches, stopping by the request deadline (design s10-1).

`BatchedTimeoutRerank` is the LlamaIndex postprocessor of the retrieval chain (s9-1). It sends
the first `top_n` fused candidates (`rerank_n` of the KB config) to the process-wide
`RerankRunner` as (question, title path + body) pairs - no other metadata (s10-1) - and returns
them sorted by rerank score, ties in fusion order. The rerank score goes into the node's
metadata (`rerank_score`, later `Chunk.rerank_score`); `NodeWithScore.score` keeps the fusion
score. Candidates after `top_n` are dropped: they were never candidates for the reranker.

`RerankRunner` scores in batches of `batch_size` (8) in a thread pool behind a semaphore of
`max_concurrency` slots (reranking is CPU bound; several requests at once slow each other
down). The time spent waiting for a slot counts against the request's deadline. Before each
batch, the first one included, it looks at the time left: when the deadline has passed, or the
batch would not finish before it, it stops. "Would not finish" is judged by the runner's time
per pair, a moving average over the batches of all requests, seeded by a warm-up batch when the
process runner is created, so it also reflects how busy the CPU is. Then the scored candidates
come first, sorted by rerank score, the unscored ones follow in fusion order, and the trace says
`rerank_timeout` (s16-8). The caller never waits past the deadline: if a batch is still running
then (it took longer than estimated), the runner returns what was scored so far and the batch
finishes in the background (its slot is held until it does).

The model (sentence-transformers `CrossEncoder`, bge-reranker-base) is loaded once per process
from the local Hugging Face cache (`local_files_only`), with sigmoid as the activation, so scores
are in [0, 1] (s10-2: the threshold depends on it; scores outside are refused). All knowledge
bases share it, so rerank scores compare across them (s8-3). Loading sets torch's thread count
for the process (`reranker_torch_threads`): slots x threads should not exceed the physical
cores, else concurrent reranks fight over them.
"""

from __future__ import annotations

import json
import logging
import math
import threading
import time
from collections.abc import Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, ClassVar, Protocol, runtime_checkable

from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import MetadataMode, NodeWithScore, QueryBundle
from pydantic import Field, PrivateAttr

from rag.config import ServiceSettings
from rag.retrieve.deadline import Deadline
from rag.retrieve.query import HybridQuery
from rag.types import Chunk, RouteHit

logger = logging.getLogger(__name__)

RERANK_SCORE = "rerank_score"
"""Node metadata key of the rerank score (moved to `Chunk.rerank_score` at the layer boundary)."""
TITLE_SEPARATOR = " > "
ESTIMATE_WEIGHT = 0.3
"""Weight of the newest batch in the moving average of the time per pair."""
WARMUP_QUESTION = "How does retrieval-augmented generation find the passages it cites?"
WARMUP_PASSAGE = "Guide > Retrieval\n" + " ".join(
    ["Retrieval-augmented generation searches an index, reranks the passages it finds and"]
    + ["gives the best ones to a language model, which cites them in its answer."] * 16
)
"""About 300 tokens with the question: near the median pair of the HF docs (266 tokens)."""


@runtime_checkable
class PairScorer(Protocol):
    """Scores (question, passage) pairs; one score in [0, 1] per pair, in order."""

    def predict(self, pairs: Sequence[tuple[str, str]]) -> Sequence[float]: ...


class RerankScaleError(RuntimeError):
    """The scorer returned a score outside [0, 1] (or not a number): it is not the sigmoid
    scale the rerank threshold was set on (s10-2), so its results cannot be judged."""


def rerank_text(title_path: str | Sequence[str], body: str) -> str:
    """The passage the cross-encoder reads: the title path, a line break, the body (s10-1)."""
    title = title_path if isinstance(title_path, str) else TITLE_SEPARATOR.join(title_path)
    return f"{title}\n{body}" if title else body


# ---- the model -----------------------------------------------------------------------------


def cross_encoder_kwargs(settings: ServiceSettings) -> dict[str, Any]:
    """Every `CrossEncoder` argument we set (besides the sigmoid activation, which needs torch):
    nothing is left to the library's defaults."""
    return {
        "device": settings.reranker_device,
        "max_length": settings.reranker_max_length,
        "local_files_only": True,
        "backend": "torch",
        "trust_remote_code": False,
    }


class CrossEncoderScorer:
    """`PairScorer` over a sentence-transformers `CrossEncoder` with sigmoid activation."""

    def __init__(self, model: Any, *, name: str, load_s: float = 0.0, torch_threads: int = 0):
        self.model = model
        self.name = name
        self.load_s = load_s
        """Seconds to load the model (imports included)."""
        self.torch_threads = torch_threads

    @staticmethod
    def load(model: str, *, torch_threads: int, **kwargs: Any) -> CrossEncoderScorer:
        """Load `model` with `kwargs` (see `cross_encoder_kwargs`) after setting torch's thread
        count for the process to `torch_threads`. Imports torch and sentence-transformers here,
        so modules that only define the chain stay quick to import."""
        if torch_threads < 1:
            raise ValueError(f"torch_threads must be >= 1, got {torch_threads}")
        start = time.perf_counter()
        import torch
        from sentence_transformers import CrossEncoder

        torch.set_num_threads(torch_threads)
        encoder = CrossEncoder(model, activation_fn=torch.nn.Sigmoid(), **kwargs)
        scorer = CrossEncoderScorer(
            encoder, name=model, load_s=time.perf_counter() - start, torch_threads=torch_threads
        )
        logger.info(
            "reranker %s loaded in %.1f s (%d torch threads)", model, scorer.load_s, torch_threads
        )
        return scorer

    def predict(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        scores = self.model.predict(
            [list(p) for p in pairs],
            batch_size=max(1, len(pairs)),
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return [float(s) for s in scores]


def _check_scores(scores: Sequence[float], expected: int) -> list[float]:
    values = [float(s) for s in scores]
    if len(values) != expected:
        raise RerankScaleError(f"scorer returned {len(values)} scores for {expected} pairs")
    for value in values:
        if not (math.isfinite(value) and 0.0 <= value <= 1.0):
            raise RerankScaleError(
                f"rerank score {value!r} is outside [0, 1]: the scorer is not on the sigmoid "
                "scale the threshold was set on (s10-2)"
            )
    return values


# ---- running reranks -----------------------------------------------------------------------


@dataclass(frozen=True)
class RerankRun:
    """One rerank: `scores[i]` belongs to `texts[i]`; texts after `len(scores)` were not scored."""

    scores: list[float]
    timed_out: bool
    """Not every text was scored (deadline, or no slot before the deadline)."""
    queue_ms: float
    """Time spent waiting for a slot."""
    score_ms: float
    batches: int


@dataclass
class _Progress:
    scores: list[float] = field(default_factory=list)
    batches: int = 0
    stop: threading.Event = field(default_factory=threading.Event)
    lock: threading.Lock = field(default_factory=threading.Lock)


class RerankRunner:
    """Batches, deadline checks and the concurrency limit around a `PairScorer`.

    One per process (`get_rerank_runner`), shared by every knowledge base's chain, so
    `max_concurrency` bounds all reranks of the process.
    """

    def __init__(
        self,
        scorer: PairScorer,
        *,
        batch_size: int,
        max_concurrency: int,
        per_pair_s: float | None = None,
    ) -> None:
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        if max_concurrency < 1:
            raise ValueError(f"max_concurrency must be >= 1, got {max_concurrency}")
        self.scorer = scorer
        self.batch_size = batch_size
        self.max_concurrency = max_concurrency
        self.per_pair_s = per_pair_s
        """Moving average of the seconds per pair over all batches (None before the first)."""
        self.warmup_s = 0.0
        self._estimate_lock = threading.Lock()
        self._slots = threading.BoundedSemaphore(max_concurrency)
        # As many threads as slots: a request holding a slot never waits for a thread.
        self._pool = ThreadPoolExecutor(max_workers=max_concurrency, thread_name_prefix="rerank")

    def warm_up(self) -> None:
        """Score one batch of realistic pairs (unused) in a pool thread: the first call of a
        model is slower, and its time seeds the per-pair estimate (replacing any earlier)."""
        pairs = [(WARMUP_QUESTION, WARMUP_PASSAGE)] * self.batch_size

        def run() -> float:
            began = time.monotonic()
            _check_scores(self.scorer.predict(pairs), len(pairs))
            return time.monotonic() - began

        with self._slot():
            self.warmup_s = self._pool.submit(run).result()
        with self._estimate_lock:
            self.per_pair_s = self.warmup_s / len(pairs)

    @contextmanager
    def _slot(self) -> Iterator[None]:
        self._slots.acquire()
        try:
            yield
        finally:
            self._slots.release()

    def _record(self, seconds: float, pairs: int) -> None:
        measured = seconds / pairs
        with self._estimate_lock:
            if self.per_pair_s is None:
                self.per_pair_s = measured
            else:
                self.per_pair_s += ESTIMATE_WEIGHT * (measured - self.per_pair_s)

    def rerank(
        self, question: str, texts: Sequence[str], deadline: Deadline | None = None
    ) -> RerankRun:
        """Score (question, text) pairs in order until done or out of time (module doc).
        Without a deadline every text is scored. Scorer errors are raised."""
        texts = list(texts)
        if not texts:
            return RerankRun(scores=[], timed_out=False, queue_ms=0.0, score_ms=0.0, batches=0)
        began = time.perf_counter()
        if deadline is None:
            acquired = self._slots.acquire()
        else:
            wait_s = deadline.remaining_s()
            acquired = wait_s > 0 and self._slots.acquire(timeout=wait_s)
        queue_ms = (time.perf_counter() - began) * 1000.0
        if not acquired:
            logger.warning("no rerank slot before the deadline (waited %.0f ms)", queue_ms)
            return RerankRun(scores=[], timed_out=True, queue_ms=queue_ms, score_ms=0.0, batches=0)

        progress = _Progress()
        try:
            future = self._pool.submit(self._run_batches, question, texts, deadline, progress)
        except BaseException:
            self._slots.release()
            raise
        started = time.perf_counter()
        try:
            future.result(timeout=None if deadline is None else deadline.remaining_s())
        except TimeoutError:
            # A batch is still running at the deadline: return what is scored; the batch ends
            # in the background and its scores are not used.
            progress.stop.set()
        score_ms = (time.perf_counter() - started) * 1000.0
        with progress.lock:
            progress.stop.set()
            scores = list(progress.scores)
            batches = progress.batches
        return RerankRun(
            scores=scores,
            timed_out=len(scores) < len(texts),
            queue_ms=queue_ms,
            score_ms=score_ms,
            batches=batches,
        )

    def _run_batches(
        self,
        question: str,
        texts: list[str],
        deadline: Deadline | None,
        progress: _Progress,
    ) -> None:
        clock = deadline.clock if deadline is not None else time.monotonic
        try:
            for start in range(0, len(texts), self.batch_size):
                batch = texts[start : start + self.batch_size]
                if progress.stop.is_set():
                    return
                if deadline is not None:
                    remaining = deadline.remaining_s()
                    estimate = self.per_pair_s
                    if remaining <= 0 or (
                        estimate is not None and remaining < estimate * len(batch)
                    ):
                        return
                began = clock()
                scores = _check_scores(
                    self.scorer.predict([(question, text) for text in batch]), len(batch)
                )
                self._record(clock() - began, len(batch))
                with progress.lock:
                    if progress.stop.is_set():
                        return
                    progress.scores.extend(scores)
                    progress.batches += 1
        finally:
            self._slots.release()

    def close(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)


# ---- one model and one runner per process ---------------------------------------------------

_LOAD_LOCK = threading.Lock()
_RUNNER_LOCK = threading.Lock()
_SCORERS: dict[str, PairScorer] = {}
_RUNNERS: dict[tuple[int, int, int], RerankRunner] = {}


def load_cross_encoder(settings: ServiceSettings) -> PairScorer:
    """The process's cross-encoder for these settings, loaded on the first call (service start;
    one to two minutes on this machine, mostly imports) and reused after that."""
    kwargs = cross_encoder_kwargs(settings)
    threads = settings.reranker_torch_threads
    key = json.dumps([settings.reranker_model, kwargs, threads], sort_keys=True)
    with _LOAD_LOCK:
        scorer = _SCORERS.get(key)
        if scorer is None:
            scorer = CrossEncoderScorer.load(
                settings.reranker_model, torch_threads=threads, **kwargs
            )
            _SCORERS[key] = scorer
        return scorer


def get_rerank_runner(settings: ServiceSettings) -> RerankRunner:
    """The process's `RerankRunner` (model loaded once; one semaphore for every chain),
    warmed up with one batch when it is created."""
    scorer = load_cross_encoder(settings)
    key = (id(scorer), settings.rerank_batch_size, settings.rerank_max_concurrency)
    with _RUNNER_LOCK:
        runner = _RUNNERS.get(key)
        if runner is None:
            runner = RerankRunner(
                scorer,
                batch_size=settings.rerank_batch_size,
                max_concurrency=settings.rerank_max_concurrency,
            )
            runner.warm_up()
            logger.info(
                "rerank runner: %d slots, batch %d, warm-up %.2f s (%.0f ms per pair)",
                runner.max_concurrency,
                runner.batch_size,
                runner.warmup_s,
                (runner.per_pair_s or 0.0) * 1000.0,
            )
            _RUNNERS[key] = runner
        return runner


def reset_rerank_runners() -> None:
    """Forget the process's model and runners (tests)."""
    with _RUNNER_LOCK:
        for runner in _RUNNERS.values():
            runner.close()
        _RUNNERS.clear()
    with _LOAD_LOCK:
        _SCORERS.clear()


# ---- LlamaIndex postprocessor and the eval adapter ---------------------------------------------


class BatchedTimeoutRerank(BaseNodePostprocessor):
    """Rerank the first `top_n` nodes with the shared runner (see the module doc).

    With a `HybridQuery` the request's deadline applies and the trace gets the reranker's own
    list, `rerank_timeout` and the queue time; with a plain `QueryBundle` every candidate is
    scored.
    """

    timing_name: ClassVar[str] = "rerank"

    top_n: int = Field(default=30, gt=0, description="Candidates sent to the reranker.")
    _runner: RerankRunner = PrivateAttr()

    def __init__(self, *, runner: RerankRunner, top_n: int = 30, **kwargs: Any) -> None:
        super().__init__(top_n=top_n, **kwargs)
        self._runner = runner

    @property
    def runner(self) -> RerankRunner:
        return self._runner

    @classmethod
    def class_name(cls) -> str:
        return "BatchedTimeoutRerank"

    def _postprocess_nodes(
        self, nodes: list[NodeWithScore], query_bundle: QueryBundle | None = None
    ) -> list[NodeWithScore]:
        if query_bundle is None:
            raise ValueError("BatchedTimeoutRerank needs the question (query_bundle)")
        query = query_bundle if isinstance(query_bundle, HybridQuery) else None
        window = list(nodes[: self.top_n])
        texts = [
            rerank_text(
                str(item.node.metadata.get("title_path", "") or ""),
                item.node.get_content(metadata_mode=MetadataMode.NONE),
            )
            for item in window
        ]
        run = self._runner.rerank(
            query_bundle.query_str, texts, query.deadline if query is not None else None
        )
        count = len(run.scores)
        for item, score in zip(window, run.scores, strict=False):
            item.node.metadata[RERANK_SCORE] = score
        for item in window[count:]:
            item.node.metadata.pop(RERANK_SCORE, None)
        # sorted() is stable: equal rerank scores keep the fusion order.
        order = sorted(range(count), key=lambda i: -run.scores[i])
        scored = [window[i] for i in order]
        if query is not None:
            trace = query.trace
            trace.rerank_candidates = len(window)
            trace.rerank_timeout = run.timed_out
            trace.rerank = [
                RouteHit(
                    chunk_id=item.node.node_id,
                    doc_id=str(item.node.metadata.get("doc_id", "")),
                    rank=rank,
                    score=item.node.metadata[RERANK_SCORE],
                )
                for rank, item in enumerate(scored, start=1)
            ]
            trace.timings_ms["rerank_queue"] = run.queue_ms
        if run.timed_out:
            logger.warning(
                "rerank_timeout: %d of %d candidates scored (queue %.0f ms, scoring %.0f ms)",
                count,
                len(window),
                run.queue_ms,
                run.score_ms,
            )
        return scored + window[count:]


class ChunkReranker:
    """The P0-10 `Reranker` protocol over a runner: `(question, chunks) -> chunks` sorted by
    rerank score (set on each returned copy), no deadline. Used by the rerank latency curve."""

    def __init__(self, runner: RerankRunner) -> None:
        self.runner = runner

    def __call__(self, question: str, candidates: Sequence[Chunk]) -> list[Chunk]:
        texts = [rerank_text(c.title_path, c.text) for c in candidates]
        run = self.runner.rerank(question, texts)
        scored = [
            c.model_copy(update={"rerank_score": s})
            for c, s in zip(candidates, run.scores, strict=False)
        ]
        order = sorted(range(len(scored)), key=lambda i: -run.scores[i])
        return [scored[i] for i in order] + list(candidates[len(scored) :])
