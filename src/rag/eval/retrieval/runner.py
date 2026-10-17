"""Run a retriever over a question set and write the run record (design s14-1, s14-4, s14-6).

record = run_retrieval_eval(retriever, load_question_set(path, kb=cfg.kb), kb_config=cfg)
write_run(record, public=is_public_kb_file(cfg_path, kb_root))
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import time
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from rag.config import REPO_ROOT, KnowledgeBaseConfig, ServiceSettings, repo_path
from rag.eval.questions import LabeledQuestion, QuestionSet, content_sha256
from rag.eval.retrieval.metrics import (
    Level,
    cap_ks,
    compute_metrics,
    first_relevant_rank,
    is_binary_metric,
    metric_names,
    ranked_ids,
)
from rag.eval.retrieval.protocol import Retriever
from rag.eval.retrieval.record import (
    FINAL,
    HitRecord,
    LatencySummary,
    LevelSummary,
    Provenance,
    QuestionRecord,
    QuestionSetInfo,
    RunInfo,
    RunRecord,
)
from rag.eval.retrieval.report import render_run_report
from rag.eval.stats import wilson_interval
from rag.types import Chunk, RetrievalResult, RouteHit

DEFAULT_KS: tuple[int, ...] = (1, 3, 5, 10)
REPORTS_ROOT = "evals/reports"
_LABEL_RE = re.compile(r"[^a-z0-9_-]+")


def make_run_id(kb: str, label: str = "", now: datetime | None = None) -> str:
    """`<UTC time>_<kb>[_<label>]`, e.g. `20261017T093000Z_hf_docs_rerank20`: sorts by time."""
    now = (now or datetime.now(UTC)).astimezone(UTC)
    slug = _LABEL_RE.sub("-", label.lower()).strip("-")
    return f"{now:%Y%m%dT%H%M%SZ}_{kb}" + (f"_{slug}" if slug else "")


# Variables that point git at another repository (set e.g. inside git hooks); dropped so that
# `git -C <repo_root>` always describes `repo_root` (`git rev-parse --local-env-vars`).
_GIT_REPO_ENV = {
    "GIT_ALTERNATE_OBJECT_DIRECTORIES",
    "GIT_COMMON_DIR",
    "GIT_DIR",
    "GIT_GRAFT_FILE",
    "GIT_IMPLICIT_WORK_TREE",
    "GIT_INDEX_FILE",
    "GIT_NO_REPLACE_OBJECTS",
    "GIT_OBJECT_DIRECTORY",
    "GIT_PREFIX",
    "GIT_REPLACE_REF_BASE",
    "GIT_SHALLOW_FILE",
    "GIT_WORK_TREE",
}


def git_env() -> dict[str, str]:
    """The current environment without the variables that redirect git to another repo."""
    return {k: v for k, v in os.environ.items() if k not in _GIT_REPO_ENV}


def _git(repo_root: Path, *args: str) -> str | None:
    try:
        done = subprocess.run(
            ["git", "-C", str(repo_root), *args],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
            env=git_env(),
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout


def collect_provenance(
    *,
    settings: ServiceSettings | None = None,
    models: dict[str, str] | None = None,
    repo_root: Path = REPO_ROOT,
) -> Provenance:
    """Git commit and dirty flag, `uv.lock` hash, model ids (default: embed and reranker).

    Dirty means changed or untracked files anywhere except under `evals/reports/` (the reports
    are outputs of runs, not inputs).
    """
    settings = settings if settings is not None else ServiceSettings()
    commit = _git(repo_root, "rev-parse", "HEAD")
    status = _git(
        repo_root,
        "status",
        "--porcelain",
        "--untracked-files=all",
        "--",
        ".",
        f":(exclude){REPORTS_ROOT}",
    )
    lock = repo_root / "uv.lock"
    return Provenance(
        git_commit=commit.strip() if commit else None,
        git_dirty=bool(status.strip()) if status is not None else None,
        lock_sha256=content_sha256(lock.read_bytes()) if lock.is_file() else None,
        models=models
        if models is not None
        else {
            "embed": settings.embed_model,
            "embed_dim": str(settings.embed_dim),
            "reranker": settings.reranker_model,
        },
    )


def _relevant(question: LabeledQuestion, level: Level) -> list[str]:
    return question.relevant_pages if level == "page" else question.relevant_chunks


def _score_question(
    question: LabeledQuestion,
    result: RetrievalResult | None,
    *,
    lists: Sequence[str],
    top_k: int,
    ks: Sequence[int],
    latency_ms: float,
    error: str | None,
) -> QuestionRecord:
    chunks: list[Chunk] = result.chunks[:top_k] if result else []
    routes: dict[str, list[RouteHit]] = {
        name: sorted((result.routes.get(name, []) if result else []), key=lambda h: h.rank)
        for name in lists
        if name != FINAL
    }
    route_rank = {name: {h.chunk_id: h.rank for h in hits} for name, hits in routes.items()}
    hits = [
        HitRecord(
            chunk_id=c.chunk_id,
            doc_id=c.doc_id,
            score=c.score,
            route_ranks={name: ranks.get(c.chunk_id) for name, ranks in route_rank.items()},
        )
        for c in chunks
    ]
    metrics: dict[str, dict[str, dict[str, float]]] = {}
    first: dict[str, dict[str, int | None]] = {}
    for name in lists:
        items: Sequence[Chunk | RouteHit] = chunks if name == FINAL else routes[name]
        metrics[name], first[name] = {}, {}
        for level in ("page", "chunk"):
            relevant = set(_relevant(question, level))
            ranked = ranked_ids(items, level)
            values = compute_metrics(ranked, relevant, ks)
            if values is not None:
                metrics[name][level] = values
                first[name][level] = first_relevant_rank(ranked, relevant)
    return QuestionRecord(
        qid=question.qid,
        question=question.question,
        relevant_pages=question.relevant_pages,
        relevant_chunks=question.relevant_chunks,
        hits=hits,
        routes={name: hits[: max(ks)] for name, hits in routes.items()},
        metrics=metrics,
        first_relevant_rank=first,
        latency_ms=latency_ms,
        timings_ms=result.timings_ms if result else {},
        degraded=result.degraded if result else False,
        rerank_timeout=result.rerank_timeout if result else False,
        error=error,
    )


def _summarize(
    questions: Sequence[QuestionRecord], lists: Sequence[str], ks: Sequence[int]
) -> dict[str, dict[str, LevelSummary]]:
    summary: dict[str, dict[str, LevelSummary]] = {}
    for name in lists:
        summary[name] = {}
        for level in ("page", "chunk"):
            rows = [q.metrics[name][level] for q in questions if level in q.metrics[name]]
            if not rows:
                continue
            means = {m: float(np.mean([r[m] for r in rows])) for m in metric_names(ks)}
            wilson = {
                m: wilson_interval(round(sum(r[m] for r in rows)), len(rows))
                for m in metric_names(ks)
                if is_binary_metric(m)
            }
            summary[name][level] = LevelSummary(n=len(rows), means=means, hit_wilson=wilson)
    return summary


def latency_summary(latencies_ms: Sequence[float]) -> LatencySummary:
    """P50 and P95 by linear interpolation between the closest ranks."""
    values = np.asarray(latencies_ms, dtype=float)
    if values.size == 0:
        return LatencySummary(p50_ms=0.0, p95_ms=0.0, mean_ms=0.0, max_ms=0.0)
    p50, p95 = np.percentile(values, [50, 95])
    return LatencySummary(
        p50_ms=float(p50),
        p95_ms=float(p95),
        mean_ms=float(values.mean()),
        max_ms=float(values.max()),
    )


def _distinct[T](values: Sequence[T]) -> list[T]:
    return list(dict.fromkeys(values))


def run_retrieval_eval(
    retriever: Retriever,
    question_set: QuestionSet,
    *,
    kb_config: KnowledgeBaseConfig,
    top_k: int | None = None,
    ks: Sequence[int] = DEFAULT_KS,
    label: str = "",
    provenance: Provenance | None = None,
    clock: Callable[[], float] = time.perf_counter,
    now: datetime | None = None,
) -> RunRecord:
    """Ask every question once, in file order, and score the final list and every route.

    `top_k` defaults to the KB's `retrieval.final_k`; a k above `top_k` is capped at `top_k`
    with a warning (`cap_ks`), for the routes too, so every column is cut at the same depth. A
    retriever exception is recorded on the question (`error`) and the question scores 0
    everywhere; the run goes on. `provenance` defaults to `collect_provenance()`.
    """
    question_set.check_kb(kb_config.kb)
    top_k = top_k if top_k is not None else kb_config.retrieval.final_k
    ks = cap_ks(ks, top_k)
    now = (now or datetime.now(UTC)).astimezone(UTC)
    provenance = provenance if provenance is not None else collect_provenance()

    raw: list[tuple[LabeledQuestion, RetrievalResult | None, float, str | None]] = []
    for question in question_set.questions:
        start = clock()
        try:
            result: RetrievalResult | None = retriever.retrieve(
                question.question, kb=kb_config.kb, top_k=top_k
            )
            error = None
        except Exception as exc:  # recorded per question; one failure must not end the run
            result, error = None, f"{type(exc).__name__}: {exc}"
        raw.append((question, result, (clock() - start) * 1000.0, error))

    results = [r for _, r, _, _ in raw if r is not None]
    route_names = _distinct([name for r in results for name in r.routes])
    lists = [FINAL, *route_names]
    questions = [
        _score_question(q, r, lists=lists, top_k=top_k, ks=ks, latency_ms=ms, error=err)
        for q, r, ms, err in raw
    ]
    config = kb_config.model_dump(mode="json")
    config_json = json.dumps(config, sort_keys=True, separators=(",", ":"))
    info = RunInfo(
        **provenance.model_dump(),
        run_id=make_run_id(kb_config.kb, label, now),
        created_at=now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        kb=kb_config.kb,
        label=label,
        kb_config=config,
        kb_config_sha256=hashlib.sha256(config_json.encode("utf-8")).hexdigest(),
        index_versions=_distinct([r.index_version for r in results]),
        data_generations=_distinct([r.data_generation for r in results]),
        question_set=QuestionSetInfo(
            path=question_set.path, sha256=question_set.sha256, count=len(question_set.questions)
        ),
        top_k=top_k,
        ks=list(ks),
        lists=lists,
    )
    missed = [
        q.qid
        for q in questions
        if all(q.first_relevant_rank[name].get("page", 0) is None for name in lists)
    ]
    return RunRecord(
        run=info,
        summary=_summarize(questions, lists, ks),
        latency=latency_summary([q.latency_ms for q in questions]),
        counts={
            "questions": len(questions),
            "errors": sum(q.error is not None for q in questions),
            "degraded": sum(q.degraded for q in questions),
            "rerank_timeout": sum(q.rerank_timeout for q in questions),
        },
        missed_by_all_lists=missed,
        questions=questions,
    )


def run_dir(run_id: str, *, public: bool, reports_root: str | Path = REPORTS_ROOT) -> Path:
    """`<reports_root>/public/<run_id>` or the git-ignored `<reports_root>/private/<run_id>`."""
    return repo_path(reports_root) / ("public" if public else "private") / run_id


def write_run(record: RunRecord, *, public: bool, reports_root: str | Path = REPORTS_ROOT) -> Path:
    """Write `results.json` and `report.md` into a new run directory and return it.

    `public` must be decided by the caller from where the KB config lives (`kb/public/` or
    not, `rag.config.kb.is_public_kb_file`): reports of private knowledge bases quote their
    questions and doc ids, so they go to the git-ignored `private/` directory.
    """
    record = record.model_copy(update={"run": record.run.model_copy(update={"public": public})})
    out = run_dir(record.run.run_id, public=public, reports_root=reports_root)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.mkdir()  # FileExistsError: never overwrite an earlier run
    (out / "results.json").write_text(
        record.model_dump_json(indent=1) + "\n", encoding="utf-8", newline="\n"
    )
    (out / "report.md").write_text(render_run_report(record), encoding="utf-8", newline="\n")
    return out


def load_run(run: str | Path, *, reports_root: str | Path = REPORTS_ROOT) -> RunRecord:
    """Load a run from its directory, its `results.json`, or its run id under `reports_root`."""
    path = Path(run)
    candidates = [path, repo_path(path)]
    if len(path.parts) == 1:
        candidates += [
            run_dir(str(run), public=p, reports_root=reports_root) for p in (True, False)
        ]
    for candidate in candidates:
        file = candidate / "results.json" if candidate.is_dir() else candidate
        if file.is_file():
            return RunRecord.model_validate_json(file.read_text(encoding="utf-8"))
    raise FileNotFoundError(
        f"no eval run {str(run)!r} (looked in {reports_root}/public and private)"
    )
