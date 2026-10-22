"""Answer every question by the standard RAG path and record it (spec P1-04, design s14-1).

    contexts = RetrieverContexts(make_search_pipeline(...), kb="hf_docs", top_k=8)
    run = await run_answer_eval(items, contexts, kb_config=cfg, model=make_chat_model("online"))
    write_answer_run(run, out_dir)

Per question: the contexts source gives the chunks (the search pipeline: fusion, rerank,
threshold; or the planted page of a security question), `build_context` numbers them,
`generate_answer` answers. Questions run one after the other, so the latencies are those of
one request at a time, and every ledger line written meanwhile (chat, and query embeddings the
cache missed) is that question's.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from langchain_core.language_models import BaseChatModel

from rag.answer import build_context, generate_answer
from rag.config import KnowledgeBaseConfig
from rag.eval.answer.compliance import format_compliance
from rag.eval.answer.record import (
    AnswerItem,
    AnswerQuestionRecord,
    AnswerRunRecord,
    RunKind,
    Usage,
    usage_by_kind,
)
from rag.eval.answer.usage import collect_ledger_rows
from rag.eval.questions import QuestionSet
from rag.eval.retrieval.protocol import Retriever
from rag.eval.retrieval.record import Provenance
from rag.eval.retrieval.runner import collect_provenance, latency_summary, make_run_id
from rag.types import Chunk

REPORTS_ROOT = "data/reports/answer-eval"


@dataclass
class RetrievedContext:
    """What the answer step gets for one question."""

    chunks: list[Chunk]
    not_found: bool = False
    degraded: bool = False
    rerank_timeout: bool = False
    index_version: str = ""
    data_generation: int = 0
    timings_ms: dict[str, float] = field(default_factory=dict)


class ContextSource(Protocol):
    def __call__(self, item: AnswerItem) -> RetrievedContext: ...


class RetrieverContexts:
    """Chunks from a retriever: the search pipeline search_docs uses (P0-08, P0-13 settings)."""

    def __init__(self, retriever: Retriever, *, kb: str, top_k: int) -> None:
        self.retriever = retriever
        self.kb = kb
        self.top_k = top_k

    def __call__(self, item: AnswerItem) -> RetrievedContext:
        result = self.retriever.retrieve(item.question, kb=self.kb, top_k=self.top_k)
        return RetrievedContext(
            chunks=list(result.chunks),
            not_found=result.not_found,
            degraded=result.degraded,
            rerank_timeout=result.rerank_timeout,
            index_version=result.index_version,
            data_generation=result.data_generation,
            timings_ms=dict(result.timings_ms),
        )


class StoredContexts:
    """The chunks an earlier answer run gave each question, in the same order (spec P7-01).

    Two chat models answered from one stored run see the same fragments, so a difference between
    them is the model's, not the retrieval's (and no reranker runs). The answer step rebuilds the
    context from these chunks with the KB's budget, as it did for the stored run."""

    def __init__(self, run: AnswerRunRecord) -> None:
        self.run_id = run.run_id
        self._by_qid = {q.qid: q for q in run.questions}

    def __call__(self, item: AnswerItem) -> RetrievedContext:
        record = self._by_qid.get(item.qid)
        if record is None:
            raise KeyError(f"{item.qid}: not in the stored answer run {self.run_id}")
        answer = record.answer
        return RetrievedContext(
            chunks=list(answer.chunks) if answer is not None else [],
            not_found=record.retrieval_not_found,
            degraded=record.degraded,
            rerank_timeout=record.rerank_timeout,
            index_version=answer.index_version if answer is not None else "",
            data_generation=answer.data_generation if answer is not None else 0,
        )


def items_from_question_set(question_set: QuestionSet) -> list[AnswerItem]:
    return [
        AnswerItem(
            qid=q.qid, question=q.question, reference=q.answer, relevant_pages=q.relevant_pages
        )
        for q in question_set.questions
    ]


async def _answer_one(
    item: AnswerItem,
    contexts: ContextSource,
    *,
    kb_config: KnowledgeBaseConfig,
    model: BaseChatModel,
    clock: Callable[[], float],
) -> AnswerQuestionRecord:
    with collect_ledger_rows() as rows:
        start = clock()
        try:
            ctx = await asyncio.to_thread(contexts, item)
        except Exception as exc:  # recorded per question; one failure must not end the run
            elapsed = (clock() - start) * 1000.0
            return AnswerQuestionRecord.from_answer(
                qid=item.qid,
                question=item.question,
                reference=item.reference,
                relevant_pages=item.relevant_pages,
                answer=None,
                error=f"{type(exc).__name__}: {exc}",
                retrieval_ms=elapsed,
                total_ms=elapsed,
                usage=Usage.from_rows(rows),
                usage_by_kind=usage_by_kind(rows),
            )
        retrieved = clock()
        answer_context = build_context(
            [ctx.chunks], kb_config.answer.context_tokens, metadata_keys=kb_config.metadata.llm
        )
        answer = await generate_answer(
            item.question,
            answer_context,
            kb=kb_config.kb,
            config=kb_config.answer,
            model=model,
            request_id=f"answer-eval-{item.qid}",
        )
        done = clock()
        # The caller copies what retrieval knows into the answer (spec P1-02 open issue).
        answer = answer.model_copy(
            update={
                "degraded": ctx.degraded,
                "rerank_timeout": ctx.rerank_timeout,
                "index_version": ctx.index_version,
                "data_generation": ctx.data_generation,
                "timings_ms": {**ctx.timings_ms, **answer.timings_ms},
            }
        )
    return AnswerQuestionRecord.from_answer(
        qid=item.qid,
        question=item.question,
        reference=item.reference,
        relevant_pages=item.relevant_pages,
        answer=answer,
        error=None,
        retrieval_not_found=ctx.not_found,
        degraded=ctx.degraded,
        rerank_timeout=ctx.rerank_timeout,
        retrieval_ms=(retrieved - start) * 1000.0,
        generation_ms=(done - retrieved) * 1000.0,
        total_ms=(done - start) * 1000.0,
        usage=Usage.from_rows(rows),
        usage_by_kind=usage_by_kind(rows),
    )


async def run_answer_eval(
    items: Sequence[AnswerItem],
    contexts: ContextSource,
    *,
    kb_config: KnowledgeBaseConfig,
    model: BaseChatModel,
    chat_model: str,
    label: str = "",
    kind: RunKind = "hf",
    question_set: dict[str, object] | None = None,
    provenance: Provenance | None = None,
    clock: Callable[[], float] = time.perf_counter,
    now: datetime | None = None,
) -> AnswerRunRecord:
    """Answer every item once, in order, and summarize. `chat_model` is the model id recorded."""
    now = (now or datetime.now(UTC)).astimezone(UTC)
    provenance = provenance if provenance is not None else collect_provenance()
    records: list[AnswerQuestionRecord] = []
    for item in items:
        records.append(
            await _answer_one(item, contexts, kb_config=kb_config, model=model, clock=clock)
        )
    config = kb_config.model_dump(mode="json")
    config_json = json.dumps(config, sort_keys=True, separators=(",", ":"))
    total = Usage()
    for r in records:
        total = total + r.usage
    kinds: dict[str, Usage] = {}
    for r in records:
        for name, usage in r.usage_by_kind.items():
            kinds[name] = kinds.get(name, Usage()) + usage
    answered = [r for r in records if r.error is None]
    return AnswerRunRecord(
        **provenance.model_dump(),
        run_id=make_run_id(kb_config.kb, label, now),
        created_at=now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        kb=kb_config.kb,
        label=label,
        kind=kind,
        kb_config=config,
        kb_config_sha256=hashlib.sha256(config_json.encode("utf-8")).hexdigest(),
        chat_model=chat_model,
        question_set=dict(question_set or {"count": len(items)}),
        questions=records,
        compliance=format_compliance(records),
        usage=total,
        usage_by_kind=dict(sorted(kinds.items())),
        latency={
            "retrieval": latency_summary([r.retrieval_ms for r in records]),
            "generation": latency_summary([r.generation_ms for r in answered]),
            "total": latency_summary([r.total_ms for r in records]),
        },
    )


def render_answer_summary(run: AnswerRunRecord) -> str:
    c = run.compliance
    lines = [
        f"# Answer run {run.run_id}",
        "",
        f"- kb: {run.kb}, kind: {run.kind}, chat model: {run.chat_model} ({run.chat_profile})",
        f"- commit: {run.git_commit} (dirty: {run.git_dirty})",
        f"- questions: {c.n_questions}",
        f"- usage: {run.usage.calls} calls, prompt {run.usage.prompt_tokens}, completion "
        f"{run.usage.completion_tokens} (reasoning {run.usage.reasoning_tokens}), "
        f"{run.usage.cny:.4f} CNY",
        "",
        "| measure | value |",
        "|---|---|",
        f"| answered (found) | {c.answered.text()} |",
        f"| found=false | {c.found_false.text()} |",
        f"| every sentence cited, of answered | {c.fully_cited.text()} |",
        f"| every sentence cited, of all | {c.fully_cited_all.text()} |",
        f"| answers with unknown numbers | {c.with_unknown_markers.text()} |",
        f"| unknown numbers in total | {c.unknown_markers_total} |",
        f"| answers with an unsupported sentence | {c.with_unsupported.text()} |",
        f"| unsupported sentences | {c.unsupported_sentences.text()} |",
        f"| answer_parse_failed | {c.parse_failed.text()} |",
        f"| needed the retry | {c.retried.text()} |",
        f"| gateway errors | {c.generation_errors} |",
        f"| retrieval not_found | {c.retrieval_not_found} |",
        f"| retrieval errors | {c.retrieval_errors} |",
        "",
        "| step | P50 ms | P95 ms | max ms |",
        "|---|---|---|---|",
    ]
    for name, lat in run.latency.items():
        lines.append(f"| {name} | {lat.p50_ms:.0f} | {lat.p95_ms:.0f} | {lat.max_ms:.0f} |")
    return "\n".join(lines) + "\n"


def write_answer_run(run: AnswerRunRecord, out_root: str | Path) -> Path:
    """`<out_root>/answers/<run_id>/results.json` and `summary.md`; never overwrites a run."""
    out = Path(out_root) / "answers" / run.run_id
    out.parent.mkdir(parents=True, exist_ok=True)
    out.mkdir()
    (out / "results.json").write_text(
        run.model_dump_json(indent=1) + "\n", encoding="utf-8", newline="\n"
    )
    (out / "summary.md").write_text(render_answer_summary(run), encoding="utf-8", newline="\n")
    return out


def load_answer_run(path: str | Path) -> AnswerRunRecord:
    file = Path(path)
    if file.is_dir():
        file = file / "results.json"
    return AnswerRunRecord.model_validate_json(file.read_text(encoding="utf-8"))
