"""Spec P1-04: answer runner, format compliance, cost projection, security answers."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest
from langchain_core.messages import AIMessage, BaseMessage

from rag.answer.prompts import NOT_FOUND
from rag.config import REPO_ROOT, ModelPrice, load_kb_config
from rag.cost import CostLedger
from rag.eval.answer import (
    AnswerItem,
    RetrieverContexts,
    format_compliance,
    project_cost,
    run_answer_eval,
)
from rag.eval.answer.security import PlantedContexts, security_items, security_report
from rag.eval.retrieval.record import Provenance
from rag.eval.security import load_security_set
from rag.testing import FakeRetriever, StreamingFakeChatModel
from rag.types import Chunk, RetrievalResult

KB_FILE = REPO_ROOT / "kb" / "public" / "hf_docs.yaml"
PROVENANCE = Provenance(git_commit="abc", git_dirty=False, lock_sha256="x", models={})


def _cfg() -> Any:
    return load_kb_config(KB_FILE, kb_root=REPO_ROOT / "kb")


def _chunk(n: int, doc: str = "hf_docs:a.md") -> Chunk:
    return Chunk(
        chunk_id=f"{doc}#{n}",
        doc_id=doc,
        kb="hf_docs",
        text=f"Fact number {n} about the library.",
        title_path=["Guide"],
        source_rel=doc.split(":", 1)[1],
        score=1.0 / n,
        rerank_score=0.9,
    )


def _result(chunks: list[Chunk], **kw: Any) -> RetrievalResult:
    return RetrievalResult(request_id="r", kb="hf_docs", chunks=chunks, **kw)


class LedgerFakeChat(StreamingFakeChatModel):
    """Streams scripted replies and writes one ledger line per call, like the real gateway."""

    ledger_path: str = ""
    prompt_tokens: int = 1000
    completion_tokens: int = 300

    def _next_message(self, messages: list[BaseMessage]) -> AIMessage:
        message = super()._next_message(messages)
        ledger = CostLedger(
            self.ledger_path,
            prices={"fake-chat": ModelPrice(input_cny_per_m=2.0, output_cny_per_m=8.0)},
            budget_stop_cny=28.0,
        )
        ledger.record(
            kind="chat",
            model="fake-chat",
            prompt_tokens=self.prompt_tokens,
            completion_tokens=self.completion_tokens,
            reasoning_tokens=200,
        )
        return message


def _items() -> list[AnswerItem]:
    return [
        AnswerItem(
            qid="q1",
            question="What is fact one?",
            reference="Fact one.",
            relevant_pages=["hf_docs:a.md"],
        ),
        AnswerItem(
            qid="q2",
            question="What is fact two?",
            reference="Fact two.",
            relevant_pages=["hf_docs:b.md"],
        ),
    ]


def _run(items: list[AnswerItem], retriever: FakeRetriever, model: Any) -> Any:
    cfg = _cfg()
    return asyncio.run(
        run_answer_eval(
            items,
            RetrieverContexts(retriever, kb=cfg.kb, top_k=cfg.retrieval.final_k),
            kb_config=cfg,
            model=model,
            label="test",
            provenance=PROVENANCE,
            chat_model="fake-chat",
        )
    )


def test_runner_records_answer_citations_usage(tmp_path: Path) -> None:
    retriever = FakeRetriever(
        {
            "What is fact one?": _result([_chunk(1), _chunk(2), _chunk(3, "hf_docs:c.md")]),
            "What is fact two?": _result([_chunk(1, "hf_docs:b.md")]),
        }
    )
    model = LedgerFakeChat(
        responses=[
            "Fact one is first [1]. It has a second part [3]. Also [9] this.",
            "Fact two [1].",
        ],
        ledger_path=str(tmp_path / "ledger.jsonl"),
    )
    run = _run(_items(), retriever, model)
    assert [q.qid for q in run.questions] == ["q1", "q2"]
    q1 = run.questions[0]
    assert q1.found is True
    assert q1.text.startswith("Fact one is first")
    assert q1.context_chunk_ids == ["hf_docs:a.md#1", "hf_docs:a.md#2", "hf_docs:c.md#3"]
    assert q1.cited_chunk_ids == ["hf_docs:a.md#1", "hf_docs:c.md#3"]
    assert q1.cited_pages == ["hf_docs:a.md", "hf_docs:c.md"]
    assert q1.cites_relevant_page is True
    assert q1.removed_markers == [9]
    assert q1.unsupported_sentences == 1
    assert q1.sentences == 3
    assert q1.parse_attempts == 1 and q1.answer_parse_failed is False
    assert q1.usage.calls == 1
    assert q1.usage.prompt_tokens == 1000 and q1.usage.completion_tokens == 300
    assert q1.usage.reasoning_tokens == 200
    assert q1.usage.cny == pytest.approx((1000 * 2 + 300 * 8) / 1e6)
    assert q1.retrieval_ms >= 0 and q1.generation_ms >= 0 and q1.total_ms >= q1.generation_ms
    assert run.usage.calls == 2
    assert run.usage.cny == pytest.approx(2 * (1000 * 2 + 300 * 8) / 1e6)
    assert run.chat_model == "fake-chat"
    assert run.git_commit == "abc"
    assert run.compliance.n_questions == 2
    assert run.questions[1].cites_relevant_page is True


def test_runner_not_found_and_retrieval_error(tmp_path: Path) -> None:
    items = [
        *_items(),
        AnswerItem(qid="q3", question="Third?", reference="Three.", relevant_pages=[]),
        AnswerItem(qid="q4", question="Fourth?", reference="Four.", relevant_pages=[]),
    ]
    retriever = FakeRetriever(
        {
            "What is fact one?": _result([], not_found=True),
            "What is fact two?": RuntimeError("es down"),
            "Third?": _result([_chunk(1)], degraded=True),
            "Fourth?": _result([_chunk(1)]),
        }
    )
    model = LedgerFakeChat(
        responses=[NOT_FOUND["en"], "no marker here", "still none"],
        ledger_path=str(tmp_path / "ledger.jsonl"),
    )
    run = _run(items, retriever, model)
    q1, q2, q3, q4 = run.questions
    assert q1.retrieval_not_found and q1.found is False and q1.usage.calls == 0
    assert q2.error is not None and "es down" in q2.error and q2.answer is None
    assert q3.found is False and q3.degraded is True and q3.answer is not None
    assert q3.answer.degraded is True  # copied from the retrieval result (P1-02 open issue)
    assert q4.answer_parse_failed is True and q4.parse_attempts == 2
    assert q4.usage.calls == 2
    assert len(model.calls) == 3


def test_format_compliance_counts_and_intervals(tmp_path: Path) -> None:
    questions = {f"Q{i}?": _result([_chunk(1), _chunk(2)]) for i in range(5)}
    retriever = FakeRetriever(questions)
    replies = [
        "All cited [1]. Second cited [2].",  # fully cited
        "Cited [1]. Not cited.",  # one unsupported sentence
        "Cited [1] and unknown [7].",  # unknown number
        NOT_FOUND["en"],  # found=false
        "no marker",  # parse failed after retry
        "no marker again",
    ]
    model = LedgerFakeChat(responses=replies, ledger_path=str(tmp_path / "l.jsonl"))
    items = [AnswerItem(qid=f"q{i}", question=f"Q{i}?", reference="r") for i in range(5)]
    run = _run(items, retriever, model)
    c = format_compliance(run.questions)
    assert c.n_questions == 5
    assert (c.answered.count, c.answered.n) == (3, 5)
    assert (c.found_false.count, c.found_false.n) == (2, 5)
    assert (c.fully_cited.count, c.fully_cited.n) == (2, 3)
    assert (c.with_unknown_markers.count, c.with_unknown_markers.n) == (1, 5)
    assert c.unknown_markers_total == 1
    assert (c.unsupported_sentences.count, c.unsupported_sentences.n) == (1, 5)
    assert (c.with_unsupported.count, c.with_unsupported.n) == (1, 3)
    assert (c.parse_failed.count, c.parse_failed.n) == (1, 5)
    assert (c.retried.count, c.retried.n) == (1, 5)
    assert c.fully_cited.rate == pytest.approx(2 / 3)
    low, high = c.fully_cited.low, c.fully_cited.high
    assert 0 < low < 2 / 3 < high < 1
    assert c == run.compliance


def test_project_cost_from_pilot(tmp_path: Path) -> None:
    retriever = FakeRetriever({f"Q{i}?": _result([_chunk(1)]) for i in range(2)})
    model = LedgerFakeChat(responses=["A [1].", "B [1]."], ledger_path=str(tmp_path / "l.jsonl"))
    items = [AnswerItem(qid=f"q{i}", question=f"Q{i}?", reference="r") for i in range(2)]
    run = _run(items, retriever, model)
    per_answer = (1000 * 2 + 300 * 8) / 1e6
    p = project_cost(
        run, questions=65, judge_cny_per_call=0.001, judge_calls_per_answer=4, security_questions=30
    )
    assert p.per_answer_cny == pytest.approx(per_answer)
    assert p.answers_cny == pytest.approx(65 * per_answer)
    assert p.judge_cny == pytest.approx(65 * 4 * 0.001)
    assert p.security_cny == pytest.approx(30 * per_answer)
    assert p.total_cny == pytest.approx(p.answers_cny + p.judge_cny + p.security_cny)
    assert p.mean_prompt_tokens == pytest.approx(1000)
    assert p.mean_completion_tokens == pytest.approx(300)


def test_security_answers_scored(tmp_path: Path) -> None:
    sset = load_security_set()
    cfg = _cfg()
    items = security_items(sset)[:2]
    assert items[0].qid == sset.questions[0].qid
    contexts = PlantedContexts(sset, cfg, tokenizer=str.split)
    q0, q1 = sset.questions[0], sset.questions[1]
    followed = sset.pages[q1.doc].injection.followed_example
    model = LedgerFakeChat(
        responses=[f"{q0.answer.rstrip('.')} [1].", followed],
        ledger_path=str(tmp_path / "l.jsonl"),
    )
    run = asyncio.run(
        run_answer_eval(
            items,
            contexts,
            kb_config=cfg,
            model=model,
            label="security",
            provenance=PROVENANCE,
            chat_model="fake-chat",
            kind="security",
        )
    )
    assert run.kind == "security"
    assert all(q.context_chunk_ids for q in run.questions)
    assert run.questions[0].relevant_pages == [run.questions[0].context_chunk_ids[0].split("#")[0]]
    report = security_report(sset, run)
    assert report.missing == len(sset.questions) - 2
    by_qid = {s.qid: s for s in report.scores}
    assert by_qid[q0.qid].outcome == "compliant"
    assert by_qid[q1.qid].outcome == "violated"
