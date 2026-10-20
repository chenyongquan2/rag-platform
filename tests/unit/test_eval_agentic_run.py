"""Spec P3-05 AC3-AC6: running the two systems (standard RAG, the ask_docs graph) over items.

Scripted retrieval and fake chat models that write ledger lines; nothing calls a network.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from rag.agent import AgentRuntime
from rag.eval.agentic.records import SystemRecord, done_qids, read_records
from rag.eval.agentic.runner import Tracer, run_agentic, run_standard
from rag.eval.agentic.sets import EvalItem
from rag.eval.answer.runner import RetrieverContexts
from rag.observe import shutdown_observability
from rag.testing.agent import (
    ScriptedSearchBackend,
    agent_settings,
    judge_reply,
    kb_config,
    multi_query_reply,
    pages_result,
    scored_result,
)
from rag.testing.ledger_chat import LedgerChat, LedgerStreamingChat
from rag.types import AnswerStatus

KB = "hf_docs"
BELOW = scored_result(KB, [0.1], pages=[9])


@pytest.fixture(autouse=True)
def _no_active_observability() -> Iterator[None]:
    shutdown_observability()
    yield
    shutdown_observability()


def _item(qid: str, question: str, gold: list[str] | None = None, **kw: Any) -> EvalItem:
    pages = gold if gold is not None else []
    fields: dict[str, Any] = {
        "qid": qid,
        "qtype": "single_hop" if pages else "no_answer",
        "set": "x",
        "question": question,
        "reference": "ref" if pages else "",
        "relevant_pages": pages,
        "answerable": bool(pages),
    }
    return EvalItem(**{**fields, **kw})


def _answer_model(tmp_path: Path, *replies: str) -> LedgerStreamingChat:
    return LedgerStreamingChat(
        responses=list(replies),
        default_response="Use it [1].",
        ledger_path=str(tmp_path / "ledger.jsonl"),
    )


def _ledger_model(tmp_path: Path, profile: str, *replies: Any) -> LedgerChat:
    return LedgerChat(
        responses=list(replies), ledger_path=str(tmp_path / "ledger.jsonl"), profile=profile
    )


# ---- standard RAG -------------------------------------------------------------------------------


def test_run_standard_records(tmp_path: Path) -> None:
    backend = ScriptedSearchBackend(
        KB,
        by_question={
            "How to push?": scored_result(KB, [0.9, 0.8, 0.7], pages=[1, 2, 3]),
            "Where is X?": BELOW,  # nothing above the threshold: not found, no model call
        },
    )
    model = _answer_model(tmp_path, "Use push_to_hub [1].")
    items = [
        _item("q1", "How to push?", [f"{KB}:p1.md"]),
        _item("q2", "Where is X?", [f"{KB}:p9.md"]),
    ]
    out = tmp_path / "standard.jsonl"
    stats = asyncio.run(
        run_standard(
            items,
            records_path=out,
            contexts=RetrieverContexts(backend, kb=KB, top_k=8),
            kb_config=kb_config(KB),
            model=model,
        )
    )
    assert (stats.ran, stats.skipped, stats.pending) == (2, 0, 0)
    first, second = read_records(out)
    assert (first.system, first.qid, first.qtype) == ("standard", "q1", "single_hop")
    assert first.answered and first.found and first.status == "answered"
    assert first.text == "Use push_to_hub."
    assert first.context_pages == [f"{KB}:p1.md", f"{KB}:p2.md", f"{KB}:p3.md"]
    assert first.cited_pages == [f"{KB}:p1.md"]
    assert (first.sentences, first.unsupported_sentences) == (1, 0)
    # One chat call, written to the ledger by the fake: tokens, CNY and profile come from it.
    assert first.usage.calls == 1 and first.usage.cny > 0
    assert first.usage_by_profile["online"].calls == 1
    assert first.total_ms >= first.steps_ms["retrieval"] >= 0
    assert first.rounds == 0 and first.round_info == []
    # Nothing passed the threshold: the fixed not-found answer, no model call.
    assert (second.found, second.answered, second.status) == (False, False, "not_found")
    assert second.retrieval_not_found and second.usage.calls == 0 and second.context_pages == []
    assert len(model.calls) == 1


def test_run_standard_retrieval_error(tmp_path: Path) -> None:
    backend = ScriptedSearchBackend(KB, raises=RuntimeError("es down"))
    out = tmp_path / "standard.jsonl"
    stats = asyncio.run(
        run_standard(
            [_item("q1", "How to push?", [f"{KB}:p1.md"]), _item("q2", "Other?")],
            records_path=out,
            contexts=RetrieverContexts(backend, kb=KB, top_k=8),
            kb_config=kb_config(KB),
            model=_answer_model(tmp_path),
        )
    )
    assert stats.ran == 2  # the failure of the first did not end the run
    records = read_records(out)
    assert all(r.error and "es down" in r.error and not r.answered for r in records)
    assert records[0].status == "error"


# ---- the ask_docs graph -------------------------------------------------------------------------


def _runtime(tracer: Tracer) -> AgentRuntime:
    return AgentRuntime.from_settings(agent_settings(), hooks=tracer.hooks())


def test_run_agentic_records_rounds_and_hooks(tmp_path: Path) -> None:
    backend = ScriptedSearchBackend(
        KB,
        by_question={
            "gamma delta": scored_result(KB, [0.5], pages=[7]),  # passes, but not the page
            "gamma epsilon": scored_result(KB, [0.8], pages=[3]),  # the gold page
            "gamma zeta": scored_result(KB, [0.2], pages=[5]),  # below the threshold
        },
    )
    helper = _ledger_model(
        tmp_path, "cheap", multi_query_reply((1, ["gamma epsilon", "gamma zeta"]))
    )
    judge = _ledger_model(tmp_path, "cheap", judge_reply(False), judge_reply(True))
    answer = _answer_model(tmp_path, "Use it [1].")
    tracer = Tracer()
    out = tmp_path / "agentic.jsonl"
    cfg = kb_config(KB, judge=True, rewrite=True, declarative=False)
    asyncio.run(
        run_agentic(
            [_item("q1", "gamma delta", [f"{KB}:p3.md"])],
            records_path=out,
            runtime=_runtime(tracer),
            retriever=backend,
            kb_config=cfg,
            tracer=tracer,
            model=answer,
            helper_model=helper,
            judge_model=judge,
        )
    )
    (rec,) = read_records(out)
    assert rec.system == "agentic" and rec.answered and rec.found
    assert (rec.rounds, rec.sufficiency, rec.stop_reason) == (2, "enough", None)
    assert not rec.coverage_unconfirmed and not rec.degraded and rec.fallbacks == []
    # Every search of every round, with the pages it returned (below-threshold ones are not).
    assert [(r.round, r.query or r.sub_question, r.pages, r.not_found) for r in rec.round_info] == [
        (1, "gamma delta", [f"{KB}:p7.md"], False),
        (2, "gamma epsilon", [f"{KB}:p3.md"], False),
        (2, "gamma zeta", [], True),
    ]
    assert [r.kind for r in rec.round_info] == ["original", "multi_query", "multi_query"]
    assert f"{KB}:p3.md" in rec.context_pages
    # What the hooks decided, per round.
    verdicts = [(e.round, e.data["verdict"]) for e in rec.events if e.step == "judge"]
    assert verdicts == [(1, "partly_enough"), (2, "enough")]
    rewrites = [e for e in rec.events if e.step == "rewrite"]
    assert [e.round for e in rewrites] == [1, 2]
    assert rewrites[1].data["technique"] == "multi_query"
    assert rewrites[1].data["queries"] == ["gamma epsilon", "gamma zeta"]
    # Model calls by profile: rewrite + two judge calls on the cheap profile, one answer online.
    assert rec.usage_by_profile["cheap"].calls == 3 and rec.usage_by_profile["online"].calls == 1
    assert rec.usage.calls == 4 and rec.usage.cny > 0
    assert set(rec.steps_ms) >= {"route", "rewrite", "retrieve", "judge", "answer"}
    assert rec.tokens_used > 0 and rec.total_ms > 0


def test_agentic_clarification_is_refusal(tmp_path: Path) -> None:
    backend = ScriptedSearchBackend(KB)
    tracer = Tracer()
    out = tmp_path / "agentic.jsonl"
    asyncio.run(
        run_agentic(
            [_item("q1", "How do I fix this error?", [f"{KB}:p1.md"])],
            records_path=out,
            runtime=_runtime(tracer),
            retriever=backend,
            kb_config=kb_config(KB, clarify=True),
            tracer=tracer,
            model=_answer_model(tmp_path),
            helper_model=_ledger_model(tmp_path, "cheap"),
        )
    )
    (rec,) = read_records(out)
    assert rec.status == AnswerStatus.NEED_CLARIFICATION.value
    assert rec.clarification and not rec.found and not rec.answered
    assert rec.text == ""  # nothing the judge could read as an answer
    assert rec.clarify_question and rec.rounds == 0
    assert backend.calls == []  # the graph was not resumed, nothing was retrieved


def test_agentic_error_is_recorded_and_run_goes_on(tmp_path: Path) -> None:
    backend = ScriptedSearchBackend(KB, raises=RuntimeError("es down"))
    tracer = Tracer()
    out = tmp_path / "agentic.jsonl"
    stats = asyncio.run(
        run_agentic(
            [_item("q1", "gamma delta", [f"{KB}:p1.md"]), _item("q2", "alpha beta")],
            records_path=out,
            runtime=_runtime(tracer),
            retriever=backend,
            kb_config=kb_config(KB),
            tracer=tracer,
            model=_answer_model(tmp_path),
        )
    )
    assert stats.ran == 2
    records = read_records(out)
    assert all(r.status == "error" and r.error and not r.answered for r in records)


# ---- resume and the spend limit -----------------------------------------------------------------


def test_resume_skips_finished(tmp_path: Path) -> None:
    backend = ScriptedSearchBackend(KB, default=pages_result(KB, [1, 2]))
    items = [_item(f"q{i}", f"Question {i}?", [f"{KB}:p1.md"]) for i in (1, 2, 3)]
    out = tmp_path / "standard.jsonl"

    def run(chosen: list[EvalItem], model: LedgerStreamingChat) -> Any:
        return asyncio.run(
            run_standard(
                chosen,
                records_path=out,
                contexts=RetrieverContexts(backend, kb=KB, top_k=8),
                kb_config=kb_config(KB),
                model=model,
            )
        )

    run(items[:1], _answer_model(tmp_path))
    assert done_qids(out) == {"q1"}
    model = _answer_model(tmp_path)
    stats = run(items, model)
    assert (stats.ran, stats.skipped) == (2, 1)
    assert len(model.calls) == 2  # only the two new questions were answered
    assert [r.qid for r in read_records(out)] == ["q1", "q2", "q3"]


def test_a_cut_off_last_line_is_ignored_and_redone(tmp_path: Path) -> None:
    backend = ScriptedSearchBackend(KB, default=pages_result(KB, [1]))
    items = [_item(f"q{i}", f"Question {i}?", [f"{KB}:p1.md"]) for i in (1, 2)]
    out = tmp_path / "standard.jsonl"
    asyncio.run(
        run_standard(
            items[:1],
            records_path=out,
            contexts=RetrieverContexts(backend, kb=KB, top_k=8),
            kb_config=kb_config(KB),
            model=_answer_model(tmp_path),
        )
    )
    with out.open("a", encoding="utf-8", newline="") as fh:
        fh.write('{"system": "standard", "qid": "q2", "ques')  # a crash in mid-write
    assert done_qids(out) == {"q1"}
    stats = asyncio.run(
        run_standard(
            items,
            records_path=out,
            contexts=RetrieverContexts(backend, kb=KB, top_k=8),
            kb_config=kb_config(KB),
            model=_answer_model(tmp_path),
        )
    )
    assert (stats.ran, stats.skipped) == (1, 1)
    assert [r.qid for r in read_records(out)] == ["q1", "q2"]
    assert all(json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()[:2])


def test_spend_limit_stops_run(tmp_path: Path) -> None:
    backend = ScriptedSearchBackend(KB, default=pages_result(KB, [1, 2]))
    items = [_item(f"q{i}", f"Question {i}?", [f"{KB}:p1.md"]) for i in (1, 2, 3, 4)]
    out = tmp_path / "standard.jsonl"
    model = LedgerStreamingChat(
        responses=["Use it [1]."] * 4,
        ledger_path=str(tmp_path / "ledger.jsonl"),
        prompt_tokens=100_000,  # 0.2 + 0.0024 CNY per answer
    )
    stats = asyncio.run(
        run_standard(
            items,
            records_path=out,
            contexts=RetrieverContexts(backend, kb=KB, top_k=8),
            kb_config=kb_config(KB),
            model=model,
            max_cny=0.3,
        )
    )
    # 0.2 spent after the first question (< 0.3): the second runs; 0.4 >= 0.3: stop.
    assert (stats.ran, stats.pending, stats.stopped) == (2, 2, "budget")
    assert stats.spent_cny == pytest.approx(0.4048, abs=1e-3)
    assert [r.qid for r in read_records(out)] == ["q1", "q2"]
    # Spend already counted elsewhere (the other system, the judge) is part of the limit.
    stats = asyncio.run(
        run_standard(
            items,
            records_path=out,
            contexts=RetrieverContexts(backend, kb=KB, top_k=8),
            kb_config=kb_config(KB),
            model=model,
            max_cny=0.5,
            spent_before=0.2,
        )
    )
    assert (stats.ran, stats.pending, stats.stopped) == (0, 2, "budget")
    assert isinstance(
        SystemRecord.model_validate_json(out.read_text("utf-8").splitlines()[0]), SystemRecord
    )


def test_spend_limit_boundary_is_inclusive(tmp_path: Path) -> None:
    """The run stops once the spend has *reached* the limit, not only when it passed it."""
    backend = ScriptedSearchBackend(KB, default=pages_result(KB, [1]))
    items = [_item("q1", "Question 1?", [f"{KB}:p1.md"])]

    def run(limit: float) -> Any:
        return asyncio.run(
            run_standard(
                items,
                records_path=tmp_path / f"standard-{limit}.jsonl",
                contexts=RetrieverContexts(backend, kb=KB, top_k=8),
                kb_config=kb_config(KB),
                model=_answer_model(tmp_path),
                max_cny=limit,
                spent_before=0.5,
            )
        )

    assert (run(0.5).ran, run(0.5).stopped) == (0, "budget")  # exactly at the limit: stop
    assert run(0.5000001).ran == 1  # just below it: go on


def test_retry_errors_reruns_only_the_errored_questions(tmp_path: Path) -> None:
    backend = ScriptedSearchBackend(KB, default=pages_result(KB, [1]))
    items = [_item(f"q{i}", f"Question {i}?", [f"{KB}:p1.md"]) for i in (1, 2, 3)]
    out = tmp_path / "standard.jsonl"
    failing = RetrieverContexts(
        ScriptedSearchBackend(KB, raises=RuntimeError("es down")), kb=KB, top_k=8
    )
    asyncio.run(
        run_standard(
            items[:1], records_path=out, contexts=failing, kb_config=kb_config(KB),
            model=_answer_model(tmp_path),
        )
    )  # fmt: skip
    asyncio.run(
        run_standard(
            items[1:], records_path=out,
            contexts=RetrieverContexts(backend, kb=KB, top_k=8),
            kb_config=kb_config(KB), model=_answer_model(tmp_path),
        )
    )  # fmt: skip
    assert [r.status for r in read_records(out)] == ["error", "answered", "answered"]

    def again(retry: bool) -> Any:
        return asyncio.run(
            run_standard(
                items, records_path=out, contexts=RetrieverContexts(backend, kb=KB, top_k=8),
                kb_config=kb_config(KB), model=_answer_model(tmp_path), retry_errors=retry,
            )
        )  # fmt: skip

    assert again(False).ran == 0  # an errored question counts as done unless asked
    stats = again(True)
    assert (stats.ran, stats.skipped) == (1, 2)
    records = read_records(out)
    assert [r.qid for r in records] == ["q2", "q3", "q1"] and all(
        r.status == "answered" for r in records
    )


def test_usage_by_profile_names_embedding_and_unprofiled_calls() -> None:
    from rag.eval.agentic.records import usage_by_profile

    rows = [
        {"kind": "chat", "profile": "cheap", "cny": 0.1, "prompt_tokens": 10},
        {"kind": "chat", "profile": "online", "cny": 0.2},
        {"kind": "chat", "cny": 0.3},  # a call with no profile
        {"kind": "embedding", "profile": "ignored", "cny": 0.01},
        {"kind": "embedding", "cny": 0.01},
    ]
    result = usage_by_profile(rows)
    assert sorted(result) == ["chat", "cheap", "embedding", "online"]
    assert result["embedding"].calls == 2 and result["chat"].calls == 1
    assert result["cheap"].prompt_tokens == 10
