"""Spec P3-07: the 150-question re-run keeps what the citation check needs, and a new run of one
system is compared with an older run of the same system on the same questions.

Scripted retrieval and fake chat models; nothing calls a network.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from rag.agent import AgentRuntime
from rag.eval.agentic import cli
from rag.eval.agentic.records import (
    JudgeLine,
    SystemRecord,
    append_line,
    judge_file,
    read_records,
    system_file,
)
from rag.eval.agentic.runner import Tracer, run_agentic
from rag.eval.agentic.runs import compare_runs, render_compare_runs
from rag.eval.agentic.sets import EvalItem
from rag.eval.answer.record import Usage
from rag.eval.support.items import load_p3_05
from rag.observe import shutdown_observability
from rag.testing.agent import ScriptedSearchBackend, agent_settings, kb_config, scored_result
from rag.testing.ledger_chat import LedgerChat, LedgerStreamingChat

KB = "hf_docs"


@pytest.fixture(autouse=True)
def _no_active_observability() -> Iterator[None]:
    shutdown_observability()
    yield
    shutdown_observability()


# ---- AC1: the graph's record keeps sentence citations ---------------------------------------


def test_agentic_record_keeps_sentence_citations(tmp_path: Path) -> None:
    backend = ScriptedSearchBackend(
        KB, by_question={"How to push?": scored_result(KB, [0.99, 0.8], pages=[1, 2])}
    )
    ledger = str(tmp_path / "ledger.jsonl")
    answer = LedgerStreamingChat(
        responses=["Use push_to_hub [1]. It uploads the model [2]."], ledger_path=ledger
    )
    helper = LedgerChat(responses=[], ledger_path=ledger, profile="cheap")
    tracer = Tracer()
    out = tmp_path / "agentic" / "records.jsonl"
    item = EvalItem(
        qid="q1",
        qtype="single_hop",
        set="x",
        question="How to push?",
        reference="r",
        relevant_pages=[f"{KB}:p1.md"],
        answerable=True,
    )
    asyncio.run(
        run_agentic(
            [item],
            records_path=out,
            runtime=AgentRuntime.from_settings(agent_settings(), hooks=tracer.hooks()),
            retriever=backend,
            kb_config=kb_config(KB, judge=False),
            tracer=tracer,
            model=answer,
            helper_model=helper,
            judge_model=helper,
        )
    )
    (rec,) = read_records(out)
    assert rec.answered
    assert [s.text for s in rec.answer_sentences] == ["Use push_to_hub.", "It uploads the model."]
    cited = [c.chunk_id for s in rec.answer_sentences for c in s.citations]
    assert len(cited) == 2 and len(set(cited)) == 2
    assert [c.chunk_id for c in rec.cited_chunks] == cited  # first-cited order, each once
    assert all(c.text for c in rec.cited_chunks)
    # The P1-06 loader turns it into a judgeable item (the citation check runs on the graph too).
    (judgeable,) = [i for i in load_p3_05(tmp_path) if i.system == "agentic"]
    assert judgeable.skip_reason is None
    assert [c.chunk_id for c in judgeable.chunks] == cited


# ---- AC2: a new run against an older run of the same system ----------------------------------


def _rec(qid: str, qtype: str, *, answered: bool, ms: float, cny: float, **kw: Any) -> SystemRecord:
    answerable = qtype != "no_answer"
    usage = Usage(calls=2, cny=cny)
    return SystemRecord(
        system=kw.pop("system", "agentic"),
        qid=qid,
        qtype=qtype,
        question=qid,
        answerable=answerable,
        relevant_pages=[f"{KB}:p1.md"] if answerable else [],
        status=kw.pop("status", "answered" if answered else "not_found"),
        found=answered,
        answered=answered,
        text="t" if answered else "",
        total_ms=ms,
        usage=usage,
        usage_by_kind={
            "chat": Usage(calls=1, cny=cny - 0.001),
            "embedding": Usage(calls=1, cny=0.001),
        },
        **kw,
    )


def _line(qid: str, score: int, system: str = "agentic") -> JudgeLine:
    return JudgeLine(system=system, qid=qid, run=1, score=score)  # type: ignore[arg-type]


def _runs() -> tuple[list[SystemRecord], list[SystemRecord], list[JudgeLine], list[JudgeLine]]:
    before = [
        _rec("s1", "single_hop", answered=True, ms=10_000, cny=0.010),
        _rec("s2", "single_hop", answered=True, ms=12_000, cny=0.012),
        _rec("s3", "single_hop", answered=True, ms=14_000, cny=0.014),
        _rec("n1", "no_answer", answered=False, ms=20_000, cny=0.020),
        _rec("n2", "no_answer", answered=False, ms=18_000, cny=0.020, status="error"),
        _rec("n3", "no_answer", answered=False, ms=5_000, cny=0.002),
        _rec("x9", "two_hop", answered=True, ms=1_000, cny=0.001),  # not in the new run
    ]
    after = [
        _rec("s1", "single_hop", answered=True, ms=6_000, cny=0.006),
        _rec("s2", "single_hop", answered=True, ms=7_000, cny=0.006),
        _rec("s3", "single_hop", answered=True, ms=8_000, cny=0.006),
        _rec("n1", "no_answer", answered=True, ms=9_000, cny=0.010),
        _rec("n2", "no_answer", answered=False, ms=9_000, cny=0.010),
        # Not answered because the answer model timed out: a refusal, but not a clean one.
        _rec(
            "n3", "no_answer", answered=False, ms=9_000, cny=0.010, generation_error="llm_timeout"
        ),
    ]
    before_lines = [_line("s1", 3), _line("s2", 2), _line("s3", 3), _line("x9", 3)]
    after_lines = [_line("s1", 3), _line("s2", 3), _line("s3", 1)]
    return before, after, before_lines, after_lines


def test_compare_runs_per_type() -> None:
    before, after, bl, al = _runs()
    report = compare_runs(before, after, bl, al, system="agentic")
    single = report.correct["single_hop"]
    assert (single.n, single.a.count, single.b.count) == (3, 2, 2)
    assert (single.improved, single.worsened) == (1, 1)  # s2 2 -> 3, s3 3 -> 1
    assert single.p_value == 1.0
    assert "two_hop" not in report.correct  # x9 is not shared
    refusal = report.refusal
    assert (refusal.n, refusal.a.count, refusal.b.count) == (3, 3, 2)
    assert refusal.worsened == 1  # n1 refused before, answered now
    clean = report.clean_refusal
    # n2 errored before and n3's answer model timed out now: neither is a clean refusal there.
    assert (clean.a.count, clean.b.count) == (2, 1)
    assert (clean.improved, clean.worsened) == (1, 2)
    rows = {(r.label, r.qtype): r for r in report.cost}
    assert rows[("before", "single_hop")].p50_ms == 12_000
    assert rows[("after", "single_hop")].mean_cny == pytest.approx(0.006)
    assert rows[("after", "single_hop")].mean_chat_cny == pytest.approx(0.005)
    assert rows[("before", "all")].n == 6  # shared questions only
    assert report.shared == 6 and report.only_before == 1 and report.only_after == 0
    text = render_compare_runs(report)
    assert "single_hop" in text and "McNemar" in text and "refusal" in text.lower()
    assert "after lower" in text and "B lower" not in text  # the verdicts name the runs


# ---- AC3: the command ---------------------------------------------------------------------------


def test_compare_runs_cli(tmp_path: Path) -> None:
    before, after, bl, al = _runs()
    for out, records, lines in ((tmp_path / "old", before, bl), (tmp_path / "new", after, al)):
        for r in records:
            append_line(system_file(out, "agentic"), r)
        for x in lines:
            append_line(judge_file(out), x)
    md = tmp_path / "runs.md"
    argv = ["compare-runs", "--before", str(tmp_path / "old"), "--after", str(tmp_path / "new")]
    assert cli.main([*argv, "--system", "agentic", "--md", str(md)]) == 0
    assert "single_hop" in md.read_text(encoding="utf-8")
    assert cli.main([*argv, "--system", "standard"]) == 2  # no standard records in either


# ---- AC4: an eval-only override of one kb config key ------------------------------------------


def test_kb_override() -> None:
    from rag.eval.agentic.cli import apply_kb_overrides
    from rag.eval.retrieval.cli import UsageError

    cfg = kb_config(KB)
    assert cfg.agent.round2_on_not_enough is True
    new = apply_kb_overrides(cfg, ["agent.round2_on_not_enough=false", "agent.multi_queries=2"])
    assert new.agent.round2_on_not_enough is False and new.agent.multi_queries == 2
    assert new.model_dump(exclude={"agent"}) == cfg.model_dump(exclude={"agent"})
    assert cfg.agent.round2_on_not_enough is True  # the original is untouched
    assert apply_kb_overrides(cfg, None) == cfg
    for bad in ["agent.no_such_key=1", "agent.round2_on_not_enough", "agent.max_rounds=0"]:
        with pytest.raises(UsageError):
            apply_kb_overrides(cfg, [bad])
