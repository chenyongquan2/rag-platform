"""Spec P1-06 AC6 to AC14: metrics, loaders, run, planted errors, human sheet, CLI, P3 records."""

from __future__ import annotations

import asyncio
import csv
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from langchain_core.messages import AIMessage

from rag.answer.support import (
    SUPPORT_TOOL,
    PairVerdict,
    SentenceVerdict,
    SupportJudgeConfig,
    SupportResult,
)
from rag.config import ServiceSettings
from rag.eval.agentic.records import SystemRecord, read_records
from rag.eval.agentic.runner import run_standard
from rag.eval.agentic.sets import EvalItem
from rag.eval.answer.record import (
    AnswerQuestionRecord,
    AnswerRunRecord,
    FormatCompliance,
    Proportion,
    Usage,
)
from rag.eval.answer.runner import RetrieverContexts
from rag.eval.support import cli
from rag.eval.support.human import KEY_NAME, SHEET_NAME, export_support_sheet
from rag.eval.support.items import load_p1_04, load_p3_05
from rag.eval.support.metrics import by_group, paired_all_supported, stability, summarize
from rag.eval.support.planted import plant
from rag.eval.support.records import SupportItem, SupportLine, latest_lines, read_lines
from rag.eval.support.runner import run_support
from rag.observe import shutdown_observability
from rag.testing import FakeChatModel
from rag.testing.agent import ScriptedSearchBackend, kb_config, scored_result
from rag.testing.ledger_chat import LedgerChat, LedgerStreamingChat
from rag.types import Answer, AnswerSentence, Chunk, Citation


@pytest.fixture(autouse=True)
def _no_active_observability() -> Iterator[None]:
    shutdown_observability()
    yield
    shutdown_observability()


def _chunk(cid: str, text: str) -> Chunk:
    return Chunk(chunk_id=cid, doc_id=cid.split("#")[0], kb="hf_docs", text=text, title_path=["T"])


def _sent(text: str, *cids: str) -> AnswerSentence:
    return AnswerSentence(
        text=text, citations=[Citation(marker=i + 1, chunk_id=c) for i, c in enumerate(cids)]
    )


def _verdict(index: int, *labels: str, text: str = "s") -> SentenceVerdict:
    return SentenceVerdict(
        index=index,
        text=text,
        pairs=[
            PairVerdict(sentence=index, fragment=i + 1, chunk_id=f"c{i}", label=label)  # type: ignore[arg-type]
            for i, label in enumerate(labels)
        ],
    )


def _line(
    qid: str,
    *sentences: SentenceVerdict,
    system: str = "standard",
    qtype: str = "single_hop",
    run: int = 1,
    cny: float = 0.0,
) -> SupportLine:
    item = SupportItem(source="t", system=system, qid=qid, qtype=qtype, question="q?")
    return SupportLine(
        **item.key_fields(),
        run=run,
        item=item,
        result=SupportResult(sentences=list(sentences)),
        usage=Usage(calls=1, cny=cny),
    )


# ---- metrics ------------------------------------------------------------------------------------


def test_support_metrics() -> None:
    lines = [
        # 2 sentences, pairs: supported, partly, not_supported
        _line("q1", _verdict(1, "supported", "partly"), _verdict(2, "not_supported")),
        # every sentence supported; one unknown pair beside a supported one
        _line("q2", _verdict(1, "supported", "unknown")),
        # an uncited sentence and an unknown one
        _line("q3", _verdict(1), _verdict(2, "unknown")),
        _line("q4", _verdict(1, "supported"), system="agentic", qtype="two_hop"),
    ]
    s = summarize(lines[:3])
    assert (s.answers, s.pairs_unknown, s.sentences_unknown, s.uncited) == (3, 2, 1, 1)
    assert s.precision == Proportion.of(2, 4)  # supported pairs of the known pairs
    assert s.partly == Proportion.of(1, 4)
    assert s.precision_lenient == Proportion.of(3, 4)
    assert s.recall == Proportion.of(2, 4)  # known sentences: q1 s1, q1 s2, q2 s1, q3 s1
    assert s.unsupported == Proportion.of(2, 4)  # q1 s2 and the uncited q3 s1
    assert s.all_supported == Proportion.of(1, 3)  # q2 yes; q1 no; q3 no (uncited sentence)
    groups = by_group(lines)
    assert set(groups) == {"standard / single_hop", "agentic / two_hop", "standard", "agentic"}
    assert groups["agentic"].precision == Proportion.of(1, 1)
    # A skipped item counts as skipped, nothing else.
    skipped = SupportLine(
        **SupportItem(
            source="t",
            system="standard",
            qid="q9",
            qtype="single_hop",
            question="q",
            skip_reason="not_found",
        ).key_fields(),
        run=1,
        item=SupportItem(
            source="t",
            system="standard",
            qid="q9",
            qtype="single_hop",
            question="q",
            skip_reason="not_found",
        ),
    )
    assert summarize([*lines[:3], skipped]).skipped == 1
    assert summarize([*lines[:3], skipped]).answers == 3


def test_paired_all_supported() -> None:
    a = [
        _line("q1", _verdict(1, "supported")),
        _line("q2", _verdict(1, "not_supported")),
        _line("q3", _verdict(1, "supported")),
        _line("q4", _verdict(1, "unknown")),
        _line("q5", _verdict(1, "supported")),  # only in a
    ]
    b = [
        _line("q1", _verdict(1, "supported"), system="agentic"),
        _line("q2", _verdict(1, "supported"), system="agentic"),
        _line("q3", _verdict(1, "partly"), system="agentic"),
        _line("q4", _verdict(1, "supported"), system="agentic"),
    ]
    p = paired_all_supported(a, b)
    assert (p.n, p.both, p.b_only, p.a_only, p.neither, p.unknown, p.unpaired) == (
        3,
        1,
        1,
        1,
        0,
        1,
        1,
    )
    assert p.p_value == 1.0
    many_a = [_line(f"q{i}", _verdict(1, "not_supported")) for i in range(8)]
    many_b = [_line(f"q{i}", _verdict(1, "supported"), system="agentic") for i in range(8)]
    assert paired_all_supported(many_a, many_b).p_value == pytest.approx(2 / 2**8)


def test_support_stability() -> None:
    r1 = [
        _line("q1", _verdict(1, "supported", "partly"), _verdict(2, "supported")),
        _line("q2", _verdict(1, "unknown")),
        _line("q3", _verdict(1, "supported")),
    ]
    r2 = [
        _line("q1", _verdict(1, "supported", "not_supported"), _verdict(2, "partly"), run=2),
        _line("q2", _verdict(1, "supported"), run=2),
        _line("q3", _verdict(1, "supported"), run=2),
    ]
    st = stability(r1, r2)
    assert st.pairs == Proportion.of(
        2, 4
    )  # q1 (1, c1) and (2, c0) changed; q2's pair is unknown in run 1
    assert st.pairs_unknown == 1
    assert st.sentences == Proportion.of(1, 3)  # q1 s2 supported -> partly
    assert st.answers == Proportion.of(
        1, 2
    )  # q1: all supported in run 1, not in run 2; q2 (unknown in run 1) is left out
    assert [c["qid"] for c in st.changed] == ["q1", "q1"]


# ---- loaders ------------------------------------------------------------------------------------


def _compliance() -> FormatCompliance:
    z = Proportion.of(0, 0)
    return FormatCompliance(
        n_questions=0,
        answered=z,
        found_false=z,
        fully_cited=z,
        fully_cited_all=z,
        with_unknown_markers=z,
        unknown_markers_total=0,
        with_unsupported=z,
        unsupported_sentences=z,
        parse_failed=z,
        retried=z,
        generation_errors=0,
        retrieval_not_found=0,
        retrieval_errors=0,
    )


def _answer_run(tmp_path: Path) -> Path:
    chunks = [_chunk("p:a#0", "BERT base has 12 layers."), _chunk("p:b#0", "Other.")]
    good = Answer(
        request_id="r1",
        kb="hf_docs",
        found=True,
        sentences=[_sent("BERT base has 12 layers.", "p:a#0")],
        chunks=chunks,
    )
    missing = Answer(request_id="r2", kb="hf_docs", found=False, chunks=chunks)
    run = AnswerRunRecord(
        run_id="r",
        created_at="t",
        kb="hf_docs",
        questions=[
            AnswerQuestionRecord(qid="hf:001", question="How many?", answer=good, found=True),
            AnswerQuestionRecord(qid="hf:002", question="What?", answer=missing),
            AnswerQuestionRecord(qid="hf:003", question="Who?", answer=None, error="X: y"),
        ],
        compliance=_compliance(),
    )
    path = tmp_path / "results.json"
    path.write_text(run.model_dump_json(), encoding="utf-8", newline="")
    return path


def _system_record(system: str, qid: str, **kw: Any) -> SystemRecord:
    return SystemRecord(system=system, qid=qid, qtype="single_hop", question="q?", **kw)  # type: ignore[arg-type]


def test_loaders(tmp_path: Path) -> None:
    items = load_p1_04(_answer_run(tmp_path))
    assert [(i.qid, i.system, i.qtype, i.skip_reason) for i in items] == [
        ("hf:001", "standard", "single_hop", None),
        ("hf:002", "standard", "single_hop", "not_found"),
        ("hf:003", "standard", "single_hop", "no_answer"),
    ]
    assert [c.chunk_id for c in items[0].chunks] == ["p:a#0", "p:b#0"]
    assert items[0].question == "How many?" and items[0].source == "p1-04"

    out = tmp_path / "p3"
    for system in ("standard", "agentic"):
        path = out / system / "records.jsonl"
        path.parent.mkdir(parents=True)
        recs = [
            _system_record(system, "q1", answered=True, found=True, text="Old answer."),
            _system_record(system, "q2"),  # not answered
            _system_record(
                system,
                "q3",
                answered=True,
                found=True,
                text="New.",
                answer_sentences=[_sent("New.", "p:a#0")],
                cited_chunks=[_chunk("p:a#0", "New.")],
            ),
        ]
        path.write_text("".join(r.model_dump_json() + "\n" for r in recs), encoding="utf-8")
    items = load_p3_05(out)
    assert [(i.system, i.qid, i.skip_reason) for i in items] == [
        ("standard", "q1", "no_sentence_citations"),
        ("standard", "q2", "no_answer"),
        ("standard", "q3", None),
        ("agentic", "q1", "no_sentence_citations"),
        ("agentic", "q2", "no_answer"),
        ("agentic", "q3", None),
    ]


# ---- run ----------------------------------------------------------------------------------------


def _reply(*judgements: tuple[int, int, str]) -> AIMessage:
    args = {
        "judgements": [
            {"sentence": s, "fragment": f, "reason": "r", "verdict": v} for s, f, v in judgements
        ]
    }
    return AIMessage(content="", tool_calls=[{"name": SUPPORT_TOOL, "args": args, "id": "c"}])


def _items(n: int) -> list[SupportItem]:
    chunk = _chunk("p:a#0", "Fact 12.")
    return [
        SupportItem(
            source="t",
            system="standard",
            qid=f"q{i}",
            qtype="single_hop",
            question="q?",
            sentences=[_sent(f"Fact {i}.", "p:a#0")],
            chunks=[chunk],
        )
        for i in range(n)
    ]


def _ledger_judge(tmp_path: Path, *replies: Any) -> LedgerChat:
    return LedgerChat(
        responses=list(replies),
        ledger_path=str(tmp_path / "ledger.jsonl"),
        profile="offline",
        prompt_tokens=1000,
        completion_tokens=100,
    )  # 0.0028 CNY per call at the fake price


def test_run_resume_and_cost_cap(tmp_path: Path) -> None:
    out = tmp_path / "lines.jsonl"
    items = [
        *_items(3),
        SupportItem(
            source="t",
            system="standard",
            qid="skip",
            qtype="x",
            question="q",
            skip_reason="not_found",
        ),
    ]
    cfg = SupportJudgeConfig(concurrency=1)
    model = _ledger_judge(tmp_path, _reply((1, 1, "supported")), AIMessage(content="no tool"))
    stats = asyncio.run(
        run_support(items, run=1, model=model, config=cfg, out_path=out, limit=2, max_cny=1.0)
    )
    assert (stats.ran, stats.done_before, stats.pending) == (2, 0, 2)
    lines = read_lines(out)
    assert [(x.qid, x.result.sentences[0].status if x.result else None) for x in lines] == [
        ("q0", "supported"),
        ("q1", "unknown"),
    ]
    assert lines[0].usage.calls == 1 and lines[0].usage.cny > 0

    # Resume: done keys are skipped; the skipped item gets a line without a call.
    model = _ledger_judge(tmp_path, _reply((1, 1, "supported")))
    stats = asyncio.run(run_support(items, run=1, model=model, config=cfg, out_path=out))
    assert (stats.ran, stats.done_before, stats.pending) == (2, 2, 0)
    assert len(model.calls) == 1
    assert read_lines(out)[-1].item.skip_reason == "not_found"

    # --retry-unknown asks q1 again; the newest line of a key wins.
    model = _ledger_judge(tmp_path, _reply((1, 1, "not_supported")))
    stats = asyncio.run(
        run_support(items, run=1, model=model, config=cfg, out_path=out, retry_unknown=True)
    )
    assert stats.ran == 1
    latest = {x.qid: x for x in latest_lines(read_lines(out))}
    assert latest["q1"].result is not None
    assert latest["q1"].result.sentences[0].status == "not_supported"

    # Spend cap: stops before the call that would start at or past the cap.
    out2 = tmp_path / "capped.jsonl"
    model = _ledger_judge(tmp_path, *[_reply((1, 1, "supported"))] * 3)
    stats = asyncio.run(
        run_support(_items(3), run=1, model=model, config=cfg, out_path=out2, max_cny=0.005)
    )
    assert stats.stopped == "max_cny" and stats.ran == 2 and len(model.calls) == 2
    stats = asyncio.run(
        run_support(
            _items(3),
            run=1,
            model=model,
            config=cfg,
            out_path=out2,
            max_cny=0.001,
            spent_before=0.0,
        )
    )
    assert stats.stopped == "max_cny" and stats.ran == 0  # the lines in the file count


def test_cli_refuses_past_ledger_stop(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ledger = tmp_path / "ledger.jsonl"
    ledger.write_text(json.dumps({"kind": "chat", "cny": 27.9}) + "\n", encoding="utf-8")
    settings = ServiceSettings(cost_ledger_path=str(ledger), chat_backend="fake")
    results = _answer_run(tmp_path)
    called: list[int] = []
    monkeypatch.setattr(cli, "build_support_model", lambda s, c: called.append(1))
    code = cli.main(
        [
            "run",
            "--source",
            "p1-04",
            "--path",
            str(results),
            "--out",
            str(tmp_path / "o"),
            "--estimate-cny",
            "0.2",
        ],
        settings=settings,
    )
    assert code == 2 and called == []


# ---- planted errors -----------------------------------------------------------------------------


def _judged_line(qid: str, sentence: str, chunk: Chunk) -> SupportLine:
    item = SupportItem(
        source="p1-04",
        system="standard",
        qid=qid,
        qtype="single_hop",
        question="q?",
        sentences=[_sent(sentence, chunk.chunk_id), _sent("Other claim.", chunk.chunk_id)],
        chunks=[chunk],
    )
    verdicts = [
        SentenceVerdict(
            index=1,
            text=sentence,
            pairs=[PairVerdict(sentence=1, fragment=1, chunk_id=chunk.chunk_id, label="supported")],
        ),
        SentenceVerdict(
            index=2,
            text="Other claim.",
            pairs=[
                PairVerdict(sentence=2, fragment=1, chunk_id=chunk.chunk_id, label="not_supported")
            ],
        ),
    ]
    return SupportLine(
        **item.key_fields(), run=1, item=item, result=SupportResult(sentences=verdicts)
    )


def test_plant_errors() -> None:
    lines = [
        _judged_line(
            f"q{i}",
            f"Model {i} has {12 + i} layers.",
            _chunk(f"p:{i}#0", f"Model {i} has {12 + i} layers and x86_64 builds."),
        )
        for i in range(6)
    ]
    before = [x.model_dump_json() for x in lines]
    planted = plant(lines, n=4, seed=0)
    assert len(planted) == 4
    assert [x.model_dump_json() for x in lines] == before  # originals untouched
    kinds = sorted(p.planted.kind for p in planted if p.planted)
    assert kinds == ["change_number", "change_number", "swap_fragment", "swap_fragment"]
    for p in planted:
        assert p.planted is not None and p.source == "planted"
        assert p.qid == f"planted-{p.planted.kind}:{p.planted.original_qid}"
        original = next(x for x in lines if x.qid == p.planted.original_qid)
        changed = p.sentences[p.planted.sentence - 1]
        assert p.planted.sentence == 1  # only sentences the judge called supported are used
        if p.planted.kind == "change_number":
            assert changed.text != original.item.sentences[0].text
            assert changed.citations == original.item.sentences[0].citations
            assert "x86_64" not in changed.text
        else:
            assert changed.text == original.item.sentences[0].text
            (cit,) = changed.citations
            assert cit.chunk_id == p.planted.chunk_id != original.item.chunks[0].chunk_id
            assert cit.chunk_id in {c.chunk_id for c in p.chunks}
        assert p.sentences[1] == original.item.sentences[1]
    assert plant(lines, n=4, seed=0) == planted  # seeded

    # Only a number the cited fragment also has is changed (2019 is not in it), to a value the
    # fragment does not have (24 is, so 12 becomes 13).
    one = _judged_line(
        "y", "Model y has 12 layers since 2019.", _chunk("p:y#0", "Model y: 12 layers, 24 heads.")
    )
    (changed,) = plant([one], n=1, seed=0)
    assert changed.sentences[0].text == "Model y has 13 layers since 2019."


# ---- human sheet --------------------------------------------------------------------------------


def test_support_human_sheet_is_blind(tmp_path: Path) -> None:
    lines = []
    for i in range(30):
        chunk = _chunk(f"p:{i}#0", f"Fragment text {i}.")
        lines.append(_judged_line(f"q{i:02d}", f"Sentence {i}.", chunk))
    sheet, key = export_support_sheet(lines, n=40, seed=0, out_dir=tmp_path)
    assert (sheet.name, key.name) == (SHEET_NAME, KEY_NAME)
    with sheet.open(encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 40
    header = set(rows[0])
    assert not any(w in h for h in header for w in ("judge", "system", "qid", "verdict_"))
    assert all(r["human_label"] == "" for r in rows)
    assert {"sentence", "fragment_text", "question"} <= header
    with key.open(encoding="utf-8-sig", newline="") as fh:
        keys = list(csv.DictReader(fh))
    assert [k["sample_id"] for k in keys] == [r["sample_id"] for r in rows]
    # Every pair the judge did not call supported is in the sheet (here 30, under half of 40 is
    # 20: at most 20 of them, the rest drawn from the supported pairs).
    labels = [k["judge_label_run1"] for k in keys]
    assert labels.count("not_supported") == 20 and labels.count("supported") == 20
    again, _ = export_support_sheet(lines, n=40, seed=0, out_dir=tmp_path / "b")
    assert again.read_text(encoding="utf-8-sig") == sheet.read_text(encoding="utf-8-sig")


# ---- CLI ----------------------------------------------------------------------------------------


def test_support_cli_with_fakes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ledger = tmp_path / "ledger.jsonl"
    settings = ServiceSettings(cost_ledger_path=str(ledger), chat_backend="fake")
    results = _answer_run(tmp_path)
    out = tmp_path / "out"
    monkeypatch.setattr(
        cli,
        "build_support_model",
        lambda s, c: FakeChatModel(default_response=_reply((1, 1, "supported"))),
    )
    for run in ("1", "2"):
        assert (
            cli.main(
                [
                    "run",
                    "--source",
                    "p1-04",
                    "--path",
                    str(results),
                    "--out",
                    str(out),
                    "--run",
                    run,
                ],
                settings=settings,
            )
            == 0
        )
    lines = read_lines(out / "lines.jsonl")
    assert [(x.qid, x.run) for x in lines if x.item.skip_reason is None] == [
        ("hf:001", 1),
        ("hf:001", 2),
    ]
    assert cli.main(["plant", "--out", str(out), "--n", "1"], settings=settings) == 0
    assert any(x.item.planted is not None for x in read_lines(out / "lines.jsonl"))
    assert cli.main(["human-sheet", "--out", str(out), "--n", "1"]) == 0
    assert (out / "human" / SHEET_NAME).is_file()
    md = tmp_path / "report.md"
    assert cli.main(["report", "--out", str(out), "--md", str(md)]) == 0
    text = md.read_text(encoding="utf-8")
    for needle in ("precision", "recall", "stability", "planted", "standard / single_hop"):
        assert needle in text


# ---- P3 records keep what level 2 needs ---------------------------------------------------------


def test_records_keep_sentence_citations(tmp_path: Path) -> None:
    kb = "hf_docs"
    backend = ScriptedSearchBackend(
        kb, by_question={"How to push?": scored_result(kb, [0.9, 0.8], pages=[1, 2])}
    )
    model = LedgerStreamingChat(
        responses=["Use push_to_hub [1]."], ledger_path=str(tmp_path / "ledger.jsonl")
    )
    item = EvalItem(
        qid="q1",
        qtype="single_hop",
        set="x",
        question="How to push?",
        reference="r",
        relevant_pages=[f"{kb}:p1.md"],
        answerable=True,
    )
    out = tmp_path / "standard.jsonl"
    asyncio.run(
        run_standard(
            [item],
            records_path=out,
            contexts=RetrieverContexts(backend, kb=kb, top_k=8),
            kb_config=kb_config(kb),
            model=model,
        )
    )
    (rec,) = read_records(out)
    assert [s.text for s in rec.answer_sentences] == ["Use push_to_hub."]
    cited = rec.answer_sentences[0].citations[0].chunk_id
    assert [c.chunk_id for c in rec.cited_chunks] == [cited]  # only the cited chunk is kept
    assert rec.cited_chunks[0].text
