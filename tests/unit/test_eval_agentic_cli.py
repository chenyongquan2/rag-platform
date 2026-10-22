"""Spec P3-05 AC13, AC14: human sheet, report, cost projection and the CLI with fakes."""

from __future__ import annotations

import csv
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from rag.agent import AgentRuntime
from rag.config import ServiceSettings
from rag.eval.agentic import cli
from rag.eval.agentic.human import export_human_sheet
from rag.eval.agentic.records import (
    JudgeLine,
    SystemRecord,
    read_judge_lines,
    read_records,
    system_file,
)
from rag.eval.agentic.report import project_cost, render_report
from rag.eval.agentic.runner import RunStats
from rag.eval.agentic.sets import EvalItem
from rag.eval.answer.record import Usage
from rag.observe import shutdown_observability
from rag.testing.agent import ScriptedSearchBackend, agent_settings, pages_result
from rag.testing.ledger_chat import LedgerChat, LedgerStreamingChat


@pytest.fixture(autouse=True)
def _no_active_observability() -> Iterator[None]:
    shutdown_observability()
    yield
    shutdown_observability()


TAG = {"standard": "Alpha", "agentic": "Beta"}  # the answers do not name their system


def _pair(qtype: str, n: int) -> list[SystemRecord]:
    out = []
    for i in range(n):
        for system in ("standard", "agentic"):
            out.append(
                SystemRecord(
                    system=system,  # type: ignore[arg-type]
                    qid=f"{qtype}:{i:02d}",
                    qtype=qtype,
                    question=f"Question {qtype} {i}?",
                    reference=f"Reference {i}.",
                    answerable=qtype != "no_answer",
                    answered=i % 7 != 0,
                    found=i % 7 != 0,
                    text=f"{TAG[system]} says {i}." if i % 7 else "",
                )
            )
    return out


def test_human_sheet_blind_pair(tmp_path: Path) -> None:
    records = [*_pair("single_hop", 40), *_pair("two_hop", 20), *_pair("colloquial", 15)]
    sheet, key = export_human_sheet(records, n=30, seed=0, out_dir=tmp_path)
    rows = list(csv.DictReader(sheet.open(encoding="utf-8-sig", newline="")))
    keys = list(csv.DictReader(key.open(encoding="utf-8-sig", newline="")))
    assert len(rows) == len(keys) == 30
    assert list(rows[0]) == [
        "sample_id",
        "question",
        "reference_answer",
        "answer_1",
        "answer_2",
        "score_1_3_answer_1",
        "score_1_3_answer_2",
        "comment",
    ]
    text = sheet.read_text(encoding="utf-8-sig")
    assert "standard" not in text and "agentic" not in text  # nothing says which system
    assert all(r["score_1_3_answer_1"] == "" and r["score_1_3_answer_2"] == "" for r in rows)
    # The key says which system wrote which answer, and the order is mixed.
    firsts = {k["answer_1_system"] for k in keys}
    assert firsts == {"standard", "agentic"}
    for r, k in zip(rows, keys, strict=True):
        assert r["sample_id"] == k["sample_id"]
        assert {k["answer_1_system"], k["answer_2_system"]} == {"standard", "agentic"}
        assert r["answer_1"].startswith(TAG[k["answer_1_system"]]) or "(no answer" in r["answer_1"]
    # Spread over the types: 12 single-hop, 10 two-hop, 8 colloquial.
    by_type = [k["qid"].split(":")[0] for k in keys]
    assert (by_type.count("single_hop"), by_type.count("two_hop"), by_type.count("colloquial")) == (
        12,
        10,
        8,
    )
    # Deterministic for a seed; another seed picks other questions.
    again, _ = export_human_sheet(records, n=30, seed=0, out_dir=tmp_path / "again")
    assert again.read_text(encoding="utf-8-sig") == text
    other, _ = export_human_sheet(records, n=30, seed=1, out_dir=tmp_path / "other")
    assert other.read_text(encoding="utf-8-sig") != text
    assert (tmp_path / "human_rubric.txt").is_file()


def test_report_renders_the_tables() -> None:
    records = [*_pair("single_hop", 10), *_pair("no_answer", 4)]
    lines = [
        JudgeLine(system=s, qid=f"single_hop:{i:02d}", run=run, score=3 if i % 3 else 2)  # type: ignore[arg-type]
        for s in ("standard", "agentic")
        for i in range(10)
        for run in (1, 2)
    ]
    text = render_report(records, lines)
    for needle in (
        "single_hop",
        "no_answer",
        "refused cleanly",
        "McNemar",
        "direction only",
        "Wilson",
        "minimum detectable",
        "P95",
        "round",
    ):
        assert needle in text, needle


def _rec(system: str, qtype: str, cny: float, qid: str) -> SystemRecord:
    return SystemRecord(
        system=system,  # type: ignore[arg-type]
        qid=qid,
        qtype=qtype,
        question="q",
        answerable=qtype != "no_answer",
        usage=Usage(calls=1, cny=cny),
    )


def test_project_cost_from_pilot() -> None:
    pilot = [
        _rec("standard", "single_hop", 0.005, "a"),
        _rec("standard", "no_answer", 0.001, "b"),
        _rec("agentic", "single_hop", 0.03, "a"),
        _rec("agentic", "no_answer", 0.05, "b"),
    ]
    sizes = {"single_hop": 65, "two_hop": 20, "colloquial": 15, "no_answer": 50}
    p = project_cost(pilot, sizes, judge_cny_per_call=0.0014, judge_passes=2)
    # A: 65 x 0.005 + 50 x 0.001 + the 35 types without a pilot at the mean of A (0.003).
    assert p.by_system["standard"] == pytest.approx(65 * 0.005 + 50 * 0.001 + 35 * 0.003)
    assert p.by_system["agentic"] == pytest.approx(65 * 0.03 + 50 * 0.05 + 35 * 0.04)
    # Judge: 100 answerable questions x 2 systems x 2 passes.
    assert p.judge == pytest.approx(100 * 2 * 2 * 0.0014)
    assert p.total == pytest.approx(sum(p.by_system.values()) + p.judge)
    # What is left once the pilot's questions are done.
    assert p.remaining_by_system["agentic"] == pytest.approx(p.by_system["agentic"] - 0.08)


# ---- the CLI ------------------------------------------------------------------------------------


GOLD = "hf_docs:p1.md"
ITEMS = [
    EvalItem(
        qid="a1",
        qtype="single_hop",
        set="x",
        question="gamma delta",
        reference="Use it.",
        relevant_pages=[GOLD],
    ),
    EvalItem(qid="n1", qtype="no_answer", set="y", question="alpha beta", answerable=False),
]


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    ledger = str(tmp_path / "ledger.jsonl")
    backend = ScriptedSearchBackend("hf_docs", default=pages_result("hf_docs", [1, 2]))
    monkeypatch.setattr(cli, "load_items", lambda kb, **kw: ITEMS)
    monkeypatch.setattr(cli, "build_retriever", lambda cfg, settings, **kw: backend)
    monkeypatch.setattr(
        cli,
        "build_answer_model",
        lambda settings: LedgerStreamingChat(default_response="Use it [1].", ledger_path=ledger),
    )
    monkeypatch.setattr(
        cli, "build_agent_models", lambda settings: (
            LedgerStreamingChat(default_response="Use it [1].", ledger_path=ledger),
            LedgerChat(default_response="x", ledger_path=ledger),
            LedgerChat(default_response="x", ledger_path=ledger),
        ),
    )  # fmt: skip
    monkeypatch.setattr(
        cli,
        "build_judge_model",
        lambda settings, config: LedgerChat(
            default_response="Feedback: right. [RESULT] 3", ledger_path=ledger, profile="offline"
        ),
    )
    monkeypatch.setattr(
        cli,
        "build_runtime",
        lambda settings, hooks: AgentRuntime.from_settings(agent_settings(), hooks=hooks),
    )
    settings = ServiceSettings(
        chat_backend="fake",
        otel_mode="off",
        business_record_path="",
        log_dir="",
        cost_ledger_path=ledger,
    )
    return {"out": tmp_path / "out", "settings": settings}


def _main(env: dict[str, Any], *argv: str) -> int:
    return cli.main([*argv, "--out", str(env["out"])], settings=env["settings"])


def test_cli_with_fakes(env: dict[str, Any], capsys: pytest.CaptureFixture[str]) -> None:
    out: Path = env["out"]
    assert _main(env, "run", "--system", "both", "--kb", "hf_docs") == 0
    standard = read_records(system_file(out, "standard"))
    agentic = read_records(system_file(out, "agentic"))
    assert [r.qid for r in standard] == [r.qid for r in agentic] == ["a1", "n1"]
    assert all(r.answered for r in standard)
    # A second run does nothing: every question is in the files.
    assert _main(env, "run", "--system", "both", "--kb", "hf_docs") == 0
    assert len(read_records(system_file(out, "agentic"))) == 2

    assert _main(env, "judge", "--runs", "2") == 0
    lines = read_judge_lines(out / "judge.jsonl")
    assert len(lines) == 4  # one answerable question, two systems, two passes
    assert all(x.score == 3 for x in lines if x.system == "standard")

    assert _main(env, "report", "--md", str(out / "report.md")) == 0
    assert "single_hop" in (out / "report.md").read_text(encoding="utf-8")
    assert _main(env, "human-sheet", "--n", "30") == 0
    assert (out / "human" / "human_sheet.csv").is_file()
    assert _main(env, "project") == 0
    assert "total" in capsys.readouterr().out.lower()


def test_cli_rejects_a_run_past_the_stop_line(env: dict[str, Any]) -> None:
    settings = env["settings"].model_copy(update={"budget_stop_cny": 0.0001})
    ledger = Path(settings.cost_ledger_path)
    ledger.write_text('{"kind": "chat", "cny": 1.0}\n', encoding="utf-8")
    assert (
        cli.main(["run", "--system", "standard", "--out", str(env["out"])], settings=settings) == 2
    )


def test_cli_ledger_stop_line_is_inclusive(env: dict[str, Any]) -> None:
    """A ledger at exactly the stop line leaves no room: refuse; a cent below it: go on."""
    ledger = Path(env["settings"].cost_ledger_path)
    settings = env["settings"].model_copy(update={"budget_stop_cny": 28.0})
    argv = ["run", "--system", "standard", "--out", str(env["out"])]
    ledger.write_text('{"kind": "chat", "cny": 28.0}\n', encoding="utf-8")
    assert cli.main(argv, settings=settings) == 2
    below = ledger.with_name("below.jsonl")  # a file of its own: the ledger caches its total
    below.write_text('{"kind": "chat", "cny": 27.99}\n', encoding="utf-8")
    settings = settings.model_copy(update={"cost_ledger_path": str(below)})
    assert cli.main(argv, settings=settings) == 0


def test_human_sheet_always_holds_the_questions_where_the_scores_differ(tmp_path: Path) -> None:
    records = [*_pair("single_hop", 40), *_pair("two_hop", 20), *_pair("colloquial", 15)]
    differ = {"single_hop:31", "two_hop:17", "colloquial:09"}
    lines = [
        JudgeLine(
            system=r.system, qid=r.qid, run=1,
            score=3 if r.system == "standard" or r.qid not in differ else 1,
        )
        for r in records
        if r.answerable
    ]  # fmt: skip
    for seed in range(5):  # whatever the random part picks, these three are in
        _sheet, key = export_human_sheet(
            records, n=30, seed=seed, out_dir=tmp_path / str(seed), lines=lines
        )
        qids = {k["qid"] for k in csv.DictReader(key.open(encoding="utf-8-sig", newline=""))}
        assert differ <= qids and len(qids) == 30
    # The first line of the rubric file says how the rows were chosen.
    assert "differ" in (tmp_path / "0" / "human_rubric.txt").read_text(encoding="utf-8")


def test_cli_selects_questions_and_passes_the_deadline(
    env: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: dict[str, Any] = {}

    async def spy(items: Any, **kw: Any) -> Any:
        seen["qids"] = [i.qid for i in items]
        seen["deadline_ms"] = kw["deadline_ms"]
        return RunStats()

    monkeypatch.setattr(cli, "run_agentic", spy)
    assert _main(env, "run", "--system", "agentic", "--qids", "n1", "--deadline-ms", "60000") == 0
    assert seen == {"qids": ["n1"], "deadline_ms": 60000}
    assert _main(env, "run", "--system", "agentic", "--qids", "nope") == 2  # unknown question id


def test_run_applies_and_records_a_kb_override(
    env: dict[str, Any], monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Spec P3-07: `--kb-override` reaches the runner and is written down with the run."""
    import json

    seen: dict[str, Any] = {}
    real = cli.run_agentic

    async def spy(items: Any, **kw: Any) -> RunStats:
        seen["cfg"] = kw["kb_config"]
        return await real(items, **kw)

    monkeypatch.setattr(cli, "run_agentic", spy)
    override = "agent.round2_on_not_enough=false"
    argv = ["run", "--system", "agentic", "--kb", "hf_docs", "--kb-override", override]
    assert _main(env, *argv) == 0
    assert seen["cfg"].agent.round2_on_not_enough is False
    assert override in capsys.readouterr().out
    (line,) = [
        json.loads(x) for x in (env["out"] / "runs.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert line["system"] == "agentic" and line["kb_overrides"] == [override]
    assert line["questions"] == 2 and line["started"]
