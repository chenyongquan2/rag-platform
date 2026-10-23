# ruff: noqa: RUF001 (Chinese report text on purpose)
"""Spec EVAL-3 AC1, AC10, AC11, AC12: the three commands with fakes (no model, no network)."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from rag.agent import AgentRuntime
from rag.config import ServiceSettings
from rag.eval.agentic import cli
from rag.eval.agentic.records import SYSTEMS, read_judge_lines, read_records, system_file
from rag.eval.agentic.repeat import cli as repeat_cli
from rag.eval.agentic.repeat.passes import load_manifest
from rag.eval.agentic.repeat.sample import read_question_list, write_question_list
from rag.eval.agentic.runner import RunStats
from rag.eval.agentic.sets import EvalItem
from rag.observe import shutdown_observability
from rag.testing.agent import ScriptedSearchBackend, agent_settings, pages_result
from rag.testing.ledger_chat import LedgerChat, LedgerStreamingChat
from rag.testing.repeat import passes_from, qlist_for, write_pass


@pytest.fixture(autouse=True)
def _no_active_observability() -> Iterator[None]:
    shutdown_observability()
    yield
    shutdown_observability()


GOLD = "hf_docs:p1.md"
QIDS = ["single_hop:00", "single_hop:01", "hard:00", "ambiguous:00"]


def _item(qid: str) -> EvalItem:
    qtype = qid.split(":")[0]
    ambiguous = qtype == "ambiguous"
    return EvalItem(
        qid=qid,
        qtype=qtype,  # type: ignore[arg-type]
        set="x",
        question="gamma delta",
        reference="" if ambiguous else "Use it.",
        relevant_pages=[] if ambiguous else [GOLD],
        answerable=not ambiguous,
    )


ITEMS = [_item(q) for q in QIDS]


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    ledger = str(tmp_path / "ledger.jsonl")
    backend = ScriptedSearchBackend("hf_docs", default=pages_result("hf_docs", [1, 2]))
    for module in (cli, repeat_cli):
        monkeypatch.setattr(module, "load_items", lambda kb, **kw: ITEMS)
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
        budget_cny=39.0,
        budget_stop_cny=37.0,
    )
    # Pass 1 on disk: old questions in one directory, a hard question in the step-2 directory.
    stored = passes_from(
        {
            "standard": {q: "c" for q in QIDS},
            "agentic": {q: "c" for q in QIDS},
        }
    )[0]
    old, new = tmp_path / "p3-07", tmp_path / "step2"
    write_pass(old, stored, only={"single_hop:00", "single_hop:01"})
    write_pass(new, stored, only={"hard:00", "ambiguous:00"})
    qfile = tmp_path / "list.json"
    write_question_list(qfile, qlist_for(QIDS))
    return {
        "base": tmp_path / "base",
        "settings": settings,
        "old": old,
        "new": new,
        "list": qfile,
        "ledger": Path(ledger),
    }


def _argv(env: dict[str, Any], pass_no: int, *extra: str) -> list[str]:
    return [
        "repeat-run",
        "--base",
        str(env["base"]),
        "--pass",
        str(pass_no),
        "--list",
        str(env["list"]),
        "--pass1-old",
        str(env["old"]),
        "--pass1-new",
        str(env["new"]),
        *extra,
    ]


def _spy(monkeypatch: pytest.MonkeyPatch, events: list[str]) -> None:
    real_warm = repeat_cli.warm_up
    real_std, real_agent = cli.run_standard, cli.run_agentic

    def warm(*a: Any, **kw: Any) -> Any:
        events.append("warm")
        return real_warm(*a, **kw)

    async def std(*a: Any, **kw: Any) -> RunStats:
        events.append("standard")
        return await real_std(*a, **kw)

    async def agent(*a: Any, **kw: Any) -> RunStats:
        events.append("agentic")
        return await real_agent(*a, **kw)

    monkeypatch.setattr(repeat_cli, "warm_up", warm)
    monkeypatch.setattr(cli, "run_standard", std)
    monkeypatch.setattr(cli, "run_agentic", agent)


def test_run_prints_estimate_first_and_refuses_over_the_stop_line(
    env: dict[str, Any], monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    events: list[str] = []
    _spy(monkeypatch, events)
    env["ledger"].write_text('{"kind": "chat", "cny": 36.999}\n', encoding="utf-8")
    code = cli.main(_argv(env, 2), settings=env["settings"])
    out = capsys.readouterr()
    assert code == 2
    assert "estimated cost of this pass" in out.out  # the estimate comes first
    assert "stop line" in out.err
    assert events == []  # no warm-up, no run
    assert not (env["base"] / "pass2").exists()


def test_estimate_only_runs_nothing(
    env: dict[str, Any], monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    events: list[str] = []
    _spy(monkeypatch, events)
    assert cli.main(_argv(env, 2, "--estimate-only"), settings=env["settings"]) == 0
    out = capsys.readouterr().out
    assert "estimated cost of this pass" in out and "ledger: spent" in out
    assert "standard:" in out and "agentic:" in out
    assert events == [] and not (env["base"] / "pass2").exists()
    saved = load_manifest(env["base"])  # the paths are remembered for the next commands
    assert saved.pass1_old == [str(env["old"])] and saved.question_list == str(env["list"])


def test_run_pass_with_fakes_orders_systems_and_warms_up(
    env: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[str] = []
    _spy(monkeypatch, events)
    assert cli.main(_argv(env, 2), settings=env["settings"]) == 0
    assert events == ["warm", "standard", "agentic"]
    pass2 = env["base"] / "pass2"
    for system in SYSTEMS:
        assert [r.qid for r in read_records(system_file(pass2, system))] == QIDS
    judged = read_judge_lines(pass2 / "judge.jsonl")
    assert len(judged) == 6 and all(x.run == 1 for x in judged)  # 3 answerable questions, 2 systems
    first = json.loads((pass2 / "runs.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert first["pass"] == 2

    # Pass 3 runs the graph first; the paths come from the manifest this time.
    events.clear()
    again = ["repeat-run", "--base", str(env["base"]), "--pass", "3"]
    assert cli.main(again, settings=env["settings"]) == 0
    assert events == ["warm", "agentic", "standard"]
    assert len(read_records(system_file(env["base"] / "pass3", "agentic"))) == 4

    # A finished pass is not run again, and nothing is warmed up for it.
    events.clear()
    assert cli.main(again, settings=env["settings"]) == 0
    assert events == []
    assert len(read_records(system_file(env["base"] / "pass3", "agentic"))) == 4


def test_run_needs_a_pass_after_the_stored_one_and_known_sources(
    env: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(_argv(env, 1), settings=env["settings"]) == 2  # pass 1 is the stored run
    bare = ["repeat-run", "--base", str(env["base"] / "other"), "--pass", "2"]
    assert cli.main(bare, settings=env["settings"]) == 2  # no list and no pass 1 given anywhere
    assert "--pass1-old" in capsys.readouterr().err


def test_a_question_missing_from_the_sets_is_refused(
    env: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(repeat_cli, "load_items", lambda kb, **kw: ITEMS[:2])
    assert cli.main(_argv(env, 2, "--estimate-only"), settings=env["settings"]) == 2


def test_analyze_writes_report_and_stable_file(env: dict[str, Any], tmp_path: Path) -> None:
    ql = qlist_for(QIDS)
    spec = {
        "standard": {
            "single_hop:00": "ccc",
            "single_hop:01": "cwc",
            "hard:00": "cwc",
            "ambiguous:00": "cwc",
        },
        "agentic": {
            "single_hop:00": "ccc",
            "single_hop:01": "ccc",
            "hard:00": "cxc",
            "ambiguous:00": "ccc",
        },
    }
    data = passes_from(spec)
    base = env["base"]
    # Pass 1 where `repeat-run` finds it, later passes in BASE/passN.
    p1_old, p1_new = tmp_path / "p1-old", tmp_path / "p1-new"
    write_pass(p1_old, data[0], only={"single_hop:00", "single_hop:01"})
    write_pass(p1_new, data[0], only={"hard:00", "ambiguous:00"})
    write_pass(base / "pass2", data[1])
    write_pass(base / "pass3", data[2])
    report = tmp_path / "docs" / "reports" / "eval-upgrade-report.md"
    stable = tmp_path / "stable.json"
    argv = [
        "repeat-analyze",
        "--base",
        str(base),
        "--list",
        str(env["list"]),
        "--pass1-old",
        str(p1_old),
        "--pass1-new",
        str(p1_new),
        "--report",
        str(report),
        "--stable-out",
        str(stable),
    ]
    assert cli.main(argv) == 0
    text = report.read_text(encoding="utf-8")
    assert "## 第 3 步" in text and "hard:00" in text  # the cut record is listed
    payload = json.loads(stable.read_text(encoding="utf-8"))
    assert payload["agentic"]["single_hop"] == ["single_hop:00", "single_hop:01"]
    assert payload["standard"]["single_hop"] == ["single_hop:00"]
    assert payload["agentic"]["ambiguous"] == ["ambiguous:00"]
    assert ql.seed == payload["question_list_seed"]

    # Running it again replaces the section and nothing else.
    report.write_text(
        report.read_text(encoding="utf-8") + "\n## 第 4 步：别的\n\n内容\n", encoding="utf-8"
    )
    assert cli.main(argv) == 0
    again = report.read_text(encoding="utf-8")
    assert again.count("<!-- eval-upgrade:step3:begin -->") == 1
    assert "## 第 4 步：别的" in again

    # Paths are read from the manifest when `repeat-run` saved them: a missing pass is an error.
    (base / "pass3" / "judge.jsonl").unlink()
    for system in SYSTEMS:
        system_file(base / "pass3", system).unlink()
    assert cli.main(argv) == 2


def test_analyze_without_the_second_pass_is_an_error(env: dict[str, Any]) -> None:
    argv = ["repeat-analyze", "--base", str(env["base"]), "--list", str(env["list"])]
    assert cli.main(argv) == 2


def test_sample_command_writes_and_checks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    real = read_question_list()
    out = tmp_path / "list.json"
    assert cli.main(["repeat-sample", "--out", str(out)]) == 0
    assert read_question_list(out) == real
    assert cli.main(["repeat-sample", "--check"]) == 0  # the shipped file is what the seed draws
    assert (
        cli.main(["repeat-sample", "--check", "--seed", "1"]) == 2
    )  # another seed draws another list
    assert cli.main(["repeat-sample", "--check", "--out", str(tmp_path / "none.json")]) == 2


def test_run_reports_an_incomplete_pass_and_does_not_judge(
    env: dict[str, Any], monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    async def stopped(items: Any, **kw: Any) -> RunStats:
        return RunStats(pending=len(items), stopped="budget")

    monkeypatch.setattr(cli, "run_standard", stopped)
    monkeypatch.setattr(cli, "run_agentic", stopped)
    assert cli.main(_argv(env, 2), settings=env["settings"]) == 1
    assert "not complete" in capsys.readouterr().err
    assert not (env["base"] / "pass2" / "judge.jsonl").exists()
