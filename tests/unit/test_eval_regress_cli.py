"""Spec EVAL-4 AC7 to AC13: the `regress` command with fake runs (no model, no network)."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from rag.config import ServiceSettings
from rag.eval.agentic import cli
from rag.eval.agentic.records import JudgeLine, append_line, read_records, system_file
from rag.eval.agentic.regress import cli as regress_cli
from rag.eval.agentic.regress.suite import build_suite, write_baseline, write_suite
from rag.eval.agentic.sets import EvalItem
from rag.observe import shutdown_observability
from rag.testing.repeat import SCORES, passes_from, qlist_for, record

QIDS = [
    "single_hop:00",
    "single_hop:01",
    "single_hop:02",
    "two_hop:00",
    "two_hop:01",
    "colloquial:00",
    "hard:00",
    "hard:01",
    "ambiguous:00",
]
STANDARD_STABLE = {"single_hop:00", "single_hop:01", "hard:00"}


@pytest.fixture(autouse=True)
def _no_active_observability() -> Iterator[None]:
    shutdown_observability()
    yield
    shutdown_observability()


def _item(qid: str) -> EvalItem:
    qtype = qid.split(":")[0]
    ambiguous = qtype == "ambiguous"
    return EvalItem(
        qid=qid,
        qtype=qtype,  # type: ignore[arg-type]
        set="x",
        question=f"Question {qid}?",
        reference="" if ambiguous else "Reference.",
        relevant_pages=[] if ambiguous else ["hf_docs:p.md"],
        answerable=not ambiguous,
    )


ITEMS = [_item(q) for q in QIDS]


class Fake:
    """Replaces the runner and the judge: writes the records and judge lines a script says.

    `script[(system, qid)]` is one outcome character per attempt (see `rag.testing.repeat`); the
    last one repeats; the default is `c`. Every record costs `step` CNY in the ledger."""

    def __init__(self, ledger: Path, step: float = 0.01) -> None:
        self.script: dict[tuple[str, str], str] = {}
        self.calls: list[tuple[int, str, list[str]]] = []
        self.args: list[Any] = []
        self.events: list[str] = []
        self.ledger, self.step = ledger, step
        self._outcome: dict[tuple[str, str, str], str] = {}

    def run(self, args: Any, settings: Any) -> int:
        self.args.append(args)
        out = Path(args.out)
        attempt = int(out.name.removeprefix("attempt"))
        qids = [q for q in args.qids.split(",") if q]
        self.calls.append((attempt, args.system, qids))
        self.events.append(f"run:{args.system}")
        for qid in qids:
            row = self.script.get((args.system, qid), "c")
            outcome = row[min(attempt - 1, len(row) - 1)]
            if outcome == "n":
                continue
            self._outcome[(str(out), args.system, qid)] = outcome
            append_line(
                system_file(out, args.system), record(args.system, qid, outcome, cny=self.step)
            )
            with self.ledger.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({"kind": "chat", "cny": self.step}) + "\n")
        return 0

    def judge(self, args: Any, settings: Any) -> int:
        out = Path(args.out)
        self.events.append("judge")
        for system in ("standard", "agentic"):
            for r in read_records(system_file(out, system)):
                score = SCORES.get(self._outcome[(str(out), system, r.qid)])
                if r.answerable and score is not None:
                    append_line(
                        out / "judge.jsonl",
                        JudgeLine(system=system, qid=r.qid, run=1, score=score),  # type: ignore[arg-type]
                    )
        return 0


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    ledger = tmp_path / "ledger.jsonl"
    ledger.write_text('{"kind": "chat", "cny": 30.0}\n', encoding="utf-8")
    fake = Fake(ledger)
    monkeypatch.setattr(regress_cli, "load_items", lambda kb, **kw: ITEMS)
    monkeypatch.setattr(cli, "_run", fake.run)
    monkeypatch.setattr(cli, "_judge", fake.judge)
    monkeypatch.setattr(
        cli, "build_retriever", lambda cfg, settings, **kw: SimpleNamespace(close=lambda: None)
    )
    monkeypatch.setattr(regress_cli, "warm_up", lambda *a, **kw: fake.events.append("warm"))
    spec = {
        "agentic": {q: "ccc" for q in QIDS},
        "standard": {q: "ccc" if q in STANDARD_STABLE else "cwc" for q in QIDS},
    }
    rs, baseline = build_suite(
        passes_from(spec),
        qlist=qlist_for(QIDS),
        items=ITEMS,
        seed=1,
        date="2026-10-23",
        max_per_type={},
        sources={},
    )
    set_path = tmp_path / "set.json"
    write_suite(set_path, rs)
    write_baseline(tmp_path / "set.baseline.json", baseline)
    settings = ServiceSettings(
        chat_backend="fake",
        otel_mode="off",
        business_record_path="",
        log_dir="",
        cost_ledger_path=str(ledger),
        budget_cny=39.0,
        budget_stop_cny=37.0,
    )
    return {
        "fake": fake,
        "set": set_path,
        "out": tmp_path / "out",
        "settings": settings,
        "ledger": ledger,
        "tmp": tmp_path,
        "baseline": baseline,
        "order": rs.qids(),
    }


def _argv(env: dict[str, Any], *extra: str) -> list[str]:
    return ["regress", "--set", str(env["set"]), "--out", str(env["out"]), *extra]


def _go(env: dict[str, Any], *extra: str) -> int:
    return cli.main(_argv(env, *extra), settings=env["settings"])


def _result(env: dict[str, Any], out: Path | None = None) -> dict[str, Any]:
    return json.loads(((out or env["out"]) / "result.json").read_text(encoding="utf-8"))


def test_estimate_first_and_refusal_over_the_stop_line(
    env: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    env["ledger"].write_text('{"kind": "chat", "cny": 36.95}\n', encoding="utf-8")
    assert _go(env) == 2
    out = capsys.readouterr()
    assert "estimated cost of this run" in out.out
    assert out.out.index("estimated cost") < out.out.index("ledger: spent")
    assert "0.1080" in out.out  # 9 questions x 0.01 from the baseline, 20% margin
    assert "stop line" in out.err
    assert env["fake"].events == [] and not env["out"].exists()  # no warm-up, no run, no model


def test_estimate_only_runs_nothing(
    env: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    assert _go(env, "--estimate-only") == 0
    out = capsys.readouterr().out
    assert "estimated cost of this run" in out and "ledger: spent" in out
    assert "agentic: 0.0900" in out and "standard" not in out
    assert env["fake"].events == [] and not env["out"].exists()


def test_all_correct_passes_in_one_attempt(
    env: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    assert _go(env) == 0
    fake: Fake = env["fake"]
    assert fake.events == ["warm", "run:agentic", "judge"]
    assert fake.calls == [(1, "agentic", env["order"])]
    assert "RESULT: PASS" in capsys.readouterr().out
    result = _result(env)
    assert result["verdict"] == "PASS"
    assert {q["status"] for q in result["questions"]} == {"pass"}
    assert result["spent_cny"] == pytest.approx(0.09)


def test_regression_is_reported_and_only_flagged_questions_rerun(
    env: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    fake: Fake = env["fake"]
    fake.script[("agentic", "hard:00")] = "w"  # wrong in every attempt
    fake.script[("agentic", "two_hop:01")] = "wcw"  # wrong, right, wrong
    assert _go(env) == 1
    assert fake.calls == [
        (1, "agentic", env["order"]),
        (2, "agentic", ["two_hop:01", "hard:00"]),
        (3, "agentic", ["two_hop:01"]),  # hard:00 was decided after two wrong attempts
    ]
    assert "RESULT: REGRESSION" in capsys.readouterr().out
    result = _result(env)
    status = {q["qid"]: q["status"] for q in result["questions"]}
    assert status["hard:00"] == "regression" and status["two_hop:01"] == "regression"
    assert status["single_hop:00"] == "pass"
    assert result["counts"]["regression"] == 2


def test_a_single_miss_is_recovered(
    env: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    fake: Fake = env["fake"]
    fake.script[("agentic", "hard:00")] = "wcc"
    assert _go(env) == 0
    assert fake.calls == [
        (1, "agentic", env["order"]),
        (2, "agentic", ["hard:00"]),
        (3, "agentic", ["hard:00"]),
    ]
    out = capsys.readouterr().out
    assert "RESULT: PASS" in out and "recovered" in out and "hard:00" in out


def test_run_problems_are_inconclusive_not_regression(
    env: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    fake: Fake = env["fake"]
    fake.script[("agentic", "hard:00")] = "x"  # cut by the deadline in every attempt
    fake.script[("agentic", "hard:01")] = "t"  # answer model timeout in every attempt
    assert _go(env) == 3
    out = capsys.readouterr().out
    assert "RESULT: INCONCLUSIVE" in out and "run problem" in out
    result = _result(env)
    status = {q["qid"]: q["status"] for q in result["questions"]}
    assert status["hard:00"] == status["hard:01"] == "run_problem"
    assert result["counts"]["regression"] == 0
    assert len(fake.calls) == 5  # one attempt after the other, up to --max-attempts

    # A regression found on another question is still reported, with the problems beside it.
    fake.calls.clear()
    fake.script[("agentic", "single_hop:00")] = "w"
    assert _go(env, "--out", str(env["tmp"] / "out2")) == 1
    assert "RESULT: REGRESSION" in capsys.readouterr().out
    again = _result(env, env["tmp"] / "out2")
    assert again["counts"]["regression"] == 1 and again["counts"]["run_problem"] == 2


def test_broken_environment_stops_early(
    env: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    fake: Fake = env["fake"]
    for qid in QIDS:
        fake.script[("agentic", qid)] = "e"  # every run fails
    assert _go(env) == 3
    assert len(fake.calls) == 1  # no blind retrying of a broken environment
    out = capsys.readouterr().out
    assert "RESULT: INCONCLUSIVE" in out and "environment" in out


def test_rerun_stage_respects_the_stop_line(
    env: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    fake: Fake = env["fake"]
    fake.step = 0.2  # the first attempt of 9 questions takes the ledger past the line
    for rows in env["baseline"].entries.values():
        for passes in rows.values():
            for p in passes:
                p.cny = 0.1
    write_baseline(env["tmp"] / "set.baseline.json", env["baseline"])
    env["ledger"].write_text('{"kind": "chat", "cny": 35.9}\n', encoding="utf-8")
    fake.script[("agentic", "hard:00")] = "w"
    assert _go(env) == 3
    assert [c[0] for c in fake.calls] == [1]  # the second stage was not run
    out = capsys.readouterr().out
    assert "stop line" in out and "RESULT: INCONCLUSIVE" in out
    status = {q["qid"]: q["status"] for q in _result(env)["questions"]}
    assert status["hard:00"] == "run_problem"


def test_with_standard_guards_the_stable_standard_questions(env: dict[str, Any]) -> None:
    fake: Fake = env["fake"]
    fake.script[("standard", "single_hop:01")] = "w"
    assert _go(env, "--with-standard") == 1
    first = [c for c in fake.calls if c[0] == 1]
    assert (1, "agentic", env["order"]) in first
    wanted = [q for q in env["order"] if q in STANDARD_STABLE]
    assert (1, "standard", wanted) in first
    bad = [
        (q["system"], q["qid"]) for q in _result(env)["questions"] if q["status"] == "regression"
    ]
    assert bad == [("standard", "single_hop:01")]


def test_result_files_label_overrides(
    env: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    assert _go(env, "--kb-override", "retrieval.threshold=0.9") == 0
    result = _result(env)
    assert result["overrides"] == ["retrieval.threshold=0.9"]
    assert result["set"] == "regression_v1" and result["set_date"] == "2026-10-23"
    assert "code_revision" in result and result["spent_cny"] > 0
    text = (env["out"] / "result.md").read_text(encoding="utf-8")
    assert "NOT the committed configuration" in text and "retrieval.threshold=0.9" in text
    assert all(a.kb_override == ["retrieval.threshold=0.9"] for a in env["fake"].args)
    assert "NOT the committed configuration" in capsys.readouterr().out


def test_unknown_override_and_changed_dataset_are_refused(
    env: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    assert _go(env, "--kb-override", "retrieval.nope=1") == 2
    changed = [i.model_copy(update={"question": "Another?"}) for i in ITEMS]
    monkeypatch.setattr(regress_cli, "load_items", lambda kb, **kw: changed)
    assert _go(env, "--estimate-only") == 2
    assert env["fake"].events == []


def test_refusals_before_anything_runs(env: dict[str, Any], tmp_path: Path) -> None:
    fake: Fake = env["fake"]
    assert _go(env, "--max-attempts", "2") == 2  # the rule looks at three clean attempts
    assert _go(env, "--kb", "mt5") == 2  # the set is for another knowledge base
    wrong = env["baseline"].model_copy(update={"set_name": "regression_v0"})
    write_baseline(env["tmp"] / "set.baseline.json", wrong)
    assert _go(env) == 2  # a baseline of another set
    write_baseline(env["tmp"] / "set.baseline.json", env["baseline"])
    del env["baseline"].entries[QIDS[0]]["agentic"]
    write_baseline(env["tmp"] / "set.baseline.json", env["baseline"])
    assert _go(env) == 2  # a question without its passes: stopped before any money is spent
    assert fake.events == []
    # An output directory that already holds records would be reused by the runner.
    stale = tmp_path / "old"
    append_line(system_file(stale / "attempt1", "agentic"), record("agentic", QIDS[0], "c"))
    assert _go(env, "--out", str(stale)) == 2
    assert fake.events == []


def test_spend_limit_of_the_run(env: dict[str, Any], capsys: pytest.CaptureFixture[str]) -> None:
    fake: Fake = env["fake"]
    for rows in env["baseline"].entries.values():
        for passes in rows.values():
            for p in passes:
                p.cny = 0.1
    write_baseline(env["tmp"] / "set.baseline.json", env["baseline"])
    assert _go(env, "--max-cny", "0.5") == 2  # the first attempt alone is estimated at 1.08
    assert fake.events == []
    fake.step = 0.12  # the first attempt really costs 1.08; the second one would pass the limit
    fake.script[("agentic", "hard:00")] = "w"
    assert _go(env, "--max-cny", "1.1") == 3
    assert [c[0] for c in fake.calls] == [1]
    assert "spend limit" in capsys.readouterr().out


def test_warm_up_failure_stops_the_run_as_not_decided(
    env: dict[str, Any], monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def broken(*a: Any, **kw: Any) -> None:
        raise regress_cli.UsageError("2 searches still fail")

    monkeypatch.setattr(regress_cli, "warm_up", broken)
    assert _go(env) == 2
    assert "run problem, not a regression" in capsys.readouterr().err
    assert env["fake"].calls == []


def test_regress_build_writes_the_files_once(
    env: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    from rag.eval.agentic.regress.suite import read_baseline, read_suite
    from rag.eval.agentic.repeat.passes import RepeatManifest, save_manifest
    from rag.eval.agentic.repeat.sample import write_question_list
    from rag.testing.repeat import write_pass

    tmp: Path = env["tmp"]
    base = tmp / "step3"
    spec = {
        "agentic": {q: "ccc" if q != "hard:01" else "cwc" for q in QIDS},
        "standard": {q: "ccc" for q in QIDS},
    }
    passes = passes_from(spec)
    write_pass(tmp / "p1-old", passes[0], only=set(QIDS[:5]))
    write_pass(tmp / "p1-new", passes[0], only=set(QIDS[5:]))
    write_pass(base / "pass2", passes[1])
    write_pass(base / "pass3", passes[2])
    write_question_list(tmp / "list.json", qlist_for(QIDS))
    save_manifest(
        base,
        RepeatManifest(
            question_list=str(tmp / "list.json"),
            pass1_old=[str(tmp / "p1-old")],
            pass1_new=[str(tmp / "p1-new")],
        ),
    )
    monkeypatch.setattr(regress_cli.repeat_cli, "load_items", lambda kb, **kw: ITEMS)
    out, base_out = tmp / "built" / "set.json", tmp / "built" / "set.baseline.json"
    argv = [
        "regress-build",
        "--base",
        str(base),
        "--out",
        str(out),
        "--baseline-out",
        str(base_out),
    ]
    assert cli.main([*argv, "--max-single-hop", "2", "--seed", "9"]) == 0
    rs = read_suite(out)
    assert rs.counts["single_hop"] == 2 and "hard:01" not in rs.qids()  # not stable: one miss
    assert rs.seed == 9 and rs.qids_for("standard") == rs.qids()
    assert set(read_baseline(base_out).entries) == set(rs.qids())
    assert cli.main(argv) == 2  # a set is not rewritten once written
    assert cli.main([*argv, "--force"]) == 0
