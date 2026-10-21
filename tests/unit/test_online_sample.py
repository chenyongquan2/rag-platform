"""Spec P5-02 AC8: judge a random sample of recent real requests (no gold labels)."""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from rag.eval.online_sample import (
    ONLINE_SAMPLE_HARD_CAP,
    Candidate,
    load_candidates,
    run_online_sample,
    sample_candidates,
)
from rag.eval.stats import wilson_interval
from rag.observe import shutdown_observability
from rag.testing import memory_observability, metric_points, observe_settings
from rag.testing.agent import judge_reply
from rag.testing.fakes import FakeChatModel

NOW = datetime(2026, 10, 21, 12, 0, 0, tzinfo=UTC)


@pytest.fixture(autouse=True)
def _no_active_observability() -> Iterator[None]:
    shutdown_observability()
    yield
    shutdown_observability()


def _record(
    request_id: str,
    *,
    tool: str = "search_docs",
    minutes_ago: float = 5,
    final: Sequence[str] = ("hf_docs:a.md#0",),
    flags: Sequence[str] = (),
    extra: Mapping[str, Any] | None = None,
    kb: str = "hf_docs",
) -> dict[str, Any]:
    key = "question" if tool == "ask_docs" else "query"
    default_extra = {key: f"question of {request_id}"}
    return {
        "schema_version": 1,
        "time": (NOW - timedelta(minutes=minutes_ago)).isoformat(),
        "request_id": request_id,
        "tool": tool,
        "kb": kb,
        "final": [{"chunk_id": c, "score": 0.9} for c in final],
        "flags": list(flags),
        "extra": dict(extra) if extra is not None else default_extra,
    }


def _write(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), "utf-8")


class Chunks:
    """Stands in for Elasticsearch: the text of a chunk id."""

    def __init__(self, texts: Mapping[str, str] | None = None) -> None:
        self.asked: list[tuple[str, tuple[str, ...]]] = []
        self.texts = texts

    def fetch(self, kb: str, chunk_ids: Sequence[str]) -> dict[str, tuple[list[str], str]]:
        self.asked.append((kb, tuple(chunk_ids)))
        return {
            c: (["Guide", "Page"], (self.texts or {}).get(c, f"text of {c}")) for c in chunk_ids
        }


def test_candidates_are_recent_answered_requests_with_a_question_and_chunks(tmp_path: Path) -> None:
    path = tmp_path / "business.jsonl"
    _write(
        path,
        [
            _record("ok-search"),
            _record("ok-ask", tool="ask_docs"),
            _record("old", minutes_ago=60 * 24 * 9),  # outside the window
            _record("failed", flags=["internal_error"]),  # errors are not judged
            _record("degraded", flags=["degraded"]),  # a degraded answer is still an answer
            _record("no-chunks", final=[]),
            _record("no-question", extra={}),
            _record("feedback", tool="feedback", extra={"useful": True}),
            _record("list", tool="list_knowledge_bases", extra={"knowledge_bases": ["a"]}),
            _record("cancelled", flags=["client_cancelled"]),
        ],
    )
    got = load_candidates(path, now=NOW, window_hours=24 * 7)
    assert sorted(c.request_id for c in got) == ["degraded", "ok-ask", "ok-search"]
    ask = next(c for c in got if c.request_id == "ok-ask")
    assert ask == Candidate(
        "ok-ask", "ask_docs", "hf_docs", "question of ok-ask", ("hf_docs:a.md#0",)
    )


def test_candidates_also_come_from_the_rotated_files(tmp_path: Path) -> None:
    path = tmp_path / "business.jsonl"
    _write(path, [_record("new")])
    _write(tmp_path / "business.jsonl.1", [_record("older")])
    (tmp_path / "business.jsonl.2").write_text("not json\n" + json.dumps(_record("oldest")) + "\n")
    got = load_candidates(path, now=NOW, window_hours=24)
    assert sorted(c.request_id for c in got) == ["new", "older", "oldest"]  # a bad line is skipped


def test_a_missing_file_is_an_empty_pool(tmp_path: Path) -> None:
    assert load_candidates(tmp_path / "none.jsonl", now=NOW, window_hours=24) == []


def test_the_sample_is_random_but_repeatable_and_capped() -> None:
    pool = [Candidate(f"r{i}", "search_docs", "hf_docs", "q", ("c",)) for i in range(100)]
    a = sample_candidates(pool, 20, seed=1)
    b = sample_candidates(pool, 20, seed=1)
    c = sample_candidates(pool, 20, seed=2)
    assert len(a) == 20 and a == b and a != c and len({x.request_id for x in a}) == 20
    assert len(sample_candidates(pool[:5], 20, seed=1)) == 5  # fewer than asked: all of them
    assert len(sample_candidates(pool, 10_000, seed=1)) == ONLINE_SAMPLE_HARD_CAP
    assert sample_candidates(pool, 0, seed=1) == []


def _pool_file(tmp_path: Path, n: int) -> Path:
    path = tmp_path / "business.jsonl"
    _write(path, [_record(f"r{i}", final=(f"hf_docs:p{i}.md#0",)) for i in range(n)])
    return path


def test_the_judge_is_called_once_per_sampled_request_and_the_summary_has_a_wilson_interval(
    tmp_path: Path,
) -> None:
    path = _pool_file(tmp_path, 12)
    verdicts = [True, True, True, False, True, True, False, True]  # 8 requests are sampled
    model = FakeChatModel(responses=[judge_reply(v) for v in verdicts])
    chunks = Chunks()
    out = tmp_path / "online"
    with memory_observability() as obs:
        summary = run_online_sample(
            path,
            n=8,
            seed=3,
            now=NOW,
            model=model,
            chunks=chunks,
            out_dir=out,
            settings=observe_settings(),
            budget_check=lambda cny: 0.0,
        )
        points = metric_points(obs, "rag.online_eval.score_mean")
    assert len(model.calls) == 8 and len(chunks.asked) == 8
    assert (summary.n_sampled, summary.n_scored, summary.n_failed) == (8, 8, 0)
    assert summary.grounded == 6 and summary.rate == pytest.approx(0.75)
    low, high = wilson_interval(6, 8)
    assert (summary.wilson_low, summary.wilson_high) == pytest.approx((low, high))
    assert [p[1].value for p in points] == [pytest.approx(0.75)]
    # The file: summary and per-request lines, with no fragment text.
    (written,) = out.glob("online-*.json")
    data = json.loads(written.read_text("utf-8"))
    assert data["rate"] == pytest.approx(0.75) and data["wilson"] == pytest.approx([low, high])
    assert len(data["items"]) == 8 and data["items"][0]["request_id"].startswith("r")
    assert "text of" not in written.read_text("utf-8")
    assert data["measures"].startswith("whether the passages")  # says what it does not measure
    assert "answer correctness" in " ".join(data["limits"])


def test_the_judge_gets_the_question_and_the_chunk_text(tmp_path: Path) -> None:
    path = tmp_path / "business.jsonl"
    _write(
        path, [_record("one", extra={"query": "how to push a model"}, final=("hf_docs:x.md#2",))]
    )
    model = FakeChatModel(responses=[judge_reply(True)])
    run_online_sample(
        path,
        n=5,
        seed=0,
        now=NOW,
        model=model,
        chunks=Chunks({"hf_docs:x.md#2": "Call push_to_hub after login."}),
        out_dir=tmp_path / "o",
        settings=observe_settings(),
        budget_check=lambda cny: 0.0,
    )
    prompt = "\n".join(str(m.content) for m in model.calls[0])
    assert "how to push a model" in prompt and "Call push_to_hub after login." in prompt
    assert "untrusted" in prompt  # the fragments are in the marked block


def test_a_failed_judge_call_is_counted_not_scored(tmp_path: Path) -> None:
    path = _pool_file(tmp_path, 3)
    model = FakeChatModel(
        responses=[judge_reply(True), "not a tool call at all", judge_reply(False)]
    )
    with memory_observability():
        summary = run_online_sample(
            path,
            n=3,
            seed=0,
            now=NOW,
            model=model,
            chunks=Chunks(),
            out_dir=tmp_path / "o",
            settings=observe_settings(),
            budget_check=lambda cny: 0.0,
        )
    assert (summary.n_sampled, summary.n_scored, summary.n_failed) == (3, 2, 1)
    assert summary.rate == pytest.approx(0.5)


def test_chunks_that_are_gone_are_skipped_and_counted(tmp_path: Path) -> None:
    path = _pool_file(tmp_path, 2)

    class Gone(Chunks):
        def fetch(self, kb: str, chunk_ids: Sequence[str]) -> dict[str, tuple[list[str], str]]:
            return {}

    model = FakeChatModel(responses=[])
    with memory_observability():
        summary = run_online_sample(
            path,
            n=2,
            seed=0,
            now=NOW,
            model=model,
            chunks=Gone(),
            out_dir=tmp_path / "o",
            settings=observe_settings(),
            budget_check=lambda cny: 0.0,
        )
    assert model.calls == []  # nothing to judge, nothing paid
    assert (summary.n_scored, summary.n_no_text) == (0, 2) and summary.rate is None


def test_an_empty_pool_writes_a_summary_and_no_gauge(tmp_path: Path) -> None:
    with memory_observability() as obs:
        summary = run_online_sample(
            tmp_path / "none.jsonl",
            n=20,
            seed=0,
            now=NOW,
            model=FakeChatModel(responses=[]),
            chunks=Chunks(),
            out_dir=tmp_path / "o",
            settings=observe_settings(),
            budget_check=lambda cny: 0.0,
        )
        assert metric_points(obs, "rag.online_eval.score_mean") == []
    assert summary.n_pool == 0 and summary.n_sampled == 0 and summary.rate is None
    assert len(list((tmp_path / "o").glob("online-*.json"))) == 1


def test_the_budget_is_checked_before_any_call(tmp_path: Path) -> None:
    from rag.cost import BudgetExceeded

    def refuse(cny: float) -> float:
        raise BudgetExceeded("over the stop")

    model = FakeChatModel(responses=[judge_reply(True)])
    with pytest.raises(BudgetExceeded):
        run_online_sample(
            _pool_file(tmp_path, 3),
            n=3,
            seed=0,
            now=NOW,
            model=model,
            chunks=Chunks(),
            out_dir=tmp_path / "o",
            settings=observe_settings(),
            budget_check=refuse,
        )
    assert model.calls == []
