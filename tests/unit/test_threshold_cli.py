"""Spec P2-02 AC7-AC9: `collect` and `sweep` of `python -m rag.eval.threshold`.

The real chain (Elasticsearch, gateway, cross-encoder) is replaced by `FakeRetriever` through the
`build_retriever` seam of the CLI module.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from rag.config import KnowledgeBaseConfig, ServiceSettings
from rag.eval.questions import LabeledQuestion, save_question_set
from rag.eval.retrieval.record import Provenance, QuestionSetInfo
from rag.eval.threshold import cli
from rag.eval.threshold.records import Candidate, ScoreFile, ScoreRecord, load_scores, write_scores
from rag.testing import FakeRetriever
from rag.types import RetrievalResult, RouteHit

KB = "hf_docs"
KB_YAML = """\
kb: hf_docs
source: {connector: hf_dataset, limit: null}
language: en
retrieval: {rerank_n: 10, final_k: 8, threshold: 0.3}
"""
ANSWERABLE = [
    LabeledQuestion(qid="q1", question="how to log in?", answer="a", relevant_pages=["hf_docs:a"]),
    LabeledQuestion(qid="q2", question="where are logs?", answer="b", relevant_pages=["hf_docs:b"]),
]
NO_ANSWER = [LabeledQuestion(qid="n1", question="capital of Australia?")]


def _result(scores: dict[str, float], *, timeout: bool = False) -> RetrievalResult:
    """A result whose reranker list holds one candidate per page in `scores` (best first); the
    final list is empty, as it is when a threshold dropped everything."""
    rerank = [
        RouteHit(chunk_id=f"hf_docs:{p}#0", doc_id=f"hf_docs:{p}", rank=i + 1, score=s)
        for i, (p, s) in enumerate(sorted(scores.items(), key=lambda kv: -kv[1]))
    ]
    return RetrievalResult(request_id="r", kb=KB, rerank=rerank, rerank_timeout=timeout)


class _Built:
    def __init__(self, results: dict[str, RetrievalResult]) -> None:
        self.calls: list[dict[str, Any]] = []
        self.retriever = FakeRetriever(results)  # type: ignore[arg-type]

    def __call__(
        self, cfg: KnowledgeBaseConfig, settings: ServiceSettings, **kw: Any
    ) -> FakeRetriever:
        self.calls.append({"cfg": cfg, "settings": settings, **kw})
        return self.retriever


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    kb_dir = tmp_path / "kb"
    (kb_dir / "public").mkdir(parents=True)
    (kb_dir / "public" / "hf_docs.yaml").write_text(KB_YAML, encoding="utf-8")
    answerable = tmp_path / "answerable.jsonl"
    no_answer = tmp_path / "no_answer.jsonl"
    save_question_set(ANSWERABLE, answerable)
    save_question_set(NO_ANSWER, no_answer)
    built = _Built(
        {
            "how to log in?": _result({"a": 0.95, "x": 0.1}),
            "where are logs?": _result({"y": 0.8, "b": 0.4}),
            "capital of Australia?": _result({"z": 0.5, "w": 0.01}),
        }
    )
    monkeypatch.setattr(cli, "build_retriever", built)
    monkeypatch.setattr(cli, "reranker_revision", lambda model: "abc123")
    settings = ServiceSettings(_env_file=None, kb_dir=str(kb_dir))
    return {
        "settings": settings,
        "built": built,
        "tmp": tmp_path,
        "answerable": answerable,
        "no_answer": no_answer,
    }


def _collect(env: dict[str, Any], name: str, kind: str, *extra: str) -> tuple[int, Path]:
    out = env["tmp"] / f"{name}.json"
    argv = ["collect", "--kb", KB, "--questions", str(env[name]), "--kind", kind, "--out", str(out)]
    code = cli.main([*argv, *extra], settings=env["settings"])
    return code, out


def test_collect_records_all_candidates_with_threshold_zero(env: dict[str, Any]) -> None:
    code, out = _collect(env, "answerable", "answerable", "--deadline-ms", "60000")
    assert code == 0
    call = env["built"].calls[-1]
    assert call["cfg"].retrieval.threshold == 0.0  # the YAML's 0.3 is overridden
    assert call["cfg"].retrieval.rerank_n == 10
    assert call["rerank"] is True
    assert call["settings"].search_deadline_ms == 60000
    scores = load_scores(out)
    assert scores.kb == KB and scores.rerank_n == 10
    assert scores.question_set.count == 2
    assert scores.provenance.models["reranker_revision"] == "abc123"
    first, second = scores.records
    assert (first.qid, first.kind, first.relevant_pages) == ("q1", "answerable", ["hf_docs:a"])
    assert [(c.doc_id, c.score) for c in first.candidates] == [
        ("hf_docs:a", 0.95),
        ("hf_docs:x", 0.1),
    ]  # every reranked candidate, also those far below the YAML's threshold
    assert second.candidates[1].score == 0.4

    code, out = _collect(env, "no_answer", "no_answer")
    assert code == 0
    (only,) = load_scores(out).records
    assert (only.kind, only.relevant_pages) == ("no_answer", [])


def test_collect_rejects_rerank_timeout(
    env: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    env["built"].retriever.results["where are logs?"] = _result({"b": 0.4}, timeout=True)
    code, out = _collect(env, "answerable", "answerable")
    assert code == 1
    err = capsys.readouterr().err
    assert err.startswith("error:") and "q2" in err and "rerank_timeout" in err
    assert not out.exists()  # a partial score file is never written


def test_collect_checks_labels_against_kind(
    env: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    code, _ = _collect(env, "answerable", "no_answer")  # labelled questions are not no-answer
    assert code == 2
    assert "no_answer" in capsys.readouterr().err
    code, _ = _collect(env, "no_answer", "answerable")  # an answerable question needs a label
    assert code == 2


def _score_file(path: Path, records: list[ScoreRecord]) -> Path:
    write_scores(
        ScoreFile(
            created_at="2026-10-19T00:00:00+00:00",
            kb=KB,
            rerank_n=10,
            question_set=QuestionSetInfo(path="x.jsonl", sha256="0" * 64, count=len(records)),
            provenance=Provenance(),
            records=records,
        ),
        path,
    )
    return path


def _r(qid: str, top: float, kind: str = "answerable") -> ScoreRecord:
    return ScoreRecord(
        qid=qid,
        question=qid,
        kind=kind,  # type: ignore[arg-type]
        relevant_pages=["hf_docs:p"] if kind == "answerable" else [],
        candidates=[Candidate(chunk_id="hf_docs:p#0", doc_id="hf_docs:p", score=top)],
    )


@pytest.fixture
def files(tmp_path: Path) -> dict[str, Path]:
    # dev tops: 0.9 x8, 0.4, 0.2 -> false refusals 0 at 0.1, 1 at 0.3, 2 at 0.5, 0.7, 0.9
    dev = [_r(f"d{i}", 0.9) for i in range(8)] + [_r("d8", 0.4), _r("d9", 0.2)]
    public = [_r(f"p{i}", 0.95) for i in range(5)] + [_r("p5", 0.35)]
    noans_a = [_r("n0", 0.05, "no_answer"), _r("n1", 0.6, "no_answer")]
    noans_b = [_r("m0", 0.45, "no_answer"), _r("m1", 0.99, "no_answer")]
    return {
        "dev": _score_file(tmp_path / "dev.json", dev),
        "public": _score_file(tmp_path / "public.json", public),
        "na1": _score_file(tmp_path / "na1.json", noans_a),
        "na2": _score_file(tmp_path / "na2.json", noans_b),
    }


def test_sweep_prints_table_and_recommendation(
    files: dict[str, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    out = files["dev"].parent / "sweep.md"
    code = cli.main(
        [
            "sweep",
            "--dev",
            str(files["dev"]),
            "--check",
            f"public={files['public']}",
            "--no-answer",
            f"original={files['na1']}",
            "--no-answer",
            f"extra={files['na2']}",
            "--thresholds",
            "0.1:0.9:0.2",
            "--max-false-refusal",
            "0.1",
            "--out",
            str(out),
        ]
    )
    assert code == 0
    text = capsys.readouterr().out
    assert out.read_text(encoding="utf-8").strip() == text.strip()
    # dev false refusals at 0.1, 0.3, 0.5, 0.7, 0.9: 0, 1, 2, 2, 2 of 10
    assert "| 0.30 | 1/10" in text
    assert "public" in text and "original" in text and "extra" in text
    assert "all no-answer" in text  # pooled column over the 2 + 2 questions
    assert "Recommended threshold: 0.30" in text
    assert "10%" in text and "observed" in text  # the rule is stated with its cap


def test_check_and_no_answer_sets_do_not_move_the_recommendation(
    files: dict[str, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    base = ["sweep", "--dev", str(files["dev"]), "--thresholds", "0.1:0.9:0.2"]
    cap = ["--max-false-refusal", "0.1"]
    assert cli.main([*base, *cap]) == 0
    alone = capsys.readouterr().out
    others = ["--check", f"public={files['public']}", "--no-answer", f"na={files['na1']}"]
    assert cli.main([*base, *cap, *others]) == 0
    with_others = capsys.readouterr().out
    # public would already refuse 1 of 6 at 0.4, but only dev chooses the threshold.
    assert "Recommended threshold: 0.30" in alone
    assert "Recommended threshold: 0.30" in with_others


def test_sweep_says_so_when_no_threshold_fits(
    files: dict[str, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    argv = ["sweep", "--dev", str(files["dev"]), "--thresholds", "0.5,0.9"]
    assert cli.main([*argv, "--max-false-refusal", "0"]) == 0
    assert "No threshold" in capsys.readouterr().out


def test_sweep_usage_errors_exit_2(
    files: dict[str, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    dev = str(files["dev"])
    bad_lines = [
        ["sweep", "--dev", dev, "--thresholds", "0.9:0.1:0.1"],
        ["sweep", "--dev", dev, "--max-false-refusal", "1.5"],
        ["sweep", "--dev", dev, "--check", "public"],  # not NAME=FILE
        ["sweep", "--dev", dev, "--check", f"public={files['public']}.missing"],
        ["sweep", "--dev", dev, "--no-answer", f"na={files['public']}"],  # answerable file given
        ["sweep", "--dev", str(files["na1"])],  # no-answer file given as dev
        ["sweep", "--dev", dev, "--check", f"dev={files['public']}"],  # name taken by --dev
    ]
    for argv in bad_lines:
        assert cli.main(argv) == 2, argv
        err = capsys.readouterr().err
        assert err.startswith("error:") and err.count("\n") == 1, (argv, err)
    with pytest.raises(SystemExit) as exc:
        cli.main(["sweep", "--dev", dev, "--bound", "median"])
    assert exc.value.code == 2
