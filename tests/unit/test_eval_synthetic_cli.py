"""Spec P1-05 AC10: the synthetic question CLI end to end with fake pages and chat models."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest
from langchain_core.messages import AIMessage, BaseMessage

from rag.config import ServiceSettings
from rag.eval.questions import load_question_set
from rag.eval.synthetic import cli
from rag.eval.synthetic.sampler import Page
from rag.testing import FakeChatModel

SENTENCE = (
    "The library {w} keeps a cache of downloaded files in a folder under the home directory, "
    "so the second run of a script does not fetch the same weights again.\n"
)


class EchoGenerator(FakeChatModel):
    """Writes a different question for every span (the span's page number)."""

    def _next_message(self, messages: list[BaseMessage]) -> AIMessage:
        self._calls.append(list(messages))
        found = re.search(r"Page number (\d+)", str(messages[-1].content))
        n = found.group(1) if found else "?"
        return AIMessage(
            content=f"Factoid question: Which cache folder does tool {n} use?\nAnswer: Folder {n}."
        )


class Critic(FakeChatModel):
    """Rates every question 5, except relevance 2 for page numbers ending in 7."""

    def _next_message(self, messages: list[BaseMessage]) -> AIMessage:
        self._calls.append(list(messages))
        text = str(messages[-1].content)
        low = "useful" in text and re.search(r"tool \d*7 use\?", text)
        return AIMessage(content=f"Evaluation: ok.\nTotal rating: {2 if low else 5}")


def _pages() -> list[Page]:
    pages = []
    for i in range(60):
        repo = "huggingface/transformers" if i % 3 else "gradio-app/gradio"
        text = f"Page number {i}.\n" + "".join(SENTENCE.format(w=f"w{i}_{k}") for k in range(8))
        pages.append(
            Page(
                doc_id=f"hf_docs:{repo}/blob/main/p{i}.md",
                source_rel=f"{repo}/blob/main/p{i}.md",
                doc_type="markdown",
                language="en",
                text=text,
            )
        )
    return pages


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    monkeypatch.setattr(cli, "load_pages", lambda cfg, settings: _pages())
    monkeypatch.setattr(cli, "build_generator_model", lambda settings, config: EchoGenerator())
    monkeypatch.setattr(cli, "build_critic_model", lambda settings, config: Critic())
    settings = ServiceSettings(
        chat_backend="fake",
        otel_mode="off",
        business_record_path="",
        log_dir="",
        cost_ledger_path=str(tmp_path / "ledger.jsonl"),
    )
    return {"settings": settings, "tmp": tmp_path}


def _main(env: dict[str, Any], *argv: str) -> int:
    return cli.main(list(argv), settings=env["settings"])


def test_cli_end_to_end_with_fakes(env: dict[str, Any], capsys: pytest.CaptureFixture[str]) -> None:
    tmp: Path = env["tmp"]
    reports = tmp / "reports"
    assert (
        _main(
            env,
            "sample",
            "--kb",
            "hf_docs",
            "--n",
            "40",
            "--seed",
            "0",
            "--span-chars",
            "5000",
            "--min-chars",
            "100",
            "--label",
            "t",
            "--out-dir",
            str(reports),
        )
        == 0
    )
    (run,) = list(reports.iterdir())
    spans = (run / "spans.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(spans) == 40
    sample = json.loads((run / "sample.json").read_text(encoding="utf-8"))
    assert sample["config"]["seed"] == 0 and sample["allocation"]

    assert _main(env, "generate", "--run", str(run), "--limit", "10", "--concurrency", "1") == 0
    assert len((run / "generations.jsonl").read_text(encoding="utf-8").splitlines()) == 10
    assert _main(env, "generate", "--run", str(run), "--concurrency", "2") == 0  # resumes
    gens = [json.loads(x) for x in (run / "generations.jsonl").read_text("utf-8").splitlines()]
    assert len(gens) == 40 and len({g["span_id"] for g in gens}) == 40

    assert _main(env, "critique", "--run", str(run), "--concurrency", "2") == 0
    critiques = (run / "critiques.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(critiques) == 120
    assert _main(env, "critique", "--run", str(run)) == 0  # nothing left to score
    assert len((run / "critiques.jsonl").read_text(encoding="utf-8").splitlines()) == 120

    assert _main(env, "project", "--run", str(run), "--n", "420") == 0
    assert "projected" in capsys.readouterr().out

    out = tmp / "datasets"
    assert (
        _main(
            env,
            "build",
            "--run",
            str(run),
            "--kb",
            "hf_docs",
            "--seed",
            "0",
            "--min-each",
            "15",
            "--out-dir",
            str(out),
        )
        == 0
    )
    dev = load_question_set(out / "synthetic_dev.jsonl", kb="hf_docs")
    holdout = load_question_set(out / "synthetic_holdout.jsonl", kb="hf_docs")
    rejected = sum(1 for g in gens if re.search(r"tool \d*7 use\?", g["question"]))
    assert len(dev.questions) + len(holdout.questions) == 40 - rejected
    assert {q.qid for q in dev.questions}.isdisjoint({q.qid for q in holdout.questions})
    q = dev.questions[0]
    assert q.relevant_pages[0].startswith("hf_docs:") and q.answer
    assert q.meta["span"] and q.meta["scores"]["groundedness"] == 5

    manifest = json.loads((out / "synthetic_manifest.json").read_text(encoding="utf-8"))
    assert manifest["generator"]["model"] == "deepseek-v4-flash"
    assert manifest["critic"]["model"] == "qwen/qwen3.6-flash"
    assert manifest["prompt_versions"]["generation"] and manifest["prompt_versions"]["critique"]
    assert manifest["seeds"] == {"sample": 0, "split": 0}
    assert manifest["thresholds"]["min_relevance"] == 4
    assert manifest["counts"]["generated"] == 40
    assert manifest["counts"]["kept"] == 40 - rejected
    assert manifest["rejections"]["relevance_low"] == rejected
    assert manifest["files"]["synthetic_dev.jsonl"]["sha256"] == dev.sha256
    assert manifest["files"]["synthetic_holdout.jsonl"]["questions"] == len(holdout.questions)
    assert (run / "filter_report.json").exists()
