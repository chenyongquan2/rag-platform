"""Spec P1-05 AC10: the synthetic question CLI end to end with fake pages and chat models."""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path
from typing import Any

import pytest
from langchain_core.messages import AIMessage, BaseMessage
from pydantic import Field

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
    """Writes a different question for every span (the span's page number) and records the
    event loop of every call: the real model's HTTP clients fail on a second loop."""

    loops: set[int] = Field(default_factory=set)

    async def _agenerate(self, messages: list[BaseMessage], *args: Any, **kwargs: Any) -> Any:
        self.loops.add(id(asyncio.get_running_loop()))
        return self._generate(messages)

    def _next_message(self, messages: list[BaseMessage]) -> AIMessage:
        self._calls.append(list(messages))
        found = re.search(r"Page number (\d+)", str(messages[-1].content))
        n = found.group(1) if found else "?"
        return AIMessage(
            content=f"Factoid question: Where does library w{n}a keep files for package w{n}b?\n"
            f"Answer: Folder {n}."
        )


class Critic(FakeChatModel):
    """Rates every question 5, except relevance 2 for page numbers ending in 7."""

    def _next_message(self, messages: list[BaseMessage]) -> AIMessage:
        self._calls.append(list(messages))
        text = str(messages[-1].content)
        low = "useful" in text and re.search(r"library w\d*7a ", text)
        return AIMessage(content=f"Evaluation: ok.\nTotal rating: {2 if low else 5}")


def _pages() -> list[Page]:
    pages = []
    for i in range(60):
        repo = "huggingface/transformers" if i % 3 else "gradio-app/gradio"
        text = f"Page number {i}.\n" + "".join(SENTENCE.format(w=f"w{i}_{k}") for k in range(8))
        if i == 5:  # a Chinese page whose language metadata says English
            text += "这个页面是中文的介绍缓存目录在哪里 " * 20
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
    generators: list[EchoGenerator] = []

    def build_generator(settings: ServiceSettings, config: Any) -> EchoGenerator:
        generators.append(EchoGenerator())
        return generators[-1]

    monkeypatch.setattr(cli, "build_generator_model", build_generator)
    monkeypatch.setattr(cli, "build_critic_model", lambda settings, config: Critic())
    settings = ServiceSettings(
        chat_backend="fake",
        otel_mode="off",
        business_record_path="",
        log_dir="",
        cost_ledger_path=str(tmp_path / "ledger.jsonl"),
    )
    return {"settings": settings, "tmp": tmp_path, "generators": generators}


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
    assert sample["batches"][0]["config"]["seed"] == 0 and sample["batches"][0]["allocation"]

    assert _main(env, "generate", "--run", str(run), "--limit", "10", "--concurrency", "1") == 0
    assert len((run / "generations.jsonl").read_text(encoding="utf-8").splitlines()) == 10
    # a pilot run extended with spans of pages it does not have yet
    extend = ["--kb", "hf_docs", "--n", "10", "--seed", "5", "--span-chars", "5000"]
    assert _main(env, "sample", *extend, "--min-chars", "100", "--extend", str(run)) == 0
    spans = [json.loads(x) for x in (run / "spans.jsonl").read_text("utf-8").splitlines()]
    assert len(spans) == 50 and len({s["doc_id"] for s in spans}) == 50
    # resumes; several batches share one event loop
    assert _main(env, "generate", "--run", str(run), "--concurrency", "2", "--batch", "7") == 0
    assert len(env["generators"][-1].loops) == 1
    gens = [json.loads(x) for x in (run / "generations.jsonl").read_text("utf-8").splitlines()]
    assert len(gens) == 50 and len({g["span_id"] for g in gens}) == 50

    assert _main(env, "critique", "--run", str(run), "--concurrency", "2") == 0
    critiques = (run / "critiques.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(critiques) == 150
    assert _main(env, "critique", "--run", str(run)) == 0  # nothing left to score
    assert len((run / "critiques.jsonl").read_text(encoding="utf-8").splitlines()) == 150

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
    rejected = sum(1 for g in gens if re.search(r"library w\d*7a ", g["question"]))
    # the Chinese page's question is dropped after the split and listed in the manifest
    chinese = [f"synthetic:{g['span_id']}" for g in gens if "w5a " in g["question"]]
    meta = json.loads((out / "synthetic_manifest.json").read_text(encoding="utf-8"))
    assert meta["dropped_after_split"]["non_english_span"] == chinese
    assert len(dev.questions) + len(holdout.questions) == 50 - rejected - len(chinese)
    assert {q.qid for q in dev.questions}.isdisjoint({q.qid for q in holdout.questions})
    q = dev.questions[0]
    assert q.relevant_pages[0].startswith("hf_docs:") and q.answer
    assert q.meta["span"] and q.meta["scores"]["groundedness"] == 5
    assert q.relevant_pages[0] == q.meta["source_page"]
    assert len(q.relevant_pages) == 1 + q.meta["pages_added"]

    manifest = json.loads((out / "synthetic_manifest.json").read_text(encoding="utf-8"))
    assert manifest["generator"]["model"] == "deepseek-v4-flash"
    assert manifest["critic"]["model"] == "qwen/qwen3.6-flash"
    assert manifest["prompt_versions"]["generation"] and manifest["prompt_versions"]["critique"]
    assert manifest["seeds"] == {"sample": [0, 5], "split": 0}
    assert manifest["thresholds"]["min_relevance"] == 4
    assert manifest["counts"]["generated"] == 50
    assert manifest["counts"]["kept"] == 50 - rejected
    assert manifest["rejections"]["relevance_low"] == rejected
    assert manifest["files"]["synthetic_dev.jsonl"]["sha256"] == dev.sha256
    assert manifest["files"]["synthetic_holdout.jsonl"]["questions"] == len(holdout.questions)
    assert (run / "filter_report.json").exists()
    assert manifest["labels"]["rule"] and "near_duplicate_groups" in manifest
    # near-duplicate groups (none expected here) stay in one set
    for group in manifest["near_duplicate_groups"]:
        assert set(group) <= {q.qid for q in dev.questions} or set(group) <= {
            q.qid for q in holdout.questions
        }
