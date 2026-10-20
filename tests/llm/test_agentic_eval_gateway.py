"""Spec P3-05 AC15: one real question through standard RAG and the real ask_docs graph.

Needs the Docker services (Elasticsearch, Redis), the local reranker and the model gateway.
Run once when a spec says so: `uv run pytest -m llm tests/llm/test_agentic_eval_gateway.py -s`.
One question, both systems: about 0.02 CNY (budget of this test: 0.05).
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from rag.config import ServiceSettings
from rag.eval.agentic import cli
from rag.eval.agentic.records import read_records
from rag.eval.agentic.runner import Tracer, run_agentic, run_standard
from rag.eval.agentic.sets import load_items
from rag.eval.answer.runner import RetrieverContexts
from rag.eval.retrieval.cli import _kb_config

pytestmark = pytest.mark.llm


def test_one_question_both_systems(tmp_path: Path) -> None:
    settings = ServiceSettings()
    if settings.chat_backend != "gateway" or not settings.llm_base_url:
        pytest.skip("the chat gateway is not configured (LLM_BASE_URL, CHAT_BACKEND)")
    cfg, _ = _kb_config(settings, "hf_docs")
    items = load_items("hf_docs", take={"single_hop": 1})
    retriever = cli.build_retriever(cfg, settings, index_prefix="")
    try:
        standard = tmp_path / "standard.jsonl"
        asyncio.run(
            run_standard(
                items,
                records_path=standard,
                contexts=RetrieverContexts(retriever, kb=cfg.kb, top_k=cfg.retrieval.final_k),
                kb_config=cfg,
                model=cli.build_answer_model(settings),
            )
        )
        tracer = Tracer()
        agentic = tmp_path / "agentic.jsonl"
        asyncio.run(
            run_agentic(
                items,
                records_path=agentic,
                runtime=cli.build_runtime(settings, tracer.hooks()),
                retriever=retriever,
                kb_config=cfg,
                tracer=tracer,
            )
        )
    finally:
        close = getattr(retriever, "close", None)
        if callable(close):
            close()
    (a,), (b,) = read_records(standard), read_records(agentic)
    print(a.text, "|", b.text, "|", b.rounds, b.sufficiency, a.usage.cny, b.usage.cny)
    assert a.answered and b.answered
    assert b.rounds >= 1 and b.events  # the hooks recorded the route, rewrite and judge steps
    assert a.usage.calls >= 1 and b.usage.calls >= 2
    assert a.usage.cny + b.usage.cny < 0.05
