"""Spec P0-09 AC15: an MCP client starts the README command `uv run python -m rag.serve stdio`.

The SDK's stdio client spawns the server as a coding assistant would, lists the tools, calls
list_knowledge_bases (answers while the reranker loads) and search_docs (waits for the load),
then checks the business record line the server wrote and that the request's span reached
Jaeger through the OTel Collector. Fake query vectors on `fake_kb_hf_docs`: no model cost.
"""

from __future__ import annotations

import json
import os
import shutil
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import anyio
import pytest
import yaml
from elasticsearch import Elasticsearch
from mcp import ClientSession, StdioServerParameters, stdio_client

from rag.config import ServiceSettings, repo_path
from rag.eval.questions import load_question_set
from rag.testing.server import server_env

pytestmark = pytest.mark.integration

REPO_ROOT = repo_path(".")
LOAD_TIMEOUT_S = 300.0
TOOLS = {"search_docs", "ask_docs", "get_context", "list_knowledge_bases"}
QUESTION_SET = "evals/datasets/hf_docs/huggingface_doc_qa_eval.jsonl"
KNOWN_QID = "huggingface_doc_qa_eval:025"
KNOWN_PAGE = "hf_docs:huggingface/hf-endpoints-documentation/blob/main/docs/source/guides/logs.mdx"


def known_question() -> str:
    questions = load_question_set(repo_path(QUESTION_SET), kb="hf_docs").questions
    return next(q.question for q in questions if q.qid == KNOWN_QID)


def require_fake_index(es: Elasticsearch) -> None:
    if not es.indices.exists_alias(name="fake_kb_hf_docs"):
        pytest.skip("alias fake_kb_hf_docs is missing (run the ingest job with EMBED_BACKEND=fake)")


def require_reranker(settings: ServiceSettings) -> None:
    from huggingface_hub import try_to_load_from_cache

    if not isinstance(try_to_load_from_cache(settings.reranker_model, "config.json"), str):
        pytest.skip(f"{settings.reranker_model} is not in the local Hugging Face cache")


def _jaeger_url() -> str:
    compose = yaml.safe_load((REPO_ROOT / "docker-compose.yml").read_text("utf-8"))
    for spec in compose["services"]["jaeger"].get("ports", []):
        host_ip, host_port, port = str(spec).split(":")
        if int(port) == 16686:
            return f"http://{host_ip}:{host_port}"
    raise AssertionError("Jaeger query port is not published")


def _get(url: str) -> tuple[int, bytes]:
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as exc:
        return exc.code, b""


def _span_names(body: bytes) -> set[str]:
    return {
        span["name"]
        for rs in json.loads(body)["result"]["resourceSpans"]
        for ss in rs["scopeSpans"]
        for span in ss["spans"]
    }


def _wait_for_trace(jaeger: str, trace_id: str, timeout_s: float = 60.0) -> set[str]:
    deadline = time.monotonic() + timeout_s
    names: set[str] = set()
    while time.monotonic() < deadline:
        status, body = _get(f"{jaeger}/api/v3/traces/{trace_id}")
        if status == 200:
            names = _span_names(body)
            if {"search_docs", "retrieve"} <= names:
                break
        time.sleep(1.0)
    return names


def test_mcp_stdio_end_to_end(es: Elasticsearch, tmp_path: Path) -> None:
    require_fake_index(es)
    require_reranker(ServiceSettings())
    jaeger = _jaeger_url()
    try:
        _get(jaeger + "/")
    except OSError as exc:
        pytest.skip(f"Jaeger not reachable ({exc})")
    uv = shutil.which("uv")
    assert uv, "uv is not on PATH"
    records = tmp_path / "business.jsonl"
    params = StdioServerParameters(
        command=uv,
        args=["run", "python", "-m", "rag.serve", "stdio"],
        cwd=REPO_ROOT,
        # The stdio client passes only a safe subset of the environment; settings come from the
        # repo's .env (cwd) plus these. `server_env` turns the result cache off: its entries
        # (an hour) would be served to the next run of this test, with no retrieval and no trace.
        env=server_env(
            EMBED_BACKEND="fake",
            SEARCH_INDEX_PREFIX="fake_",
            HF_HUB_OFFLINE="1",
            BUSINESS_RECORD_PATH=str(records),
            LOG_DIR=str(tmp_path / "logs"),
            OTEL_AUTO_INSTRUMENT="false",
            **({"HF_HOME": os.environ["HF_HOME"]} if "HF_HOME" in os.environ else {}),
        ),
    )
    question = known_question()

    async def session_run() -> dict[str, Any]:
        out: dict[str, Any] = {}
        errlog = (tmp_path / "server-stderr.log").open("w", encoding="utf-8")
        try:
            async with (
                stdio_client(params, errlog=errlog) as (read, write),
                ClientSession(read, write) as session,
            ):
                await session.initialize()
                out["tools"] = {t.name for t in (await session.list_tools()).tools}
                listing = await session.call_tool("list_knowledge_bases", {})
                out["listing"] = listing.structured_content
                began = time.monotonic()
                while True:  # the first searches wait for the reranker load, then not_ready
                    t0 = time.perf_counter()
                    result = await session.call_tool(
                        "search_docs",
                        {"kb": "hf_docs", "query": question, "top_k": 5, "deadline_ms": 60_000},
                    )
                    out["search_s"] = time.perf_counter() - t0
                    data = result.structured_content or {}
                    if not (result.is_error and data.get("error") == "not_ready"):
                        break
                    assert time.monotonic() - began < LOAD_TIMEOUT_S, "server never got ready"
                out["load_wait_s"] = time.monotonic() - began
                out["search"] = data
                out["search_error"] = result.is_error
                t0 = time.perf_counter()
                again = await session.call_tool(
                    "search_docs", {"kb": "hf_docs", "query": question, "deadline_ms": 60_000}
                )
                out["warm_search_s"] = time.perf_counter() - t0
                out["warm_error"] = again.is_error
        finally:
            errlog.close()
        return out

    out = anyio.run(session_run)
    stderr = (tmp_path / "server-stderr.log").read_text("utf-8", errors="replace")
    assert out["tools"] == TOOLS, stderr[-2000:]
    assert "hf_docs" in {k["kb"] for k in out["listing"]["knowledge_bases"]}
    assert out["search_error"] is False, out["search"]
    search = out["search"]
    assert "routes" not in search and search["chunks"][0]["doc_id"] == KNOWN_PAGE
    assert all(c["untrusted"] is True for c in search["chunks"])
    assert out["warm_error"] is False
    print(
        f"\nstdio search_docs: first {out['search_s'] * 1000:.0f} ms after waiting "
        f"{out['load_wait_s']:.0f} s for the load, warm {out['warm_search_s'] * 1000:.0f} ms"
    )

    lines = [json.loads(x) for x in records.read_text("utf-8").splitlines() if x.strip()]
    record = next(r for r in lines if r["request_id"] == search["request_id"])
    assert record["tool"] == "search_docs" and record["trace_id"]
    assert {r["tool"] for r in lines} >= {"list_knowledge_bases", "search_docs"}
    names = _wait_for_trace(jaeger, record["trace_id"])
    assert {"search_docs", "retrieve"} <= names, f"trace {record['trace_id']} not in Jaeger"
