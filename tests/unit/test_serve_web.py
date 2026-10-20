"""Spec P3-04 AC21-AC25: the page, its files and security headers, the knowledge-base list it
shows, the feedback endpoint, and the JavaScript of the page (the SSE parser and the rendering of
untrusted text), which `tests/web/ui_test.js` checks with node when node is installed."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from collections.abc import Iterator
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

import pytest
from starlette.testclient import TestClient

from rag.config import AccessConfig, KnowledgeBaseConfig
from rag.observe import shutdown_observability
from rag.serve.http_app import build_http_app
from rag.serve.service import SearchService
from rag.testing import StreamingFakeChatModel, memory_observability
from rag.testing.agent import ScriptedSearchBackend, agent_settings
from rag.testing.serve import FakePipelineBuilder

REPO = Path(__file__).resolve().parents[2]
WEB = REPO / "src" / "rag" / "serve" / "web"
JS_FILES = ("sse.js", "render.js", "app.js")
NODE = shutil.which("node")


@pytest.fixture(autouse=True)
def _no_active_observability() -> Iterator[None]:
    shutdown_observability()
    yield
    shutdown_observability()


def _service(**settings: Any) -> SearchService:
    configs = {
        "hf_docs": KnowledgeBaseConfig(
            kb="hf_docs",
            description="Hugging Face library documentation.",
            language="en",
            access=AccessConfig(callers=["*"]),
        ),
        "secret_kb": KnowledgeBaseConfig(
            kb="secret_kb", description="Someone else's pages", access=AccessConfig(callers=["bob"])
        ),
    }
    service = SearchService(
        agent_settings(**settings),
        configs=configs,
        builder=FakePipelineBuilder(
            {"hf_docs": ScriptedSearchBackend("hf_docs"), "secret_kb": ScriptedSearchBackend("s")}
        ),
        answer_model=StreamingFakeChatModel(default_response="Fine [1]."),
    )
    service.load()
    return service


# ---- AC21 -----------------------------------------------------------------------------------


def test_root_serves_the_page_with_security_headers() -> None:
    with TestClient(build_http_app(_service())) as client:
        page = client.get("/")
        assert page.status_code == 200
        assert page.headers["content-type"].startswith("text/html")
        csp = page.headers["content-security-policy"]
        assert "default-src 'self'" in csp and "script-src 'self'" in csp
        assert "'unsafe-inline'" not in csp and "'unsafe-eval'" not in csp
        assert "frame-ancestors 'none'" in csp and "object-src 'none'" in csp
        assert "form-action 'none'" in csp and "base-uri 'none'" in csp
        assert page.headers["x-content-type-options"] == "nosniff"
        assert page.headers["referrer-policy"] == "no-referrer"
        assert "<title>" in page.text and 'id="question"' in page.text
        for name, kind in (
            ("sse.js", "text/javascript"),
            ("render.js", "text/javascript"),
            ("app.js", "text/javascript"),
            ("app.css", "text/css"),
        ):
            r = client.get(f"/ui/{name}")
            assert r.status_code == 200 and r.headers["content-type"].startswith(kind), name
            assert r.headers["x-content-type-options"] == "nosniff"
            assert r.text == (WEB / name).read_text("utf-8")
        # Nothing else is served from that folder, and no path leaves it.
        for bad in ("/ui/index.html", "/ui/../service.py", "/ui/%2e%2e/service.py", "/ui/nope.js"):
            assert client.get(bad).status_code == 404, bad
        assert client.post("/", json={}).status_code == 405


class _Scan(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.problems: list[str] = []
        self.refs: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        names = {k for k, _ in attrs}
        if tag == "script" and "src" not in names:
            self.problems.append("inline script")
        if tag == "style":
            self.problems.append("inline style element")
        for key, value in attrs:
            if key.startswith("on") or key == "style":
                self.problems.append(f"{tag} has {key}")
            if key in ("src", "href", "action") and value:
                self.refs.append(value)


def test_the_page_loads_nothing_from_outside_and_has_no_inline_code() -> None:
    scan = _Scan()
    scan.feed((WEB / "index.html").read_text("utf-8"))
    assert scan.problems == []
    assert scan.refs and all(ref.startswith("/ui/") for ref in scan.refs), scan.refs
    for name in (*JS_FILES, "app.css"):
        text = (WEB / name).read_text("utf-8")
        assert not re.search(r"https?://|//cdn|@import|url\(", text), name  # no outside fetch


# ---- AC22 -----------------------------------------------------------------------------------


def test_the_knowledge_base_list_hides_what_the_caller_may_not_use() -> None:
    with TestClient(build_http_app(_service())) as client:
        body = client.get("/v1/knowledge_bases").json()
    assert [k["kb"] for k in body["knowledge_bases"]] == ["hf_docs"]
    assert set(body["knowledge_bases"][0]) >= {"kb", "description", "available"}
    with TestClient(build_http_app(_service(service_caller="bob"))) as client:
        both = client.get("/v1/knowledge_bases").json()
    assert sorted(k["kb"] for k in both["knowledge_bases"]) == ["hf_docs", "secret_kb"]


# ---- AC23 -----------------------------------------------------------------------------------


def test_feedback_writes_one_record(tmp_path: Path) -> None:
    path = tmp_path / "records.jsonl"
    with (
        memory_observability(business_record_path=str(path)),
        TestClient(build_http_app(_service(business_record_path=str(path)))) as client,
    ):
        r = client.post(
            "/v1/feedback",
            json={"request_id": "req-42", "useful": False, "comment": "  wrong version  "},
        )
        assert r.status_code == 200 and r.json() == {"request_id": "req-42", "recorded": True}
        ok = client.post("/v1/feedback", json={"request_id": "req-43", "useful": True})
        assert ok.status_code == 200
    first, second = [json.loads(line) for line in path.read_text("utf-8").splitlines()]
    # Keyed by the rated request, so it joins with the answer's record.
    assert first["request_id"] == "req-42" and first["tool"] == "feedback"
    assert first["extra"]["useful"] is False and first["extra"]["comment"] == "wrong version"
    assert first["caller"] == "local" and first["flags"] == []
    assert second["extra"]["useful"] is True and second["extra"]["comment"] is None
    assert set(first["extra"]) == {"useful", "comment", "feedback_request_id"}


BAD_FEEDBACK: list[Any] = [
    b"not json",
    {"request_id": "r1"},  # no verdict
    {"useful": True},  # no request
    {"request_id": "r1", "useful": "yes"},  # not a boolean
    {"request_id": "r1", "useful": 1},
    {"request_id": "r1", "useful": True, "comment": 5},
    {"request_id": "r1", "useful": True, "comment": "x" * 501},
    {"request_id": "has space", "useful": True},
    {"request_id": "a" * 65, "useful": True},
    {"request_id": "abc\n", "useful": True},  # `$` would let a trailing newline through
    {"request_id": "abc\r\n", "useful": True},
    {"request_id": "", "useful": True},
    {"request_id": "r1", "useful": True, "extra": "field"},
]


@pytest.mark.parametrize("body", BAD_FEEDBACK)
def test_feedback_rejects_bad_bodies(body: Any, tmp_path: Path) -> None:
    path = tmp_path / "records.jsonl"
    with (
        memory_observability(business_record_path=str(path)),
        TestClient(build_http_app(_service(business_record_path=str(path)))) as client,
    ):
        kwargs = {"content": body} if isinstance(body, bytes) else {"json": body}
        r = client.post("/v1/feedback", headers={"Content-Type": "application/json"}, **kwargs)
    assert r.status_code == 400 and r.json()["error"] == "invalid_request"
    assert set(r.json()) == {"error", "request_id", "message"}
    rows = [json.loads(line) for line in path.read_text("utf-8").splitlines()]
    assert all(row["flags"] == ["invalid_request"] for row in rows)  # a refusal, not a verdict
    assert not any(row["extra"].get("useful") is not None for row in rows)


def test_feedback_comment_of_exactly_the_limit_is_kept(tmp_path: Path) -> None:
    path = tmp_path / "records.jsonl"
    with (
        memory_observability(business_record_path=str(path)),
        TestClient(build_http_app(_service(business_record_path=str(path)))) as client,
    ):
        r = client.post(
            "/v1/feedback", json={"request_id": "r1", "useful": True, "comment": "é" * 500}
        )
    assert r.status_code == 200
    assert json.loads(path.read_text("utf-8"))["extra"]["comment"] == "é" * 500


# ---- AC24, AC25: the page's JavaScript ------------------------------------------------------


def _node(suite: str) -> subprocess.CompletedProcess[str]:
    assert NODE is not None
    return subprocess.run(
        [NODE, str(REPO / "tests" / "web" / "ui_test.js"), suite],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=60,
        check=False,
    )


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_sse_parser_in_node() -> None:
    done = _node("sse")
    assert done.returncode == 0, done.stdout + done.stderr
    assert "sse: ok" in done.stdout


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_rendering_untrusted_text_creates_no_elements() -> None:
    done = _node("render")
    assert done.returncode == 0, done.stdout + done.stderr
    assert "render: ok" in done.stdout


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_the_pages_own_code_in_node() -> None:
    # app.js against a fake document and a scripted server: the streamed answer, a marker that
    # opens its passage, a question asked back, errors, a cut stream, a retry, cancel, feedback.
    done = _node("app")
    assert done.returncode == 0, done.stdout + done.stderr
    assert "app: ok" in done.stdout


FORBIDDEN = (
    r"\.innerHTML",
    r"\.outerHTML",
    r"insertAdjacentHTML",
    r"document\.write",
    r"\beval\s*\(",
    r"new\s+Function",
    r"\.srcdoc",
    r"setAttribute\(\s*['\"]on",
    r"setAttribute\(\s*['\"]style",
    r"javascript:",
    r"createContextualFragment",
    r"DOMParser",
)


def test_app_js_never_sets_html() -> None:
    for name in JS_FILES:
        text = (WEB / name).read_text("utf-8")
        for pattern in FORBIDDEN:
            assert not re.search(pattern, text), f"{name}: {pattern}"


def test_a_trailing_newline_is_not_part_of_a_session_id() -> None:
    with TestClient(build_http_app(_service())) as client:
        base = {"kb": "hf_docs", "question": "q"}
        for route in ("/v1/ask_docs", "/v1/ask_docs/stream"):
            r = client.post(route, json={**base, "session_id": "s1\n"})
            assert r.status_code == 400, route
        # The same request without the newline is accepted.
        assert client.post("/v1/ask_docs", json={**base, "session_id": "s1"}).status_code == 200
