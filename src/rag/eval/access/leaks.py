"""Permission isolation check: can a caller see passages of a knowledge base it may not use?
(spec P8-01, design s16-4: handing a document to the wrong caller is worse than a wrong answer).

The check talks to the HTTP API as each caller (a bearer token per caller in OAuth mode), so it
covers the whole way a request takes: token, scope -> caller, KB access, the ACL filter of the
retrieval layer, the result cache and the stream. For every caller and question it calls

- `search_docs` and `ask_docs` (plain and streamed) on every KB of the question set;
- `get_context` with every chunk id another caller was given from a KB this caller may not use;
- `list_knowledge_bases`;
- and first, when a cache question is given (one no caller asked before), the same search twice
  as one caller and once as a caller
  with other ACL groups (P5-01: the cache key carries the groups' hash; a hit across groups
  would hand over the first caller's passages).

A leaked chunk is a chunk id in any response (passages, sources, nearest misses, below-threshold
passages, stream events) whose knowledge base the caller may not use, or that belongs to another
KB than the one asked. A request to a hidden KB must be refused like an unknown one
(`kb_not_found`); a hidden KB answered with 200 counts as one exposure even without chunks. The
target is 0 leaks and 0 exposures.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

ALLOWED = "allowed"
DENIED = "denied"


class HttpClient(Protocol):
    """What the check needs of a client: Starlette's `TestClient` or an `httpx.Client` with a
    base URL."""

    def post(self, url: str, *, json: Any = ..., headers: Any = ...) -> Any: ...

    def get(self, url: str, *, headers: Any = ...) -> Any: ...


@dataclass(frozen=True)
class CallerSpec:
    name: str
    headers: Mapping[str, str]
    visible: frozenset[str]


@dataclass(frozen=True)
class AccessCase:
    kb: str
    question: str


@dataclass
class Check:
    caller: str
    tool: str
    kb: str
    question: str
    expected: str
    status: int
    error: str | None
    chunks: int
    leaked: list[str] = field(default_factory=list)
    exposed: bool = False
    degraded: bool = False
    cache_outcome: str | None = None
    note: str = ""

    @property
    def wrongly_denied(self) -> bool:
        return self.expected == ALLOWED and self.status != 200


def chunk_ids(data: Any) -> Iterator[str]:
    """Every `chunk_id` value anywhere in a JSON value."""
    if isinstance(data, Mapping):
        for key, value in data.items():
            if key == "chunk_id" and isinstance(value, str):
                yield value
            else:
                yield from chunk_ids(value)
    elif isinstance(data, list):
        for item in data:
            yield from chunk_ids(item)


def kb_of(chunk_id: str) -> str:
    return chunk_id.partition(":")[0]


def sse_events(text: str) -> list[tuple[str, Any]]:
    """(event, data) of a Server-Sent Events body; comment lines are skipped."""
    events: list[tuple[str, Any]] = []
    name, data = "message", []
    for line in text.splitlines():
        if not line:
            if data:
                raw = "\n".join(data)
                try:
                    events.append((name, json.loads(raw)))
                except ValueError:
                    events.append((name, raw))
            name, data = "message", []
        elif line.startswith(":"):
            continue
        elif line.startswith("event:"):
            name = line[6:].strip()
        elif line.startswith("data:"):
            data.append(line[5:].lstrip())
    return events


@dataclass
class LeakReport:
    checks: list[Check]
    cache_works: bool | None = None

    @property
    def leaked_chunks(self) -> int:
        return sum(len(c.leaked) for c in self.checks)

    @property
    def exposures(self) -> int:
        return sum(c.exposed for c in self.checks)

    @property
    def wrong_denials(self) -> int:
        return sum(c.wrongly_denied for c in self.checks)

    def summary(self) -> dict[str, Any]:
        by: dict[str, dict[str, int]] = {}
        for c in self.checks:
            row = by.setdefault(
                f"{c.caller}/{c.tool}/{c.expected}",
                {"requests": 0, "chunks": 0, "leaked": 0, "exposed": 0, "wrongly_denied": 0},
            )
            row["requests"] += 1
            row["chunks"] += c.chunks
            row["leaked"] += len(c.leaked)
            row["exposed"] += int(c.exposed)
            row["wrongly_denied"] += int(c.wrongly_denied)
        return {
            "requests": len(self.checks),
            "chunks_returned": sum(c.chunks for c in self.checks),
            "leaked_chunks": self.leaked_chunks,
            "exposures": self.exposures,
            "wrong_denials": self.wrong_denials,
            "degraded": sum(c.degraded for c in self.checks),
            "cache_works": self.cache_works,
            "by_caller_tool": by,
        }

    def markdown(self) -> str:
        s = self.summary()
        lines = [
            f"- requests: {s['requests']}, chunks returned: {s['chunks_returned']}",
            f"- leaked chunks: {s['leaked_chunks']}, exposures of a hidden KB: {s['exposures']}",
            f"- allowed requests that failed: {s['wrong_denials']}; degraded: {s['degraded']}",
            f"- cache shares nothing across groups, and works within one: {s['cache_works']}",
            "",
            "| caller / tool / expected | requests | chunks | leaked | exposed | failed |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for key, row in sorted(s["by_caller_tool"].items()):
            lines.append(
                f"| {key} | {row['requests']} | {row['chunks']} | {row['leaked']} | "
                f"{row['exposed']} | {row['wrongly_denied']} |"
            )
        return "\n".join(lines)

    def to_json(self) -> dict[str, Any]:
        return {"summary": self.summary(), "checks": [asdict(c) for c in self.checks]}


def _json(response: Any) -> Any:
    try:
        return response.json()
    except ValueError:
        return None


class LeakChecker:
    def __init__(self, client: HttpClient, callers: Sequence[CallerSpec]) -> None:
        self.client = client
        self.callers = list(callers)
        self.checks: list[Check] = []
        self.seen: dict[str, set[str]] = {}  # kb -> chunk ids some allowed caller was given

    def _record(
        self,
        caller: CallerSpec,
        tool: str,
        kb: str,
        question: str,
        status: int,
        body: Any,
        ids: Iterable[str],
        **extra: Any,
    ) -> Check:
        ids = list(dict.fromkeys(ids))
        expected = ALLOWED if kb in caller.visible else DENIED
        leaked = [i for i in ids if kb_of(i) not in caller.visible or kb_of(i) != kb]
        error = body.get("error") if isinstance(body, Mapping) else None
        exposed = expected == DENIED and (status == 200 or error not in ("kb_not_found", None))
        check = Check(
            caller=caller.name,
            tool=tool,
            kb=kb,
            question=question,
            expected=expected,
            status=status,
            error=error,
            chunks=len(ids),
            leaked=leaked,
            exposed=exposed,
            degraded=bool(body.get("degraded")) if isinstance(body, Mapping) else False,
            **extra,
        )
        if expected == ALLOWED and status == 200:
            self.seen.setdefault(kb, set()).update(i for i in ids if kb_of(i) == kb)
        self.checks.append(check)
        return check

    def search(self, caller: CallerSpec, case: AccessCase) -> Check:
        r = self.client.post(
            "/v1/search_docs",
            json={"kb": case.kb, "query": case.question},
            headers=dict(caller.headers),
        )
        body = _json(r)
        outcome = (body.get("cache") or {}).get("outcome") if isinstance(body, Mapping) else None
        return self._record(
            caller,
            "search_docs",
            case.kb,
            case.question,
            r.status_code,
            body,
            chunk_ids(body),
            cache_outcome=outcome,
        )

    def ask(self, caller: CallerSpec, case: AccessCase) -> Check:
        r = self.client.post(
            "/v1/ask_docs",
            json={"kb": case.kb, "question": case.question},
            headers=dict(caller.headers),
        )
        body = _json(r)
        return self._record(
            caller, "ask_docs", case.kb, case.question, r.status_code, body, chunk_ids(body)
        )

    def stream(self, caller: CallerSpec, case: AccessCase) -> Check:
        r = self.client.post(
            "/v1/ask_docs/stream",
            json={"kb": case.kb, "question": case.question},
            headers=dict(caller.headers),
        )
        if r.headers.get("content-type", "").startswith("text/event-stream"):
            events = sse_events(r.text)
            ids = [i for _, data in events for i in chunk_ids(data)]
            last = next((d for n, d in reversed(events) if n in ("answer", "error")), {})
            body: Any = last if isinstance(last, Mapping) else {}
        else:
            body = _json(r)
            ids = list(chunk_ids(body))
        return self._record(
            caller, "ask_docs_stream", case.kb, case.question, r.status_code, body, ids
        )

    def contexts(self) -> None:
        """Every caller asks for the context of every chunk id seen in a KB it may not use."""
        for caller in self.callers:
            for kb, ids in sorted(self.seen.items()):
                if kb in caller.visible:
                    continue
                for chunk_id in sorted(ids):
                    r = self.client.post(
                        "/v1/get_context",
                        json={"chunk_id": chunk_id, "mode": "neighbors"},
                        headers=dict(caller.headers),
                    )
                    body = _json(r)
                    self._record(
                        caller, "get_context", kb, chunk_id, r.status_code, body, chunk_ids(body)
                    )

    def listing(self, caller: CallerSpec) -> Check:
        r = self.client.get("/v1/knowledge_bases", headers=dict(caller.headers))
        body = _json(r) or {}
        names = [k.get("kb") for k in body.get("knowledge_bases", [])]
        hidden = [n for n in names if n not in caller.visible]
        check = Check(
            caller=caller.name,
            tool="list_knowledge_bases",
            kb="",
            question="",
            expected=ALLOWED,
            status=r.status_code,
            error=body.get("error"),
            chunks=0,
            leaked=[f"{n}:<kb name>" for n in hidden],
        )
        self.checks.append(check)
        return check

    def cache_isolation(self, first: CallerSpec, second: CallerSpec, case: AccessCase) -> bool:
        """`first` searches twice (the second must be a cache hit, or the cache is not on), then
        `second`, whose ACL groups differ, searches the same: it must not be served `first`'s
        entry. A shared entry counts as one exposure."""
        self.search(first, case)
        again = self.search(first, case)
        other = self.search(second, case)
        works = again.cache_outcome == "hit"
        if other.cache_outcome in ("hit", "shared"):
            other.exposed = True
            other.note = "served from another caller's cache entry"
        return works and not other.exposed


def run_leak_check(
    client: HttpClient,
    callers: Sequence[CallerSpec],
    cases: Sequence[AccessCase],
    *,
    ask: bool = True,
    cache_case: AccessCase | None = None,
) -> LeakReport:
    checker = LeakChecker(client, callers)
    cache_works = None
    if cache_case is not None and len(callers) >= 2:
        # First, before either caller has asked anything: an entry of the second caller's own
        # would otherwise look like a shared one.
        cache_works = checker.cache_isolation(callers[0], callers[1], cache_case)
    for caller in callers:
        checker.listing(caller)
        for case in cases:
            checker.search(caller, case)
            if ask:
                checker.ask(caller, case)
                checker.stream(caller, case)
    checker.contexts()
    return LeakReport(checker.checks, cache_works=cache_works)
