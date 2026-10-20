"""Server-Sent Events of the streaming ask_docs endpoint (design s16-9; spec P3-04).

An event is `event: <name>\ndata: <one line of JSON>\n\n`; a heartbeat is a comment line
(`: keep-alive`) that clients ignore and proxies count as traffic. The names and shapes are in
the spec and the README; `SearchService.ask_docs_stream` produces `StreamEvent`s and
`Heartbeat`s, the HTTP layer writes them with `encode`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

EVENT_NAMES = ("progress", "token", "answer", "need_clarification", "error", "done")
"""The events of a stream, in the order they can appear (`error` and `done` end it)."""


@dataclass(frozen=True)
class StreamEvent:
    name: str
    data: dict[str, Any]


class Heartbeat:
    """A comment line sent while the run has nothing to say, so a proxy that closes idle
    connections keeps the stream open."""

    def __repr__(self) -> str:
        return "HEARTBEAT"


HEARTBEAT = Heartbeat()


def encode(item: StreamEvent | Heartbeat) -> bytes:
    """The bytes of an event or a heartbeat. The JSON is one line (`json.dumps` escapes line
    breaks inside strings), so one `data:` line carries it."""
    if isinstance(item, Heartbeat):
        return b": keep-alive\n\n"
    data = json.dumps(item.data, ensure_ascii=False, separators=(",", ":"))
    return f"event: {item.name}\ndata: {data}\n\n".encode()
