"""Create external dependencies from connection strings (design s16-7).

Business code receives only the interface; the scheme of the URL picks the implementation.
For `file://` and `sqlite://`, everything after `scheme://` is a filesystem path, relative to
the working directory unless absolute (`file://data/docstore/hf_docs.json`, `file://C:/x.json`).
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import re
import sqlite3
import threading
import time
from collections.abc import AsyncIterator, Callable, Iterator
from contextlib import (
    AbstractAsyncContextManager,
    AbstractContextManager,
    asynccontextmanager,
    contextmanager,
)
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from filelock import FileLock
from filelock import Timeout as FileLockTimeout
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.storage.kvstore import SimpleKVStore

if TYPE_CHECKING:
    from elasticsearch import Elasticsearch
    from langgraph.checkpoint.base import BaseCheckpointSaver
    from langgraph.checkpoint.serde.base import SerializerProtocol
    from redis import Redis

    from rag.agent.session import SessionLocks, SessionStore

_SCHEME_RE = re.compile(r"^([a-zA-Z][a-zA-Z0-9+.-]*)://(.*)$")


def split_url(url: str) -> tuple[str, str]:
    """Return `(scheme, rest)`; raise ValueError when the URL has no `scheme://`."""
    match = _SCHEME_RE.match(url)
    if not match:
        raise ValueError(f"connection string needs a scheme like file:// or sqlite://: {url!r}")
    return match.group(1).lower(), match.group(2)


def url_path(url: str) -> Path:
    """Filesystem path of a `file://` or `sqlite://` URL."""
    _, rest = split_url(url)
    if not rest:
        raise ValueError(f"connection string has no path: {url!r}")
    if re.match(r"^/[A-Za-z]:/", rest):  # file:///C:/x -> C:/x
        rest = rest[1:]
    return Path(rest)


def _unsupported(what: str, scheme: str, supported: str) -> NotImplementedError:
    return NotImplementedError(
        f"{what}: scheme {scheme!r} is not implemented yet (supported: {supported})"
    )


# ---- Elasticsearch and Redis ----------------------------------------------------------------


def make_es_client(
    url: str, *, request_timeout: float = 30.0, max_retries: int = 2
) -> Elasticsearch:
    """`http://` or `https://` -> Elasticsearch client. Does not connect until the first request.

    `user:password@` in the URL becomes basic auth. Timeouts and retries are explicit; the
    query path passes a shorter `request_timeout` than ingestion (design s16-2).
    """
    scheme, _ = split_url(url)
    if scheme not in ("http", "https"):
        raise _unsupported("Elasticsearch client", scheme, "http, https")
    from elasticsearch import Elasticsearch

    return Elasticsearch(
        url,
        request_timeout=request_timeout,
        max_retries=max_retries,
        retry_on_timeout=False,
    )


def make_redis_client(url: str, *, socket_timeout: float = 5.0) -> Redis:
    """`redis://` or `rediss://` (TLS) -> Redis client with str responses.

    Does not connect until the first command. `socket_timeout` bounds both connecting and each
    command (redis-py waits forever by default).
    """
    scheme, _ = split_url(url)
    if scheme not in ("redis", "rediss"):
        raise _unsupported("Redis client", scheme, "redis, rediss")
    from redis import Redis

    return Redis.from_url(
        url,
        socket_timeout=socket_timeout,
        socket_connect_timeout=socket_timeout,
        decode_responses=True,
    )


# ---- docstore -------------------------------------------------------------------------------


class JsonFileDocumentStore(SimpleDocumentStore):
    """LlamaIndex SimpleDocumentStore bound to one JSON file; `persist()` writes back to it.

    The file is written to `<name>.tmp` and then renamed over the old one, so a crash while
    saving leaves the last saved docstore intact (it decides what ingestion deletes, s7-4).
    `get_meta`/`set_meta` keep a few values of our own in the same file (which index the
    docstore belongs to). `empty=True` starts empty and overwrites the file on `persist()`.
    """

    META_COLLECTION = "rag/meta"

    def __init__(self, path: Path, *, empty: bool = False) -> None:
        self.path = path
        if path.exists() and not empty:
            kvstore = SimpleKVStore.from_persist_path(str(path))
        else:
            kvstore = SimpleKVStore()
        super().__init__(kvstore)

    def get_meta(self, key: str) -> Any:
        entry = self._kvstore.get(key, collection=self.META_COLLECTION)
        return entry.get("value") if entry else None

    def set_meta(self, key: str, value: Any) -> None:
        self._kvstore.put(key, {"value": value}, collection=self.META_COLLECTION)

    def persist(self, persist_path: str | None = None, fs=None) -> None:  # type: ignore[no-untyped-def, override]
        if fs is not None:
            raise NotImplementedError("JsonFileDocumentStore writes local files only")
        target = Path(persist_path) if persist_path else self.path
        tmp = target.with_name(target.name + ".tmp")
        try:
            super().persist(str(tmp))
            os.replace(tmp, target)
        finally:
            tmp.unlink(missing_ok=True)


def make_docstore(url: str, namespace: str) -> SimpleDocumentStore:
    """Docstore for one knowledge base (`namespace`).

    `file://<dir>` -> `<dir>/<namespace>.json`. Postgres comes later (design s16-7).
    """
    if not re.fullmatch(r"[A-Za-z0-9_-]+", namespace):
        raise ValueError(f"docstore namespace must be a plain name: {namespace!r}")
    scheme, _ = split_url(url)
    if scheme == "file":
        return JsonFileDocumentStore(url_path(url) / f"{namespace}.json")
    raise _unsupported("docstore", scheme, "file")


# ---- LangGraph checkpointer -----------------------------------------------------------------


def make_checkpointer(url: str) -> AbstractContextManager[BaseCheckpointSaver]:
    """`sqlite://<path>` -> SqliteSaver, `memory://` -> InMemorySaver.

    Returns a context manager that opens and closes the underlying connection.
    """
    scheme, _ = split_url(url)
    if scheme == "sqlite":
        return _sqlite_checkpointer(url_path(url))
    if scheme == "memory":
        from langgraph.checkpoint.memory import InMemorySaver

        return InMemorySaver()
    raise _unsupported("checkpointer", scheme, "sqlite, memory")


@contextmanager
def _sqlite_checkpointer(path: Path) -> Iterator[BaseCheckpointSaver]:
    from langgraph.checkpoint.sqlite import SqliteSaver

    path.parent.mkdir(parents=True, exist_ok=True)
    with SqliteSaver.from_conn_string(str(path)) as saver:
        yield saver


# ---- locks (ingest lock, session lock) ------------------------------------------------------


class LockTimeoutError(TimeoutError):
    """The lock was not acquired within the timeout."""


class LockProvider(Protocol):
    def lock(self, name: str, timeout: float | None = None) -> AbstractContextManager[None]:
        """Hold the lock called `name` inside the `with` block; `timeout=None` waits forever."""
        ...


class InProcessLockProvider:
    """Named `threading.Lock`s; correct only while one process serves everything.

    A name's lock exists only while someone holds or waits for it, so the registry does not
    grow with the number of distinct names (one per session id).
    """

    def __init__(self) -> None:
        self._guard = threading.Lock()
        # name -> [lock, number of holders and waiters]
        self._locks: dict[str, list[Any]] = {}

    def __len__(self) -> int:
        """Number of names currently held or waited for."""
        with self._guard:
            return len(self._locks)

    @contextmanager
    def lock(self, name: str, timeout: float | None = None) -> Iterator[None]:
        with self._guard:
            entry = self._locks.setdefault(name, [threading.Lock(), 0])
            entry[1] += 1
        lk: threading.Lock = entry[0]
        try:
            if not lk.acquire(timeout=-1 if timeout is None else timeout):
                raise LockTimeoutError(f"lock {name!r} not acquired within {timeout}s")
            try:
                yield
            finally:
                lk.release()
        finally:
            with self._guard:
                entry[1] -= 1
                if entry[1] == 0:
                    del self._locks[name]


class FileLockProvider:
    """One lock file per name in a directory; works across processes on one machine.

    Meant for a small, fixed set of names (the ingest lock of each knowledge base): Windows
    removes the lock file on release, Linux leaves an empty file per name.
    """

    def __init__(self, directory: Path) -> None:
        self.directory = directory

    def _path(self, name: str) -> Path:
        safe = re.sub(r"[^A-Za-z0-9._-]", "_", name)[:80]
        digest = hashlib.sha1(name.encode("utf-8")).hexdigest()[:10]
        return self.directory / f"{safe}.{digest}.lock"

    @contextmanager
    def lock(self, name: str, timeout: float | None = None) -> Iterator[None]:
        self.directory.mkdir(parents=True, exist_ok=True)
        flock = FileLock(self._path(name))
        try:
            flock.acquire(timeout=-1 if timeout is None else timeout)
        except FileLockTimeout as exc:
            raise LockTimeoutError(f"lock {name!r} not acquired within {timeout}s") from exc
        try:
            yield
        finally:
            flock.release()


def make_lock_provider(url: str) -> LockProvider:
    """`file://<dir>` -> file locks, `memory://` -> in-process locks. Redis/Postgres later."""
    scheme, _ = split_url(url)
    if scheme == "file":
        return FileLockProvider(url_path(url))
    if scheme == "memory":
        return InProcessLockProvider()
    raise _unsupported("lock provider", scheme, "file, memory")


# ---- ask_docs: async checkpointer, session history, session locks (spec P3-01) --------------


_SQLITE_INIT_LOCK = threading.Lock()
_SQLITE_READY: set[str] = set()
_SQLITE_INIT_ATTEMPTS = 40
_SQLITE_INIT_WAIT_S = 0.25


def init_sqlite_once(path: Path, init: Callable[[sqlite3.Connection], None]) -> None:
    """Run `init(connection)` (journal mode, tables) once per database file in this process,
    under one lock. Switching to WAL needs the database to itself and the busy timeout does not
    wait for it: requests that all open a brand-new file at once failed with "database is
    locked". Here the first caller does it, the others wait for the lock and find it done; a
    second process doing the same is covered by retrying while the database is locked."""
    key = str(path.resolve())
    if key in _SQLITE_READY and path.exists():
        return
    with _SQLITE_INIT_LOCK:
        if key in _SQLITE_READY and path.exists():
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        for attempt in range(_SQLITE_INIT_ATTEMPTS):
            conn = sqlite3.connect(path, timeout=5.0)
            try:
                init(conn)
                conn.commit()
                break
            except sqlite3.OperationalError as exc:
                if "locked" not in str(exc) or attempt == _SQLITE_INIT_ATTEMPTS - 1:
                    raise
                time.sleep(_SQLITE_INIT_WAIT_S)
            finally:
                conn.close()
        _SQLITE_READY.add(key)


def _init_checkpoint_db(conn: sqlite3.Connection) -> None:
    from langgraph.checkpoint.sqlite import SqliteSaver

    SqliteSaver(conn).setup()  # WAL mode and the tables, the same schema as the async saver


class CheckpointerSource(Protocol):
    def prepare(self) -> None:
        """Create the database before the first request (idempotent)."""
        ...

    def open(self) -> AbstractAsyncContextManager[BaseCheckpointSaver]:
        """A saver for the `async with` block (the ask_docs graph is async)."""
        ...


class _SqliteCheckpointerSource:
    """A connection per request, closed with the block: nothing is tied to one event loop and
    no thread outlives the request. WAL mode and a busy timeout let requests of different
    sessions write their checkpoints at the same time."""

    def __init__(self, path: Path, serde: SerializerProtocol | None) -> None:
        self.path = path
        self.serde = serde

    def prepare(self) -> None:
        init_sqlite_once(self.path, _init_checkpoint_db)

    @asynccontextmanager
    async def open(self) -> AsyncIterator[BaseCheckpointSaver]:
        import aiosqlite
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

        await asyncio.to_thread(self.prepare)  # a no-op after the first request
        async with aiosqlite.connect(self.path) as conn:
            await conn.execute("PRAGMA busy_timeout=5000")
            saver = AsyncSqliteSaver(conn, serde=self.serde)
            # Tables and WAL mode exist (`prepare`); the saver's own setup would run its DDL
            # and an ALTER TABLE again on every request.
            saver.is_setup = True
            yield saver


class _MemoryCheckpointerSource:
    def __init__(self, serde: SerializerProtocol | None) -> None:
        from langgraph.checkpoint.memory import InMemorySaver

        self._saver = InMemorySaver(serde=serde)

    def prepare(self) -> None:
        pass

    @asynccontextmanager
    async def open(self) -> AsyncIterator[BaseCheckpointSaver]:
        yield self._saver


def make_checkpointer_source(
    url: str, *, serde: SerializerProtocol | None = None
) -> CheckpointerSource:
    """`sqlite://<path>` -> an async SQLite saver per `open()`, `memory://` -> one in-memory
    saver kept by the source. `serde` lists the types the state may hold (the serializer
    refuses others in strict mode)."""
    scheme, _ = split_url(url)
    if scheme == "sqlite":
        return _SqliteCheckpointerSource(url_path(url), serde)
    if scheme == "memory":
        return _MemoryCheckpointerSource(serde)
    raise _unsupported("async checkpointer", scheme, "sqlite, memory")


def make_session_store(url: str, *, keep_turns: int) -> SessionStore:
    """`sqlite://<path>` -> turns in a SQLite file, `memory://` -> in process memory; the last
    `keep_turns` turns of each session are kept (0 keeps none). Postgres later (design s16-7)."""
    from rag.agent.session import MemorySessionStore, SqliteSessionStore

    scheme, _ = split_url(url)
    if scheme == "sqlite":
        return SqliteSessionStore(url_path(url), keep_turns)
    if scheme == "memory":
        return MemorySessionStore(keep_turns)
    raise _unsupported("session store", scheme, "sqlite, memory")


def make_session_locks(url: str) -> SessionLocks:
    """`memory://` -> locks inside this process, `file://<dir>` -> also file locks across the
    processes of this machine. Redis or Postgres locks come with several instances."""
    from rag.agent.session import SessionLocks

    scheme, _ = split_url(url)
    if scheme == "memory":
        return SessionLocks()
    if scheme == "file":
        return SessionLocks(FileLockProvider(url_path(url)))
    raise _unsupported("session lock", scheme, "file, memory")
