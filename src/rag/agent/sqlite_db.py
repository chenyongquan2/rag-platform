"""The tables of the session SQLite file (spec P3-01, P3-02).

One file (`session_store_url`) holds the turns of every session and the resume tokens of
questions asked back. Both stores create it through `rag.factories.init_sqlite_once`, which runs
the setup once per file and process, so whichever store opens the file first must create every
table: this one function does.
"""

from __future__ import annotations

import sqlite3


def init_agent_db(conn: sqlite3.Connection) -> None:
    """WAL mode (other sessions' writes do not wait for each other) and the tables."""
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(
        "CREATE TABLE IF NOT EXISTS turns ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL, "
        "request_id TEXT NOT NULL, question TEXT NOT NULL, answer TEXT NOT NULL, "
        "at TEXT NOT NULL)"
    )
    conn.execute("CREATE INDEX IF NOT EXISTS turns_session ON turns (session_id, id)")
    # `token_hash`: the SHA-256 of the token; the token itself is never stored.
    # `status`: pending (can be claimed), resuming (a call is running the thread, since
    # `claimed_at`) or used.
    conn.execute(
        "CREATE TABLE IF NOT EXISTS resumes ("
        "token_hash TEXT PRIMARY KEY, caller_id TEXT NOT NULL, kb TEXT NOT NULL, "
        "session_id TEXT, thread_id TEXT NOT NULL, status TEXT NOT NULL, "
        "expires_at REAL NOT NULL, claimed_at REAL)"
    )
    # A file made by the first build of this table has no `claimed_at`.
    columns = {row[1] for row in conn.execute("PRAGMA table_info(resumes)")}
    if "claimed_at" not in columns:
        conn.execute("ALTER TABLE resumes ADD COLUMN claimed_at REAL")
