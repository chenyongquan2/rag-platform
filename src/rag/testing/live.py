"""A real HTTP server in a thread, for tests and the dev mode that must see real sockets: an ASGI
test client does not tell the app when the client goes away; a closed socket does (spec P3-04).

    with LiveServer(app) as server:
        server.url            # http://127.0.0.1:<port>, an ephemeral port
        server.port
"""

from __future__ import annotations

import socket
import threading
import time
from types import TracebackType
from typing import Any

import uvicorn


class LiveServer:
    """`uvicorn` serving `app` on 127.0.0.1 and a free port, started and stopped as a context
    manager. Lifespan runs, so the app's start-up and shutdown code runs too."""

    def __init__(self, app: Any, *, startup_timeout_s: float = 20.0) -> None:
        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        self.port: int = sock.getsockname()[1]
        sock.close()
        config = uvicorn.Config(
            app, host="127.0.0.1", port=self.port, log_config=None, log_level="warning"
        )
        self._server = uvicorn.Server(config)
        self._thread = threading.Thread(target=self._server.run, name="live-server", daemon=True)
        self._startup_timeout_s = startup_timeout_s

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def __enter__(self) -> LiveServer:
        self._thread.start()
        deadline = time.monotonic() + self._startup_timeout_s
        while not self._server.started:
            if not self._thread.is_alive() or time.monotonic() > deadline:
                raise RuntimeError("the live server did not start")
            time.sleep(0.01)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self._server.should_exit = True
        self._thread.join(timeout=20.0)
