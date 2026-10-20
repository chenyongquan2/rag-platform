"""The HTTP server process (spec P3-04, review round 1).

    server = make_server(app, host, port, service=service, settings=settings)
    server.run()

`uvicorn.Server` with two changes for streams that stay open for a long time:

- `shutdown` first tells the service (`SearchService.begin_shutdown`), which cancels the runs of
  the open streams and ends each stream with a `server_shutdown` event. uvicorn waits for open
  responses before it runs the app's own shutdown code (the lifespan), so waiting for that would
  leave a Ctrl-C hanging for as long as the longest request (up to a minute);
- `timeout_graceful_shutdown` (`SERVICE_SHUTDOWN_TIMEOUT_S`) is a few seconds, after which uvicorn
  cancels whatever is still open (a client that keeps its connection without reading, a
  non-stream request still running).
"""

from __future__ import annotations

import socket
from typing import Any

import uvicorn

from rag.config import ServiceSettings
from rag.serve.service import SearchService


class ShutdownAwareServer(uvicorn.Server):
    def __init__(self, config: uvicorn.Config, service: SearchService | None) -> None:
        super().__init__(config)
        self._service = service

    async def shutdown(self, sockets: list[socket.socket] | None = None) -> None:
        if self._service is not None:
            self._service.begin_shutdown()
        await super().shutdown(sockets)


def make_server(
    app: Any,
    host: str,
    port: int,
    *,
    service: SearchService | None,
    settings: ServiceSettings,
) -> ShutdownAwareServer:
    # log_config=None keeps our logging setup (stderr + log file) instead of uvicorn's.
    config = uvicorn.Config(
        app,
        host=host,
        port=port,
        log_config=None,
        log_level="warning",
        timeout_graceful_shutdown=settings.service_shutdown_timeout_s,
    )
    return ShutdownAwareServer(config, service)
