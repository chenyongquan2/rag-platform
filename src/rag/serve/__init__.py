"""The search service: MCP tools and HTTP API over the retrieval layer (design s16-1, spec P0-09).

`SearchService` (`rag.serve.service`) does the work; `build_mcp_server` (`rag.serve.mcp_server`)
and `build_http_app` (`rag.serve.http_app`) are the two ways in. Run with
`uv run python -m rag.serve stdio` or `uv run python -m rag.serve http`.
"""
