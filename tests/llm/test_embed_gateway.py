"""Spec P0-05 AC16: two short texts through the real gateway (costs well under 0.001 CNY).

Run only when a spec asks for it: `uv run pytest -m llm tests/llm/test_embed_gateway.py`.
Reads the gateway URL and key from `.env` and appends its calls to the real cost ledger.
"""

from __future__ import annotations

import math

import pytest

from rag.config import ServiceSettings
from rag.cost import BudgetExceeded, CostLedger
from rag.embed import InMemoryEmbeddingCache, make_embedding

pytestmark = pytest.mark.llm

TEXTS = [
    "Retrieval-augmented generation combines search with a language model.",
    "检索增强生成把检索和大模型结合起来。",
]


def test_real_gateway_embeds_two_texts() -> None:
    settings = ServiceSettings()
    if settings.embed_backend != "gateway" or not settings.embed_base_url:
        pytest.skip("the embedding gateway is not configured (EMBED_BASE_URL, EMBED_BACKEND)")
    ledger = CostLedger.from_settings(settings)
    try:
        ledger.check_budget(ledger.estimate(settings.embed_model, 3 * sum(map(len, TEXTS))))
    except BudgetExceeded as exc:
        pytest.skip(str(exc))
    before = len(ledger.entries())

    ingest = make_embedding("ingest", settings, cache=InMemoryEmbeddingCache())
    vectors = ingest.get_text_embedding_batch(TEXTS)  # one request with two inputs
    assert [len(v) for v in vectors] == [settings.embed_dim, settings.embed_dim] == [1024, 1024]
    for vector in vectors:
        assert math.isclose(math.sqrt(sum(x * x for x in vector)), 1.0, rel_tol=1e-3)
    assert vectors[0] != vectors[1]

    # The query profile (own client, own cache, a second request) gives the identical vector
    # for the identical text: same model, dims and prefix on both sides.
    query = make_embedding("query", settings, cache=InMemoryEmbeddingCache())
    assert query.get_query_embedding(TEXTS[0]) == vectors[0]

    # A cache hit returns the identical vector and makes no request.
    assert ingest.get_text_embedding(TEXTS[0]) == vectors[0]

    rows = ledger.entries()[before:]
    assert [(r["profile"], r["inputs"]) for r in rows] == [("ingest", 2), ("query", 1)]
    assert all(r["usage_estimated"] is False and r["prompt_tokens"] > 0 for r in rows)
