"""Embedding client for the OpenAI-compatible model gateway (spec P0-05, design s7-4, s17-1).

A LlamaIndex `BaseEmbedding`, so `IngestionPipeline` can use it as a transformation, that posts
to `<base_url>/embeddings` with `httpx`. Ingestion and query use the same class with the same
model, dimensions and text prefix; the profiles differ only in timeout and retries (design s16-2).

- At most `embed_batch_size` texts per request; vectors come back in input order.
- Timeouts, connection errors, 408, 429 and 5xx are retried with exponential backoff
  (`Retry-After` wins when the gateway sends it); other errors fail at once.
- Before each request the cost ledger is checked (`BudgetExceeded` stops the request); every
  2xx answer appends one ledger line with the tokens from `usage` (estimated from the number of
  characters when the answer has no usage).
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable, Sequence
from typing import Any, Literal

import httpx
from llama_index.core.base.embeddings.base import BaseEmbedding, Embedding
from pydantic import Field, PrivateAttr

from rag.cost import CostLedger

logger = logging.getLogger(__name__)

EmbedProfile = Literal["ingest", "query"]

_RETRYABLE_STATUS = frozenset({408, 429})
_SNIPPET_CHARS = 300


class EmbeddingError(RuntimeError):
    """The gateway did not return usable vectors.

    `retryable` says whether the cause was transient (then the retries are used up);
    `status_code` is the HTTP status, or None when no answer arrived.
    """

    def __init__(self, message: str, *, retryable: bool, status_code: int | None = None) -> None:
        super().__init__(message)
        self.retryable = retryable
        self.status_code = status_code


def _is_retryable_status(status: int) -> bool:
    return status in _RETRYABLE_STATUS or status >= 500


def _retry_after_s(response: httpx.Response) -> float | None:
    value = response.headers.get("Retry-After")
    if value is None:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:  # an HTTP date; use the computed backoff instead
        return None


def _snippet(response: httpx.Response) -> str:
    text = response.text.strip().replace("\n", " ")
    return text[:_SNIPPET_CHARS] + ("..." if len(text) > _SNIPPET_CHARS else "")


class GatewayEmbedding(BaseEmbedding):
    """Embeddings from an OpenAI-compatible `/embeddings` endpoint. Build it with
    `rag.embed.make_embedding`; the constructor is for tests and the factory."""

    model_name: str
    dim: int = Field(gt=0)
    base_url: str
    text_prefix: str = ""
    profile: EmbedProfile
    timeout_s: float = Field(gt=0)
    max_retries: int = Field(ge=0)
    backoff_s: float = Field(ge=0)
    backoff_max_s: float = Field(ge=0)

    _ledger: CostLedger = PrivateAttr()
    _http: httpx.Client = PrivateAttr()
    _sleep: Callable[[float], None] = PrivateAttr()

    def __init__(
        self,
        *,
        api_key: str,
        ledger: CostLedger,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
        **data: Any,
    ) -> None:
        super().__init__(**data)
        if not self.base_url:
            raise ValueError("EMBED_BASE_URL is not set; set it in .env or use EMBED_BACKEND=fake")
        if not api_key:
            raise ValueError("EMBED_API_KEY is not set; set it in .env or use EMBED_BACKEND=fake")
        ledger.price(self.model_name)  # no price -> fail now, before any money is spent
        self._ledger = ledger
        self._sleep = sleep
        self._http = httpx.Client(
            base_url=self.base_url.rstrip("/") + "/",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=self.timeout_s,
            transport=transport,
        )

    @classmethod
    def class_name(cls) -> str:
        return "GatewayEmbedding"

    def to_payload(self) -> dict[str, Any]:
        return {**super().to_payload(), "dim": self.dim, "profile": self.profile}

    def close(self) -> None:
        self._http.close()

    # ---- LlamaIndex hooks -------------------------------------------------------------------

    def _get_query_embedding(self, query: str) -> Embedding:
        return self._embed([query])[0]

    async def _aget_query_embedding(self, query: str) -> Embedding:
        return await asyncio.to_thread(self._get_query_embedding, query)

    def _get_text_embedding(self, text: str) -> Embedding:
        return self._embed([text])[0]

    async def _aget_text_embedding(self, text: str) -> Embedding:
        return await asyncio.to_thread(self._get_text_embedding, text)

    def _get_text_embeddings(self, texts: list[str]) -> list[Embedding]:
        return self._embed(texts)

    async def _aget_text_embeddings(self, texts: list[str]) -> list[Embedding]:
        return await asyncio.to_thread(self._get_text_embeddings, texts)

    # ---- requests ---------------------------------------------------------------------------

    def _embed(self, texts: Sequence[str]) -> list[Embedding]:
        vectors: list[Embedding] = []
        for start in range(0, len(texts), self.embed_batch_size):
            vectors.extend(self._request(list(texts[start : start + self.embed_batch_size])))
        return vectors

    def _request(self, texts: list[str]) -> list[Embedding]:
        inputs = [self.text_prefix + t for t in texts]
        # About one token per character or fewer for English and Chinese: an upper bound.
        estimated_tokens = sum(len(t) for t in inputs)
        self._ledger.check_budget(self._ledger.estimate(self.model_name, estimated_tokens))
        body = {
            "model": self.model_name,
            "input": inputs,
            "dimensions": self.dim,
            "encoding_format": "float",
        }
        attempts = self.max_retries + 1
        for attempt in range(attempts):
            retry_after: float | None = None
            cause: Exception | None = None
            try:
                response = self._http.post("embeddings", json=body)
            except httpx.TransportError as exc:  # timeouts, refused or reset connections
                error = EmbeddingError(
                    f"embedding request failed: {type(exc).__name__}: {exc}", retryable=True
                )
                cause = exc
            else:
                if response.is_success:
                    return self._parse(response, inputs, estimated_tokens)
                error = EmbeddingError(
                    f"embedding gateway answered HTTP {response.status_code}: {_snippet(response)}",
                    retryable=_is_retryable_status(response.status_code),
                    status_code=response.status_code,
                )
                retry_after = _retry_after_s(response)
            if not error.retryable:
                raise error from cause
            if attempt + 1 == attempts:
                raise EmbeddingError(
                    f"{error} (gave up after {attempts} attempts)",
                    retryable=True,
                    status_code=error.status_code,
                ) from cause
            delay = self._delay(attempt, retry_after)
            logger.warning(
                "embedding (%s): %s; retry %d of %d in %.1f s",
                self.profile,
                error,
                attempt + 1,
                self.max_retries,
                delay,
            )
            self._sleep(delay)
        raise AssertionError("unreachable")  # pragma: no cover

    def _delay(self, attempt: int, retry_after: float | None) -> float:
        wait = retry_after if retry_after is not None else self.backoff_s * 2**attempt
        return min(wait, self.backoff_max_s)

    def _parse(
        self, response: httpx.Response, inputs: list[str], estimated_tokens: int
    ) -> list[Embedding]:
        status = response.status_code
        try:
            payload = response.json()
        except ValueError:
            self._record(len(inputs), estimated_tokens, estimated=True)
            raise EmbeddingError(
                f"embedding gateway answer is not JSON: {_snippet(response)}",
                retryable=False,
                status_code=status,
            ) from None
        if not isinstance(payload, dict):
            payload = {}
        usage = payload.get("usage")
        tokens = None
        if isinstance(usage, dict):
            tokens = usage.get("prompt_tokens") or usage.get("total_tokens")
        if isinstance(tokens, int) and tokens >= 0:
            self._record(len(inputs), tokens, estimated=False)
        else:
            self._record(len(inputs), estimated_tokens, estimated=True)

        data = payload.get("data")
        items = data if isinstance(data, list) else []
        if len(items) != len(inputs) or not all(
            isinstance(i, dict) and isinstance(i.get("embedding"), list) for i in items
        ):
            raise EmbeddingError(
                f"embedding gateway returned {len(items)} vectors for {len(inputs)} inputs",
                retryable=False,
                status_code=status,
            )
        items = sorted(items, key=lambda item: item.get("index", 0))
        vectors = [[float(x) for x in item["embedding"]] for item in items]
        for vector in vectors:
            if len(vector) != self.dim:
                raise EmbeddingError(
                    f"embedding dimension {len(vector)} differs from the configured {self.dim} "
                    f"(model {self.model_name})",
                    retryable=False,
                    status_code=status,
                )
        return vectors

    def _record(self, inputs: int, tokens: int, *, estimated: bool) -> None:
        self._ledger.record(
            kind="embedding",
            model=self.model_name,
            prompt_tokens=tokens,
            profile=self.profile,
            inputs=inputs,
            usage_estimated=estimated,
        )
