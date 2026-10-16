"""httpx transports that put every chat model call on the cost ledger (spec P1-01).

`make_chat_model` gives `ChatOpenAI` httpx clients whose transport is one of these. They sit below
the openai SDK, so they see every HTTP attempt, including the SDK's own retries, and they work
the same for `invoke`, `stream`, structured output, sync and async calls.

- Before an attempt is sent, its estimated cost is reserved in the ledger (`CostLedger.reserve`,
  spec P0-05); over the budget stop `BudgetExceeded` is raised and nothing is sent. The estimate
  is an upper bound: one token per character of the request body, plus the request's
  `max_completion_tokens` (reasoning and answer together).
- Every 2xx answer appends one `chat` line with the tokens from `usage` (for a stream, the usage
  event at its end). Output tokens include reasoning tokens. Without usage (no field, a stream
  closed before its end) the line is estimated; so is a read timeout or a dropped connection
  after the request was sent, since the gateway may have done and billed the work. Connect
  failures and 4xx/5xx answers write no line (the gateway did not process the request).
- A retryable error answer whose `Retry-After` is longer than the profile's `retry_after_max_s`
  gets `x-should-retry: false`, so the SDK raises at once instead of waiting (design s16-2).
- The API key echoed in an error body is masked before the SDK puts the body into its exception.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Iterator
from contextlib import ExitStack
from typing import Any

import httpx

from rag.cost import CostLedger

# Raised after the request went out: the gateway may have done (and billed) the work.
_FAILED_AFTER_SEND = (httpx.ReadTimeout, httpx.ReadError, httpx.RemoteProtocolError)
_RETRYABLE_STATUS = frozenset({408, 409, 429})  # and every 5xx, like the openai SDK
# Output estimate for a request without max_completion_tokens (the factory always sets it).
_UNCAPPED_OUTPUT_TOKENS = 32_768
# The body is passed on decoded, so these no longer describe it.
_DROP_HEADERS = frozenset({"content-encoding", "content-length", "transfer-encoding"})
_CLOSED_EARLY = "closed before usage"


def parse_usage(usage: object) -> tuple[int, int, int] | None:
    """(prompt tokens, output tokens including reasoning, reasoning tokens) from an OpenAI
    `usage` object, or None when it has no token counts.

    OpenAI and DeepSeek count reasoning tokens inside `completion_tokens`. A gateway that does
    not shows them on top in `total_tokens`; then they are added to the output tokens.
    """
    if not isinstance(usage, dict):
        return None
    prompt = usage.get("prompt_tokens")
    completion = usage.get("completion_tokens")
    if not isinstance(prompt, int) or not isinstance(completion, int):
        return None
    details = usage.get("completion_tokens_details")
    reasoning = details.get("reasoning_tokens") if isinstance(details, dict) else None
    reasoning = reasoning if isinstance(reasoning, int) and reasoning > 0 else 0
    total = usage.get("total_tokens")
    output = completion
    if reasoning and isinstance(total, int) and total >= prompt + completion + reasoning:
        output = completion + reasoning
    return prompt, output, reasoning


def _retry_after_s(headers: httpx.Headers) -> float | None:
    """The wait the gateway asks for, read like the openai SDK reads it (seconds only)."""
    for name, scale in (("retry-after-ms", 1000.0), ("retry-after", 1.0)):
        value = headers.get(name)
        if value is None:
            continue
        try:
            return max(0.0, float(value) / scale)
        except ValueError:  # an HTTP date: the SDK then uses its own backoff
            return None
    return None


def _bearer_token(request: httpx.Request) -> str:
    scheme, _, token = request.headers.get("authorization", "").partition(" ")
    return token.strip() if scheme.lower() == "bearer" else ""


class _Meter:
    """Accounting of one HTTP attempt: holds its reservation, reads its usage, writes its line."""

    def __init__(self, ledger: CostLedger, request: httpx.Request, *, profile: str) -> None:
        self._ledger = ledger
        self._profile = profile
        text = request.content.decode("utf-8", "replace")
        try:
            body = json.loads(text)
        except ValueError:
            body = None
        if not isinstance(body, dict):
            body = {}
        self.model = str(body.get("model") or "")
        self.stream = bool(body.get("stream"))
        cap = body.get("max_completion_tokens") or body.get("max_tokens")
        self.estimated_prompt = len(text)
        self.estimated_output = cap if isinstance(cap, int) and cap > 0 else _UNCAPPED_OUTPUT_TOKENS
        self._reservation = ExitStack()
        self._price_estimated = False
        self._pending = bytearray()  # stream: an unfinished SSE line; otherwise the whole body
        self._usage: tuple[int, int, int] | None = None
        self._served_model: str | None = None
        self._done = False

    def reserve(self) -> None:
        """Reserve the estimate; raises `BudgetExceeded` or `UnknownModelPriceError`."""
        self._price_estimated = self._ledger.price(self.model).estimated
        cny = self._ledger.estimate(self.model, self.estimated_prompt, self.estimated_output)
        self._reservation.enter_context(self._ledger.reserve(cny))

    def feed(self, data: bytes) -> None:
        self._pending.extend(data)
        if not self.stream:
            return
        *lines, rest = bytes(self._pending).split(b"\n")
        self._pending = bytearray(rest)
        for line in lines:
            self._read_event(line.strip())

    def _read_event(self, line: bytes) -> None:
        if not line.startswith(b"data:"):
            return
        data = line[5:].strip()
        if data == b"[DONE]" or (b'"usage"' not in data and self._served_model is not None):
            return
        try:
            event = json.loads(data)
        except ValueError:
            return
        if isinstance(event, dict):
            self._read_answer(event)

    def _read_answer(self, answer: dict[str, Any]) -> None:
        if self._served_model is None and isinstance(answer.get("model"), str):
            self._served_model = answer["model"]
        usage = parse_usage(answer.get("usage"))
        if usage is not None:
            self._usage = usage

    def finish(self, error: str | None = None) -> None:
        """Write the attempt's line (real usage if seen, else estimated) and release it."""
        if self._done:
            return
        self._done = True
        try:
            if not self.stream and error is None:
                try:
                    answer = json.loads(bytes(self._pending))
                except ValueError:
                    answer = None
                if isinstance(answer, dict):
                    self._read_answer(answer)
            self._record(error)
        finally:
            self._reservation.close()

    def release(self) -> None:
        """Release without a line: the gateway did not process the request."""
        if not self._done:
            self._done = True
            self._reservation.close()

    def _record(self, error: str | None) -> None:
        if self._usage is not None:
            prompt, output, reasoning = self._usage
        else:
            prompt, output, reasoning = self.estimated_prompt, self.estimated_output, 0
        extra: dict[str, Any] = {
            "profile": self._profile,
            "reasoning_tokens": reasoning,
            "usage_estimated": self._usage is None,
            "stream": self.stream,
            "price_estimated": self._price_estimated,
        }
        if self._served_model is not None:
            extra["served_model"] = self._served_model
        if error is not None and self._usage is None:
            extra["error"] = error
        self._ledger.record(
            kind="chat", model=self.model, prompt_tokens=prompt, completion_tokens=output, **extra
        )


class _Shared:
    """What the sync and async transports have in common."""

    def __init__(self, ledger: CostLedger, *, profile: str, retry_after_max_s: float) -> None:
        self.ledger = ledger
        self.profile = profile
        self.retry_after_max_s = retry_after_max_s

    def meter(self, request: httpx.Request) -> _Meter:
        meter = _Meter(self.ledger, request, profile=self.profile)
        meter.reserve()
        return meter

    @staticmethod
    def passed_headers(response: httpx.Response) -> list[tuple[str, str]]:
        return [(k, v) for k, v in response.headers.multi_items() if k.lower() not in _DROP_HEADERS]

    def error_response(
        self, request: httpx.Request, response: httpx.Response, content: bytes
    ) -> httpx.Response:
        headers = self.passed_headers(response)
        status = response.status_code
        retry_after = _retry_after_s(response.headers)
        if (
            (status in _RETRYABLE_STATUS or status >= 500)
            and retry_after is not None
            and retry_after > self.retry_after_max_s
        ):
            headers.append(("x-should-retry", "false"))
        token = _bearer_token(request)
        if token and token.encode() in content:
            content = content.replace(token.encode(), b"***")
        return httpx.Response(
            status,
            headers=headers,
            content=content,
            extensions=response.extensions,
            request=request,
        )


class _MeteredStream(httpx.SyncByteStream):
    def __init__(self, response: httpx.Response, meter: _Meter) -> None:
        self._response = response
        self._meter = meter

    def __iter__(self) -> Iterator[bytes]:
        try:
            for chunk in self._response.iter_bytes():
                self._meter.feed(chunk)
                yield chunk
        except _FAILED_AFTER_SEND as exc:
            self._meter.finish(type(exc).__name__)
            raise
        self._meter.finish()

    def close(self) -> None:
        try:
            self._response.close()
        finally:
            self._meter.finish(_CLOSED_EARLY)


class _AsyncMeteredStream(httpx.AsyncByteStream):
    def __init__(self, response: httpx.Response, meter: _Meter) -> None:
        self._response = response
        self._meter = meter

    async def __aiter__(self) -> AsyncIterator[bytes]:
        try:
            async for chunk in self._response.aiter_bytes():
                self._meter.feed(chunk)
                yield chunk
        except _FAILED_AFTER_SEND as exc:
            self._meter.finish(type(exc).__name__)
            raise
        self._meter.finish()

    async def aclose(self) -> None:
        try:
            await self._response.aclose()
        finally:
            self._meter.finish(_CLOSED_EARLY)


class LedgerTransport(httpx.BaseTransport):
    """Sync transport: reserve, send through an inner client, record (see the module doc).

    The inner `httpx.Client` reads proxies from the environment (`HTTPS_PROXY`, `NO_PROXY`) like
    any default client; the outer client, which has this transport, would ignore them. Each
    request keeps the timeout the openai SDK set on it. `transport` replaces the network in tests.
    """

    def __init__(
        self,
        ledger: CostLedger,
        *,
        profile: str,
        retry_after_max_s: float,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._shared = _Shared(ledger, profile=profile, retry_after_max_s=retry_after_max_s)
        self._client = httpx.Client(transport=transport)

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        meter = self._shared.meter(request)
        try:
            response = self._client.send(request, stream=True)
        except _FAILED_AFTER_SEND as exc:
            meter.finish(type(exc).__name__)
            raise
        except BaseException:
            meter.release()
            raise
        if not response.is_success:
            try:
                content = response.read()
            finally:
                response.close()
                meter.release()
            return self._shared.error_response(request, response, content)
        return httpx.Response(
            response.status_code,
            headers=self._shared.passed_headers(response),
            stream=_MeteredStream(response, meter),
            extensions=response.extensions,
            request=request,
        )

    def close(self) -> None:
        self._client.close()


class AsyncLedgerTransport(httpx.AsyncBaseTransport):
    """Async twin of `LedgerTransport`."""

    def __init__(
        self,
        ledger: CostLedger,
        *,
        profile: str,
        retry_after_max_s: float,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._shared = _Shared(ledger, profile=profile, retry_after_max_s=retry_after_max_s)
        self._client = httpx.AsyncClient(transport=transport)

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        meter = self._shared.meter(request)
        try:
            response = await self._client.send(request, stream=True)
        except _FAILED_AFTER_SEND as exc:
            meter.finish(type(exc).__name__)
            raise
        except BaseException:
            meter.release()
            raise
        if not response.is_success:
            try:
                content = await response.aread()
            finally:
                await response.aclose()
                meter.release()
            return self._shared.error_response(request, response, content)
        return httpx.Response(
            response.status_code,
            headers=self._shared.passed_headers(response),
            stream=_AsyncMeteredStream(response, meter),
            extensions=response.extensions,
            request=request,
        )

    async def aclose(self) -> None:
        await self._client.aclose()
