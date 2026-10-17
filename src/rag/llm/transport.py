"""httpx transports that retry chat model calls and put every attempt on the cost ledger
(spec P1-01).

`make_chat_model` gives `ChatOpenAI` httpx clients whose transport is one of these, and turns
the openai SDK's own retries off: one SDK request is one call to `handle_request`, and the
transport owns the retries, so it can keep one deadline per call. The transports work the same
for `invoke`, `stream`, structured output, sync and async calls.

- Retries (design s16-2, s16-8): failures before the gateway got the request (connect errors
  and timeouts, write errors) and 408, 429 and 5xx answers are retried up to `max_retries`
  times, waiting `backoff_s` * 2**n up to `backoff_max_s`. A longer `Retry-After` ends the call.
  A read timeout or dropped connection after the request was sent is not retried: the gateway
  may still be generating, and billing, the first answer. With `deadline_s`, every attempt's
  timeouts are cut to the time left and no wait may reach the deadline.
- Before an attempt is sent, its estimated cost is reserved in the ledger (`CostLedger.reserve`,
  spec P0-05); over the budget stop `BudgetExceeded` is raised and nothing is sent. The estimate
  is an upper bound: one token per character of the request body as input, and the request's
  `max_completion_tokens` (reasoning and answer together) times `n` answers as output.
- Every 2xx answer appends one `chat` line with the tokens from `usage` (for a stream, the usage
  event at its end). Output tokens include reasoning tokens. Without usage (no field, a stream
  closed before its end) the line is estimated. So is an attempt that failed or was cancelled
  (`asyncio.CancelledError`, `KeyboardInterrupt`) after it was handed to the network, since
  the gateway may have done and billed the work. Connect failures and 4xx/5xx answers write no
  line (the gateway did not process the request).
- The API key echoed in an error body is masked before the SDK puts the body into its exception.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator
from contextlib import ExitStack
from typing import Any

import httpx

from rag.cost import CostLedger

logger = logging.getLogger(__name__)

# Raised before the gateway had the whole request: safe to send again, nothing billed.
_FAILED_BEFORE_SEND = (
    httpx.ConnectError,
    httpx.ConnectTimeout,
    httpx.PoolTimeout,
    httpx.WriteError,
    httpx.WriteTimeout,
)
# Raised after the request went out: the gateway may have done (and billed) the work.
_FAILED_AFTER_SEND = (httpx.ReadTimeout, httpx.ReadError, httpx.RemoteProtocolError)
_RETRYABLE_STATUS = frozenset({408, 429})  # and every 5xx
# Output estimate for a request without max_completion_tokens (the factory always sets it).
_UNCAPPED_OUTPUT_TOKENS = 32_768
# The body is passed on decoded, so these no longer describe it.
_DROP_HEADERS = frozenset({"content-encoding", "content-length", "transfer-encoding"})
_CLOSED_EARLY = "closed before usage"
_SHORTEST_TIMEOUT_S = 0.001


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
    """The wait the gateway asks for, in seconds (`retry-after-ms` or `retry-after`)."""
    for name, scale in (("retry-after-ms", 1000.0), ("retry-after", 1.0)):
        value = headers.get(name)
        if value is None:
            continue
        try:
            return max(0.0, float(value) / scale)
        except ValueError:  # an HTTP date: use the computed backoff instead
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
        if not isinstance(cap, int) or cap <= 0:
            cap = _UNCAPPED_OUTPUT_TOKENS
        answers = body.get("n")
        if not isinstance(answers, int) or answers < 1:
            answers = 1
        self.estimated_prompt = len(text)
        self.estimated_output = cap * answers
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


class _Policy:
    """Retries, deadline and accounting shared by the sync and async transports."""

    def __init__(
        self,
        ledger: CostLedger,
        *,
        profile: str,
        max_retries: int,
        backoff_s: float,
        backoff_max_s: float,
        deadline_s: float | None,
        clock: Callable[[], float],
    ) -> None:
        self.ledger = ledger
        self.profile = profile
        self.max_retries = max_retries
        self.backoff_s = backoff_s
        self.backoff_max_s = backoff_max_s
        self.deadline_s = deadline_s
        self.clock = clock

    def start(self, request: httpx.Request) -> tuple[float, dict[str, Any]]:
        """Start time and the timeouts the SDK set on the request."""
        return self.clock(), dict(request.extensions.get("timeout") or {})

    def prepare(self, request: httpx.Request, start: float, timeouts: dict[str, Any]) -> _Meter:
        """Cut the attempt's timeouts to the time left before the deadline, then reserve."""
        if self.deadline_s is not None:
            left = max(self.deadline_s - (self.clock() - start), _SHORTEST_TIMEOUT_S)
            cut = {k: left if v is None else min(v, left) for k, v in timeouts.items()}
            request.extensions = {**request.extensions, "timeout": cut}
        meter = _Meter(self.ledger, request, profile=self.profile)
        meter.reserve()
        return meter

    def wait_before_retry(
        self, attempt: int, start: float, cause: object, retry_after: float | None = None
    ) -> float | None:
        """Seconds to wait before attempt `attempt + 1`, or None to give up."""
        if attempt >= self.max_retries:
            return None
        if retry_after is not None and retry_after > self.backoff_max_s:
            return None
        wait = retry_after
        if wait is None:
            wait = min(self.backoff_s * 2**attempt, self.backoff_max_s)
        if self.deadline_s is not None and self.clock() - start + wait >= self.deadline_s:
            return None
        logger.warning(
            "chat model (%s): %s; retry %d of %d in %.1f s",
            self.profile,
            cause,
            attempt + 1,
            self.max_retries,
            wait,
        )
        return wait

    @staticmethod
    def retryable(response: httpx.Response) -> bool:
        return response.status_code in _RETRYABLE_STATUS or response.status_code >= 500

    @staticmethod
    def metered(
        request: httpx.Request,
        response: httpx.Response,
        stream: httpx.SyncByteStream | httpx.AsyncByteStream,
    ) -> httpx.Response:
        return httpx.Response(
            response.status_code,
            headers=_passed_headers(response),
            stream=stream,
            extensions=response.extensions,
            request=request,
        )

    @staticmethod
    def error_response(
        request: httpx.Request, response: httpx.Response, content: bytes
    ) -> httpx.Response:
        token = _bearer_token(request)
        if token and token.encode() in content:
            content = content.replace(token.encode(), b"***")
        return httpx.Response(
            response.status_code,
            headers=_passed_headers(response),
            content=content,
            extensions=response.extensions,
            request=request,
        )


def _passed_headers(response: httpx.Response) -> list[tuple[str, str]]:
    return [(k, v) for k, v in response.headers.multi_items() if k.lower() not in _DROP_HEADERS]


def _interrupted(exc: BaseException) -> bool:
    """Cancelled or interrupted (not an error, not the consumer closing a generator)."""
    return not isinstance(exc, Exception | GeneratorExit)


class _MeteredStream(httpx.SyncByteStream):
    def __init__(self, response: httpx.Response, meter: _Meter) -> None:
        self._response = response
        self._meter = meter

    def __iter__(self) -> Iterator[bytes]:
        try:
            for chunk in self._response.iter_bytes():
                self._meter.feed(chunk)
                yield chunk
        except BaseException as exc:
            if isinstance(exc, _FAILED_AFTER_SEND) or _interrupted(exc):
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
        except BaseException as exc:
            if isinstance(exc, _FAILED_AFTER_SEND) or _interrupted(exc):
                self._meter.finish(type(exc).__name__)
            raise
        self._meter.finish()

    async def aclose(self) -> None:
        try:
            await self._response.aclose()
        finally:
            self._meter.finish(_CLOSED_EARLY)


class LedgerTransport(httpx.BaseTransport):
    """Sync transport: retry, reserve, send through an inner client, record (module doc).

    The inner `httpx.Client` reads proxies from the environment (`HTTPS_PROXY`, `NO_PROXY`) like
    any default client; the outer client, which has this transport, would ignore them. Each
    attempt keeps the timeouts the openai SDK set on the request, cut to the deadline.
    `transport`, `clock` and `sleep` replace the network and the time in tests.
    """

    def __init__(
        self,
        ledger: CostLedger,
        *,
        profile: str,
        max_retries: int,
        backoff_s: float,
        backoff_max_s: float,
        deadline_s: float | None,
        transport: httpx.BaseTransport | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._policy = _Policy(
            ledger,
            profile=profile,
            max_retries=max_retries,
            backoff_s=backoff_s,
            backoff_max_s=backoff_max_s,
            deadline_s=deadline_s,
            clock=clock,
        )
        self._sleep = sleep
        self._client = httpx.Client(transport=transport)

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        policy = self._policy
        start, timeouts = policy.start(request)
        attempt = 0
        while True:
            meter = policy.prepare(request, start, timeouts)
            try:
                response = self._client.send(request, stream=True)
            except _FAILED_BEFORE_SEND as exc:
                meter.release()
                wait = policy.wait_before_retry(attempt, start, repr(exc))
                if wait is None:
                    raise
            except BaseException as exc:
                if isinstance(exc, _FAILED_AFTER_SEND) or _interrupted(exc):
                    meter.finish(type(exc).__name__)
                else:
                    meter.release()
                raise
            else:
                if response.is_success:
                    return policy.metered(request, response, _MeteredStream(response, meter))
                try:
                    content = response.read()
                finally:
                    response.close()
                    meter.release()
                wait = None
                if policy.retryable(response):
                    wait = policy.wait_before_retry(
                        attempt,
                        start,
                        f"HTTP {response.status_code}",
                        _retry_after_s(response.headers),
                    )
                if wait is None:
                    return policy.error_response(request, response, content)
            self._sleep(wait)
            attempt += 1

    def close(self) -> None:
        self._client.close()


class AsyncLedgerTransport(httpx.AsyncBaseTransport):
    """Async twin of `LedgerTransport`."""

    def __init__(
        self,
        ledger: CostLedger,
        *,
        profile: str,
        max_retries: int,
        backoff_s: float,
        backoff_max_s: float,
        deadline_s: float | None,
        transport: httpx.AsyncBaseTransport | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self._policy = _Policy(
            ledger,
            profile=profile,
            max_retries=max_retries,
            backoff_s=backoff_s,
            backoff_max_s=backoff_max_s,
            deadline_s=deadline_s,
            clock=clock,
        )
        self._sleep = sleep
        self._client = httpx.AsyncClient(transport=transport)

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        policy = self._policy
        start, timeouts = policy.start(request)
        attempt = 0
        while True:
            meter = policy.prepare(request, start, timeouts)
            try:
                response = await self._client.send(request, stream=True)
            except _FAILED_BEFORE_SEND as exc:
                meter.release()
                wait = policy.wait_before_retry(attempt, start, repr(exc))
                if wait is None:
                    raise
            except BaseException as exc:
                if isinstance(exc, _FAILED_AFTER_SEND) or _interrupted(exc):
                    meter.finish(type(exc).__name__)
                else:
                    meter.release()
                raise
            else:
                if response.is_success:
                    return policy.metered(request, response, _AsyncMeteredStream(response, meter))
                try:
                    content = await response.aread()
                finally:
                    await response.aclose()
                    meter.release()
                wait = None
                if policy.retryable(response):
                    wait = policy.wait_before_retry(
                        attempt,
                        start,
                        f"HTTP {response.status_code}",
                        _retry_after_s(response.headers),
                    )
                if wait is None:
                    return policy.error_response(request, response, content)
            await self._sleep(wait)
            attempt += 1

    async def aclose(self) -> None:
        await self._client.aclose()
