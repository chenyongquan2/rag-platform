"""The one place that builds chat models (spec P1-01, design s11-1, s16-2, s16-7).

`make_chat_model(profile)` returns a LangChain chat model from `init_chat_model` talking to the
OpenAI-compatible gateway. The three profiles share the gateway and, by default, the model
(`chat_model`); they differ in connect and read timeouts, deadline, retries, waits and output
token cap. Every call goes through `LedgerTransport`, which does the retries (the openai SDK's
are off) within the profile's deadline, reserves each attempt's estimated cost before sending
and writes its real usage to the cost ledger afterwards.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import get_args

import httpx
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel

from rag.config import ChatProfile, ChatProfileName, ServiceSettings
from rag.cost import CostLedger
from rag.llm.transport import AsyncLedgerTransport, LedgerTransport
from rag.testing.fakes import FakeChatModel

PROFILES: tuple[str, ...] = get_args(ChatProfileName)


def resolve_chat_profile(profile: ChatProfileName, settings: ServiceSettings) -> ChatProfile:
    """The settings of `profile`, with an empty `model` replaced by `chat_model`."""
    if profile not in PROFILES:
        raise ValueError(f"chat profile must be one of {', '.join(PROFILES)}; got {profile!r}")
    config: ChatProfile = getattr(settings, f"chat_{profile}")
    return config if config.model else config.model_copy(update={"model": settings.chat_model})


def make_chat_model(
    profile: ChatProfileName,
    settings: ServiceSettings | None = None,
    *,
    ledger: CostLedger | None = None,
    transport: httpx.BaseTransport | None = None,
    async_transport: httpx.AsyncBaseTransport | None = None,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    max_tokens: int | None = None,
) -> BaseChatModel:
    """Chat model for `profile` (`online`, `offline` or `cheap`).

    `chat_backend=gateway`: a `ChatOpenAI` built by `init_chat_model` with every parameter set
    from the profile (model, connect and read timeouts, `max_tokens`, temperature),
    `stream_usage` on so streams end with a usage event, and `max_retries=0`: the ledger
    transport retries instead. `chat_backend=fake`: a `FakeChatModel` that answers
    `chat_fake_reply` to every call; it sends nothing and writes no ledger line.
    Fails before any request when the gateway URL or key is missing or the model has no price.
    `ledger`, `transport`, `async_transport`, `clock` and `sleep` (sync retries) are for tests.
    `max_tokens` replaces the profile's output cap for this model only (spec P3-06: the graph's
    answer call); None keeps the profile's.
    """
    settings = settings or ServiceSettings()
    config = resolve_chat_profile(profile, settings)
    if max_tokens is not None:
        if max_tokens < 1:
            raise ValueError(f"max_tokens must be >= 1, got {max_tokens}")
        config = config.model_copy(update={"max_tokens": max_tokens})
    if settings.chat_backend == "fake":
        return FakeChatModel(default_response=settings.chat_fake_reply)
    if not settings.llm_base_url:
        raise ValueError("LLM_BASE_URL is not set; set it in .env or use CHAT_BACKEND=fake")
    api_key = settings.llm_api_key.get_secret_value()
    if not api_key:
        raise ValueError("LLM_API_KEY is not set; set it in .env or use CHAT_BACKEND=fake")
    ledger = ledger or CostLedger.from_settings(settings)
    ledger.price(config.model)  # no price -> fail now, before any money is spent

    policy = {
        "ledger": ledger,
        "profile": profile,
        "max_retries": config.max_retries,
        "backoff_s": config.backoff_s,
        "backoff_max_s": config.backoff_max_s,
        "deadline_s": config.deadline_s,
        "clock": clock,
    }
    http_client = httpx.Client(
        transport=LedgerTransport(transport=transport, sleep=sleep, **policy)
    )
    http_async_client = httpx.AsyncClient(
        transport=AsyncLedgerTransport(transport=async_transport, **policy)
    )
    return init_chat_model(
        config.model,
        model_provider="openai",
        base_url=settings.llm_base_url,
        api_key=api_key,
        timeout=httpx.Timeout(config.read_timeout_s, connect=config.connect_timeout_s),
        max_retries=0,
        max_tokens=config.max_tokens,
        temperature=config.temperature,
        stream_usage=True,
        use_responses_api=False,
        http_client=http_client,
        http_async_client=http_async_client,
        # We pass both clients, so nothing is injected; () also stops the warning that says so.
        http_socket_options=(),
    )
