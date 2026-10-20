"""One structured call of the small model for the route and rewrite steps (spec P3-02, s11-1).

    plan = await ask_helper(ctx, RewritePlan, messages)      # raises HelperFailed(reason)

The model is the `agent_rewrite_profile` chat model (default `cheap`) from the ordinary factory,
so every call goes through the retry and cost ledger transport. `ask_helper` adds what the
graph needs around it:

- the call is counted against the request's token cap, with the answer's tokens kept back
  (`ctx.guard`): a call that would eat them is refused before it is sent;
- it must end within `agent_rewrite_timeout_ms` and never use the time kept for the answer;
- structured output with `chat_structured_output_method` (function calling on the gateway);
- every failure becomes `HelperFailed` with a short reason, so a step can fall back to the
  original question and the request goes on: `token_cap`, `no_time`, `timeout`, `unavailable`
  (no model could be made), `model_error`, `parse_failed`, `budget_exceeded`, `error`.

Tokens, model calls and cost of the call are written to the current span.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import threading
from collections.abc import Iterator
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any

from langchain_core.messages import BaseMessage
from opentelemetry import trace
from pydantic import BaseModel

from rag.agent.budget import RequestTokenCapExceeded
from rag.cost import add_record_listener
from rag.llm import chat_error_category, structured_output
from rag.observe import attributes as attr
from rag.types import ErrorCategory

if TYPE_CHECKING:
    from rag.agent.context import AgentContext
    from rag.agent.state import AgentState

logger = logging.getLogger(__name__)


class HelperFailed(Exception):  # noqa: N818 - a step's fallback signal, not an error to report
    """The small model could not be used; `reason` says why (see the module doc)."""

    def __init__(self, reason: str, detail: str = "") -> None:
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason


_REASON: dict[ErrorCategory, str] = {
    ErrorCategory.STRUCTURED_OUTPUT_FAILED: "parse_failed",
    ErrorCategory.LLM_TIMEOUT: "timeout",
    ErrorCategory.LLM_ERROR: "model_error",
    ErrorCategory.BUDGET_EXCEEDED: "budget_exceeded",
}

# ---- ledger lines written by this task's model calls (for the span) --------------------------

_ROWS: ContextVar[list[dict[str, Any]] | None] = ContextVar("helper_ledger_rows", default=None)
_LISTENER_LOCK = threading.Lock()
_listener_installed = False


def _collect(row: dict[str, Any]) -> None:
    rows = _ROWS.get()
    if rows is not None:
        rows.append(row)


@contextlib.contextmanager
def _collect_ledger_rows() -> Iterator[list[dict[str, Any]]]:
    global _listener_installed
    with _LISTENER_LOCK:
        if not _listener_installed:
            add_record_listener(_collect)
            _listener_installed = True
    rows: list[dict[str, Any]] = []
    token = _ROWS.set(rows)
    try:
        yield rows
    finally:
        _ROWS.reset(token)


def _usage_attributes(rows: list[dict[str, Any]], tokens: int) -> dict[str, Any]:
    return {
        attr.NODE_TOKENS: tokens,
        attr.NODE_MODEL_CALLS: len(rows),
        attr.NODE_COST_CNY: round(sum(float(r.get("cny") or 0.0) for r in rows), 10),
        attr.GEN_AI_USAGE_INPUT_TOKENS: sum(int(r.get("prompt_tokens") or 0) for r in rows),
        attr.GEN_AI_USAGE_OUTPUT_TOKENS: sum(int(r.get("completion_tokens") or 0) for r in rows),
        attr.NODE_REASONING_TOKENS: sum(int(r.get("reasoning_tokens") or 0) for r in rows),
    }


def history_lines(state: AgentState, ctx: AgentContext) -> list[str]:
    """The earlier turns of the session as `Q:` and `A:` lines for a small-model prompt: the
    last `agent_rewrite_history_turns` turns, each text cut to `agent_rewrite_history_chars`
    (history is context for rewriting, not an archive). Empty without history."""
    turns = list(state.get("history") or [])
    keep = ctx.settings.agent_rewrite_history_turns
    if not turns or keep == 0:
        return []
    limit = ctx.settings.agent_rewrite_history_chars
    lines: list[str] = []
    for turn in turns[-keep:]:
        lines.append(f"Q: {turn.question[:limit]}")
        lines.append(f"A: {turn.answer[:limit]}")
    return lines


async def ask_helper[SchemaT: BaseModel](
    ctx: AgentContext,
    schema: type[SchemaT],
    messages: list[BaseMessage],
    *,
    timeout_ms: int | None = None,
    model: Any = None,
) -> SchemaT:
    """`schema` filled in by the small model from `messages`; `HelperFailed` when it cannot.

    `timeout_ms` is the longest the call may take (default `agent_rewrite_timeout_ms`; the
    judge passes its own); `model` is the chat model or the function that makes it (default:
    the helper model of the context)."""
    left_ms = ctx.deadline.remaining_ms() - ctx.reserve_ms
    allowed_ms = ctx.settings.agent_rewrite_timeout_ms if timeout_ms is None else timeout_ms
    timeout_ms = min(allowed_ms, left_ms)
    if timeout_ms <= 0:
        raise HelperFailed("no_time", "the time kept for the answer is all that is left")
    span = trace.get_current_span()
    spent_before = ctx.budget.spent
    try:
        try:
            chat = ctx.helper_model_instance() if model is None else ctx.model_instance(model)
        except Exception as exc:  # no gateway configured, unknown profile, no price
            logger.warning("no helper model: %s", exc)
            raise HelperFailed("unavailable", str(exc)) from exc
        runnable = ctx.guard(structured_output(chat, schema, settings=ctx.settings))
        with _collect_ledger_rows() as rows:
            try:
                async with asyncio.timeout(timeout_ms / 1000.0) as limit:
                    return await runnable.ainvoke(messages)
            except TimeoutError as exc:
                if limit.expired():
                    raise HelperFailed("timeout", f"no answer within {timeout_ms:.0f} ms") from exc
                raise
            finally:
                span.set_attributes(_usage_attributes(rows, ctx.budget.spent - spent_before))
    except HelperFailed:
        raise
    except RequestTokenCapExceeded as exc:
        raise HelperFailed("token_cap", str(exc)) from exc
    except Exception as exc:
        category = chat_error_category(exc)
        if category is None:
            logger.warning("helper model call failed unexpectedly", exc_info=True)
            raise HelperFailed("error", f"{type(exc).__name__}: {exc}") from exc
        logger.warning("helper model call failed (%s): %s", category, exc)
        raise HelperFailed(_REASON.get(category, "error"), str(exc)[:200]) from exc
