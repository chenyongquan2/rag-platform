"""Per-request token cap (design s16-4, s16-8; spec P3-01).

    budget = TokenBudget(cap=20_000, answer_reserve=6_000)
    model = guard_model(chat_model, budget)               # the answer's calls
    helper = guard_model(cheap_model, budget, reserve=budget.answer_reserve)  # calls before it

A guarded model counts the tokens of every call (the gateway's usage when the reply has it, else
an estimate of prompt and reply, `rag.answer.context.estimate_tokens`) and refuses a call, before
sending it, once the tokens spent plus the `reserve` of that call reach the cap. Calls that come
before the answer (rewrite, judge; P3-02, P3-03) use a reserve, so they stop early enough for the
answer to still fit. The refusal is `RequestTokenCapExceeded`, a `BudgetExceeded`: the answer
layer and `chat_error_category` already treat that as "no model answer, chunks only".
"""

from __future__ import annotations

import threading
from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage
from langchain_core.outputs import LLMResult
from langchain_core.runnables import Runnable

from rag.answer.context import estimate_tokens
from rag.cost import BudgetExceeded


class RequestTokenCapExceeded(BudgetExceeded):
    """The request has spent its token cap (`agent_token_cap`): no further model call."""


class TokenBudget:
    """Tokens one request may spend on model calls. `cap_hit` is set when a call was refused."""

    def __init__(self, cap: int, answer_reserve: int = 0) -> None:
        self.cap = cap
        self.answer_reserve = answer_reserve
        self.spent = 0
        self.cap_hit = False
        self._lock = threading.Lock()

    def add(self, tokens: int) -> None:
        with self._lock:
            self.spent += max(0, tokens)

    def remaining(self) -> int:
        return max(0, self.cap - self.spent)

    def exhausted(self, *, reserve: int = 0) -> bool:
        """True when a call that must leave `reserve` tokens for later could not be made."""
        return self.spent + reserve >= self.cap

    def check(self, *, reserve: int = 0) -> None:
        if self.exhausted(reserve=reserve):
            self.cap_hit = True
            raise RequestTokenCapExceeded(
                f"token cap {self.cap} reached ({self.spent} spent, {reserve} reserved)"
            )


def _text(content: Any) -> str:
    return content if isinstance(content, str) else str(content)


class _TokenCounter(BaseCallbackHandler):
    """LangChain callback: checks the budget before a chat model call, counts it after.

    `raise_error` makes LangChain pass the refusal on to the caller instead of logging it, and
    `run_inline` runs it in the event loop's thread, in order."""

    raise_error = True
    run_inline = True

    def __init__(self, budget: TokenBudget, reserve: int) -> None:
        self.budget = budget
        self.reserve = reserve
        self._prompt_estimate: dict[UUID, int] = {}

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[BaseMessage]],
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        self.budget.check(reserve=self.reserve)
        self._prompt_estimate[run_id] = sum(
            estimate_tokens(_text(m.content)) for batch in messages for m in batch
        )

    def on_llm_end(self, response: LLMResult, *, run_id: UUID, **kwargs: Any) -> None:
        estimate = self._prompt_estimate.pop(run_id, 0)
        reported = 0
        has_usage = False
        for generations in response.generations:
            for generation in generations:
                usage = getattr(getattr(generation, "message", None), "usage_metadata", None)
                if usage:
                    has_usage = True
                    reported += int(
                        usage.get("total_tokens")
                        or usage.get("input_tokens", 0) + usage.get("output_tokens", 0)
                    )
                else:
                    estimate += estimate_tokens(generation.text)
        self.budget.add(reported if has_usage else estimate)

    def on_llm_error(
        self, error: BaseException, *, run_id: UUID, **kwargs: Any
    ) -> None:  # the call did not complete: nothing is counted
        self._prompt_estimate.pop(run_id, None)


def guard_model(model: Runnable[Any, Any], budget: TokenBudget, *, reserve: int = 0) -> Any:
    """`model` with the budget attached: calls are counted, and refused (before they are sent)
    with `RequestTokenCapExceeded` once `budget.exhausted(reserve=reserve)`."""
    return model.with_config(callbacks=[_TokenCounter(budget, reserve)])


__all__ = ["RequestTokenCapExceeded", "TokenBudget", "guard_model"]
