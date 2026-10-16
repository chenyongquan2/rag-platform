"""Structured output for nodes nobody reads directly: router, rewrite, sufficiency, judge
(spec P1-01, design s11-1). Answers to users are numbered plain text instead (P1-02).
"""

from __future__ import annotations

from typing import Any, Literal

from langchain_core.language_models import BaseChatModel, LanguageModelInput
from langchain_core.runnables import Runnable, RunnableLambda
from pydantic import BaseModel, ValidationError

from rag.config import ServiceSettings

StructuredOutputMethod = Literal["function_calling", "json_schema", "json_mode"]

_SNIPPET_CHARS = 200


class StructuredOutputError(ValueError):
    """The model's answer could not be turned into the requested Pydantic object."""


def structured_output[SchemaT: BaseModel](
    model: BaseChatModel,
    schema: type[SchemaT],
    *,
    method: StructuredOutputMethod | None = None,
    settings: ServiceSettings | None = None,
) -> Runnable[LanguageModelInput, SchemaT]:
    """`model.with_structured_output(schema, method=...)` that returns a `schema` instance or
    raises `StructuredOutputError` (category `structured_output_failed`).

    `method` defaults to `chat_structured_output_method` in the settings; it is always passed
    explicitly because langchain-openai's own default (`json_schema`) is not what we chose.
    LangChain returns None when the model answers without the tool call; that is an error here.
    """
    if method is None:
        method = (settings or ServiceSettings()).chat_structured_output_method
    raw = model.with_structured_output(schema, method=method, include_raw=True)
    name = schema.__name__

    def unwrap(result: dict[str, Any]) -> SchemaT:
        error = result.get("parsing_error")
        parsed = result.get("parsed")
        if error is not None:
            raise StructuredOutputError(
                f"the model's answer is not a valid {name}: {error}"
            ) from error
        if parsed is None:
            raw_message = result.get("raw")
            content = str(getattr(raw_message, "content", ""))[:_SNIPPET_CHARS]
            raise StructuredOutputError(f"the model returned no {name} (answer: {content!r})")
        if isinstance(parsed, schema):
            return parsed
        try:
            return schema.model_validate(parsed)
        except ValidationError as exc:
            raise StructuredOutputError(f"the model's answer is not a valid {name}: {exc}") from exc

    return raw | RunnableLambda(unwrap, name=f"parse_{name}")
