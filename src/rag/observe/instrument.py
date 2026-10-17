"""OpenInference auto-instrumentation of LlamaIndex and LangChain/LangGraph (spec P0-11, s15-1).

The instrumentors write spans for framework calls (embedding, retriever, chat model, graph
node) with our tracer provider. Spans nest under the current span because the instrumentors
keep the runtime context (`separate_trace_from_runtime_context=False`). Every content switch is
set here, none is left to the library default or the OPENINFERENCE_* env vars: embedding vectors
are always hidden (1024 floats per text would fill the trace file); prompts, answers and
retrieved texts follow `otel_trace_content`.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from opentelemetry.trace import TracerProvider

# Longest base64 image kept in a span (the library's own default, written out).
BASE64_IMAGE_MAX_LENGTH = 32_000


def trace_config(trace_content: bool) -> Any:
    from openinference.instrumentation import TraceConfig

    hide = not trace_content
    return TraceConfig(
        hide_llm_invocation_parameters=False,
        hide_llm_tools=False,
        hide_inputs=hide,
        hide_outputs=hide,
        hide_input_messages=hide,
        hide_output_messages=hide,
        hide_input_images=hide,
        hide_input_text=hide,
        hide_output_text=hide,
        hide_embeddings_vectors=True,
        hide_embeddings_text=hide,
        hide_prompts=hide,
        hide_choices=hide,
        hide_retrieval_documents=hide,
        enable_genai_semconv=False,
        base64_image_max_length=BASE64_IMAGE_MAX_LENGTH,
    )


def instrument_frameworks(
    tracer_provider: TracerProvider, *, trace_content: bool
) -> Callable[[], None]:
    """Instrument LlamaIndex and LangChain (LangGraph runs through LangChain callbacks) with
    `tracer_provider`; returns the function that removes the instrumentation again."""
    from openinference.instrumentation.langchain import LangChainInstrumentor
    from openinference.instrumentation.llama_index import LlamaIndexInstrumentor

    config = trace_config(trace_content)
    instrumentors = [LlamaIndexInstrumentor(), LangChainInstrumentor()]
    for instrumentor in instrumentors:
        if instrumentor.is_instrumented_by_opentelemetry:  # left over from another provider
            instrumentor.uninstrument()
        instrumentor.instrument(
            tracer_provider=tracer_provider,
            config=config,
            separate_trace_from_runtime_context=False,
        )

    def uninstrument() -> None:
        for instrumentor in instrumentors:
            if instrumentor.is_instrumented_by_opentelemetry:
                instrumentor.uninstrument()

    return uninstrument
