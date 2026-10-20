"""The service over fake retrieval and a fake model, for the page and for latency numbers (spec
P3-04): `python -m rag.serve http --dev-fake`. No gateway, no Elasticsearch, no reranker model:
the page can be tried, and the stream timed, on a machine with nothing else running.

The fake knowledge base `demo_docs` answers by keywords in the question (the route step asks back
about a question that names nothing technical, so add a name such as push_to_hub), so each thing
the page shows can be asked for:

    (any other)   an answer with two citations and one that the check removes
    xss           passages and an answer that contain markup (the page must show it as text)
    retry         the first attempt cites nothing and is written again
    slow          a long answer that takes about a minute (to press Cancel)
    nothing       no passage reaches the threshold: not found, with the closest passages
    "How do I fix this error?"   is asked back (the knowledge base clarifies)

Everything here is fake data. The server refuses to run it on an address other than the local
machine's.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage

from rag.config import AccessConfig, AgentConfig, KnowledgeBaseConfig, ServiceSettings
from rag.serve.service import SearchService
from rag.testing.agent import ScriptedSearchBackend, SlowStreamingChatModel, scored_result
from rag.testing.serve import FakePipelineBuilder
from rag.types import RetrievalResult

KB = "demo_docs"

PAGES: dict[str, tuple[str, str]] = {
    "hub": (
        "Upload a model to the Hub",
        "Call model.push_to_hub('user/name') after logging in with `huggingface-cli login`. "
        "To upload a whole folder at once use HfApi().upload_folder(folder_path=..., repo_id=...).",
    ),
    "tokenizers": (
        "Training a tokenizer",
        "A tokenizer is trained with Tokenizer.train(files, trainer). The vocabulary size is set "
        "on the trainer, for example BpeTrainer(vocab_size=30000).",
    ),
    "datasets": (
        "Loading a dataset",
        "load_dataset('name') downloads the dataset once and caches it. Pass streaming=True to "
        "read it without downloading everything first.",
    ),
}
XSS_TEXT = (
    'Do not trust this: <img src=x onerror="window.__pwned=1"> and '
    "<script>window.__pwned=1</script> and <b>bold</b>."
)

ANSWERS: dict[str, str] = {
    "default": (
        "To push a model call push_to_hub after logging in [1]. To upload a folder use "
        "upload_folder [2]. A made-up claim that cites a source that does not exist [7]."
    ),
    "xss": (
        'The answer itself contains <img src=x onerror="window.__pwned=1"> and '
        "<script>window.__pwned=1</script> [1]. Another passage says <b>bold</b> [2]."
    ),
    "slow": "This answer is written slowly so that there is time to press Cancel. " * 30
    + "It cites [1].",
    "nothing": "unused",
}


def _topic(text: str) -> str:
    lowered = text.lower()
    for word in ("xss", "retry", "slow", "nothing", "token", "dataset"):
        if word in lowered:
            return word
    return "default"


class DevSearchBackend(ScriptedSearchBackend):
    """Passages chosen by keywords, after a pause that stands for the search and rerank."""

    def __init__(self, kb: str, *, search_ms: int) -> None:
        super().__init__(kb)
        self.search_ms = search_ms

    def _result(self, question: str) -> RetrievalResult:
        topic = _topic(question)
        if topic == "nothing":
            return scored_result(self.kb, [0.12, 0.08], pages=[1, 2])
        if topic == "xss":
            result = scored_result(self.kb, [0.9, 0.7], pages=[1, 2], texts=[XSS_TEXT, XSS_TEXT])
            chunks = [
                c.model_copy(
                    update={
                        "title_path": [f"Page <img src=x onerror=window.__pwned=1> {i}"],
                        "source_url": "javascript:window.__pwned=1" if i == 2 else c.source_url,
                    }
                )
                for i, c in enumerate(result.chunks, 1)
            ]
            return result.model_copy(update={"chunks": chunks})
        keys = {"token": ["tokenizers", "hub"], "dataset": ["datasets", "hub"]}.get(
            topic, ["hub", "tokenizers", "datasets"]
        )
        result = scored_result(
            self.kb,
            [round(0.92 - 0.1 * i, 2) for i in range(len(keys))],
            pages=list(range(1, len(keys) + 1)),
            texts=[PAGES[k][1] for k in keys],
        )
        chunks = [
            c.model_copy(
                update={
                    "title_path": ["Guide", PAGES[k][0]],
                    "source_rel": f"guide/{k}.md",
                    "source_url": f"https://example.org/docs/{k}",
                }
            )
            for c, k in zip(result.chunks, keys, strict=True)
        ]
        return result.model_copy(update={"chunks": chunks})

    def retrieve(
        self,
        question: str,
        kb: str,
        top_k: int,
        *,
        filters: Mapping[str, Any] | None = None,
        deadline_ms: int | None = None,
        request_id: str | None = None,
        acl_groups: Sequence[str] | None = None,
    ) -> RetrievalResult:
        time.sleep(self.search_ms / 1000.0)
        self.default = self._result(question)
        return super().retrieve(
            question,
            kb,
            top_k,
            filters=filters,
            deadline_ms=deadline_ms,
            request_id=request_id,
            acl_groups=acl_groups,
        )


class DevChatModel(SlowStreamingChatModel):
    """The answer model: a reply per keyword in the question, written piece by piece with a pause
    between pieces. "retry": the first call (two messages) cites nothing; the second, which has
    the reminder after it, answers."""

    def _next_message(self, messages: list[BaseMessage]) -> AIMessage:
        self._calls.append(list(messages))
        user = str(messages[1].content)  # fragments, then "Question: ...", then the rules
        asked = user.rpartition("Question:")[2].split("Answer following")[0]
        topic = _topic(asked)
        if topic == "retry" and len(messages) == 2:
            return AIMessage(content="I know this one, but I will not say where it is written.")
        return AIMessage(content=ANSWERS.get(topic, ANSWERS["default"]))


def build_dev_service(
    settings: ServiceSettings, *, search_ms: int = 400, token_ms: int = 30
) -> SearchService:
    """The service over the fake knowledge base: stores in memory (nothing is written to
    `data/`), the fake backend and the fake model. `search_ms` is how long a search takes,
    `token_ms` the pause between two pieces of an answer."""
    dev = settings.model_copy(
        update={
            "checkpoint_url": "memory://",
            "session_store_url": "memory://",
            "session_lock_url": "memory://",
        }
    )
    cfg = KnowledgeBaseConfig(
        kb=KB,
        description="Demo documentation. Fake data: the page can be tried without a gateway.",
        language="en",
        access=AccessConfig(callers=["*"]),
        agent=AgentConfig(clarify=True),
    )
    backend = DevSearchBackend(KB, search_ms=search_ms)
    service = SearchService(
        dev,
        configs={KB: cfg},
        builder=FakePipelineBuilder({KB: backend}),
        answer_model=DevChatModel(delay_s=token_ms / 1000.0, chunk_chars=5),
    )
    service.load()
    return service
