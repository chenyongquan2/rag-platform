"""Model-written text for PDFs and images: figure descriptions and table summaries (spec P6-01,
design s5-3, s7-2, s16-4).

Every such text costs a model call, so every one goes through `ModelTextService`:

1. the text is looked up in `ModelTextCache` (SQLite, key: kind, model, prompt version, sha256 of
   the content), so a second read of the same image or table costs nothing, also after the
   index or the docstore was thrown away;
2. otherwise `SpendGuard` checks that the estimated cost still fits the budget of this run (and
   the cost ledger's own stop line, which `make_chat_model` enforces on every call);
3. the model is called, the real cost is added to the guard, the text is cached.

`GatewayVisionDescriber` and `GatewayTableSummarizer` are the gateway clients (LangChain chat
models from `rag.llm.make_chat_model`, so retries, the ledger and `CHAT_BACKEND=fake` work as
everywhere else). The estimates before a run (`estimate_image_cost`, `estimate_table_cost`) are
upper bounds from the price table.
"""

from __future__ import annotations

import base64
import hashlib
import io
import sqlite3
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from PIL import Image

from rag.cost import CostLedger

IMAGE_PROMPT_VERSION = "img1"
TABLE_PROMPT_VERSION = "tbl1"
"""Part of the cache key: change a prompt, bump its version, and the old texts are not reused."""

# Upper bounds used before a run (tokens). A figure of at most 1024 px costs about 1,300 image
# tokens at the gateway's vision model (measured: 240x160 px = 72 tokens; roughly 1 per 28x28 px
# block); the prompt and the caption come on top.
IMAGE_PROMPT_TOKENS = 1700
IMAGE_ANSWER_TOKENS = 300
TABLE_PROMPT_OVERHEAD = 200
TABLE_ANSWER_TOKENS = 150
CHARS_PER_TOKEN = 3.0


IMAGE_MARKER_PREFIX = "[Image description (written by a model"
IMAGE_MARKER_TEMPLATE = IMAGE_MARKER_PREFIX + ", not part of the original document): {text}]"
"""How a figure description stands in the page text: the sentence says who wrote it, so the
answer model and the reader can tell it from the document's own words (design s5-3)."""


def image_marker(text: str) -> str:
    return IMAGE_MARKER_TEMPLATE.format(text=" ".join(text.split()))


class DescribeBudgetError(RuntimeError):
    """The next model call would pass the budget of this run; nothing was sent."""


@dataclass(frozen=True)
class ModelReply:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cny: float = 0.0


@dataclass(frozen=True)
class CachedText:
    text: str
    prompt_tokens: int
    completion_tokens: int


class ImageDescriber(Protocol):
    model: str

    def describe(self, png: bytes, *, caption: str, language: str) -> ModelReply: ...


class TableSummarizer(Protocol):
    model: str

    def summarize(self, table_markdown: str, *, caption: str) -> ModelReply: ...


# ---- cache -----------------------------------------------------------------------------------


class ModelTextCache:
    """One SQLite file (`:memory:` for tests): model-written text by kind, model, prompt version
    and content hash. Same pattern as the embedding cache (`rag.embed.cache`)."""

    def __init__(self, path: str | Path) -> None:
        self._lock = threading.Lock()
        target = str(path)
        if target != ":memory:":
            Path(target).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(
            target, timeout=30, check_same_thread=False, isolation_level=None
        )
        if target != ":memory:":
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS model_text ("
            " kind TEXT NOT NULL, model TEXT NOT NULL, prompt_version TEXT NOT NULL,"
            " content_sha256 TEXT NOT NULL, text TEXT NOT NULL,"
            " prompt_tokens INTEGER NOT NULL, completion_tokens INTEGER NOT NULL,"
            " PRIMARY KEY (kind, model, prompt_version, content_sha256)"
            ") WITHOUT ROWID"
        )

    def get(
        self, kind: str, model: str, prompt_version: str, content_sha256: str
    ) -> CachedText | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT text, prompt_tokens, completion_tokens FROM model_text"
                " WHERE kind = ? AND model = ? AND prompt_version = ? AND content_sha256 = ?",
                (kind, model, prompt_version, content_sha256),
            ).fetchone()
        return CachedText(row[0], int(row[1]), int(row[2])) if row else None

    def put(
        self,
        kind: str,
        model: str,
        prompt_version: str,
        content_sha256: str,
        text: str,
        *,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO model_text VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    kind,
                    model,
                    prompt_version,
                    content_sha256,
                    text,
                    prompt_tokens,
                    completion_tokens,
                ),
            )

    def close(self) -> None:
        with self._lock:
            self._conn.close()


# ---- budget and stats ------------------------------------------------------------------------


class SpendGuard:
    """The budget of one run: `check` before a call, `add` the real cost after it."""

    def __init__(self, budget_cny: float | None) -> None:
        self.budget_cny = budget_cny
        self.spent = 0.0
        self._lock = threading.Lock()

    def check(self, estimated_cny: float) -> None:
        if self.budget_cny is None:
            return
        with self._lock:
            if self.spent + estimated_cny > self.budget_cny:
                raise DescribeBudgetError(
                    f"model text budget {self.budget_cny} CNY: spent {self.spent:.6f} CNY + "
                    f"estimated {estimated_cny:.6f} CNY for the next call would pass it; "
                    "nothing was sent (raise `budget_cny` or run with the cache filled)"
                )

    def add(self, cny: float) -> None:
        with self._lock:
            self.spent += cny


@dataclass
class DescribeStats:
    """What the reader and the table step report (no text of the documents)."""

    images_described: int = 0
    image_cache_hits: int = 0
    tables_summarised: int = 0
    table_cache_hits: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    spend_cny: float = 0.0
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "images_described": self.images_described,
            "image_cache_hits": self.image_cache_hits,
            "tables_summarised": self.tables_summarised,
            "table_cache_hits": self.table_cache_hits,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "spend_cny": round(self.spend_cny, 6),
            "errors": list(self.errors),
        }


def sha256_hex(data: bytes | str) -> str:
    return hashlib.sha256(data.encode("utf-8") if isinstance(data, str) else data).hexdigest()


class ModelTextService:
    """Cache, budget guard and stats around an image describer and a table summarizer."""

    def __init__(
        self,
        *,
        cache: ModelTextCache,
        describer: ImageDescriber | None = None,
        summarizer: TableSummarizer | None = None,
        guard: SpendGuard | None = None,
        image_estimate_cny: float = 0.0,
        table_estimate_cny: float = 0.0,
    ) -> None:
        self.cache = cache
        self.describer = describer
        self.summarizer = summarizer
        self.guard = guard or SpendGuard(None)
        self.image_estimate_cny = image_estimate_cny
        self.table_estimate_cny = table_estimate_cny
        self.stats = DescribeStats()
        self._lock = threading.Lock()

    def _record(self, reply: ModelReply) -> None:
        with self._lock:
            self.stats.prompt_tokens += reply.prompt_tokens
            self.stats.completion_tokens += reply.completion_tokens
            self.stats.spend_cny += reply.cny
        self.guard.add(reply.cny)

    def describe_image(self, png: bytes, *, caption: str, language: str) -> str:
        if self.describer is None:
            raise RuntimeError("no image describer is configured")
        key = sha256_hex(png)
        hit = self.cache.get("image", self.describer.model, IMAGE_PROMPT_VERSION, key)
        if hit is not None:
            self.stats.image_cache_hits += 1
            return hit.text
        self.guard.check(self.image_estimate_cny)
        try:
            reply = self.describer.describe(png, caption=caption, language=language)
        except Exception as exc:
            self.stats.errors.append(f"image: {type(exc).__name__}: {exc}")
            raise
        self._record(reply)
        self.cache.put(
            "image",
            self.describer.model,
            IMAGE_PROMPT_VERSION,
            key,
            reply.text,
            prompt_tokens=reply.prompt_tokens,
            completion_tokens=reply.completion_tokens,
        )
        self.stats.images_described += 1
        return reply.text

    def summarize_table(self, table_markdown: str, *, caption: str) -> str:
        if self.summarizer is None:
            raise RuntimeError("no table summarizer is configured")
        key = sha256_hex(caption + "\n" + table_markdown)
        hit = self.cache.get("table", self.summarizer.model, TABLE_PROMPT_VERSION, key)
        if hit is not None:
            self.stats.table_cache_hits += 1
            return hit.text
        self.guard.check(self.table_estimate_cny)
        try:
            reply = self.summarizer.summarize(table_markdown, caption=caption)
        except Exception as exc:
            self.stats.errors.append(f"table: {type(exc).__name__}: {exc}")
            raise
        self._record(reply)
        self.cache.put(
            "table",
            self.summarizer.model,
            TABLE_PROMPT_VERSION,
            key,
            reply.text,
            prompt_tokens=reply.prompt_tokens,
            completion_tokens=reply.completion_tokens,
        )
        self.stats.tables_summarised += 1
        return reply.text


# ---- estimates -------------------------------------------------------------------------------


def estimate_image_cost(ledger: CostLedger, model: str, images: int) -> float:
    """Upper bound in CNY for describing `images` figures that are not in the cache."""
    if images <= 0:
        ledger.price(model)
        return 0.0
    return images * ledger.estimate(model, IMAGE_PROMPT_TOKENS, IMAGE_ANSWER_TOKENS)


def estimate_table_cost(ledger: CostLedger, model: str, table_tokens: list[int]) -> float:
    """Upper bound in CNY for summarising tables of these sizes (tokens) that are not cached."""
    if not table_tokens:
        ledger.price(model)
        return 0.0
    prompt = sum(table_tokens) + TABLE_PROMPT_OVERHEAD * len(table_tokens)
    return ledger.estimate(model, prompt, TABLE_ANSWER_TOKENS * len(table_tokens))


# ---- gateway clients -------------------------------------------------------------------------

NO_THINKING: dict[str, Any] = {"enable_thinking": False}
"""Request body field that turns the thinking mode of the Qwen models off. Without it the
gateway's `qwen/qwen3.6-flash` spends the whole output cap on reasoning for a table and returns an
empty answer (`finish_reason: length`, measured in P6-01: 30 of 30 table summaries and 6 of 15
figures came back empty with a cap of 400 to 600 tokens); with it a summary takes about 35
output tokens."""

LANGUAGE_NAMES = {"en": "English", "zh": "Chinese"}

IMAGE_PROMPT = (
    "This figure comes from a technical document. Describe it so that a reader who cannot see "
    "it could find it with a search and answer questions about it. Say what kind of figure it is "
    "(chart, diagram, photo, screenshot), what the axes, boxes or labels say, what the series or "
    "parts are called, and the main trend or result in two to four sentences. Copy visible words "
    "and numbers exactly; do not guess numbers you cannot read. Write in {language}. Reply with "
    "the description only.{caption}"
)
TABLE_PROMPT = (
    "Here is a table from a technical document in Markdown. Write one or two sentences that say "
    "what the table shows: what the rows are, what the columns measure, and which values stand "
    "out. Use the exact column and row names. Do not add anything that is not in the table. "
    "Reply with the sentences only.{caption}\n\n{table}"
)


def prepare_image(png: bytes, *, max_px: int, quality: int = 85) -> bytes:
    """The image as a JPEG whose longer side is at most `max_px` (fewer image tokens, a smaller
    request body). Deterministic for the same input."""
    image = Image.open(io.BytesIO(png))
    image = image.convert("RGB")
    image.thumbnail((max_px, max_px))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    return buffer.getvalue()


def _usage(message: Any) -> tuple[int, int]:
    usage = getattr(message, "usage_metadata", None) or {}
    return int(usage.get("input_tokens", 0) or 0), int(usage.get("output_tokens", 0) or 0)


def _text(message: Any) -> str:
    content = message.content
    if isinstance(content, list):
        content = "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
    text = str(content).strip()
    if not text:
        raise ValueError("the model's answer was empty")
    return text


class GatewayVisionDescriber:
    """Describes a figure with a vision model of the gateway."""

    def __init__(
        self,
        chat: BaseChatModel,
        *,
        model: str,
        ledger: CostLedger,
        max_px: int = 1024,
        extra_body: dict[str, Any] | None = None,
    ) -> None:
        self.chat: Any = chat.bind(extra_body=extra_body) if extra_body else chat
        self.model = model
        self.ledger = ledger
        self.max_px = max_px

    def describe(self, png: bytes, *, caption: str, language: str) -> ModelReply:
        jpeg = prepare_image(png, max_px=self.max_px)
        note = f"\nThe caption in the document reads: {caption}" if caption.strip() else ""
        prompt = IMAGE_PROMPT.format(language=LANGUAGE_NAMES.get(language, "English"), caption=note)
        url = "data:image/jpeg;base64," + base64.b64encode(jpeg).decode("ascii")
        message = HumanMessage(
            content=[
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": url}},
            ]
        )
        answer = self.chat.invoke([message])
        used_in, used_out = _usage(answer)
        return ModelReply(
            _text(answer), used_in, used_out, self.ledger.estimate(self.model, used_in, used_out)
        )


class GatewayTableSummarizer:
    """Writes a one- or two-sentence summary of a table."""

    def __init__(
        self,
        chat: BaseChatModel,
        *,
        model: str,
        ledger: CostLedger,
        max_table_chars: int = 6000,
        extra_body: dict[str, Any] | None = None,
    ) -> None:
        self.chat: Any = chat.bind(extra_body=extra_body) if extra_body else chat
        self.model = model
        self.ledger = ledger
        self.max_table_chars = max_table_chars

    def summarize(self, table_markdown: str, *, caption: str) -> ModelReply:
        note = f"\nThe caption in the document reads: {caption}" if caption.strip() else ""
        prompt = TABLE_PROMPT.format(caption=note, table=table_markdown[: self.max_table_chars])
        answer = self.chat.invoke([HumanMessage(content=prompt)])
        used_in, used_out = _usage(answer)
        return ModelReply(
            _text(answer), used_in, used_out, self.ledger.estimate(self.model, used_in, used_out)
        )


# ---- the service of a process ----------------------------------------------------------------

_SHARED: dict[tuple[Any, ...], ModelTextService] = {}
_SHARED_LOCK = threading.Lock()


def reset_shared_services() -> None:
    """Forget the services of this process (tests, and a new run with a new budget)."""
    with _SHARED_LOCK:
        for service in _SHARED.values():
            service.cache.close()
        _SHARED.clear()


def shared_service(
    settings: Any = None,
    *,
    vision_model: str = "",
    text_model: str = "",
    cache_path: str | Path = "data/cache/model_text.sqlite3",
    budget_cny: float | None = None,
    max_px: int = 1024,
) -> ModelTextService:
    """The one `ModelTextService` of this process for these settings: the PDF reader (figures)
    and the `table_summary` step (tables) share its cache, budget guard and counters (one per
    cache file, budget and image size; each caller adds the model it needs), so the quality
    report adds up everything one run spent. Chat models come from `make_chat_model`
    (retries, the cost ledger and its stop line, `CHAT_BACKEND=fake`)."""
    from rag.config import ServiceSettings, repo_path
    from rag.llm.factory import make_chat_model, resolve_chat_profile

    settings = settings or ServiceSettings()
    path = repo_path(cache_path)
    key = (str(path), budget_cny, max_px)
    with _SHARED_LOCK:
        service = _SHARED.get(key)
        if service is None:
            service = ModelTextService(cache=ModelTextCache(path), guard=SpendGuard(budget_cny))
            _SHARED[key] = service
        if (vision_model and service.describer is None) or (
            text_model and service.summarizer is None
        ):
            ledger = CostLedger.from_settings(settings)

            def chat(model: str, max_tokens: int) -> BaseChatModel:
                profile = resolve_chat_profile("offline", settings).model_copy(
                    update={"model": model, "max_tokens": max_tokens}
                )
                return make_chat_model(
                    "offline", settings.model_copy(update={"chat_offline": profile}), ledger=ledger
                )

            if vision_model and service.describer is None:
                service.describer = GatewayVisionDescriber(
                    chat(vision_model, 600),
                    model=vision_model,
                    ledger=ledger,
                    max_px=max_px,
                    extra_body=NO_THINKING,
                )
                service.image_estimate_cny = estimate_image_cost(ledger, vision_model, 1)
            if text_model and service.summarizer is None:
                service.summarizer = GatewayTableSummarizer(
                    chat(text_model, 400),
                    model=text_model,
                    ledger=ledger,
                    extra_body=NO_THINKING,
                )
                service.table_estimate_cny = estimate_table_cost(ledger, text_model, [1000])
        return service
