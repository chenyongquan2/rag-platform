"""A tiny PDF through Docling and the real pipeline into Elasticsearch (P6-01 AC14).

The PDF is made in the test (fpdf2): a title, a heading, a table and an embedded chart. Docling
reads it for real (models from the local Hugging Face cache; one to two minutes on a loaded CPU,
so the conversion is done once per module). The model calls are fakes (`chat_backend=fake`) and
the embeddings are fakes of 8 dimensions; nothing costs money. Every index carries this run's
unique prefix and is deleted afterwards (see `conftest`); real indices are never touched.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
from elasticsearch import Elasticsearch
from redis import Redis

from rag.config import KnowledgeBaseConfig, ServiceSettings
from rag.index.names import KbIndexNames
from rag.ingest import KbProcessor
from rag.ingest.describe import IMAGE_MARKER_PREFIX, reset_shared_services, shared_service
from rag.ingest.job import IngestJob
from rag.registry import connectors
from rag.retrieve.search import make_hybrid_search
from rag.testing.fakes import FakeEmbedding
from rag.testing.pdf import make_test_pdf

pytestmark = pytest.mark.integration

DIMS = 8
FAKE_DESCRIPTION = "A line chart of pressure over time that rises and then levels off."


def words(text: str) -> list[int]:
    return list(range(len(text.split())))


@pytest.fixture(scope="module")
def pdf_folder(tmp_path_factory: pytest.TempPathFactory) -> Path:
    folder = tmp_path_factory.mktemp("pdf_raw")
    make_test_pdf(folder / "pumps.pdf")
    return folder


@pytest.fixture(scope="module")
def conversion_cache(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("pdf_cache")


@pytest.fixture(autouse=True)
def _fresh_model_services() -> Iterator[None]:
    reset_shared_services()
    yield
    reset_shared_services()


class Demo:
    def __init__(
        self,
        es: Elasticsearch,
        redis: Redis,
        names: KbIndexNames,
        settings: ServiceSettings,
        folder: Path,
        cache: Path,
        text_cache: Path,
        summary_mode: str,
    ) -> None:
        self.es, self.redis, self.names, self.settings = es, redis, names, settings
        self.text_cache = text_cache
        models = {"cache_path": str(text_cache), "budget_cny": 0.5}
        self.cfg = KnowledgeBaseConfig.model_validate(
            {
                "kb": names.kb,
                "source": {
                    "connector": "pdf_dir",
                    "path": str(folder),
                    "cache_dir": str(cache),
                    "docling": {"images_scale": 2.0},
                },
                "language": "en",
                "pipelines": {
                    ".pdf": {
                        "reader": "pdf",
                        "reader_params": {
                            "describe_images": "model",
                            "vision_model": "qwen/qwen3.6-flash",
                            **models,
                        },
                        "clean": ["whitespace", {"skip_empty": {"min_chars": 20}}],
                        "steps": [
                            "markdown_split",
                            "table_chunks",
                            {"token_split": {"max_tokens": 400}},
                            {"merge_tiny": {"min_tokens": 50}},
                            "mark_generated_text",
                            {
                                "table_summary": {
                                    "mode": summary_mode,
                                    "model": "qwen/qwen3.6-flash",
                                    **models,
                                }
                            },
                            "assign_chunk_ids",
                        ],
                    }
                },
                "metadata": {"embed": ["title_path"], "llm": ["title_path", "source_rel", "page"]},
                "retrieval": {"replace_table_summaries": True},
                "access": {"callers": ["*"]},
            }
        )

    def ingest(self) -> Any:
        connector = connectors.create(
            "pdf_dir", kb=self.cfg.kb, settings=self.settings, **(self.cfg.source.model_extra or {})
        )
        job = IngestJob(
            self.cfg,
            self.settings,
            es=self.es,
            redis=self.redis,
            embed_model=FakeEmbedding(dim=DIMS),
            connector=connector,
            processor=KbProcessor(self.cfg, settings=self.settings, tokenizer=words),
            names=self.names,
        )
        return job.run()

    @property
    def index(self) -> str:
        return self.names.index(1)

    def sources(self) -> dict[str, dict[str, Any]]:
        self.es.indices.refresh(index=self.index)
        hits = self.es.search(index=self.index, query={"match_all": {}}, size=100)["hits"]["hits"]
        return {h["_id"]: h["_source"] for h in hits}

    def search(self, question: str) -> list[Any]:
        search = make_hybrid_search(
            self.cfg,
            self.settings,
            acl_groups=["public"],
            index_prefix=self.names.prefix,
            es=self.es,
            embed_model=FakeEmbedding(dim=DIMS),
            routes=("bm25",),
        )
        try:
            return search.retrieve(question, kb=self.cfg.kb, top_k=10).chunks
        finally:
            search.close()


@pytest.fixture
def make_demo(
    es: Elasticsearch,
    redis_client: Redis,
    make_names: Callable[[], KbIndexNames],
    service_settings: ServiceSettings,
    tmp_path: Path,
    pdf_folder: Path,
    conversion_cache: Path,
) -> Callable[..., Demo]:
    settings = service_settings.model_copy(
        update={
            "docstore_url": f"file://{(tmp_path / 'docstore').as_posix()}",
            "lock_url": f"file://{(tmp_path / 'locks').as_posix()}",
            "ingest_report_dir": (tmp_path / "reports").as_posix(),
            "cost_ledger_path": (tmp_path / "ledger.jsonl").as_posix(),
            "embed_dim": DIMS,
            "embed_backend": "fake",
            "chat_backend": "fake",
            "chat_fake_reply": FAKE_DESCRIPTION,
        }
    )

    def factory(summary_mode: str = "rule") -> Demo:
        return Demo(
            es,
            redis_client,
            make_names(),
            settings,
            pdf_folder,
            conversion_cache,
            tmp_path / "text.sqlite3",
            summary_mode,
        )

    return factory


def test_the_pdf_becomes_page_documents_with_tables_figures_and_summaries(
    make_demo: Callable[..., Demo],
) -> None:
    demo = make_demo()
    report = demo.ingest()
    chunks = demo.sources()
    assert report.chunks_written == len(chunks)
    kb = demo.cfg.kb
    assert {c["doc_id"] for c in chunks.values()} == {f"{kb}:pumps.pdf#p1", f"{kb}:pumps.pdf#p2"}
    assert {c["page"] for c in chunks.values()} == {1, 2}
    assert {c["doc_type"] for c in chunks.values()} == {"pdf"}
    assert all(c["source_rel"] == "pumps.pdf" for c in chunks.values())
    kinds = {c["chunk_kind"] for c in chunks.values()}
    assert kinds == {"text", "table", "table_summary"}
    (table,) = [c for c in chunks.values() if c["chunk_kind"] == "table"]
    (summary,) = [c for c in chunks.values() if c["chunk_kind"] == "table_summary"]
    assert table["table_id"] == summary["table_id"] == f"{kb}:pumps.pdf#p1#t0"
    assert table["page"] == summary["page"] == 1
    assert "Zephyr" in table["content"] and "| Pump" in table["content"]
    assert summary["has_generated_text"] is True and "Columns: Pump" in summary["content"]
    assert table["title_path"].endswith("1 Introduction")
    # the figure on page 2 is described at its place, after its caption
    (figure,) = [c for c in chunks.values() if IMAGE_MARKER_PREFIX in c["content"]]
    assert figure["page"] == 2 and figure["has_generated_text"] is True
    text = figure["content"]
    assert text.index("Figure 1: Pressure rises") < text.index(FAKE_DESCRIPTION)
    assert figure["images"] and figure["images"][0].endswith("p2_f1.png")
    assert figure["title_path"].startswith("Tiny Report on Pumps")
    # chunks of the page are numbered in page order, the summary after the page's own chunks
    page_one = sorted(
        (c for c in chunks.values() if c["page"] == 1), key=lambda c: c["chunk_index"]
    )
    assert page_one[-1]["chunk_kind"] == "table_summary"


def test_a_summary_hit_returns_the_table_chunk(make_demo: Callable[..., Demo]) -> None:
    demo = make_demo()
    demo.ingest()
    # "rows" occurs only in the rule-written summary ("3 rows: Zephyr; ..."), not in the table
    found = demo.search("rows")
    kinds = [c.metadata["chunk_kind"] for c in found]
    assert "table_summary" not in kinds and "table" in kinds
    table = next(c for c in found if c.metadata["chunk_kind"] == "table")
    assert "| Zephyr" in table.text and table.metadata["page"] == 1
    assert table.metadata["matched_table_summary"].endswith("pumps.pdf#p1#2")
    assert table.metadata["has_generated_text"] is False


def test_a_second_run_changes_nothing_and_calls_no_model(make_demo: Callable[..., Demo]) -> None:
    demo = make_demo(summary_mode="model")
    first = demo.ingest()
    assert first.chunks_written > 0
    service = shared_service(
        demo.settings,
        vision_model="qwen/qwen3.6-flash",
        text_model="qwen/qwen3.6-flash",
        cache_path=demo.text_cache,
        budget_cny=0.5,
    )
    assert service.stats.images_described == 1 and service.stats.tables_summarised == 1
    reset_shared_services()  # a new process: only the cache file and the docstore are left
    second = demo.ingest()
    assert second.chunks_written == 0 and len(second.unchanged) == 2 and not second.added
    service = shared_service(
        demo.settings,
        vision_model="qwen/qwen3.6-flash",
        text_model="qwen/qwen3.6-flash",
        cache_path=demo.text_cache,
        budget_cny=0.5,
    )
    assert service.stats.images_described == 0 and service.stats.image_cache_hits == 1
    assert service.stats.tables_summarised == 0
    summary = [c for c in demo.sources().values() if c["chunk_kind"] == "table_summary"]
    assert summary and FAKE_DESCRIPTION in summary[0]["content"]
