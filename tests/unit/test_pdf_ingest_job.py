"""Spec P6-01 review round 2: a PDF that cannot be converted on a later run keeps its pages.

A PDF is indexed as page documents `<kb>:<file>#p<N>`; when a later run cannot convert the file,
the connector hands over one document for the whole file (`<kb>:<file>`) that fails to read. The
job's rule for a document that fails to read is to keep its stored copy (design s7-4); for a file
that means every stored page of it, or the pages would count as deleted and lose their chunks.
In-memory Elasticsearch, Redis and embeddings; the conversion engine is a fake.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from rag.config import KnowledgeBaseConfig, ServiceSettings
from rag.data.base import make_doc_id, split_doc_id, units_of
from rag.data.pdf.connector import PdfDirConnector
from rag.index.names import KbIndexNames
from rag.ingest import KbProcessor
from rag.ingest.job import IngestJob
from rag.testing.fakes import FakeEmbedding
from rag.testing.pdf import FakeEngine, sample_pages
from rag.testing.services import FakeElasticsearch, FakeRedis

KB = "p"
DIMS = 8


def words(text: str) -> list[str]:
    return text.split()


# ---- the id convention ------------------------------------------------------------------------


def test_split_doc_id_separates_the_unit() -> None:
    assert split_doc_id("p:x.pdf#p2") == ("p:x.pdf", "p2")
    assert split_doc_id("p:x.pdf") == ("p:x.pdf", None)
    assert split_doc_id(make_doc_id("p", "dir/c#.md")) == ("p:dir/c%23.md", None)
    assert split_doc_id(make_doc_id("p", "dir/c#.md", "p1")) == ("p:dir/c%23.md", "p1")
    assert split_doc_id("mt5:zh/a.htm") == ("mt5:zh/a.htm", None)


def test_units_of_a_file_are_its_pages_and_a_unit_only_itself() -> None:
    stored = ["p:x.pdf#p1", "p:x.pdf#p2", "p:x.pdf.bak#p1", "p:xx.pdf#p1", "p:y.pdf#p1", "p:z.md"]
    assert units_of("p:x.pdf", stored) == ["p:x.pdf#p1", "p:x.pdf#p2"]
    assert units_of("p:x.pdf#p2", stored) == ["p:x.pdf#p2"]  # a failed page keeps only itself
    assert units_of("p:z.md", stored) == ["p:z.md"]  # a file without units: itself
    assert units_of("p:gone.pdf", stored) == []


def test_the_connector_names_a_failed_file_by_the_file_part_of_its_page_ids(
    tmp_path: Path,
) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "x.pdf").write_bytes(b"%PDF x")
    good = PdfDirConnector(
        kb=KB, path=raw, cache_dir=tmp_path / "c1", engine=FakeEngine(sample_pages())
    )
    pages = [d.doc_id for d in good.documents()]
    bad = PdfDirConnector(
        kb=KB,
        path=raw,
        cache_dir=tmp_path / "c2",
        engine=FakeEngine(sample_pages(), fail_on=["x.pdf"]),
    )
    (failed,) = [d.doc_id for d in bad.documents()]
    assert {split_doc_id(p)[0] for p in pages} == {failed}
    assert units_of(failed, pages) == pages


# ---- the job ----------------------------------------------------------------------------------


def config() -> KnowledgeBaseConfig:
    return KnowledgeBaseConfig.model_validate(
        {
            "kb": KB,
            "source": {"connector": "pdf_dir"},
            "language": "en",
            "pipelines": {
                ".pdf": {
                    "reader": "pdf",
                    "reader_params": {"describe_images": "off"},
                    "clean": ["whitespace"],
                    "steps": [
                        "markdown_split",
                        "table_chunks",
                        {"token_split": {"max_tokens": 60}},
                        {"merge_tiny": {"min_tokens": 3}},
                        "mark_generated_text",
                        "assign_chunk_ids",
                    ],
                }
            },
            "access": {"callers": ["*"]},
        }
    )


class Env:
    def __init__(self, tmp: Path) -> None:
        self.tmp = tmp
        self.raw = tmp / "raw"
        self.raw.mkdir()
        (self.raw / "x.pdf").write_bytes(b"%PDF x first")
        (self.raw / "y.pdf").write_bytes(b"%PDF y first")
        self.settings = ServiceSettings(
            _env_file=None,  # type: ignore[call-arg]
            docstore_url=f"file://{(tmp / 'docstore').as_posix()}",
            lock_url=f"file://{(tmp / 'locks').as_posix()}",
            ingest_report_dir=(tmp / "reports").as_posix(),
            cost_ledger_path=(tmp / "ledger.jsonl").as_posix(),
            embed_dim=DIMS,
            embed_backend="fake",
            chat_backend="fake",
            ingest_max_read_failure_ratio=0.5,  # one file of two fails here
        )
        self.cfg = config()
        self.es = FakeElasticsearch()
        self.redis = FakeRedis()
        self.names = KbIndexNames(KB)

    def run(self, engine: FakeEngine) -> Any:
        connector = PdfDirConnector(
            kb=KB, path=self.raw, cache_dir=self.tmp / "pdfcache", engine=engine
        )
        return IngestJob(
            self.cfg,
            self.settings,
            es=self.es,
            redis=self.redis,
            embed_model=FakeEmbedding(dim=DIMS),
            connector=connector,
            processor=KbProcessor(self.cfg, settings=self.settings, tokenizer=words),
            names=self.names,
        ).run()

    def chunk_ids(self, file: str) -> list[str]:
        prefix = make_doc_id(KB, file) + "#"
        return sorted(i for i in self.es.docs.get(self.names.index(1), {}) if i.startswith(prefix))


@pytest.fixture
def env(tmp_path: Path) -> Env:
    return Env(tmp_path)


def test_a_pdf_that_fails_on_a_later_run_keeps_its_indexed_pages(env: Env) -> None:
    first = env.run(FakeEngine(sample_pages()))
    x_pages = [make_doc_id(KB, "x.pdf", f"p{n}") for n in (1, 2, 3)]
    assert set(x_pages) <= set(first.added)
    before = env.chunk_ids("x.pdf")
    assert before

    # x.pdf changes and can no longer be converted; y.pdf is as before
    (env.raw / "x.pdf").write_bytes(b"%PDF x broken now")
    second = env.run(FakeEngine(sample_pages(), fail_on=["x.pdf"]))

    assert list(second.read_failures) == [make_doc_id(KB, "x.pdf")]
    assert sorted(second.kept_old_version) == x_pages
    assert second.deleted == []
    assert set(x_pages) <= set(second.unchanged)
    assert env.chunk_ids("x.pdf") == before  # the pages of x.pdf are still searchable
    assert second.chunks_deleted == 0


def test_a_pdf_that_is_removed_still_loses_its_pages(env: Env) -> None:
    env.run(FakeEngine(sample_pages()))
    (env.raw / "x.pdf").unlink()
    second = env.run(FakeEngine(sample_pages()))
    assert sorted(second.deleted) == [make_doc_id(KB, "x.pdf", f"p{n}") for n in (1, 2, 3)]
    assert env.chunk_ids("x.pdf") == []
    assert env.chunk_ids("y.pdf")
