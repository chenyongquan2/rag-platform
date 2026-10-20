"""The ingest job over a CHM knowledge base, with in-memory Elasticsearch and Redis (spec P4-01).

The demo manual (`rag.testing.chm.demo_manual`) is read by the real `chm_dir` connector and the
real reader and rules; embeddings and the stores are fakes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from rag.config import KnowledgeBaseConfig, ServiceSettings
from rag.index.names import KbIndexNames
from rag.ingest import KbProcessor
from rag.ingest.clean import RemovalGuardError
from rag.ingest.job import IngestJob
from rag.registry import connectors
from rag.testing import chm as t
from rag.testing.chm import DEMO_WORDS, demo_kb_dict, demo_manual
from rag.testing.fakes import FakeEmbedding
from rag.testing.services import FakeElasticsearch, FakeRedis

KB = "mt5"
DIMS = 8


def words(text: str) -> list[int]:
    return list(range(len(text.split())))


def settings_for(tmp: Path) -> ServiceSettings:
    return ServiceSettings(  # type: ignore[call-arg]
        _env_file=None,
        docstore_url=f"file://{(tmp / 'docstore').as_posix()}",
        lock_url=f"file://{(tmp / 'locks').as_posix()}",
        ingest_report_dir=(tmp / "reports").as_posix(),
        cost_ledger_path=(tmp / "cost" / "ledger.jsonl").as_posix(),
        embed_dim=DIMS,
        embed_backend="fake",
    )


class Env:
    def __init__(self, tmp: Path, cfg_data: dict[str, Any] | None = None) -> None:
        self.tmp = tmp
        self.volumes = demo_manual(tmp / "manual")
        self.cfg = KnowledgeBaseConfig.model_validate(cfg_data or demo_kb_dict(self.volumes))
        self.settings = settings_for(tmp)
        self.es = FakeElasticsearch()
        self.redis = FakeRedis()
        self.names = KbIndexNames(KB)

    def job(self) -> IngestJob:
        connector = connectors.create(
            "chm_dir", kb=KB, settings=self.settings, **(self.cfg.source.model_extra or {})
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
        )

    def chunks(self) -> dict[str, dict[str, Any]]:
        return self.es.docs.get(self.names.index(1), {})


@pytest.fixture
def env(tmp_path: Path) -> Env:
    return Env(tmp_path)


def test_job_writes_chunks_with_language_page_key_and_links(env: Env) -> None:
    report = env.job().run()
    chunks = env.chunks()
    assert report.chunks_written == len(chunks) == 7  # one chunk per page that is kept
    assert {c["language"] for c in chunks.values()} == {"en", "zh"}
    assert {c["doc_type"] for c in chunks.values()} == {"chm"}
    assert all(c["acl_groups"] == ["public"] and c["kb"] == KB for c in chunks.values())
    a_en = [c for c in chunks.values() if c["doc_id"] == "mt5:en/a.htm"]
    a_zh = [c for c in chunks.values() if c["doc_id"] == "mt5:zh/a.htm"]
    assert a_en and a_zh
    assert {c["page_key"] for c in a_en + a_zh} == {"a"}
    assert a_zh[0]["title_path"].startswith("甲")  # the contents hierarchy, then the page heading
    assert a_en[0]["title_path"].startswith("A")
    assert all(c["content"] and c["embedding"] for c in chunks.values())
    # skipped as empty and as a copy: no chunks
    assert not [
        c for c in chunks.values() if c["doc_id"] in {"mt5:en/empty.htm", "mt5:en/b_copy.htm"}
    ]
    assert set(report.skipped_by_cleaning) == {"mt5:en/empty.htm", "mt5:en/b_copy.htm"}


def test_job_report_has_read_stats_and_clean_stats_without_page_text(env: Env) -> None:
    report = env.job().run()
    saved = json.loads(Path(report.report_path or "").read_text("utf-8"))
    (reader,) = saved["read_stats"].values()
    assert reader["pages"] == 9
    assert reader["encodings"] == {"en:ascii": 6, "zh:gb18030": 3}
    assert reader["fallbacks"] == 1  # the Chinese page whose charset label lies
    assert reader["html_rules"]["nav_bar"]["pages_hit"] == 9
    assert saved["clean_stats"]["skip_empty"]["units_hit"] == 1
    assert saved["clean_stats"]["dedupe"]["units_hit"] == 1
    assert saved["clean_stats"]["chm_title_echo"]["chars_removed"] > 0
    text = Path(report.report_path or "").read_text("utf-8")
    for word in DEMO_WORDS:
        assert word not in text


def test_second_run_changes_nothing_and_a_changed_page_is_updated(env: Env) -> None:
    first = env.job().run()
    again = env.job().run()
    assert again.added == [] and again.updated == [] and again.deleted == []
    assert len(again.unchanged) == len(first.added) and again.chunks_written == 0
    page = env.volumes["en"] / "a.htm"
    page.write_bytes(
        t.page_html(
            t.h1("A Page"), t.text("A sentence that is new on this page."), title="A Page"
        ).encode("cp1252")
    )
    third = env.job().run()
    assert third.updated == ["mt5:en/a.htm"] and third.added == []


def test_noise_outside_the_body_does_not_make_a_page_look_changed(env: Env) -> None:
    env.job().run()
    page = env.volumes["en"] / "a.htm"
    raw = page.read_bytes().decode("cp1252")
    page.write_bytes(raw.replace("Reference", "Other Section").encode("cp1252"))  # breadcrumbs
    again = env.job().run()
    assert again.updated == [] and again.chunks_written == 0


def test_a_tripped_removal_guard_stops_the_run_before_anything_is_written(tmp_path: Path) -> None:
    volumes = demo_manual(tmp_path / "manual")
    data = demo_kb_dict(volumes)
    data["pipelines"][".htm"]["clean"] = [
        {"skip_empty": {"min_chars": 400, "max_corpus_share": 0.1}}
    ]
    env = Env(tmp_path, data)
    with pytest.raises(RemovalGuardError, match="skip_empty"):
        env.job().run()
    assert env.chunks() == {}
    assert not (tmp_path / "docstore").exists() or not list((tmp_path / "docstore").glob("*.json"))
