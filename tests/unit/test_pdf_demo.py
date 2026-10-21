"""`kb/private/pdf_demo.yaml` and the shared layers it uses (spec P6-01 AC12, AC13)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from rag.config import KnowledgeBaseConfig, ServiceSettings, load_kb_config, load_kb_configs
from rag.config.paths import REPO_ROOT, repo_path
from rag.ingest import KbProcessor
from rag.ingest.build import INGEST_CODE_VERSION, pipeline_version
from rag.ingest.pdf import ImageReader, PdfReader
from rag.ingest.tables import TableChunks, TableSummary
from rag.retrieve.dedupe import TextDedupe
from rag.retrieve.pipeline import SearchPipeline, make_search_pipeline
from rag.retrieve.rerank import BatchedTimeoutRerank, RerankRunner
from rag.retrieve.table_groups import TableSummaryReplace
from rag.retrieve.threshold import RerankThreshold
from rag.security import Caller, kb_visible, visible_kbs
from rag.testing import FakeEmbedding
from rag.testing.rerank import FakeCrossEncoder
from rag.testing.services import FakeElasticsearch

PDF_YAML = repo_path("kb/private/pdf_demo.yaml")
KB_ROOT = repo_path("kb")


def words(text: str) -> list[int]:
    return list(range(len(text.split())))


@pytest.fixture(scope="module")
def cfg() -> KnowledgeBaseConfig:
    return load_kb_config(PDF_YAML, kb_root=KB_ROOT)


def settings() -> ServiceSettings:
    return ServiceSettings(_env_file=None)  # type: ignore[call-arg]


def test_pdf_demo_config_loads_and_names_its_source(cfg: KnowledgeBaseConfig) -> None:
    assert cfg.kb == "pdf_demo" and cfg.description
    assert cfg.source.connector == "pdf_dir"
    extra = cfg.source.model_extra or {}
    assert extra["path"] == "data/raw/pdf_demo" and extra["docling"]["ocr"] is False
    assert set(cfg.pipelines) == {".pdf", ".png", ".jpg", ".jpeg"}
    assert cfg.pipelines[".pdf"].reader == "pdf" and cfg.pipelines[".png"].reader == "image"
    assert cfg.metadata.llm == ["title_path", "source_rel", "page"]  # citations can say "page N"
    assert cfg.retrieval.replace_table_summaries is True
    assert cfg.eval.dir == "evals/datasets/pdf_demo"


def test_pdf_demo_builds_a_processor_without_calling_a_model(cfg: KnowledgeBaseConfig) -> None:
    proc = KbProcessor(cfg, settings=settings(), tokenizer=words)
    assert proc.version_for(".png") == proc.version_for(".jpg") == proc.version_for(".jpeg")
    assert proc.version_for(".pdf") != proc.version_for(".png")
    pdf_reader = proc._groups[proc.version_for(".pdf") or ""].reader
    image_reader = proc._groups[proc.version_for(".png") or ""].reader
    assert isinstance(pdf_reader, PdfReader) and isinstance(image_reader, ImageReader)
    assert pdf_reader.describe_images == "model" and pdf_reader.vision_model == "qwen/qwen3.6-flash"
    assert pdf_reader.budget_cny == 0.5
    steps = proc.steps_for(".pdf")
    kinds = [type(s) for s in steps]
    assert TableChunks in kinds and TableSummary in kinds
    names = [s.class_name() for s in steps]
    assert names.index("table_chunks") < names.index("token_split") < names.index("merge_tiny")
    assert names.index("mark_generated_text") < names.index("table_summary")
    assert names[-1] == "assign_chunk_ids"
    assert "table_chunks" not in [s.class_name() for s in proc.steps_for(".png")]


def test_pdf_demo_is_private_and_names_who_may_use_it(cfg: KnowledgeBaseConfig) -> None:
    assert cfg.access.callers == ["local"]
    local, stranger = Caller("local", ("public",)), Caller("someone-else", ("public",))
    assert kb_visible(local, cfg) and not kb_visible(stranger, cfg)
    configs = load_kb_configs(KB_ROOT)
    assert {"hf_docs", "mt5", "pdf_demo"} <= {c.kb for c in visible_kbs(local, configs)}
    assert "pdf_demo" not in {c.kb for c in visible_kbs(stranger, configs)}


def test_pdf_demo_config_without_access_is_refused(tmp_path: Path) -> None:
    data = yaml.safe_load(PDF_YAML.read_text("utf-8"))
    data.pop("access")
    private = tmp_path / "kb" / "private"
    private.mkdir(parents=True)
    (private / "pdf_demo.yaml").write_text(yaml.safe_dump(data, allow_unicode=True), "utf-8")
    with pytest.raises(ValueError, match=r"access\.callers"):
        load_kb_config(private / "pdf_demo.yaml", kb_root=tmp_path / "kb")


def test_pdf_demo_yaml_is_tracked_and_the_papers_are_not() -> None:
    import subprocess

    def tracked(path: str) -> bool:
        done = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "ls-files", "--error-unmatch", path],
            capture_output=True,
            check=False,
        )
        return done.returncode == 0

    assert tracked("kb/private/pdf_demo.yaml")
    ignored = subprocess.run(
        [
            "git",
            "-C",
            str(REPO_ROOT),
            "check-ignore",
            "data/raw/pdf_demo/x.pdf",
            "data/cache/pdf/y",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert ignored.stdout.split() == ["data/raw/pdf_demo/x.pdf", "data/cache/pdf/y"]


# ---- reader_params and the pipeline version ------------------------------------------------------


def test_pipeline_version_of_a_pipeline_without_reader_params_is_what_it_was() -> None:
    hf = load_kb_config(repo_path("kb/public/hf_docs.yaml"), kb_root=KB_ROOT)
    model = "BAAI/bge-reranker-base"
    pipe = hf.pipelines[".md"]
    payload = {
        "code": INGEST_CODE_VERSION,
        "reader": "markdown",
        "clean": [s.model_dump() for s in pipe.clean],
        "steps": [s.model_dump() for s in pipe.steps],
        "metadata": hf.metadata.model_dump(),
        "language": hf.language,
        "tokenizer": model,
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    assert (
        pipeline_version(hf, ".md", tokenizer_model=model)
        == f"v{INGEST_CODE_VERSION}-{digest[:10]}"
    )


def test_pipeline_version_changes_with_reader_params_that_change_the_text_only(
    cfg: KnowledgeBaseConfig,
) -> None:
    def version(**params: object) -> str:
        pipe = cfg.pipelines[".pdf"].model_copy(
            update={"reader_params": {**cfg.pipelines[".pdf"].reader_params, **params}}
        )
        changed = cfg.model_copy(update={"pipelines": {**cfg.pipelines, ".pdf": pipe}})
        return pipeline_version(changed, ".pdf", tokenizer_model="m")

    base = version()
    assert version(budget_cny=9.0) == base  # how a run is paid for
    assert version(cache_path="elsewhere.sqlite3") == base
    assert version(describe_images="off") != base  # the page text changes
    assert version(vision_model="other-model") != base
    assert version(min_image_px=128) != base


# ---- the other layers do not change (AC13) -----------------------------------------------------


def test_pdf_demo_uses_the_same_search_chain(cfg: KnowledgeBaseConfig) -> None:
    es = FakeElasticsearch()
    es.mappings["kb_pdf_demo_v1"] = {"_meta": {"embed_model": "fake-embedding@4"}, "properties": {}}
    es.aliases["kb_pdf_demo"] = "kb_pdf_demo_v1"
    pipeline = make_search_pipeline(
        cfg,
        ServiceSettings(_env_file=None, embed_backend="fake", embed_dim=4),  # type: ignore[call-arg]
        acl_groups=["local"],
        es=es,
        embed_model=FakeEmbedding(dim=4),
        count_tokens=lambda s: len(s.split()),
        runner=RerankRunner(FakeCrossEncoder(), batch_size=8, max_concurrency=1),
    )
    assert type(pipeline) is SearchPipeline
    assert [type(p) for p in pipeline.postprocessors] == [
        TextDedupe,
        BatchedTimeoutRerank,
        RerankThreshold,
        TableSummaryReplace,  # the table takes its summary's place after the rerank
    ]
    assert pipeline.retriever.config == cfg.retrieval


def test_no_shared_layer_knows_the_pdf_demo_knowledge_base_by_name() -> None:
    """Retrieval, rerank, answer, agent and the service run the same code for every knowledge
    base; what differs is configuration (spec P6-01: the other layers do not change)."""
    for layer in ("retrieve", "answer", "agent", "serve", "index", "llm", "embed"):
        for path in (REPO_ROOT / "src" / "rag" / layer).rglob("*.py"):
            text = path.read_text("utf-8")
            assert "pdf_demo" not in text, f"{path} names the pdf_demo knowledge base"


def test_the_processor_hands_its_settings_to_readers_and_steps_that_call_a_model(
    cfg: KnowledgeBaseConfig,
) -> None:
    """Without this a reader would read the real `.env` and call the real gateway in a run
    that was given fake-model settings (found while testing the report)."""
    mine = ServiceSettings(_env_file=None, chat_backend="fake")  # type: ignore[call-arg]
    proc = KbProcessor(cfg, settings=mine, tokenizer=words)
    reader = proc._groups[proc.version_for(".pdf") or ""].reader
    assert reader._settings is mine
    (summary,) = [s for s in proc.steps_for(".pdf") if isinstance(s, TableSummary)]
    assert summary._settings is mine
    md = load_kb_config(repo_path("kb/public/hf_docs.yaml"), kb_root=KB_ROOT)
    KbProcessor(md, settings=mine, tokenizer=words)  # readers without `settings` still build
