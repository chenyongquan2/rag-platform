"""Spec P0-04: readers and the Document contract (design s5-4).

The contract checks are written against `KbProcessor.read`, so every reader registered later (CHM,
PDF) can be run through the same functions.
"""

from __future__ import annotations

import json

import pytest
from llama_index.core.schema import Document, MetadataMode

from rag.config import REPO_ROOT, KnowledgeBaseConfig, PipelineConfig, StepConfig, load_kb_config
from rag.data import SourceDocument, make_doc_id
from rag.ingest import KbProcessor, pipeline_version, reader_name_for
from rag.ingest.readers import MarkdownReader, NotebookJsonError
from rag.registry import UnknownNameError, readers

KB = "hf_docs"
CONTRACT_FIELDS = {
    "kb",
    "doc_id",
    "source_rel",
    "unit",
    "title_path",
    "source_url",
    "doc_type",
    "language",
    "acl_groups",
    "images",
    "pipeline_version",
}
FORBIDDEN_FIELDS = {
    "mtime",
    "last_modified_date",
    "creation_date",
    "file_size",
    "size",
    "source_version",
    "ingest_run_id",
    "content_hash",
}


def words(text: str) -> list[str]:
    """Fake tokenizer: one token per whitespace-separated word."""
    return text.split()


def source(rel: str, text: str, *, version: str = "repo@abc") -> SourceDocument:
    ext = "." + rel.rsplit(".", 1)[-1]
    return SourceDocument(
        doc_id=make_doc_id(KB, rel),
        kb=KB,
        source_rel=rel,
        source_url=f"https://github.com/{rel}",
        ext=ext,
        content=text.encode("utf-8"),
        source_version=version,
    )


def md_pipeline(*, max_tokens: int = 400, min_tokens: int = 3) -> PipelineConfig:
    return PipelineConfig(
        clean=["md_clean"],
        steps=[
            "markdown_split",
            {"token_split": {"max_tokens": max_tokens}},
            {"merge_tiny": {"min_tokens": min_tokens}},
            "assign_chunk_ids",
        ],
    )


def kb_config(**pipelines: PipelineConfig) -> KnowledgeBaseConfig:
    pipes = pipelines or {".md": md_pipeline(), ".mdx": md_pipeline()}
    return KnowledgeBaseConfig(kb=KB, language="en", pipelines=pipes)


def test_reader_lookup_by_extension() -> None:
    assert "markdown" in readers
    assert reader_name_for(".md", PipelineConfig()) == "markdown"
    assert reader_name_for(".MDX", PipelineConfig()) == "markdown"
    assert reader_name_for(".ipynb", PipelineConfig(reader="markdown")) == "markdown"
    with pytest.raises(UnknownNameError, match=r"\.ipynb"):
        reader_name_for(".ipynb", PipelineConfig())

    notebook = md_pipeline().model_copy(update={"reader": "markdown"})
    cfg = kb_config(**{".md": md_pipeline(), ".ipynb": notebook})
    proc = KbProcessor(cfg, tokenizer=words)
    result = proc.read(
        [
            source("a/one.md", "# One\n\nbody"),
            source("a/two.ipynb", " Notebook\n\n```\nprint(1)\n```\n"),
            source("a/three.rst", "Three\n=====\n"),
        ]
    )
    assert [d.metadata["doc_type"] for d in result.documents] == ["markdown", "notebook"]
    assert result.skipped == {make_doc_id(KB, "a/three.rst"): "no_pipeline"}


def test_markdown_reader_rejects_json_notebook() -> None:
    reader = MarkdownReader(kb=KB, pipeline_version="v1-test", language="en")
    notebook = json.dumps({"cells": [], "metadata": {}, "nbformat": 4, "nbformat_minor": 5})
    with pytest.raises(NotebookJsonError, match="flattened"):
        reader.load_data(source("demo/run.ipynb", notebook))
    # Flattened notebook text (what the HF dataset stores) is read as Markdown.
    docs = reader.load_data(
        source("demo/run.ipynb", " Gradio Demo: x\n\n```\n!pip install gradio\n```")
    )
    assert docs[0].text.startswith(" Gradio Demo")


def test_reader_contract_fields() -> None:
    cfg = kb_config()
    proc = KbProcessor(cfg, tokenizer=words)
    (doc,) = proc.read(
        [source("huggingface/blog/blob/main/intro.md", "# Intro\n\nHello.")]
    ).documents

    assert isinstance(doc, Document)
    assert set(doc.metadata) >= CONTRACT_FIELDS
    assert not FORBIDDEN_FIELDS & set(doc.metadata)
    meta = doc.metadata
    assert doc.id_ == meta["doc_id"] == "hf_docs:huggingface/blog/blob/main/intro.md"
    assert meta["kb"] == KB
    assert meta["source_rel"] == "huggingface/blog/blob/main/intro.md"
    assert meta["unit"] is None
    assert meta["title_path"] == ""
    assert meta["source_url"] == "https://github.com/huggingface/blog/blob/main/intro.md"
    assert meta["doc_type"] == "markdown"
    assert meta["language"] == "en"
    assert meta["acl_groups"] == ["public"]
    assert meta["images"] == []  # Markdown images stay in the text for now
    assert meta["pipeline_version"] == pipeline_version(cfg, ".md")
    # Whitelist (s5-4): nothing but title_path may reach the embedding text, title_path and
    # source_rel the prompt; an empty title_path is left out too.
    assert set(doc.excluded_embed_metadata_keys) == set(meta)
    assert set(doc.excluded_llm_metadata_keys) == set(meta) - {"source_rel"}
    assert doc.get_content(MetadataMode.EMBED) == "# Intro\n\nHello."


def test_reader_stable_id_and_hash() -> None:
    cfg = kb_config()
    a = source("x/page.md", "# Page\n\nSame text.", version="repo@111")
    b = source("x/other.md", "# Other\n\nFirst version.")
    b2 = source("x/other.md", "# Other\n\nSecond version.")

    first = KbProcessor(cfg, tokenizer=words).read([a, b]).documents
    # A new read (new processor, other upstream commit, other file changed) gives the same page.
    second = (
        KbProcessor(cfg, tokenizer=words)
        .read([b2, a.model_copy(update={"source_version": "repo@222"})])
        .documents
    )
    page1 = next(d for d in first if d.id_ == a.doc_id)
    page2 = next(d for d in second if d.id_ == a.doc_id)
    assert page1.id_ == page2.id_ == "hf_docs:x/page.md"
    assert page1.hash == page2.hash
    other1 = next(d for d in first if d.id_ == b.doc_id)
    other2 = next(d for d in second if d.id_ == b.doc_id)
    assert other1.hash != other2.hash


def test_pipeline_version_tracks_config() -> None:
    base = kb_config()
    more_tokens = kb_config(**{".md": md_pipeline(max_tokens=300), ".mdx": md_pipeline()})
    fewer_rules = kb_config(
        **{
            ".md": md_pipeline().model_copy(update={"clean": [StepConfig(name="md_markup")]}),
            ".mdx": md_pipeline(),
        }
    )
    versions = {
        pipeline_version(base, ".md"),
        pipeline_version(more_tokens, ".md"),
        pipeline_version(fewer_rules, ".md"),
    }
    assert len(versions) == 3
    # Same config for two extensions: same version (they share one clean chain).
    assert pipeline_version(base, ".md") == pipeline_version(base, ".mdx")
    assert pipeline_version(base, ".md") == pipeline_version(kb_config(), ".md")

    src = source("x/page.md", "# Page\n\ntext")
    (doc_a,) = KbProcessor(base, tokenizer=words).read([src]).documents
    (doc_b,) = KbProcessor(more_tokens, tokenizer=words).read([src]).documents
    assert doc_a.text == doc_b.text
    assert doc_a.hash != doc_b.hash


def test_metadata_whitelist_embed_and_llm_text() -> None:
    cfg = kb_config()
    text = "# Guide\n\nIntro text here.\n\n## Install\n\nRun pip install now."
    proc = KbProcessor(cfg, tokenizer=words)
    docs = proc.clean(proc.read([source("lib/docs/guide.md", text)]).documents)
    chunks = proc.chunk(docs)

    install = next(c for c in chunks if "pip install" in c.text)
    assert install.metadata["title_path"] == "Guide > Install"
    assert install.get_content(MetadataMode.EMBED) == (
        f"title_path: Guide > Install\n\n{install.text}"
    )
    assert install.get_content(MetadataMode.LLM) == (
        f"title_path: Guide > Install\nsource_rel: lib/docs/guide.md\n\n{install.text}"
    )
    for chunk in chunks:
        embed = chunk.get_content(MetadataMode.EMBED)
        assert "pipeline_version" not in embed
        assert "acl_groups" not in embed
        assert "chunk_index" not in embed


def test_hf_docs_config_builds() -> None:
    cfg = load_kb_config(REPO_ROOT / "kb" / "public" / "hf_docs.yaml", kb_root=REPO_ROOT / "kb")
    proc = KbProcessor(cfg, tokenizer=words)
    assert proc.version_for(".md") == proc.version_for(".mdx") == proc.version_for(".ipynb")
    names = [type(s).class_name() for s in proc.steps_for(".ipynb")]
    assert names == [
        "markdown_split",
        "token_split",
        "merge_tiny",
        "suspicious_instruction",
        "assign_chunk_ids",
    ]
    # A flattened notebook as the dataset stores it: the lost "#" of the title is put back.
    notebook = source(
        "gradio-app/gradio/blob/main/demo/blocks_random_slider/run.ipynb",
        " Gradio Demo: blocks_random_slider\n\n\n```\n!pip install -q gradio \n```\n\n\n"
        "```\nimport gradio as gr\n\ndef func(slider_1, slider_2):\n    return slider_1 * 5\n```\n",
    )
    docs = proc.clean(proc.read([notebook]).documents)
    (chunk,) = proc.chunk(docs)
    assert chunk.metadata["doc_type"] == "notebook"
    assert chunk.metadata["title_path"] == "Gradio Demo: blocks_random_slider"
    assert chunk.node_id == notebook.doc_id + "#0"
