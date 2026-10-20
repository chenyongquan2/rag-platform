"""`kb/private/mt5.yaml`: the private MT5 manual KB config (spec P4-01 AC13; s5-1, s16-4).

The YAML holds configuration only (paths, rule names, limits): no text of the manual.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
import yaml

from rag.config import KnowledgeBaseConfig, ServiceSettings, load_kb_config, load_kb_configs
from rag.config.paths import REPO_ROOT, repo_path
from rag.index.mapping import KEYWORD_FIELDS, index_body_for_kb
from rag.ingest import KbProcessor
from rag.ingest.clean import CleanRule
from rag.security import Caller, kb_visible, visible_kbs

MT5_YAML = repo_path("kb/private/mt5.yaml")
KB_ROOT = repo_path("kb")


def words(text: str) -> list[int]:
    return list(range(len(text.split())))


@pytest.fixture(scope="module")
def cfg() -> KnowledgeBaseConfig:
    return load_kb_config(MT5_YAML, kb_root=KB_ROOT)


def test_mt5_config_loads_and_names_its_sources(cfg: KnowledgeBaseConfig) -> None:
    assert cfg.kb == "mt5"
    assert cfg.source.connector == "chm_dir"
    volumes = {v["language"]: v["path"] for v in (cfg.source.model_extra or {})["volumes"]}
    assert volumes == {"zh": "data/raw/mt5/zh", "en": "data/raw/mt5/en"}
    assert cfg.language == "auto"
    assert set(cfg.pipelines) == {".htm", ".html"}
    assert cfg.description  # callers choose a knowledge base by it


def test_mt5_config_builds_a_processor_with_the_chm_reader_and_rules(
    cfg: KnowledgeBaseConfig,
) -> None:
    proc = KbProcessor(cfg, settings=ServiceSettings(), tokenizer=words)
    assert proc.version_for(".htm") == proc.version_for(".html")  # one pipeline, two extensions
    pipe = cfg.pipelines[".htm"]
    assert pipe.reader == "chm"
    names = [s.name for s in pipe.clean]
    assert names[:2] == ["whitespace", "chm_title_echo"]
    assert "md_markup" not in names  # decoded `&lt;b&gt;` must never be cut as a tag (design s6-1)
    assert "detect_language" not in names  # the language is the volume's, not a guess
    dedupe = next(s for s in pipe.clean if s.name == "dedupe")
    # equal text under two different title paths (an Admin page and its Manager twin) stays twice
    assert dedupe.params["group_by"] == ["language", "title_path"]
    steps = [s.name for s in pipe.steps]
    assert steps[:2] == ["markdown_split", "token_split"]
    assert steps[-1] == "assign_chunk_ids"
    token_split = next(s for s in pipe.steps if s.name == "token_split")
    assert token_split.params["max_tokens"] == 400  # reranker tokens, as in hf_docs


def test_mt5_removal_guard_is_configured_on_the_rules_that_cut_text(
    cfg: KnowledgeBaseConfig,
) -> None:
    proc = KbProcessor(cfg, settings=ServiceSettings(), tokenizer=words)
    guarded = {
        name: rule
        for name, rule in ((r.name, r) for _, r in _rules(proc))
        if rule.max_page_share is not None or rule.max_corpus_share is not None
    }
    assert {"whitespace", "chm_title_echo", "skip_empty", "dedupe"} <= set(guarded)
    for rule in guarded.values():
        assert rule.max_corpus_share is not None and rule.max_corpus_share <= 0.1


def _rules(proc: KbProcessor) -> list[tuple[str, CleanRule]]:
    return [(rule.name, rule) for group in proc._groups.values() for rule in group.rules]


def test_mt5_is_private_and_names_who_may_use_it(cfg: KnowledgeBaseConfig) -> None:
    assert cfg.access.callers == ["local"]  # the local development caller (settings.service_caller)
    assert ServiceSettings().service_caller == "local"
    local = Caller("local", ("public",))
    stranger = Caller("someone-else", ("public",))
    assert kb_visible(local, cfg) and not kb_visible(stranger, cfg)
    configs = load_kb_configs(KB_ROOT)
    assert {c.kb for c in visible_kbs(local, configs)} >= {"hf_docs", "mt5"}
    assert "mt5" not in {c.kb for c in visible_kbs(stranger, configs)}
    assert "hf_docs" in {c.kb for c in visible_kbs(stranger, configs)}


def test_mt5_config_without_access_is_refused(tmp_path: Path) -> None:
    data = yaml.safe_load(MT5_YAML.read_text("utf-8"))
    data.pop("access")
    private = tmp_path / "kb" / "private"
    private.mkdir(parents=True)
    (private / "mt5.yaml").write_text(yaml.safe_dump(data, allow_unicode=True), "utf-8")
    with pytest.raises(ValueError, match=r"access\.callers"):
        load_kb_config(private / "mt5.yaml", kb_root=tmp_path / "kb")


def test_mt5_retrieval_groups_the_two_language_versions_of_a_page(cfg: KnowledgeBaseConfig) -> None:
    assert cfg.retrieval.dedupe_by_page_key is True
    hf = load_kb_config(repo_path("kb/public/hf_docs.yaml"), kb_root=KB_ROOT)
    assert hf.retrieval.dedupe_by_page_key is False  # the HF knowledge base behaves as before
    assert cfg.eval.dir == "evals/datasets/mt5"


def test_mt5_analyzer_is_a_config_key_that_reaches_the_index_mapping(
    cfg: KnowledgeBaseConfig,
) -> None:
    assert cfg.es.analyzer == "standard"  # the same as the HF knowledge base until P4-02 compares
    props = index_body_for_kb(cfg, dims=8)["mappings"]["properties"]
    assert props["content"]["analyzer"] == props["title_path"]["analyzer"] == "standard"
    data = yaml.safe_load(MT5_YAML.read_text("utf-8"))
    data["es"] = {"analyzer": "cjk"}
    data["access"] = {"callers": ["local"]}
    other = KnowledgeBaseConfig.model_validate(data)
    props = index_body_for_kb(other, dims=8)["mappings"]["properties"]
    assert props["content"]["analyzer"] == props["title_path"]["analyzer"] == "cjk"


def test_index_mapping_has_page_key_and_links_to_as_keywords() -> None:
    assert "page_key" in KEYWORD_FIELDS and "links_to" in KEYWORD_FIELDS
    props = index_body_for_kb(KnowledgeBaseConfig(kb="x"), dims=8)["mappings"]["properties"]
    assert props["page_key"] == {"type": "keyword"}
    assert props["links_to"] == {"type": "keyword"}


def test_the_config_is_tracked_and_the_manual_is_not() -> None:
    """The YAML is committed (no manual text in it); the unpacked pages, the reports and the
    docstore stay out of git (`.gitignore`)."""
    git = ["git", "-C", str(REPO_ROOT)]

    def ignored(path: str) -> bool:
        return subprocess.run([*git, "check-ignore", "-q", path], check=False).returncode == 0

    if not (REPO_ROOT / ".git").exists():
        pytest.skip("not a git checkout")
    assert not ignored("kb/private/mt5.yaml")
    assert ignored("kb/private/other.yaml")  # private configs stay untracked unless named
    for private in (
        "data/raw/mt5/zh/page.htm",
        "data/raw/mt5/_unpack.json",
        "data/reports/mt5/chm-quality-x.json",
        "data/docstore/kb_mt5_v1.json",
        "data/cache/embeddings.sqlite3",
    ):
        assert ignored(private), private
