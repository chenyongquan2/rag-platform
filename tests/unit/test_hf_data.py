"""Spec P0-03: Hugging Face data download, manifest verification, connector and eval loader.

Everything runs against a tiny parquet fixture served from a `file://` "mirror" (no network).
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from rag.config import ServiceSettings, load_kb_config
from rag.data import SourceDocument, make_doc_id
from rag.data.hf import (
    HfDataError,
    HfDatasetConnector,
    Manifest,
    ManifestMismatchError,
    NotDownloadedError,
    NotPinnedError,
    ensure_dataset,
    load_eval_questions,
    load_manifest,
    missing_sources,
    save_manifest,
    verify_file,
)
from rag.data.hf.cli import main as cli_main
from rag.registry import connectors

REPO_ROOT = Path(__file__).resolve().parents[2]
KB = "hf_docs"
CORPUS_REPO = "m-ric/huggingface_doc"
EVAL_REPO = "m-ric/huggingface_doc_qa_eval"
REVISION = "0123456789abcdef0123456789abcdef01234567"
CORPUS_FILE = "default/train/0000.parquet"
EVAL_FILE = "data/train-00000-of-00001.parquet"

CORPUS_ROWS = [
    {"text": "# Alpha\nalpha body", "source": "huggingface/transformers/blob/main/docs/alpha.md"},
    {"text": "# Beta\nbeta body", "source": "huggingface/hub-docs/blob/main/docs/hub/beta.mdx"},
    {"text": "Gamma notebook", "source": "gradio-app/gradio/blob/main/demo/gamma/run.ipynb"},
    {
        "text": "# Delta",
        "source": "huggingface/evaluate/blob/main/templates/{{ cookiecutter.slug }}/README.md",
    },
]


def _eval_row(question: str, answer: str, source_doc: str) -> dict[str, Any]:
    return {
        "context": f"context for {question}",
        "question": question,
        "answer": answer,
        "source_doc": source_doc,
        "standalone_score": 5,
        "standalone_eval": "ok",
        "relatedness_score": 5,
        "relatedness_eval": "ok",
        "relevance_score": 4,
        "relevance_eval": "ok",
    }


EVAL_ROWS = [
    _eval_row("What is alpha?", "alpha body", CORPUS_ROWS[0]["source"]),
    _eval_row("What is delta?", "Delta", CORPUS_ROWS[3]["source"]),
]


def _write_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class FakeHub:
    """A `file://` mirror laid out like `<endpoint>/datasets/<repo>/resolve/<rev>/<path>`."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.endpoint = root.as_uri()

    def path(self, repo: str, filename: str) -> Path:
        return self.root / "datasets" / repo / "resolve" / REVISION / filename

    def publish(self, repo: str, filename: str, rows: list[dict[str, Any]]) -> Path:
        path = self.path(repo, filename)
        _write_parquet(path, rows)
        return path


def _rows(path: Path) -> int:
    with pq.ParquetFile(path) as parquet:
        return parquet.metadata.num_rows


def _manifest_dict(hub: FakeHub, *, pinned: bool = True) -> dict[str, Any]:
    """A manifest describing the files the hub serves right now."""

    def file_entry(repo: str, filename: str) -> dict[str, Any]:
        path = hub.path(repo, filename)
        if not pinned:
            return {"path": filename, "size": None, "sha256": None, "rows": None}
        size, sha, rows = path.stat().st_size, _sha256(path), _rows(path)
        return {"path": filename, "size": size, "sha256": sha, "rows": rows}

    return {
        "kb": KB,
        "corpus": {
            "repo_id": CORPUS_REPO,
            "revision": REVISION,
            "ref": "refs/convert/parquet",
            "source_url": f"https://huggingface.co/datasets/{CORPUS_REPO}",
            "license": "MIT",
            "columns": {"text": "text", "path": "source"},
            "doc_url_prefix": "https://github.com/",
            "rows": _rows(hub.path(CORPUS_REPO, CORPUS_FILE)),
            "files": [file_entry(CORPUS_REPO, CORPUS_FILE)],
        },
        "eval": {
            "repo_id": EVAL_REPO,
            "revision": REVISION,
            "ref": "main",
            "source_url": f"https://huggingface.co/datasets/{EVAL_REPO}",
            "license": "Apache-2.0",
            "columns": {
                "question": "question",
                "answer": "answer",
                "source_doc": "source_doc",
                "context": "context",
            },
            "rows": _rows(hub.path(EVAL_REPO, EVAL_FILE)),
            "files": [file_entry(EVAL_REPO, EVAL_FILE)],
        },
    }


@pytest.fixture
def hub(tmp_path: Path) -> FakeHub:
    hub = FakeHub(tmp_path / "mirror")
    hub.publish(CORPUS_REPO, CORPUS_FILE, CORPUS_ROWS)
    hub.publish(EVAL_REPO, EVAL_FILE, EVAL_ROWS)
    return hub


@pytest.fixture
def manifest_path(tmp_path: Path, hub: FakeHub) -> Path:
    path = tmp_path / "manifests" / "hf_docs.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(_manifest_dict(hub), indent=2), "utf-8")
    return path


@pytest.fixture
def settings(tmp_path: Path, hub: FakeHub, monkeypatch: pytest.MonkeyPatch) -> ServiceSettings:
    for key in ("HF_ENDPOINT", "HF_DATA_DIR", "HF_TIMEOUT_S"):
        monkeypatch.delenv(key, raising=False)
    return ServiceSettings(
        _env_file=None, hf_endpoint=hub.endpoint, hf_data_dir=str(tmp_path / "cache")
    )


def _ensure(manifest: Manifest, settings: ServiceSettings, *, pin: bool = False) -> list[Any]:
    out = []
    for entry in (manifest.corpus, manifest.eval):
        assert entry is not None
        out += ensure_dataset(
            entry,
            endpoint=settings.hf_endpoint,
            data_dir=settings.hf_data_dir,
            timeout_s=settings.hf_timeout_s,
            pin=pin,
        )
    return out


def _connector(manifest_path: Path, settings: ServiceSettings, **kw: Any) -> HfDatasetConnector:
    return HfDatasetConnector(
        kb=KB, dataset=CORPUS_REPO, manifest=str(manifest_path), settings=settings, **kw
    )


# AC1
def test_make_doc_id_is_kb_plus_relative_path() -> None:
    src = "huggingface/transformers/blob/main/docs/source/en/bert.md"
    assert make_doc_id("hf_docs", src) == f"hf_docs:{src}"
    assert make_doc_id("hf_docs", "./a\\b.md") == "hf_docs:a/b.md"
    assert make_doc_id("mt5", "help/mt5.chm", unit="page.htm") == "mt5:help/mt5.chm#page.htm"
    for bad in ("", "/", "a/../b.md", "/abs/path.md"):
        with pytest.raises(ValueError):
            make_doc_id("hf_docs", bad)
    with pytest.raises(ValueError):
        make_doc_id("", "a.md")


# AC2
def test_hf_settings_defaults_and_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in ("HF_ENDPOINT", "HF_DATA_DIR", "HF_TIMEOUT_S"):
        monkeypatch.delenv(key, raising=False)
    s = ServiceSettings(_env_file=None)
    assert s.hf_endpoint == "https://hf-mirror.com"
    assert s.hf_data_dir == "data/raw/hf"
    assert s.hf_timeout_s > 0

    monkeypatch.setenv("HF_ENDPOINT", "https://example.org/hf")
    monkeypatch.setenv("HF_DATA_DIR", "D:/shared/raw/hf")
    monkeypatch.setenv("HF_TIMEOUT_S", "5")
    s = ServiceSettings(_env_file=None)
    assert (s.hf_endpoint, s.hf_data_dir, s.hf_timeout_s) == (
        "https://example.org/hf",
        "D:/shared/raw/hf",
        5,
    )


# AC3
def test_download_caches_and_verifies(
    tmp_path: Path, hub: FakeHub, manifest_path: Path, settings: ServiceSettings
) -> None:
    manifest = load_manifest(manifest_path)
    first = _ensure(manifest, settings)
    cache = Path(settings.hf_data_dir)
    expected = cache / CORPUS_REPO / REVISION / CORPUS_FILE
    assert first[0].path == expected
    assert expected.read_bytes() == hub.path(CORPUS_REPO, CORPUS_FILE).read_bytes()
    assert [f.downloaded for f in first] == [True, True]

    # A verified cached file is not fetched again: remove the mirror and ensure again.
    for f in hub.root.rglob("*.parquet"):
        f.unlink()
    second = _ensure(manifest, settings)
    assert [f.downloaded for f in second] == [False, False]

    # The mirror serves a different file: the download fails and nothing bad lands in the cache.
    other_cache = tmp_path / "cache2"
    hub.publish(CORPUS_REPO, CORPUS_FILE, CORPUS_ROWS[:2])
    with pytest.raises(ManifestMismatchError):
        ensure_dataset(
            manifest.corpus,
            endpoint=hub.endpoint,
            data_dir=str(other_cache),
            timeout_s=5,
        )
    assert not [p for p in other_cache.rglob("*") if p.is_file()]


# AC4
def test_verify_rejects_wrong_hash_size_and_rows(hub: FakeHub, manifest_path: Path) -> None:
    manifest = load_manifest(manifest_path)
    entry = manifest.corpus.files[0]
    path = hub.path(CORPUS_REPO, CORPUS_FILE)
    verify_file(path, entry)  # the real values pass

    with pytest.raises(ManifestMismatchError, match="sha256"):
        verify_file(path, entry.model_copy(update={"sha256": "0" * 64}))
    with pytest.raises(ManifestMismatchError, match="size"):
        verify_file(path, entry.model_copy(update={"size": entry.size + 1}))
    with pytest.raises(ManifestMismatchError, match="rows"):
        verify_file(path, entry.model_copy(update={"rows": 999}))
    with pytest.raises(NotDownloadedError, match="download"):
        verify_file(path.with_name("missing.parquet"), entry)
    assert issubclass(ManifestMismatchError, HfDataError)


# AC5
def test_pin_fills_unpinned_manifest_entries(
    tmp_path: Path, hub: FakeHub, settings: ServiceSettings
) -> None:
    path = tmp_path / "unpinned.json"
    path.write_text(json.dumps(_manifest_dict(hub, pinned=False)), "utf-8")
    manifest = load_manifest(path)

    with pytest.raises(NotPinnedError, match="--pin"):
        _ensure(manifest, settings)

    _ensure(manifest, settings, pin=True)
    save_manifest(manifest, path)
    reloaded = load_manifest(path)
    assert reloaded == load_manifest_from_dict(_manifest_dict(hub, pinned=True))
    assert path.read_bytes().endswith(b"\n") and b"\r\n" not in path.read_bytes()


def load_manifest_from_dict(data: dict[str, Any]) -> Manifest:
    return Manifest.model_validate(data)


# AC6
def test_doc_ids_stable_across_loads_and_row_order(
    tmp_path: Path, hub: FakeHub, manifest_path: Path, settings: ServiceSettings
) -> None:
    docs = list(_connector(manifest_path, settings).documents())
    again = list(_connector(manifest_path, settings).documents())
    assert docs == again
    assert [d.doc_id for d in docs] == [make_doc_id(KB, r["source"]) for r in CORPUS_ROWS]

    first = docs[0]
    assert isinstance(first, SourceDocument)
    assert first.kb == KB
    assert first.source_rel == CORPUS_ROWS[0]["source"]
    assert first.source_url == "https://github.com/" + CORPUS_ROWS[0]["source"]
    assert first.ext == ".md"
    assert [d.ext for d in docs] == [".md", ".mdx", ".ipynb", ".md"]
    assert first.content == CORPUS_ROWS[0]["text"].encode("utf-8")
    assert first.source_version == f"{CORPUS_REPO}@{REVISION}"

    # Shuffled rows in another revision of the file: every source keeps its id.
    shuffled = CORPUS_ROWS[:]
    random.Random(7).shuffle(shuffled)
    hub2 = FakeHub(tmp_path / "mirror2")
    hub2.publish(CORPUS_REPO, CORPUS_FILE, shuffled)
    hub2.publish(EVAL_REPO, EVAL_FILE, EVAL_ROWS)
    path2 = tmp_path / "m2.json"
    path2.write_text(json.dumps(_manifest_dict(hub2)), "utf-8")
    settings2 = settings.model_copy(
        update={"hf_endpoint": hub2.endpoint, "hf_data_dir": str(tmp_path / "cache-b")}
    )
    by_source = {d.source_rel: d.doc_id for d in _connector(path2, settings2).documents()}
    assert by_source == {d.source_rel: d.doc_id for d in docs}


# AC7
def test_limit_takes_first_n_deterministically(
    manifest_path: Path, settings: ServiceSettings
) -> None:
    all_ids = _connector(manifest_path, settings).doc_ids()
    assert len(all_ids) == len(CORPUS_ROWS) == len(set(all_ids))
    two = _connector(manifest_path, settings, limit=2).doc_ids()
    assert two == _connector(manifest_path, settings, limit=2).doc_ids() == all_ids[:2]
    assert _connector(manifest_path, settings, limit=100).doc_ids() == all_ids
    for bad in (0, -1):
        with pytest.raises(ValueError):
            _connector(manifest_path, settings, limit=bad)


def test_duplicate_sources_are_rejected(
    hub: FakeHub, tmp_path: Path, settings: ServiceSettings
) -> None:
    hub.publish(CORPUS_REPO, CORPUS_FILE, [*CORPUS_ROWS, CORPUS_ROWS[0]])
    data = _manifest_dict(hub)
    path = tmp_path / "dupes.json"
    path.write_text(json.dumps(data), "utf-8")
    with pytest.raises(HfDataError, match="duplicate"):
        _connector(path, settings).doc_ids()


# AC8
def test_hf_dataset_connector_registered_and_created_from_kb_config(
    manifest_path: Path, settings: ServiceSettings
) -> None:
    assert connectors.get("hf_dataset") is HfDatasetConnector
    cfg = load_kb_config(REPO_ROOT / "kb" / "public" / "hf_docs.yaml", kb_root=REPO_ROOT / "kb")
    params = dict(cfg.source.model_extra or {})
    assert params["manifest"] == "data/manifests/hf_docs.json"
    params["manifest"] = str(manifest_path)

    connector = connectors.create(cfg.source.connector, kb=cfg.kb, settings=settings, **params)
    assert connector.mode == "snapshot"
    assert connector.doc_ids()[0] == make_doc_id("hf_docs", CORPUS_ROWS[0]["source"])

    with pytest.raises(HfDataError, match="dataset"):
        HfDatasetConnector(
            kb=KB, dataset="someone/else", manifest=str(manifest_path), settings=settings
        )
    with pytest.raises(HfDataError, match="kb"):
        HfDatasetConnector(
            kb="other_kb", dataset=CORPUS_REPO, manifest=str(manifest_path), settings=settings
        )


# AC9
def test_eval_questions_map_to_corpus_doc_ids(
    manifest_path: Path, settings: ServiceSettings
) -> None:
    questions = load_eval_questions(manifest_path, settings=settings)
    assert [q.question for q in questions] == [r["question"] for r in EVAL_ROWS]
    assert [q.answer for q in questions] == [r["answer"] for r in EVAL_ROWS]
    assert [q.doc_id for q in questions] == [make_doc_id(KB, r["source_doc"]) for r in EVAL_ROWS]
    assert questions[0].context == EVAL_ROWS[0]["context"]
    assert len({q.qid for q in questions}) == len(questions)
    assert [q.qid for q in questions] == [
        q.qid for q in load_eval_questions(manifest_path, settings=settings)
    ]

    corpus_ids = _connector(manifest_path, settings).doc_ids()
    assert missing_sources(questions, corpus_ids) == []


def test_missing_eval_sources_reported(
    hub: FakeHub, tmp_path: Path, settings: ServiceSettings
) -> None:
    rows = [*EVAL_ROWS, _eval_row("Where is zeta?", "nowhere", "huggingface/x/blob/main/zeta.md")]
    hub.publish(EVAL_REPO, EVAL_FILE, rows)
    data = _manifest_dict(hub)
    path = tmp_path / "m.json"
    path.write_text(json.dumps(data), "utf-8")

    questions = load_eval_questions(path, settings=settings)
    corpus_ids = _connector(path, settings).doc_ids()
    missing = missing_sources(questions, corpus_ids)
    assert [q.source_doc for q in missing] == ["huggingface/x/blob/main/zeta.md"]


# AC10
def test_cli_download_and_verify_print_counts(
    hub: FakeHub,
    tmp_path: Path,
    manifest_path: Path,
    settings: ServiceSettings,
    capsys: pytest.CaptureFixture[str],
) -> None:
    common = ["--manifest", str(manifest_path), "--data-dir", settings.hf_data_dir]
    assert cli_main(["download", "--endpoint", hub.endpoint, *common]) == 0
    out = capsys.readouterr().out
    assert re.search(r"docs: 4\b", out) and re.search(r"questions: 2\b", out)

    assert cli_main(["verify", *common]) == 0
    out = capsys.readouterr().out
    assert re.search(r"docs: 4\b", out) and re.search(r"questions: 2\b", out)
    assert re.search(r"missing eval sources: 0\b", out)

    # `python -m rag.data.hf` runs the same CLI (and registers the connector only once).
    proc = subprocess.run(
        [sys.executable, "-m", "rag.data.hf", "verify", *common],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    assert "docs: 4" in proc.stdout

    # An eval source missing from the corpus makes verify fail and names the source.
    rows = [*EVAL_ROWS, _eval_row("Where is zeta?", "nowhere", "huggingface/x/blob/main/zeta.md")]
    hub.publish(EVAL_REPO, EVAL_FILE, rows)
    data = _manifest_dict(hub)
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(data), "utf-8")
    bad_common = ["--manifest", str(bad), "--data-dir", str(tmp_path / "cache-bad")]
    assert cli_main(["download", "--endpoint", hub.endpoint, *bad_common]) == 0
    capsys.readouterr()
    assert cli_main(["verify", *bad_common]) == 1
    out = capsys.readouterr().out
    assert re.search(r"missing eval sources: 1\b", out)
    assert "huggingface/x/blob/main/zeta.md" in out


# AC11
def test_committed_manifest_is_pinned() -> None:
    manifest = load_manifest(REPO_ROOT / "data" / "manifests" / "hf_docs.json")
    assert manifest.kb == "hf_docs"
    corpus, evalset = manifest.corpus, manifest.eval
    assert evalset is not None
    assert (corpus.repo_id, corpus.rows, corpus.license) == (CORPUS_REPO, 2647, "MIT")
    assert (evalset.repo_id, evalset.rows, evalset.license) == (EVAL_REPO, 65, "Apache-2.0")
    for entry in (corpus, evalset):
        assert re.fullmatch(r"[0-9a-f]{40}", entry.revision)
        assert entry.source_url == f"https://huggingface.co/datasets/{entry.repo_id}"
        assert entry.files and all(f.pinned for f in entry.files)
        assert sum(f.rows or 0 for f in entry.files) == entry.rows


def test_raw_data_dir_is_git_ignored() -> None:
    def ignored(path: str) -> bool:
        proc = subprocess.run(
            ["git", "check-ignore", "-q", "--no-index", path], cwd=REPO_ROOT, check=False
        )
        return proc.returncode == 0

    assert ignored("data/raw/hf/m-ric/huggingface_doc/rev/default/train/0000.parquet")
    assert not ignored("data/manifests/hf_docs.json")
