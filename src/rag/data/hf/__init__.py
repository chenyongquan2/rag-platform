"""Hugging Face datasets: download from a mirror, verify against the manifest, read (spec P0-03).

CLI: `uv run python -m rag.data.hf download [--pin]` and `uv run python -m rag.data.hf verify`.
"""

from rag.data.hf.connector import HfDatasetConnector, corpus_documents
from rag.data.hf.download import CachedFile, ensure_dataset, read_dataset
from rag.data.hf.evalset import EvalQuestion, eval_questions, load_eval_questions, missing_sources
from rag.data.hf.manifest import (
    DatasetEntry,
    FileEntry,
    HfDataError,
    Manifest,
    ManifestMismatchError,
    NotDownloadedError,
    NotPinnedError,
    load_manifest,
    save_manifest,
    verify_file,
)

__all__ = [
    "CachedFile",
    "DatasetEntry",
    "EvalQuestion",
    "FileEntry",
    "HfDataError",
    "HfDatasetConnector",
    "Manifest",
    "ManifestMismatchError",
    "NotDownloadedError",
    "NotPinnedError",
    "corpus_documents",
    "ensure_dataset",
    "eval_questions",
    "load_eval_questions",
    "load_manifest",
    "missing_sources",
    "read_dataset",
    "save_manifest",
    "verify_file",
]
