"""The public HF eval set (65 questions) with answers mapped to corpus doc ids (design s14-2)."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import pyarrow as pa
from pydantic import BaseModel, ConfigDict

from rag.config import ServiceSettings
from rag.data.base import make_doc_id
from rag.data.hf.download import ensure_dataset, read_dataset
from rag.data.hf.manifest import DatasetEntry, HfDataError, Manifest, load_manifest


class EvalQuestion(BaseModel):
    """One question, labelled at document (page) level: `doc_id` uses the corpus id scheme."""

    model_config = ConfigDict(frozen=True)

    qid: str
    question: str
    answer: str
    source_doc: str
    doc_id: str
    context: str
    """The passage the question was generated from (useful when checking a label)."""


def eval_questions(table: pa.Table, manifest: Manifest) -> list[EvalQuestion]:
    """Rows in file order; `qid` is `<eval repo name>:<row index>` within the pinned revision."""
    entry = _eval_entry(manifest)
    cols = {field: table.column(column).to_pylist() for field, column in entry.columns.items()}
    prefix = entry.repo_id.split("/", 1)[1]
    return [
        EvalQuestion(
            qid=f"{prefix}:{i:03d}",
            question=cols["question"][i],
            answer=cols["answer"][i],
            source_doc=cols["source_doc"][i],
            doc_id=make_doc_id(manifest.kb, cols["source_doc"][i]),
            context=cols["context"][i] or "",
        )
        for i in range(table.num_rows)
    ]


def load_eval_questions(
    manifest: Manifest | str | Path,
    *,
    settings: ServiceSettings | None = None,
    download: bool = True,
) -> list[EvalQuestion]:
    """Load the eval set of a manifest, downloading it first unless `download=False`."""
    if not isinstance(manifest, Manifest):
        manifest = load_manifest(manifest)
    settings = settings if settings is not None else ServiceSettings()
    entry = _eval_entry(manifest)
    if download:
        ensure_dataset(
            entry,
            endpoint=settings.hf_endpoint,
            data_dir=settings.hf_data_dir,
            timeout_s=settings.hf_timeout_s,
        )
    return eval_questions(read_dataset(entry, settings.hf_data_dir), manifest)


def missing_sources(
    questions: Iterable[EvalQuestion], doc_ids: Iterable[str]
) -> list[EvalQuestion]:
    """Questions whose source document is not in the corpus (a wrong label or a missing page)."""
    known = set(doc_ids)
    return [q for q in questions if q.doc_id not in known]


def _eval_entry(manifest: Manifest) -> DatasetEntry:
    if manifest.eval is None:
        raise HfDataError(f"manifest of kb {manifest.kb!r} has no eval dataset")
    return manifest.eval
