"""Logical backup and restore of a knowledge base (spec P8-01, design s16-6).

The design's backup is an Elasticsearch snapshot repository (`deploy/elasticsearch/README.md`
has the change and the steps); it needs `path.repo`, i.e. an Elasticsearch restart, which this
machine's shared Elasticsearch cannot have now. Until then a backup is these files:

    uv run python -m rag.ops.backup export  --index kb_hf_docs_v1 --out <dir>
    uv run python -m rag.ops.backup restore --from <dir>/kb_hf_docs_v1 --index <new index>
    uv run python -m rag.ops.backup verify  --source kb_hf_docs_v1 --target <new index>
    uv run python -m rag.ops.backup copy-files --out <dir>/files [--path data/docstore ...]
    uv run python -m rag.ops.backup rebind-docstore --docstore <file> --index <new> --out <file>

`export` reads the index through a point in time (a consistent view while ingestion may write)
sorted by `_shard_doc`, 1000 documents a page, with the vectors (`source_exclude_vectors=false`:
the indexes leave vectors out of `_source` by default), and writes `meta.json` (mapping,
the index settings that matter, aliases, count, uuid) and gzipped NDJSON files of `_id` and
`_source`. `restore` creates a new index (it refuses one that exists) with that mapping and
those settings, loads the documents with refresh off, then refreshes and checks the count.
`verify` compares two indexes: counts, every id, a sample of sources (vectors included) and the
top 10 of a few BM25 and kNN searches.

Restore order (what the P5-02 incident taught: a docstore bound to another index makes the next
ingest treat every page as new): 1. restore the index under a new version name; 2. copy the
docstore of the old version to the new name and bind it to the new index's uuid
(`rebind-docstore`); 3. copy the L0 embedding cache back (the next ingest then costs nothing);
4. switch the alias (`python -m rag.index switch`, spec P2-01); 5. bump the data generation, so no
cache entry of before is served. Never bind a docstore to an index that was not restored from
the same backup: the docstore then claims chunks the index does not have.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import random
import shutil
import sqlite3
import sys
import time
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rag.config import ServiceSettings, repo_path

DOCS_PER_FILE = 10_000
PAGE = 1000
KEEP_ALIVE = "5m"
# Index settings a restore keeps; the rest (uuid, creation date, version, routing) belongs to
# the old index.
_KEPT_SETTINGS = ("number_of_shards", "number_of_replicas", "analysis", "similarity", "mapping")
VERIFY_QUERIES = (
    "how to push a model to the hub",
    "tokenizer padding side",
    "load a dataset in streaming mode",
    "LoRA fine-tuning with PEFT",
    "inference endpoint logs",
)


@dataclass
class ExportResult:
    index: str
    uuid: str
    docs: int
    files: list[str]
    bytes: int
    seconds: float


@dataclass
class RestoreResult:
    index: str
    uuid: str
    docs: int
    seconds: float


@dataclass
class VerifyResult:
    source: str
    target: str
    source_count: int
    target_count: int
    ids_equal: bool
    missing: int
    extra: int
    sampled: int
    sources_equal: int
    searches: list[dict[str, Any]] = field(default_factory=list)
    seconds: float = 0.0

    @property
    def ok(self) -> bool:
        return (
            self.source_count == self.target_count
            and self.ids_equal
            and self.sources_equal == self.sampled
            and all(s["same_top10"] for s in self.searches if s["kind"] == "bm25")
        )


def _kept_settings(index_settings: Mapping[str, Any]) -> dict[str, Any]:
    return {k: index_settings[k] for k in _KEPT_SETTINGS if k in index_settings}


def iter_docs(
    es: Any, index: str, *, page: int = PAGE, source: bool = True
) -> Iterator[dict[str, Any]]:
    """Every document of `index` (one point in time), `{_id, _source}` with vectors."""
    pit = es.open_point_in_time(index=index, keep_alive=KEEP_ALIVE)["id"]
    after: list[Any] | None = None
    try:
        while True:
            kwargs: dict[str, Any] = {
                "pit": {"id": pit, "keep_alive": KEEP_ALIVE},
                "sort": [{"_shard_doc": "asc"}],
                "size": page,
                "track_total_hits": False,
            }
            if source:
                kwargs["source_exclude_vectors"] = False
            else:
                kwargs["source"] = False
            if after is not None:
                kwargs["search_after"] = after
            response = es.search(**kwargs)
            pit = response.get("pit_id", pit)
            hits = response["hits"]["hits"]
            if not hits:
                return
            for hit in hits:
                yield {"_id": hit["_id"], "_source": hit.get("_source", {})}
            after = hits[-1]["sort"]
    finally:
        es.close_point_in_time(id=pit)


def export_index(es: Any, index: str, out_dir: Path) -> ExportResult:
    began = time.perf_counter()
    target = out_dir / index
    if target.exists() and any(target.iterdir()):
        raise FileExistsError(f"{target} is not empty")
    target.mkdir(parents=True, exist_ok=True)
    settings = es.indices.get_settings(index=index)[index]["settings"]["index"]
    mappings = es.indices.get_mapping(index=index)[index]["mappings"]
    aliases = sorted(es.indices.get_alias(index=index)[index].get("aliases", {}))
    files: list[str] = []
    count = 0
    fh: Any = None
    try:
        for doc in iter_docs(es, index):
            if count % DOCS_PER_FILE == 0:
                if fh is not None:
                    fh.close()
                name = f"docs-{count // DOCS_PER_FILE:05d}.ndjson.gz"
                files.append(name)
                # One file per DOCS_PER_FILE documents, closed in `finally` below.
                fh = gzip.open(target / name, "wt", encoding="utf-8", newline="")  # noqa: SIM115
            fh.write(json.dumps(doc, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
    finally:
        if fh is not None:
            fh.close()
    meta = {
        "index": index,
        "uuid": settings["uuid"],
        "count": count,
        "exported_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "es_version": es.info()["version"]["number"],
        "aliases": aliases,
        "settings": _kept_settings(settings),
        "mappings": mappings,
        "files": files,
    }
    (target / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8", newline=""
    )
    size = sum(p.stat().st_size for p in target.iterdir())
    return ExportResult(index, settings["uuid"], count, files, size, time.perf_counter() - began)


def _read_docs(src: Path, files: Sequence[str]) -> Iterator[dict[str, Any]]:
    for name in files:
        with gzip.open(src / name, "rt", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    yield json.loads(line)


def restore_index(es: Any, src: Path, index: str, *, chunk: int = 500) -> RestoreResult:
    """Create `index` from an export and load it; refuses an index or alias that exists."""
    from elasticsearch import helpers

    began = time.perf_counter()
    meta = json.loads((src / "meta.json").read_text(encoding="utf-8"))
    if es.indices.exists(index=index) or es.indices.exists_alias(name=index):
        raise FileExistsError(f"index or alias {index} exists; restore into a new name")
    settings = {**meta["settings"], "refresh_interval": "-1", "number_of_replicas": "0"}
    es.indices.create(index=index, settings={"index": settings}, mappings=meta["mappings"])
    actions = (
        {"_op_type": "create", "_index": index, "_id": d["_id"], "_source": d["_source"]}
        for d in _read_docs(src, meta["files"])
    )
    loaded = 0
    for ok, _ in helpers.streaming_bulk(
        es, actions, chunk_size=chunk, raise_on_error=True, max_retries=2
    ):
        loaded += int(ok)
    replicas = meta["settings"].get("number_of_replicas", "0")
    es.indices.put_settings(
        index=index, settings={"index": {"refresh_interval": None, "number_of_replicas": replicas}}
    )
    es.indices.refresh(index=index)
    count = int(es.count(index=index)["count"])
    if count != meta["count"]:
        raise RuntimeError(
            f"restored {count} documents into {index}, the backup has {meta['count']}"
        )
    uuid = es.indices.get_settings(index=index)[index]["settings"]["index"]["uuid"]
    return RestoreResult(index, uuid, loaded, time.perf_counter() - began)


def _ids(es: Any, index: str) -> set[str]:
    return {d["_id"] for d in iter_docs(es, index, source=False)}


def _top(es: Any, index: str, **query: Any) -> list[tuple[str, float]]:
    hits = es.search(index=index, size=10, source=False, **query)["hits"]["hits"]
    return [(h["_id"], round(float(h["_score"]), 4)) for h in hits]


def verify_indexes(
    es: Any,
    source: str,
    target: str,
    *,
    sample: int = 200,
    queries: Sequence[str] = VERIFY_QUERIES,
    seed: int = 801,
) -> VerifyResult:
    began = time.perf_counter()
    counts = [int(es.count(index=i)["count"]) for i in (source, target)]
    a, b = _ids(es, source), _ids(es, target)
    rng = random.Random(seed)
    picked = rng.sample(sorted(a & b), min(sample, len(a & b)))
    equal = 0
    vectors: list[tuple[str, list[float]]] = []
    for doc_id in picked:
        docs = [
            es.get(index=i, id=doc_id, source_exclude_vectors=False)["_source"]
            for i in (source, target)
        ]
        equal += docs[0] == docs[1]
        if isinstance(docs[0].get("embedding"), list) and len(vectors) < 3:
            vectors.append((doc_id, docs[0]["embedding"]))
    searches: list[dict[str, Any]] = []
    for q in queries:
        tops = [_top(es, i, query={"match": {"content": q}}) for i in (source, target)]
        searches.append(
            {"kind": "bm25", "query": q, "same_top10": tops[0] == tops[1], "top": tops[0][:3]}
        )
    for doc_id, vector in vectors:
        knn = {"field": "embedding", "query_vector": vector, "k": 10, "num_candidates": 100}
        tops = [[h for h, _ in _top(es, i, knn=knn)] for i in (source, target)]
        searches.append(
            {
                "kind": "knn",
                "query": doc_id,
                "same_top10": tops[0] == tops[1],
                "overlap": len(set(tops[0]) & set(tops[1])),
                "self_first": tops[1][:1] == [doc_id],
            }
        )
    return VerifyResult(
        source=source,
        target=target,
        source_count=counts[0],
        target_count=counts[1],
        ids_equal=a == b,
        missing=len(a - b),
        extra=len(b - a),
        sampled=len(picked),
        sources_equal=equal,
        searches=searches,
        seconds=time.perf_counter() - began,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def copy_files(paths: Sequence[Path], out_dir: Path) -> list[dict[str, Any]]:
    """Copy docstores and caches: SQLite files through SQLite's backup API (a consistent copy
    while a process may be writing), other files as they are; returns a manifest."""
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, Any]] = []
    files: list[Path] = []
    for path in paths:
        files += sorted(p for p in path.rglob("*") if p.is_file()) if path.is_dir() else [path]
    for src in files:
        if src.suffix in (".tmp", ".lock"):
            continue
        dest = out_dir / src.name
        began = time.perf_counter()
        if src.suffix in (".sqlite3", ".db"):
            source = sqlite3.connect(f"file:{src.as_posix()}?mode=ro", uri=True)
            target = sqlite3.connect(dest)
            try:
                source.backup(target)
            finally:
                target.close()
                source.close()
        else:
            shutil.copy2(src, dest)
        manifest.append(
            {
                "source": str(src),
                "copy": str(dest),
                "bytes": dest.stat().st_size,
                "sha256": _sha256(dest),
                "seconds": round(time.perf_counter() - began, 3),
            }
        )
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8", newline=""
    )
    return manifest


def rebind_docstore(es: Any, docstore: Path, index: str, out: Path) -> dict[str, Any]:
    """Write a copy of `docstore` bound to `index`'s uuid (`rag/meta` `index_uuid`), so the next
    ingest into the restored index trusts it. `out` must not be `docstore` itself."""
    from rag.factories import JsonFileDocumentStore
    from rag.ingest.job import INDEX_UUID_KEY

    if out.resolve() == docstore.resolve():
        raise ValueError("write the rebound docstore to a new file (the restored index's name)")
    store = JsonFileDocumentStore(docstore)
    before = store.get_meta(INDEX_UUID_KEY)
    uuid = es.indices.get_settings(index=index)[index]["settings"]["index"]["uuid"]
    store.set_meta(INDEX_UUID_KEY, uuid)
    out.parent.mkdir(parents=True, exist_ok=True)
    store.persist(str(out))
    return {"docstore": str(out), "was": before, "now": uuid, "documents": len(store.docs)}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m rag.ops.backup", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("export")
    p.add_argument("--index", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("restore")
    p.add_argument("--from", dest="src", required=True)
    p.add_argument("--index", required=True)
    p = sub.add_parser("verify")
    p.add_argument("--source", required=True)
    p.add_argument("--target", required=True)
    p.add_argument("--sample", type=int, default=200)
    p = sub.add_parser("copy-files")
    p.add_argument("--out", required=True)
    p.add_argument(
        "--path",
        action="append",
        default=None,
        help="file or folder (default: data/docstore, data/cache)",
    )
    p = sub.add_parser("rebind-docstore")
    p.add_argument("--docstore", required=True)
    p.add_argument("--index", required=True)
    p.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    settings = ServiceSettings()
    result: Any
    if args.command == "copy-files":
        paths = [repo_path(x) for x in (args.path or ["data/docstore", "data/cache"])]
        result = copy_files(paths, repo_path(args.out))
    else:
        from rag.factories import make_es_client

        es = make_es_client(settings.es_url, request_timeout=120.0)
        try:
            if args.command == "export":
                result = asdict(export_index(es, args.index, repo_path(args.out)))
            elif args.command == "restore":
                result = asdict(restore_index(es, repo_path(args.src), args.index))
            elif args.command == "verify":
                verified = verify_indexes(es, args.source, args.target, sample=args.sample)
                result = {**asdict(verified), "ok": verified.ok}
            else:
                result = rebind_docstore(
                    es, repo_path(args.docstore), args.index, repo_path(args.out)
                )
        finally:
            es.close()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if isinstance(result, dict) and result.get("ok") is False:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
