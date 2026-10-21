"""Estimate and quality report of a PDF and image knowledge base (spec P6-01, design s6-4, s16-4).

    uv run python -m rag.ingest.pdf_report --kb pdf_demo --estimate   # no model call
    uv run python -m rag.ingest.pdf_report --kb pdf_demo              # convert, describe, chunk
    uv run python -m rag.ingest.pdf_report --kb pdf_demo --limit 2    # first 2 files

`--estimate` converts the PDFs (cached by file hash), looks every figure and every table up in the
model-text cache and prints how many model calls a run would make and what they would cost at the
most (upper bounds from the price table), whether that fits the budget of the YAML and the cost
ledger's stop line, and what the embeddings would cost. It calls no model. The plain command
runs the real read, clean and chunk steps, so it does call the models for what the cache does not
hold (and fills the cache: the ingest job that follows costs nothing for figures and tables).

The report is the generic one of `rag.ingest.report` plus: per file the pages, figures, tables
and Docling's time (pages per second), the reader's counts (figures seen, skipped as too small,
described, failed), the table steps' counts, how many chunks there are of each kind, and what the
model calls cost. Counts only: no page text and no description. Exit codes: 0 ok, 1 failed, 2 usage.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from llama_index.core.schema import BaseNode, MetadataMode
from pydantic import BaseModel, Field

from rag.config import KnowledgeBaseConfig, ServiceSettings, load_kb_configs, repo_path
from rag.cost import BudgetExceeded, CostLedger
from rag.data.pdf.connector import IMAGE_EXTENSIONS, PdfDirConnector
from rag.ingest.build import KbProcessor
from rag.ingest.clean import RemovalGuardError
from rag.ingest.describe import (
    IMAGE_PROMPT_VERSION,
    TABLE_PROMPT_VERSION,
    ModelTextCache,
    estimate_image_cost,
    estimate_table_cost,
    sha256_hex,
)
from rag.ingest.report import WARN_RATIO, QualityReport, quality_report
from rag.ingest.tables import join_table_parts
from rag.ingest.tokens import TokenizerFn
from rag.registry import connectors

DEFAULT_CACHE = "data/cache/model_text.sqlite3"
DESCRIPTION_TOKENS = 120
"""Tokens a figure description adds to the text that is embedded (estimate)."""
SUMMARY_TOKENS = 80


class RunEstimate(BaseModel):
    """What a run would call and cost, at the most (nothing is called to make this)."""

    files: int
    pages: int
    figures_total: int
    figures_skipped_small: int
    figures_cached: int
    figures_to_describe: int
    tables_total: int
    tables_cached: int
    tables_to_summarise: int
    vision_model: str
    text_model: str
    image_cost_cny: float
    table_cost_cny: float
    embed_cost_cny: float
    total_cny: float
    budget_cny: float | None
    fits_budget: bool
    note: str = ""

    def summary_lines(self) -> list[str]:
        budget = "no budget" if self.budget_cny is None else f"budget {self.budget_cny:.2f} CNY"
        return [
            f"files {self.files}, pages {self.pages}",
            f"figures: {self.figures_total} in all, {self.figures_skipped_small} too small, "
            f"{self.figures_cached} cached, {self.figures_to_describe} to describe "
            f"({self.vision_model}): up to {self.image_cost_cny:.4f} CNY",
            f"tables: {self.tables_total} in all, {self.tables_cached} cached, "
            f"{self.tables_to_summarise} to summarise ({self.text_model}): up to "
            f"{self.table_cost_cny:.4f} CNY",
            f"embeddings: up to {self.embed_cost_cny:.4f} CNY",
            f"total up to {self.total_cny:.4f} CNY; {budget}; "
            f"{'fits' if self.fits_budget else 'DOES NOT FIT'}",
        ]


def _reader_params(cfg: KnowledgeBaseConfig, ext: str) -> dict[str, Any]:
    pipe = cfg.pipelines.get(ext)
    return dict(pipe.reader_params) if pipe is not None else {}


def _table_step(cfg: KnowledgeBaseConfig) -> dict[str, Any]:
    pipe = cfg.pipelines.get(".pdf")
    for step in pipe.steps if pipe is not None else []:
        if step.name == "table_summary":
            return dict(step.params)
    return {"mode": "off"}


def _without_models(cfg: KnowledgeBaseConfig) -> KnowledgeBaseConfig:
    """The same configuration with no model call: figures are dropped, tables get no summary."""
    pipelines = {}
    for ext, pipe in cfg.pipelines.items():
        steps = [
            s.model_copy(update={"params": {**s.params, "mode": "off"}})
            if s.name == "table_summary"
            else s
            for s in pipe.steps
        ]
        params = {**pipe.reader_params, "describe_images": "off"} if pipe.reader_params else {}
        pipelines[ext] = pipe.model_copy(update={"reader_params": params, "steps": steps})
    return cfg.model_copy(update={"pipelines": pipelines})


def estimate_run(
    cfg: KnowledgeBaseConfig,
    connector: PdfDirConnector,
    settings: ServiceSettings,
    *,
    tokenizer: TokenizerFn | None = None,
) -> RunEstimate:
    """Count the model calls of a run and add up their cost (see the module doc)."""
    ledger = CostLedger.from_settings(settings)
    pdf_params = _reader_params(cfg, ".pdf") or _reader_params(cfg, ".png")
    table_params = _table_step(cfg)
    vision_model = str(pdf_params.get("vision_model", ""))
    describing = pdf_params.get("describe_images", "off") == "model" and bool(vision_model)
    text_model = str(table_params.get("model", ""))
    summarising = table_params.get("mode") == "model" and bool(text_model)
    cache_path = str(
        pdf_params.get("cache_path") or table_params.get("cache_path") or DEFAULT_CACHE
    )
    cache = ModelTextCache(repo_path(cache_path))
    budget = pdf_params.get("budget_cny")

    sources = list(connector.documents())
    files = {s.source_rel for s in sources}
    pages = sum(1 for s in sources if s.ext == ".pdf")
    seen: dict[str, bool] = {}  # image hash -> cached
    total = small = 0
    for source in sources:
        if source.ext == ".pdf":
            sizes = [
                (f["sha256"], min(int(f["width"]), int(f["height"])))
                for f in source.extra.get("figures", [])
            ]
        elif source.ext in IMAGE_EXTENSIONS:
            from io import BytesIO

            from PIL import Image

            sizes = [(sha256_hex(source.content), min(Image.open(BytesIO(source.content)).size))]
        else:
            sizes = []
        min_px = int(pdf_params.get("min_image_px", 64))
        for sha, shortest in sizes:
            total += 1
            if shortest < min_px:
                small += 1
            elif sha not in seen:
                hit = cache.get("image", vision_model, IMAGE_PROMPT_VERSION, sha) is not None
                seen[sha] = hit
    cached_figures = sum(seen.values())
    to_describe = 0 if not describing else len(seen) - cached_figures

    proc = KbProcessor(_without_models(cfg), settings=settings, tokenizer=tokenizer)
    chunks = proc.chunk(proc.clean(proc.read(sources).documents))
    tables: dict[str, list[BaseNode]] = {}
    for node in chunks:
        if node.metadata.get("chunk_kind") == "table":
            tables.setdefault(str(node.metadata["table_id"]), []).append(node)
    uncached_tokens: list[int] = []
    cached_tables = 0
    for parts in tables.values():
        caption, table = join_table_parts([p.get_content(MetadataMode.NONE) for p in parts])
        key = sha256_hex(caption + "\n" + table)
        if cache.get("table", text_model, TABLE_PROMPT_VERSION, key) is not None:
            cached_tables += 1
        else:
            uncached_tokens.append(len(proc.tokenizer(table)))
    to_summarise = uncached_tokens if summarising else []
    cache.close()

    image_cost = estimate_image_cost(ledger, vision_model, to_describe) if describing else 0.0
    table_cost = estimate_table_cost(ledger, text_model, to_summarise) if summarising else 0.0
    embed_tokens = sum(len(proc.tokenizer(c.get_content(MetadataMode.NONE))) + 16 for c in chunks)
    embed_tokens += to_describe * DESCRIPTION_TOKENS + len(tables) * SUMMARY_TOKENS
    embed_cost = ledger.estimate(settings.embed_model, embed_tokens)
    models = image_cost + table_cost
    fits = budget is None or models <= float(budget)
    try:
        ledger.check_budget(models + embed_cost)
    except BudgetExceeded:
        fits = False
    return RunEstimate(
        files=len(files),
        pages=pages,
        figures_total=total,
        figures_skipped_small=small,
        figures_cached=cached_figures,
        figures_to_describe=to_describe,
        tables_total=len(tables),
        tables_cached=cached_tables,
        tables_to_summarise=len(to_summarise),
        vision_model=vision_model,
        text_model=text_model,
        image_cost_cny=image_cost,
        table_cost_cny=table_cost,
        embed_cost_cny=embed_cost,
        total_cny=models + embed_cost,
        budget_cny=float(budget) if budget is not None else None,
        fits_budget=fits,
    )


class PdfQualityReport(BaseModel):
    kb: str
    created_at: str
    conversion: dict[str, Any]
    reader: dict[str, Any]
    tables: dict[str, Any]
    chunk_kinds: dict[str, int]
    chunks_with_generated_text: int
    quality: QualityReport
    files: dict[str, dict[str, Any]] = Field(default_factory=dict)

    def summary_lines(self) -> list[str]:
        c, r, t = self.conversion, self.reader, self.tables
        model = r.get("model_text", {})
        lines = [
            f"kb: {self.kb}",
            f"conversion: {c['files']} files, {c['pages']} pages, {c['figures']} figures, "
            f"{c['tables']} tables; Docling {c['docling_version']} took {c['engine_seconds']} s "
            f"({c['pages_per_second']} pages/s on this CPU; {c['cached_files']} files from cache)",
            f"reader: pages {r.get('pages', 0)}, images seen {r.get('images_seen', 0)}, "
            f"skipped as too small {r.get('images_skipped_small', 0)}, "
            f"failed {r.get('images_failed', 0)}",
            f"images described {model.get('images_described', 0)}, cache hits "
            f"{model.get('image_cache_hits', 0)}; tables summarised "
            f"{model.get('tables_summarised', 0)}, cache hits {model.get('table_cache_hits', 0)}; "
            f"spend {model.get('spend_cny', 0.0):.6f} CNY, tokens "
            f"{model.get('prompt_tokens', 0)} in / {model.get('completion_tokens', 0)} out",
            f"tables: {t['tables']} tables in {t['table_chunks']} chunks ({t['split_tables']} "
            f"cut into parts), {t['summary_chunks']} summary chunks, model errors "
            f"{t.get('model_errors', 0)}",
            "chunk kinds: " + ", ".join(f"{k}={v}" for k, v in sorted(self.chunk_kinds.items())),
            f"chunks with model-written text: {self.chunks_with_generated_text}",
        ]
        lines += self.quality.summary_lines()
        return lines


def _sum_counts(stats: list[dict[str, Any]]) -> dict[str, Any]:
    """Add up the counters of the readers; the model service is one object, counted once."""
    merged: dict[str, Any] = {}
    for one in stats:
        for key, value in one.items():
            if key == "model_text":
                if value and not merged.get("model_text"):
                    merged["model_text"] = value
            elif isinstance(value, int | float):
                merged[key] = merged.get(key, 0) + value
    merged.setdefault("model_text", {})
    return merged


def pdf_quality_report(
    proc: KbProcessor, connector: PdfDirConnector, *, warn_ratio: float = WARN_RATIO
) -> PdfQualityReport:
    """Read, clean and chunk everything `connector` delivers with `proc` and measure it."""
    sources = list(connector.documents())
    chunks: list[BaseNode] = []
    chunk_step = proc.chunk

    def capture(documents: Any) -> list[BaseNode]:
        out = chunk_step(documents)
        chunks.extend(out)
        return out

    proc.chunk = capture  # type: ignore[method-assign]
    generic = quality_report(proc, sources, sample_size=0, warn_ratio=warn_ratio)
    proc.chunk = chunk_step  # type: ignore[method-assign]
    reader = _sum_counts(list(proc.reader_stats().values()))

    done = connector.conversions
    engine_seconds = sum(float(v["recorded_engine_seconds"]) for v in done.values())
    pages = sum(int(v["pages"]) for v in done.values())
    conversion = {
        "files": len(done),
        "pages": pages,
        "figures": sum(int(v["figures"]) for v in done.values()),
        "tables": sum(int(v["tables"]) for v in done.values()),
        "engine_seconds": round(engine_seconds, 1),
        "pages_per_second": round(pages / engine_seconds, 3) if engine_seconds else None,
        "cached_files": sum(1 for v in done.values() if v["cached"]),
        "docling_version": next((v["docling_version"] for v in done.values()), ""),
    }
    kinds = Counter(str(c.metadata.get("chunk_kind", "text")) for c in chunks)
    parts: dict[str, int] = Counter(
        str(c.metadata["table_id"]) for c in chunks if c.metadata.get("chunk_kind") == "table"
    )
    summary = proc.step_stats().get("table_summary", {})
    tables = {
        "tables": len(parts),
        "table_chunks": kinds.get("table", 0),
        "split_tables": sum(1 for n in parts.values() if n > 1),
        "summary_chunks": kinds.get("table_summary", 0),
        "mode": summary.get("mode", "off"),
        "model_errors": summary.get("model_errors", 0),
    }
    return PdfQualityReport(
        kb=proc.cfg.kb,
        created_at=datetime.now(UTC).isoformat(timespec="seconds"),
        conversion=conversion,
        reader=reader,
        tables=tables,
        chunk_kinds=dict(kinds),
        chunks_with_generated_text=sum(1 for c in chunks if c.metadata.get("has_generated_text")),
        quality=generic,
        files={k: dict(v) for k, v in done.items()},
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m rag.ingest.pdf_report",
        description="Estimate the cost of, or report the data quality of, a PDF and image "
        "knowledge base (counts only; no ES). Without --estimate the models are called for "
        "what the cache does not hold.",
    )
    parser.add_argument("--kb", required=True, help="knowledge base name (`kb` in its YAML)")
    parser.add_argument(
        "--estimate", action="store_true", help="count the model calls and their cost; call none"
    )
    size = parser.add_mutually_exclusive_group()
    size.add_argument("--limit", type=int, help="first N files (default: the YAML's)")
    size.add_argument("--all", action="store_true", help="every file (no limit)")
    parser.add_argument("--out", help="JSON path (default: data/reports/<kb>/pdf-...-<time>.json)")
    parser.add_argument("--warn-ratio", type=float, default=WARN_RATIO)
    return parser


def main(argv: Sequence[str] | None = None, *, settings: ServiceSettings | None = None) -> int:
    args = _parser().parse_args(argv)
    settings = settings if settings is not None else ServiceSettings()
    try:
        configs = load_kb_configs(settings.kb_dir)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    cfg = configs.get(args.kb)
    if cfg is None:
        known = ", ".join(sorted(configs)) or "(none)"
        print(f"error: unknown knowledge base {args.kb!r}; known: {known}", file=sys.stderr)
        return 2
    if args.limit is not None and args.limit < 1:
        print("error: --limit must be a positive integer", file=sys.stderr)
        return 2
    extra = dict(cfg.source.model_extra or {})
    if args.all:
        extra["limit"] = None
    elif args.limit is not None:
        extra["limit"] = args.limit
    try:
        connector = connectors.create(cfg.source.connector, kb=cfg.kb, settings=settings, **extra)
        if not isinstance(connector, PdfDirConnector):
            print(f"error: {cfg.kb} does not use the pdf_dir connector", file=sys.stderr)
            return 2
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        if args.estimate:
            estimate = estimate_run(cfg, connector, settings)
            lines, payload, name = (
                estimate.summary_lines(),
                estimate.model_dump_json(indent=1),
                "estimate",
            )
        else:
            proc = KbProcessor(cfg, settings=settings)
            report = pdf_quality_report(proc, connector, warn_ratio=args.warn_ratio)
            lines, payload, name = (
                report.summary_lines(),
                report.model_dump_json(indent=1),
                "quality",
            )
    except RemovalGuardError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # one line, not a traceback
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    out = (
        Path(args.out) if args.out else repo_path(f"data/reports/{cfg.kb}/pdf-{name}-{stamp}.json")
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(payload, "utf-8")
    for line in lines:
        print(line)
    print(f"report: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
