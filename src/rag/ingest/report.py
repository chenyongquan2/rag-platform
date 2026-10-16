"""Data quality report of one ingestion (design s6-3, s6-4).

    uv run python -m rag.ingest.report --kb hf_docs              # the KB's own `limit`
    uv run python -m rag.ingest.report --kb hf_docs --limit 50
    uv run python -m rag.ingest.report --kb hf_docs --all        # every document

Reads, cleans and chunks without writing anything to ES. Prints a summary and saves the full
report as JSON (default `data/reports/<kb>/quality-<UTC time>.json`, git-ignored) so the next run
can be compared with it. Exit codes: 0 ok, 1 failed, 2 usage error.
"""

from __future__ import annotations

import argparse
import difflib
import itertools
import math
import random
import sys
import time
from collections import Counter
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from pathlib import Path

from llama_index.core.schema import BaseNode, MetadataMode
from pydantic import BaseModel, Field

from rag.config import ServiceSettings, load_kb_configs, repo_path
from rag.data import SourceDocument
from rag.ingest.build import KbProcessor
from rag.ingest.chunk import doc_id_of, metadata_tokens
from rag.ingest.clean import RuleStats
from rag.ingest.markdown import is_table_row, is_title_only
from rag.registry import connectors

WARN_RATIO = 0.05
"""A rule that removes more than this share of the input characters is marked WARN (s6-3)."""


class SkippedUnit(BaseModel):
    doc_id: str
    reason: str
    rule: str


class RuleReport(BaseModel):
    name: str
    stage: str
    units_seen: int
    units_hit: int
    chars_removed: int
    chars_added: int
    removed_ratio: float
    warn: bool


class TokenStats(BaseModel):
    min: int = 0
    p50: int = 0
    p90: int = 0
    max: int = 0
    mean: float = 0.0


class Sample(BaseModel):
    doc_id: str
    status: str
    diff: str


class QualityReport(BaseModel):
    kb: str
    created_at: str
    pipeline_versions: list[str]
    docs: int
    units: int
    units_kept: int
    read_errors: dict[str, str] = Field(default_factory=dict)
    skipped: dict[str, int] = Field(default_factory=dict)
    skipped_units: list[SkippedUnit] = Field(default_factory=list)
    garbled_units: int = 0
    duplicate_units: int = 0
    languages: dict[str, int] = Field(default_factory=dict)
    doc_types: dict[str, int] = Field(default_factory=dict)
    input_chars: int = 0
    rules: list[RuleReport] = Field(default_factory=list)
    chunks: int = 0
    max_tokens: int | None = None
    tokens: TokenStats = Field(default_factory=TokenStats)
    over_cap: int = 0
    tiny_chunks: int = 0
    title_only_chunks: int = 0
    cut_tables: int = 0
    suspicious_chunks: int = 0
    timings_s: dict[str, float] = Field(default_factory=dict)
    samples: list[Sample] = Field(default_factory=list)
    chunk_tokens: list[int] = Field(default_factory=list)

    def summary_lines(self) -> list[str]:

        def pct(n: int, d: int) -> str:
            return f"{100 * n / d:.1f}%" if d else "0%"

        lines = [
            f"kb: {self.kb}  pipeline: {', '.join(self.pipeline_versions)}",
            f"docs: {self.docs}  units: {self.units}  kept: {self.units_kept}  "
            f"read errors: {len(self.read_errors)}",
            "skipped: "
            + (", ".join(f"{k}={v}" for k, v in sorted(self.skipped.items())) or "none"),
            f"garbled units: {self.garbled_units} ({pct(self.garbled_units, self.units)})  "
            f"duplicate units: {self.duplicate_units}",
            "languages: " + ", ".join(f"{k}={v}" for k, v in self.languages.items()),
            "doc types: " + ", ".join(f"{k}={v}" for k, v in self.doc_types.items()),
            f"rules (stage, units hit, chars removed, share of {self.input_chars:,} input chars):",
        ]
        for r in self.rules:
            flag = "  WARN" if r.warn else ""
            lines.append(
                f"  {r.name:<24}{r.stage:<7}{r.units_hit:>6}{r.chars_removed:>12,}"
                f"{100 * r.removed_ratio:>7.1f}%{flag}"
            )
        t = self.tokens
        lines += [
            f"chunks: {self.chunks}  tokens min/p50/p90/max/mean: "
            f"{t.min}/{t.p50}/{t.p90}/{t.max}/{t.mean:.0f}  cap: {self.max_tokens}  "
            f"over cap: {self.over_cap}  tiny: {self.tiny_chunks}",
            f"title-only chunks: {self.title_only_chunks}  cut tables: {self.cut_tables}  "
            f"suspicious chunks: {self.suspicious_chunks}",
            "timings (s): " + ", ".join(f"{k}={v:.1f}" for k, v in self.timings_s.items()),
        ]
        return lines


def _percentile(values: Sequence[int], p: float) -> int:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(p * len(ordered)) - 1)] if ordered else 0


def _cut_tables(chunks: Sequence[BaseNode]) -> int:
    """Tables split over two or more chunks of one page (counted once per table)."""

    def lines(node: BaseNode) -> list[str]:
        return [ln for ln in node.get_content(MetadataMode.NONE).split("\n") if ln.strip()]

    count = 0
    pages: dict[str, list[BaseNode]] = {}
    for c in chunks:
        pages.setdefault(doc_id_of(c), []).append(c)
    for page in pages.values():
        inside_prev = False
        for a, b in itertools.pairwise(page):
            a_lines, b_lines = lines(a), lines(b)
            inside = (
                bool(a_lines and b_lines) and is_table_row(a_lines[-1]) and is_table_row(b_lines[0])
            )
            continues = inside_prev and all(is_table_row(ln) for ln in a_lines)
            if inside and not continues:
                count += 1
            inside_prev = inside
    return count


def _diff(before: str, after: str, max_lines: int = 80) -> str:
    lines = list(
        difflib.unified_diff(
            before.splitlines(), after.splitlines(), "before", "after", lineterm="", n=1
        )
    )
    if len(lines) > max_lines:
        lines = [*lines[:max_lines], f"... ({len(lines) - max_lines} more diff lines)"]
    return "\n".join(lines)


def _rule_reports(
    stats: Iterable[tuple[str, RuleStats]], stage: str, input_chars: int, warn_ratio: float
) -> list[RuleReport]:
    merged: dict[str, RuleStats] = {}
    for name, s in stats:
        m = merged.setdefault(name, RuleStats())
        m.units_seen += s.units_seen
        m.units_hit += s.units_hit
        m.chars_removed += s.chars_removed
        m.chars_added += s.chars_added
    out = []
    for name, s in merged.items():
        ratio = s.chars_removed / input_chars if input_chars else 0.0
        out.append(
            RuleReport(
                name=name,
                stage=stage,
                units_seen=s.units_seen,
                units_hit=s.units_hit,
                chars_removed=s.chars_removed,
                chars_added=s.chars_added,
                removed_ratio=round(ratio, 4),
                warn=stage == "clean" and ratio > warn_ratio,
            )
        )
    return out


def quality_report(
    proc: KbProcessor,
    sources: Iterable[SourceDocument],
    *,
    sample_size: int = 20,
    seed: int = 0,
    warn_ratio: float = WARN_RATIO,
) -> QualityReport:
    """Read, clean and chunk `sources` with `proc` and measure every stage (s6-4)."""
    timings: dict[str, float] = {}
    started = time.perf_counter()
    sources = list(sources)
    timings["fetch"] = time.perf_counter() - started

    started = time.perf_counter()
    read = proc.read(sources)
    timings["read"] = time.perf_counter() - started
    originals = {d.node_id: d.get_content(MetadataMode.NONE) for d in read.documents}
    input_chars = sum(len(t) for t in originals.values())

    started = time.perf_counter()
    kept = proc.clean(read.documents)
    timings["clean"] = time.perf_counter() - started

    started = time.perf_counter()
    chunks = proc.chunk(kept)
    timings["chunk"] = time.perf_counter() - started

    skipped_units = [SkippedUnit(doc_id=d, reason=r, rule="read") for d, r in read.skipped.items()]
    for name, stats in proc.rule_stats():
        skipped_units += [
            SkippedUnit(doc_id=d, reason=r, rule=name) for d, r in stats.skipped.items()
        ]

    tokenizer = proc.tokenizer
    chunk_tokens = [len(tokenizer(c.get_content(MetadataMode.NONE))) for c in chunks]
    over_cap = tiny = 0
    caps: set[int] = set()
    for c, n in zip(chunks, chunk_tokens, strict=True):
        version = c.metadata["pipeline_version"]
        cap = proc.step_param(version, "token_split", "max_tokens")
        min_tokens = proc.step_param(version, "merge_tiny", "min_tokens")
        if cap is not None:
            caps.add(cap)
            over_cap += n + metadata_tokens(c, tokenizer) > cap
        if min_tokens is not None:
            tiny += n < min_tokens

    rng = random.Random(seed)
    sample_ids = rng.sample(sorted(originals), min(sample_size, len(originals)))
    by_id = {d.node_id: d for d in read.documents}
    kept_ids = {d.node_id for d in kept}
    reasons = {s.doc_id: s.reason for s in skipped_units}
    samples = [
        Sample(
            doc_id=i,
            status="kept" if i in kept_ids else f"skipped: {reasons.get(i, '?')}",
            diff=_diff(originals[i], by_id[i].get_content(MetadataMode.NONE)),
        )
        for i in sample_ids
    ]

    t = TokenStats()
    if chunk_tokens:
        t = TokenStats(
            min=min(chunk_tokens),
            p50=_percentile(chunk_tokens, 0.5),
            p90=_percentile(chunk_tokens, 0.9),
            max=max(chunk_tokens),
            mean=round(sum(chunk_tokens) / len(chunk_tokens), 1),
        )
    return QualityReport(
        kb=proc.cfg.kb,
        created_at=datetime.now(UTC).isoformat(timespec="seconds"),
        pipeline_versions=sorted({d.metadata["pipeline_version"] for d in read.documents}),
        docs=len(sources),
        units=len(read.documents),
        units_kept=len(kept),
        read_errors=read.errors,
        skipped=dict(Counter(s.reason for s in skipped_units)),
        skipped_units=skipped_units,
        garbled_units=sum("\ufffd" in text for text in originals.values()),
        duplicate_units=sum(s.reason == "duplicate" for s in skipped_units),
        languages=dict(Counter(d.metadata.get("language") for d in kept).most_common()),
        doc_types=dict(Counter(d.metadata.get("doc_type") for d in kept).most_common()),
        input_chars=input_chars,
        rules=_rule_reports(proc.rule_stats(), "clean", input_chars, warn_ratio)
        + _rule_reports(proc.chunk_rule_stats(), "chunk", input_chars, warn_ratio),
        chunks=len(chunks),
        max_tokens=max(caps) if caps else None,
        tokens=t,
        over_cap=over_cap,
        tiny_chunks=tiny,
        title_only_chunks=sum(is_title_only(c.get_content(MetadataMode.NONE)) for c in chunks),
        cut_tables=_cut_tables(chunks),
        suspicious_chunks=sum(bool(c.metadata.get("suspicious_instruction")) for c in chunks),
        timings_s={k: round(v, 2) for k, v in timings.items()},
        samples=samples,
        chunk_tokens=chunk_tokens,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m rag.ingest.report",
        description="Read, clean and chunk a knowledge base and report data quality (no ES).",
    )
    parser.add_argument("--kb", required=True, help="knowledge base name (`kb` in its YAML)")
    size = parser.add_mutually_exclusive_group()
    size.add_argument("--limit", type=int, help="first N documents (default: the YAML's limit)")
    size.add_argument("--all", action="store_true", help="every document (no limit)")
    parser.add_argument("--out", help="JSON report path (default: data/reports/<kb>/...)")
    parser.add_argument("--samples", type=int, default=20, help="before/after diffs to keep")
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
        proc = KbProcessor(cfg, settings=settings)
        report = quality_report(proc, connector.documents(), sample_size=args.samples)
    except Exception as exc:  # report every failure as one line, not a traceback
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out = Path(args.out) if args.out else repo_path(f"data/reports/{cfg.kb}/quality-{stamp}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report.model_dump_json(indent=1), "utf-8")
    for line in report.summary_lines():
        print(line)
    print(f"report: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
