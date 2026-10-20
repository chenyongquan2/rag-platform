"""Data-quality report of a CHM knowledge base (spec P4-01, design s6-3, s6-4).

    uv run python -m rag.ingest.chm_report --kb mt5              # the YAML's `limit`
    uv run python -m rag.ingest.chm_report --kb mt5 --limit 50   # pages per volume
    uv run python -m rag.ingest.chm_report --kb mt5 --all

Reads, cleans and chunks without writing to Elasticsearch and without model calls, then prints a
summary and saves the report as JSON (default `data/reports/<kb>/chm-quality-<UTC time>.json`,
git-ignored). It is the generic report of `rag.ingest.report` (rules with characters removed,
chunk and token numbers, skipped pages and reasons) plus what is specific to a CHM: per
language the pages in and out, the encodings, the pages that are not in the contents, the
noise removed while reading, and how many pages exist in both languages.

The manual is a private vendor document, so the report holds counts and page ids only: no page
text, and the generic report's before/after samples are switched off. A removal guard that trips
(`RemovalGuardError`, design s6-3) stops the report with the rule, the page and the shares.
Exit codes: 0 ok, 1 failed (also a tripped guard), 2 usage error.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from rag.config import ServiceSettings, load_kb_configs, repo_path
from rag.data.chm.connector import ChmDirConnector
from rag.ingest.build import KbProcessor
from rag.ingest.clean import RemovalGuardError
from rag.ingest.report import WARN_RATIO, QualityReport, quality_report
from rag.registry import connectors


class VolumeReport(BaseModel):
    language: str
    pages_in: int
    pages_out: int
    """Pages left after cleaning (not skipped as empty, too short, a contents page or a copy)."""
    skipped: dict[str, int] = Field(default_factory=dict)
    encodings: dict[str, int] = Field(default_factory=dict)
    not_in_toc: int = 0
    toc_without_page: int = 0
    duplicate_page_keys: int = 0


class ReadRuleReport(BaseModel):
    name: str
    pages_hit: int
    elements: int
    chars_removed: int


class ChmQualityReport(BaseModel):
    kb: str
    created_at: str
    volumes: dict[str, VolumeReport]
    pairing: dict[str, int]
    """`both`: page keys that exist in two or more languages; `only_<language>`: in one only."""
    read_rules: list[ReadRuleReport]
    reader: dict[str, Any]
    quality: QualityReport

    def summary_lines(self) -> list[str]:
        lines = [f"kb: {self.kb}  pipeline: {', '.join(self.quality.pipeline_versions)}"]
        for lang, v in self.volumes.items():
            skipped = ", ".join(f"{k}={n}" for k, n in sorted(v.skipped.items())) or "none"
            codecs = ", ".join(f"{k}={n}" for k, n in sorted(v.encodings.items())) or "none"
            lines.append(
                f"{lang}: pages in {v.pages_in}, out {v.pages_out}, skipped {skipped}; "
                f"encodings {codecs}; not in contents {v.not_in_toc}, contents entries without "
                f"page {v.toc_without_page}, duplicate page keys {v.duplicate_page_keys}"
            )
        pairs = ", ".join(
            f"paired {n}" if k == "both" else f"only {k.removeprefix('only_')} {n}"
            for k, n in self.pairing.items()
        )
        lines.append(f"pairing: {pairs}")
        r = self.reader
        lines.append(
            f"reader: fallbacks {r.get('fallbacks', 0)}, decode errors {r.get('decode_errors', 0)} "
            f"(replacement characters {r.get('replacement_chars', 0)}), pages without title "
            f"{r.get('no_title', 0)}, without help container {r.get('no_container', 0)}, "
            f"empty pages {r.get('empty_pages', 0)}"
        )
        lines.append(
            f"fidelity: lost {r.get('lost_chars', 0)} of {r.get('visible_chars', 0)} letters and "
            f"digits of visible text, lossy pages {len(r.get('lossy_pages', []))}"
        )
        lines.append("read-time noise (pages hit, elements, visible characters removed):")
        for rule in self.read_rules:
            lines.append(
                f"  {rule.name:<18}{rule.pages_hit:>8}{rule.elements:>10}{rule.chars_removed:>12,}"
            )
        lines += self.quality.summary_lines()
        return lines


def _merge(stats: list[dict[str, Any]]) -> dict[str, Any]:
    """Add up the counts of several readers (extensions with different configuration)."""
    merged: dict[str, Any] = {}
    for one in stats:
        for key, value in one.items():
            if isinstance(value, bool) or not isinstance(value, int | dict | list):
                merged[key] = value
            elif isinstance(value, int):
                merged[key] = merged.get(key, 0) + value
            elif isinstance(value, list):
                merged[key] = [*merged.get(key, []), *value]
            else:
                inner = merged.setdefault(key, {})
                for name, count in value.items():
                    if isinstance(count, dict):
                        total = inner.setdefault(name, {})
                        for field_name, number in count.items():
                            total[field_name] = total.get(field_name, 0) + number
                    else:
                        inner[name] = inner.get(name, 0) + count
    return merged


def chm_quality_report(
    proc: KbProcessor, connector: ChmDirConnector, *, warn_ratio: float = WARN_RATIO
) -> ChmQualityReport:
    """Read, clean and chunk everything `connector` delivers with `proc` and measure it."""
    sources = list(connector.documents())
    generic = quality_report(proc, sources, sample_size=0, warn_ratio=warn_ratio)
    reader = _merge(list(proc.reader_stats().values()))
    language_of = {s.doc_id: str(s.extra.get("language", "")) for s in sources}

    skipped_by_language: dict[str, Counter[str]] = {}
    for unit in generic.skipped_units:
        skipped_by_language.setdefault(language_of.get(unit.doc_id, ""), Counter())[
            unit.reason
        ] += 1

    volumes: dict[str, VolumeReport] = {}
    for language, found in connector.volume_stats.items():
        codecs = {
            key.split(":", 1)[1]: n
            for key, n in reader.get("encodings", {}).items()
            if key.startswith(f"{language}:")
        }
        volumes[language] = VolumeReport(
            language=language,
            pages_in=sum(1 for lang in language_of.values() if lang == language),
            pages_out=generic.languages.get(language, 0),
            skipped=dict(skipped_by_language.get(language, {})),
            encodings=codecs,
            not_in_toc=len(found.not_in_toc),
            toc_without_page=len(found.toc_without_page),
            duplicate_page_keys=len(found.duplicate_page_keys),
        )

    present: dict[str, set[str]] = {}
    for s in sources:
        key = s.extra.get("page_key")
        if key:
            present.setdefault(str(key), set()).add(str(s.extra.get("language", "")))
    pairing: Counter[str] = Counter()
    for languages in present.values():
        pairing["both" if len(languages) > 1 else f"only_{next(iter(languages))}"] += 1

    read_rules = [
        ReadRuleReport(name=name, **numbers)
        for name, numbers in reader.get("html_rules", {}).items()
    ]
    return ChmQualityReport(
        kb=proc.cfg.kb,
        created_at=datetime.now(UTC).isoformat(timespec="seconds"),
        volumes=volumes,
        pairing=dict(pairing),
        read_rules=read_rules,
        reader=reader,
        quality=generic,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m rag.ingest.chm_report",
        description="Read, clean and chunk a CHM knowledge base and report data quality "
        "(counts only, no page text; no ES, no model calls).",
    )
    parser.add_argument("--kb", required=True, help="knowledge base name (`kb` in its YAML)")
    size = parser.add_mutually_exclusive_group()
    size.add_argument("--limit", type=int, help="first N pages per volume (default: the YAML's)")
    size.add_argument("--all", action="store_true", help="every page (no limit)")
    parser.add_argument("--out", help="JSON report path (default: data/reports/<kb>/...)")
    parser.add_argument(
        "--warn-ratio", type=float, default=WARN_RATIO, help="mark a rule that removes more"
    )
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
        if not isinstance(connector, ChmDirConnector):
            print(f"error: {cfg.kb} does not use the chm_dir connector", file=sys.stderr)
            return 2
        proc = KbProcessor(cfg, settings=settings)
        report = chm_quality_report(proc, connector, warn_ratio=args.warn_ratio)
    except RemovalGuardError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # report every failure as one line, not a traceback
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out = (
        Path(args.out) if args.out else repo_path(f"data/reports/{cfg.kb}/chm-quality-{stamp}.json")
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report.model_dump_json(indent=1), "utf-8")
    for line in report.summary_lines():
        print(line)
    print(f"report: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
