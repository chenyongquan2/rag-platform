"""Detector scan over a corpus: how often and where the injection patterns fire (spec P1-03).

The scan runs `scan_text` on every document's raw text (before cleaning, so HTML comments and
hidden elements are still there) and counts hits by pattern, category and view. A seeded sample
of hits with their surrounding text is kept for labelling by hand: each sample's `label` is
left empty, to be set to `tp` (a real instruction to a model) or `fp` (documentation that only
looks like one); the false-positive rate of the sample estimates the detector's precision.
"""

from __future__ import annotations

import random
from collections import Counter
from collections.abc import Iterable

from pydantic import BaseModel, Field

from rag.data import SourceDocument
from rag.ingest.injection import InjectionHit, scan_text

CONTEXT_CHARS = 160
"""Characters of raw text kept on each side of a sampled hit."""


class ScanSample(BaseModel):
    doc_id: str
    source_rel: str
    pattern: str
    category: str
    language: str
    view: str
    snippet: str
    context: str
    """Raw text around the hit; for decoded or folded views the snippet is appended."""
    label: str = ""
    """`tp` or `fp`, set by hand."""


class ScanReport(BaseModel):
    docs: int = 0
    docs_hit: int = 0
    hits: int = 0
    seconds: float = 0.0
    by_pattern: dict[str, int] = Field(default_factory=dict)
    by_category: dict[str, int] = Field(default_factory=dict)
    by_view: dict[str, int] = Field(default_factory=dict)
    hit_docs: dict[str, int] = Field(default_factory=dict)
    """doc_id -> hits, for documents with at least one hit."""
    seed: int = 0
    samples: list[ScanSample] = Field(default_factory=list)

    @property
    def false_positive_rate(self) -> float | None:
        """Share of `fp` among the labelled samples; None before any are labelled."""
        labelled = [s.label for s in self.samples if s.label in ("tp", "fp")]
        return labelled.count("fp") / len(labelled) if labelled else None

    def summary_lines(self) -> list[str]:
        lines = [
            f"docs: {self.docs}",
            f"docs hit: {self.docs_hit}",
            f"hits: {self.hits}",
            f"scan seconds: {self.seconds:.1f}",
        ]
        lines += [f"  {name}: {n}" for name, n in self.by_pattern.items()]
        return lines


def _context(text: str, hit: InjectionHit) -> str:
    start = max(0, hit.start - CONTEXT_CHARS)
    end = min(len(text), hit.start + len(hit.snippet) + CONTEXT_CHARS)
    window = text[start:end]
    if hit.snippet in window:
        return window
    return f"{window}\n[{hit.view}] {hit.snippet}"


def _count(items: Iterable[str]) -> dict[str, int]:
    return dict(sorted(Counter(items).items(), key=lambda kv: (-kv[1], kv[0])))


def scan_sources(
    sources: Iterable[SourceDocument], *, samples: int = 20, seed: int = 0
) -> ScanReport:
    """Scan every source document (see the module doc); `samples` hits are drawn with `seed`."""
    import time

    began = time.perf_counter()
    found: list[tuple[SourceDocument, str, InjectionHit]] = []
    report = ScanReport(seed=seed)
    for source in sources:
        report.docs += 1
        text = source.content.decode("utf-8", errors="replace")
        hits = scan_text(text)
        if hits:
            report.hit_docs[source.doc_id] = len(hits)
        found.extend((source, text, hit) for hit in hits)
    report.docs_hit = len(report.hit_docs)
    report.hits = len(found)
    report.by_pattern = _count(h.pattern for _, _, h in found)
    report.by_category = _count(h.category for _, _, h in found)
    report.by_view = _count(h.view for _, _, h in found)
    picked = sorted(random.Random(seed).sample(range(len(found)), min(samples, len(found))))
    for index in picked:
        source, text, hit = found[index]
        report.samples.append(
            ScanSample(
                doc_id=source.doc_id,
                source_rel=source.source_rel,
                pattern=hit.pattern,
                category=hit.category,
                language=hit.language,
                view=hit.view,
                snippet=hit.snippet,
                context=_context(text, hit),
            )
        )
    report.seconds = round(time.perf_counter() - began, 2)
    return report


def _cell(text: str, limit: int = 200) -> str:
    text = " ".join(text.split()).replace("|", "\\|")
    return text if len(text) <= limit else text[: limit - 1] + "…"


def render_scan(report: ScanReport, *, title: str) -> str:
    """Markdown: counts, then the sampled hits with an empty label column to fill in."""
    lines = [f"# Injection scan: {title}", ""]
    lines += [
        f"- docs: {report.docs}",
        f"- docs hit: {report.docs_hit}",
        f"- hits: {report.hits}",
        f"- seed: {report.seed}",
        "",
    ]
    lines += ["| pattern | hits |", "|---|---|"]
    lines += [f"| {name} | {n} |" for name, n in report.by_pattern.items()]
    lines += ["", "| category | hits |", "|---|---|"]
    lines += [f"| {name} | {n} |" for name, n in report.by_category.items()]
    lines += ["", "| view | hits |", "|---|---|"]
    lines += [f"| {name} | {n} |" for name, n in report.by_view.items()]
    lines += ["", "## Sampled hits", "", "| # | doc | pattern | view | snippet | context | label |"]
    lines.append("|---|---|---|---|---|---|---|")
    for i, s in enumerate(report.samples, start=1):
        lines.append(
            f"| {i} | {s.source_rel} | {s.pattern} | {s.view} | {_cell(s.snippet, 120)} | "
            f"{_cell(s.context)} | {s.label} |"
        )
    return "\n".join(lines) + "\n"
