# ruff: noqa: E501 (Markdown table rows)
"""Markdown report of the support check (spec P1-06): no model call here."""

from __future__ import annotations

from collections.abc import Sequence

from rag.eval.answer.record import Proportion
from rag.eval.support.metrics import by_group, paired_all_supported, stability, summarize
from rag.eval.support.records import SupportLine, latest_lines


def _p(x: Proportion) -> str:
    return x.text()


def _group_table(lines: Sequence[SupportLine]) -> list[str]:
    out = [
        "| group | answers | skipped | pairs (unknown) | precision | partly | precision with partly | sentences (unknown, uncited) | recall | unsupported sentences | every sentence supported |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name, s in by_group(lines).items():
        out.append(
            f"| {name} | {s.answers} | {s.skipped} {s.skip_reasons or ''} | {s.pairs} ({s.pairs_unknown}) | "
            f"{_p(s.precision)} | {_p(s.partly)} | {_p(s.precision_lenient)} | "
            f"{s.sentences} ({s.sentences_unknown}, {s.uncited}) | {_p(s.recall)} | {_p(s.unsupported)} | "
            f"{_p(s.all_supported)} |"
        )
    return out


def _planted(lines: Sequence[SupportLine]) -> list[str]:
    planted = [x for x in lines if x.item.planted is not None and x.result is not None]
    if not planted:
        return ["No planted errors judged yet."]
    out = [
        "| kind | original | sentence before | sentence after | cited chunk | verdict of the planted pair | sentence status |",
        "|---|---|---|---|---|---|---|",
    ]
    counts = {"not_supported": 0, "partly": 0, "supported": 0, "unknown": 0}
    for x in planted:
        p = x.item.planted
        assert p is not None and x.result is not None
        sentence = x.result.sentences[p.sentence - 1]
        labels = [q.label for q in sentence.pairs if q.chunk_id == p.chunk_id] or ["unknown"]
        status = sentence.status
        counts[status] = counts.get(status, 0) + 1
        out.append(
            f"| {p.kind} | {p.original_qid} | {p.before} | {p.after} | {p.chunk_id} | {', '.join(labels)} | {status} |"
        )
    n = len(planted)
    caught = counts["not_supported"] + counts["partly"]
    out += [
        "",
        f"- planted: {n}; caught (sentence not supported, or only partly): {Proportion.of(caught, n).text()}",
        f"- caught as not_supported: {Proportion.of(counts['not_supported'], n).text()}; "
        f"partly: {counts['partly']}; missed (still supported): {counts['supported']}; unknown: {counts['unknown']}",
    ]
    return out


def _not_supported_list(lines: Sequence[SupportLine]) -> list[str]:
    out = ["| system | qid | sentence | chunk | verdict | reason |", "|---|---|---|---|---|---|"]
    for x in lines:
        if x.result is None or x.item.planted is not None:
            continue
        for s in x.result.sentences:
            if not s.pairs and s.status == "not_supported":
                out.append(f"| {x.system} | {x.qid} | {s.text} | (none) | uncited | |")
            for q in s.pairs:
                if q.label != "supported":
                    out.append(
                        f"| {x.system} | {x.qid} | {s.text} | {q.chunk_id} | {q.label} | {q.reason or q.error or ''} |"
                    )
    return out


def render_report(all_lines: Sequence[SupportLine]) -> str:
    lines = latest_lines(all_lines)
    runs = sorted({x.run for x in lines})
    real = [x for x in lines if x.item.planted is None]
    judged = [x for x in lines if x.result is not None]
    models = sorted({(x.result.judge_model, x.result.prompt_version) for x in judged if x.result})
    text = ["# Citation check level 2 (P1-06)", ""]
    text.append(f"- judge: {', '.join(f'{m} (prompt {v})' for m, v in models) or 'n/a'}")
    text.append(f"- sources: {', '.join(sorted({x.source for x in lines}))}; runs: {runs}")
    for run in runs:
        s = summarize([x for x in lines if x.run == run])
        text.append(f"- run {run}: {s.calls} calls, {s.cny:.4f} CNY")
    text += ["", "## Metrics, run 1 (precision of pairs, recall of sentences)", ""]
    text += _group_table([x for x in real if x.run == 1])
    for run in runs[1:]:
        text += ["", f"## Metrics, run {run}", ""]
        text += _group_table([x for x in real if x.run == run])

    text += ["", "## Judge stability (run 1 against run 2, same inputs)", ""]
    if 2 in runs:
        st = stability([x for x in real if x.run == 1], [x for x in real if x.run == 2])
        text += [
            f"- pair labels that changed: {_p(st.pairs)} (pairs unknown in a run: {st.pairs_unknown})",
            f"- sentence statuses that changed: {_p(st.sentences)}",
            f"- answers whose 'every sentence supported' changed: {_p(st.answers)}",
        ]
        text += [
            f"  - {c['system']} {c['qid']} sentence {c['sentence']} {c['chunk_id']}: {c['run1']} -> {c['run2']}"
            for c in st.changed
        ]
    else:
        text.append("Only one run so far.")

    text += ["", "## Paired comparison, standard RAG (A) against ask_docs (B)", ""]
    first = [x for x in real if x.run == 1]
    a = [x for x in first if x.system == "standard" and x.source != "p1-04"]
    b = [x for x in first if x.system == "agentic"]
    if any(x.result for x in a) and any(x.result for x in b):
        p = paired_all_supported(a, b)
        text.append(
            f"- every sentence supported, on {p.n} shared questions: both {p.both}, only A {p.a_only}, "
            f"only B {p.b_only}, neither {p.neither}; unknown {p.unknown}, unpaired {p.unpaired}; "
            f"exact McNemar p = {p.p_value:.3f}"
        )
    else:
        text.append(
            "Not possible on these lines: one of the two systems has no judged answer "
            "(records without sentence citations are skipped)."
        )

    text += ["", "## Planted errors (copies of real answers)", ""]
    text += _planted([x for x in lines if x.run == 1])
    text += ["", "## Pairs the judge did not call supported (run 1)", ""]
    text += _not_supported_list(first)
    return "\n".join(text) + "\n"
