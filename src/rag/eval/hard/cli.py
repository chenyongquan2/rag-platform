"""Command line of the hard questions (spec EVAL-2).

uv run python -m rag.eval.hard check [--file hard.jsonl] [--data-dir DIR]
uv run python -m rag.eval.hard self-check --out DIR [--file hard.jsonl] [--n 10] [--seed 0]
uv run python -m rag.eval.hard clarify-report --out RECORDS_DIR

`check` re-proves from the downloaded corpus that no single page states all facts of a hard
question and that every quoted evidence is on its page (exit 1 on any problem, 2 when the corpus
is missing). `self-check` writes the owner's blind sheet and its key (for the owner's sheet use
`--out C:/workspace/rag-platform-data/human/hard_self_check`; the files stay out of git).
`clarify-report` applies the rule check (did the system ask back?) to the records of an agentic
run (`python -m rag.eval.agentic run --types ambiguous`).
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from rag.eval.agentic.records import SYSTEMS, read_records, system_file
from rag.eval.hard.clarify import clarification_section
from rag.eval.hard.corpus import corpus_report, load_corpus
from rag.eval.hard.selfcheck import export_self_check
from rag.eval.hard.spec import HARD_FILE, load_hard, validate_questions


class UsageError(Exception):
    """Bad arguments or a missing input: exit code 2, one line on stderr."""


def _check(args: argparse.Namespace) -> int:
    questions = load_hard(args.file)
    problems = validate_questions(questions)
    for line in problems:
        print(f"shape problem: {line}")
    corpus = load_corpus(args.data_dir)
    if corpus is None:
        raise UsageError("the Hugging Face corpus is not downloaded (see HF_DATA_DIR)")
    findings = corpus_report(questions, corpus)
    bad = 0
    for f in findings:
        status = "ok" if f.ok else "PROBLEM"
        pages = ", ".join(f"{n}" for n in f.key_pages.values())
        print(
            f"{f.qid}: {status} (key pages per fact: {pages or '-'}; searched {f.searched_pages})"
        )
        for p in f.problems:
            print(f"    {p}")
        bad += 0 if f.ok else 1
    print(f"{len(findings)} questions, {bad} with problems, {len(problems)} shape problems")
    return 1 if bad or problems else 0


def _self_check(args: argparse.Namespace) -> int:
    try:
        sheet, key = export_self_check(
            load_hard(args.file), n=args.n, seed=args.seed, out_dir=args.out
        )
    except ValueError as exc:
        raise UsageError(str(exc)) from exc
    print(f"wrote {sheet}\nwrote {key}\nanswer the sheet first; open the key afterwards")
    return 0


def _clarify_report(args: argparse.Namespace) -> int:
    out = Path(args.out)
    records = []
    for system in SYSTEMS:
        records += read_records(system_file(out, system))
    if not records:
        raise UsageError(f"no records under {out}")
    lines = clarification_section(records)
    if not lines:
        raise UsageError(f"no ambiguous question among the records under {out}")
    print("\n".join(lines).lstrip("\n"))
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m rag.eval.hard",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)
    p_check = sub.add_parser("check", help="check the questions against the corpus")
    p_check.add_argument("--file", default=HARD_FILE)
    p_check.add_argument("--data-dir", help="corpus directory (default: HF_DATA_DIR)")
    p_self = sub.add_parser("self-check", help="blind sheet for the owner")
    p_self.add_argument("--file", default=HARD_FILE)
    p_self.add_argument("--n", type=int, default=10)
    p_self.add_argument("--seed", type=int, default=0)
    p_self.add_argument("--out", required=True)
    p_clar = sub.add_parser(
        "clarify-report", help="did the systems ask back on ambiguous questions"
    )
    p_clar.add_argument("--out", required=True, help="records directory of an agentic run")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Exit codes: 0 ok, 1 problems found, 2 usage error."""
    args = _parser().parse_args(argv)
    try:
        return {"check": _check, "self-check": _self_check, "clarify-report": _clarify_report}[
            args.command
        ](args)
    except UsageError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
