"""A few real ask_docs requests on the pdf_demo knowledge base (spec P6-01 evaluation).

    uv run python scripts/p6_01_ask.py [qid ...] [--out FILE]

Asks the questions of evals/datasets/pdf_demo/questions.jsonl through the same `SearchService`
that serves `ask_docs` (the real chain, the real model; about 0.01 CNY a question), and prints per
question: status, the answer, the numbered sources with file, page, chunk kind and whether the
text was written by a model, the labelled page and whether a source came from it, and the cost
read from the cost ledger. Needs the Docker services and the gateway.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from rag.config import ServiceSettings
from rag.cost import CostLedger
from rag.security import Credentials
from rag.serve.service import SearchService

DEFAULT = ("pdf-t02", "pdf-c01", "pdf-f01", "pdf-i01")
QUESTIONS = (
    Path(__file__).resolve().parents[1] / "evals" / "datasets" / "pdf_demo" / "questions.jsonl"
)


def questions() -> dict[str, dict[str, Any]]:
    rows = [json.loads(ln) for ln in QUESTIONS.read_bytes().decode("utf-8").splitlines() if ln]
    return {r["qid"]: r for r in rows}


async def run(qids: list[str], settings: ServiceSettings) -> list[dict[str, Any]]:
    service = SearchService(settings)
    service.start()  # loads the reranker in a background thread
    if not service.wait_ready(timeout_s=300):
        raise SystemExit("the service did not become ready within 300 s")
    ledger = CostLedger.from_settings(settings)
    asked = questions()
    out: list[dict[str, Any]] = []
    try:
        for qid in qids:
            q = asked[qid]
            before = ledger.total_spent()
            response = await service.ask_docs(
                Credentials(transport="stdio"), kb="pdf_demo", question=q["question"]
            )
            labelled = set(q["relevant_pages"])
            sources = [
                {
                    "n": i,
                    "file": s.source_rel,
                    "page": s.metadata.get("page"),
                    "kind": s.metadata.get("chunk_kind"),
                    "generated": bool(s.metadata.get("has_generated_text")),
                    "labelled_page": s.doc_id in labelled,
                    "chars": len(s.text),
                }
                for i, s in enumerate(response.sources, start=1)
            ]
            out.append(
                {
                    "qid": qid,
                    "question": q["question"],
                    "expected": q["answer"],
                    "status": response.status,
                    "answer": response.answer,
                    "cited": sorted({c.marker for s in response.sentences for c in s.citations}),
                    "sources": sources,
                    "cost_cny": round(ledger.total_spent() - before, 6),
                    "degraded": response.degraded,
                }
            )
    finally:
        service.close()
    return out


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("qids", nargs="*", default=list(DEFAULT))
    parser.add_argument("--out", help="write the results as JSON")
    args = parser.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    results = asyncio.run(run(args.qids, ServiceSettings()))
    for r in results:
        print(f"\n== {r['qid']}: {r['question']}")
        print(f"expected: {r['expected']}")
        print(f"status: {r['status']}  cost: {r['cost_cny']} CNY  degraded: {r['degraded']}")
        print(f"answer: {r['answer']}")
        print(f"cited markers: {r['cited']}")
        for s in r["sources"]:
            print(f"  [{s['n']}] {s}")
    print(f"\ntotal cost: {sum(r['cost_cny'] for r in results):.6f} CNY")
    if args.out:
        Path(args.out).write_bytes(json.dumps(results, ensure_ascii=False, indent=1).encode())
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
