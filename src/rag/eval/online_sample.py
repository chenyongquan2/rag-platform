"""Judge a random sample of recent real requests (spec P5-02, design s15-1, s14-5).

    uv run python -m rag.eval.online_sample [--n 20] [--window-hours 168] [--seed 7]

Real requests have no gold answer, so this does not measure whether an answer is right. It asks the
sufficiency judge of the question graph (`rag.agent.judge`, the `cheap` profile, the same
instructions; P3-03 measured its verdicts at 92 of 100 on questions whose page is known) one thing:
do the passages the service returned hold the information the question asks for? For `search_docs`
the passages are the final list; for `ask_docs` they are the cited sources (the business record
keeps
no answer text). The share of "yes" with its Wilson interval goes to a JSON file in
`online_eval_report_dir` and to the gauge `rag.online_eval.score_mean`.

What the number can and cannot say is written into every report (`limits`): it sees only what was
returned (a page the search missed is invisible), a question the knowledge base cannot answer counts
as "no", and the judge itself is wrong now and then; compare it with the 7 days before, not with a
fixed target. Nothing runs it on a schedule yet.

Cost: one cheap-profile call per sampled request (a few thousandths of a CNY). The job checks the
cost ledger's stop before the first call and stops by itself after `online_eval_max_cny`.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Protocol

from langchain_core.language_models import BaseChatModel
from opentelemetry.metrics import Meter
from pydantic import BaseModel

from rag.agent.judge import SufficiencyJudgement, sufficiency_messages
from rag.config import ServiceSettings
from rag.config.paths import repo_path
from rag.eval.answer.record import Usage
from rag.eval.answer.usage import collect_ledger_rows
from rag.eval.stats import wilson_interval
from rag.index.names import KbIndexNames
from rag.llm.structured import structured_output
from rag.monitor.probe import PROBE_REQUEST_PREFIX
from rag.observe import attributes as attr
from rag.observe import get_observability
from rag.observe.records import BusinessRecord
from rag.types import Chunk

logger = logging.getLogger(__name__)

ONLINE_SAMPLE_HARD_CAP = 50
"""Never more requests than this in one run, whatever `--n` says."""
QUESTION_CHARS = 200
JUDGED_TOOLS = ("search_docs", "ask_docs")
# Flags of a request that did not produce passages worth judging.
_SKIP_FLAGS = frozenset(
    {"internal_error", "invalid_request", "client_cancelled", "kb_not_found", "not_ready"}
)

MEASURES = (
    "whether the passages the service returned (ask_docs: the cited sources) hold what the "
    "question asks for, as one judge model reads them"
)
LIMITS = (
    "answer correctness, citation precision and recall are not measured: there are no gold labels "
    "and the business record keeps no answer text",
    "a passage the search did not return is invisible: a missed page shows up only as 'not "
    "grounded'",
    "a question the knowledge base cannot answer counts as 'not grounded'",
    "the judge is wrong now and then (92 of 100 on questions with a known page in P3-03): a change "
    "of a few points is noise; compare with the previous 7 days, not with a fixed target",
    "the interval is the sampling error of N requests; it does not include the judge's error",
)


@dataclass(frozen=True)
class Candidate:
    request_id: str
    tool: str
    kb: str
    question: str
    chunk_ids: tuple[str, ...]


class ChunkSource(Protocol):
    def fetch(self, kb: str, chunk_ids: Sequence[str]) -> Mapping[str, tuple[list[str], str]]:
        """`{chunk_id: (title_path, text)}` of the chunks that still exist."""
        ...


class EsChunkSource:
    """Reads chunk text from the knowledge base's alias (read only)."""

    def __init__(self, settings: ServiceSettings) -> None:
        from rag.factories import make_es_client

        self._es = make_es_client(settings.es_url, request_timeout=10.0, max_retries=1)
        self._prefix = settings.search_index_prefix

    def fetch(self, kb: str, chunk_ids: Sequence[str]) -> dict[str, tuple[list[str], str]]:
        response = self._es.search(
            index=KbIndexNames(kb, self._prefix).alias,
            query={"terms": {"chunk_id": list(chunk_ids)}},
            size=len(chunk_ids),
            source={"includes": ["chunk_id", "title_path", "content"]},
            track_total_hits=False,
        )
        found: dict[str, tuple[list[str], str]] = {}
        for hit in response["hits"]["hits"]:
            doc = hit["_source"]
            found[str(doc["chunk_id"])] = (
                [str(p) for p in doc.get("title_path") or []],
                str(doc.get("content") or ""),
            )
        return found


class OnlineSummary(BaseModel):
    created_at: str
    n_pool: int
    n_sampled: int
    n_scored: int
    n_failed: int
    n_no_text: int
    grounded: int
    rate: float | None
    wilson_low: float | None
    wilson_high: float | None
    spent_cny: float
    path: str = ""


# ---- reading the pool ---------------------------------------------------------------------


def _files(path: Path) -> list[Path]:
    """The record file and its rotated backups (`.1` is the newest old one), newest first."""
    found = [path] if path.exists() else []
    n = 1
    while (backup := path.with_name(f"{path.name}.{n}")).exists():
        found.append(backup)
        n += 1
    return found


def _candidate(row: Mapping[str, Any], earliest: datetime) -> Candidate | None:
    try:
        record = BusinessRecord.model_validate(
            {
                **row,
                "extra": row.get("extra") or {},
                "config_version": row.get("config_version", ""),
            }
        )
    except ValueError:
        return None
    if record.tool not in JUDGED_TOOLS or _SKIP_FLAGS & set(record.flags):
        return None
    if record.request_id.startswith(PROBE_REQUEST_PREFIX):
        return None  # the scheduled probe's own fixed question, not something a user asked
    try:
        when = datetime.fromisoformat(record.time)
    except ValueError:
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=UTC)
    if when < earliest:
        return None
    key = "question" if record.tool == "ask_docs" else "query"
    question = record.extra.get(key)
    chunk_ids = tuple(item.chunk_id for item in record.final)
    if not isinstance(question, str) or not question.strip() or not chunk_ids:
        return None
    return Candidate(record.request_id, record.tool, record.kb, question.strip(), chunk_ids)


def load_candidates(path: str | Path, *, now: datetime, window_hours: float) -> list[Candidate]:
    """Requests of the last `window_hours` that returned passages and have a question."""
    earliest = now - timedelta(hours=window_hours)
    out: list[Candidate] = []
    for file in _files(Path(path)):
        with file.open("r", encoding="utf-8", newline="") as fh:
            for line in fh:
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                if isinstance(row, dict) and (item := _candidate(row, earliest)) is not None:
                    out.append(item)
    return out


def sample_candidates(pool: Sequence[Candidate], n: int, *, seed: int | None) -> list[Candidate]:
    """At most `n` (and never more than the hard cap) requests of the pool, chosen at random;
    the same seed gives the same sample."""
    n = max(0, min(n, ONLINE_SAMPLE_HARD_CAP, len(pool)))
    return random.Random(seed).sample(list(pool), n)


# ---- the metric ---------------------------------------------------------------------------


def create_online_eval_gauge(meter: Meter) -> Any:
    return meter.create_gauge(
        attr.METRIC_ONLINE_EVAL_SCORE,
        unit="",
        description="Share of sampled requests whose passages the judge found sufficient (0 to 1)",
    )


# ---- the run ------------------------------------------------------------------------------


def _judge(
    model: BaseChatModel, candidate: Candidate, chunks: list[Chunk], settings: ServiceSettings
) -> bool | None:
    """True or False from the judge; None when the call or its answer failed."""
    cap = settings.agent_judge_context_tokens
    messages = sufficiency_messages(candidate.question[: QUESTION_CHARS * 3], chunks, cap)
    try:
        verdict = structured_output(model, SufficiencyJudgement, settings=settings).invoke(messages)
    except Exception as exc:  # one bad call must not end the run: it is counted
        logger.warning("judge call failed for %s: %s", candidate.request_id, type(exc).__name__)
        return None
    if [v.n for v in verdict.sub_questions] != [1]:
        return None
    return verdict.sub_questions[0].grounded


def _default_budget_check(settings: ServiceSettings) -> Callable[[float], float]:
    from rag.cost import check_budget

    return lambda cny: check_budget(cny, settings)


def run_online_sample(
    records_path: str | Path,
    *,
    n: int,
    seed: int | None,
    now: datetime,
    model: BaseChatModel,
    chunks: ChunkSource,
    out_dir: str | Path,
    settings: ServiceSettings,
    budget_check: Callable[[float], float] | None = None,
) -> OnlineSummary:
    pool = load_candidates(records_path, now=now, window_hours=settings.online_eval_window_hours)
    sample = sample_candidates(pool, n, seed=seed)
    check = budget_check if budget_check is not None else _default_budget_check(settings)
    if sample:
        check(len(sample) * settings.online_eval_est_cny_per_call)  # raises BudgetExceeded
    items: list[dict[str, Any]] = []
    grounded = failed = no_text = 0
    spent = Usage()
    for candidate in sample:
        found = chunks.fetch(candidate.kb, candidate.chunk_ids)
        texts = [
            Chunk(
                chunk_id=cid,
                doc_id=cid.rpartition("#")[0],
                kb=candidate.kb,
                text=found[cid][1],
                title_path=found[cid][0],
            )
            for cid in candidate.chunk_ids
            if cid in found and found[cid][1].strip()
        ]
        item: dict[str, Any] = {
            "request_id": candidate.request_id,
            "tool": candidate.tool,
            "kb": candidate.kb,
            "question": candidate.question[:QUESTION_CHARS],
            "n_chunks": len(texts),
            "grounded": None,
        }
        items.append(item)
        if not texts:
            no_text += 1
            item["error"] = "no_text"
            continue
        if spent.cny >= settings.online_eval_max_cny:
            failed += 1
            item["error"] = "cost_cap"
            continue
        with collect_ledger_rows() as rows:
            verdict = _judge(model, candidate, texts, settings)
        spent = spent + Usage.from_rows(rows)
        if verdict is None:
            failed += 1
            item["error"] = "judge_failed"
        else:
            item["grounded"] = verdict
            grounded += verdict
    scored = sum(1 for item in items if item["grounded"] is not None)
    rate = grounded / scored if scored else None
    low, high = wilson_interval(grounded, scored) if scored else (None, None)
    created = now.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"online-{now.astimezone(UTC).strftime('%Y%m%dT%H%M%SZ')}.json"
    summary = OnlineSummary(
        created_at=created,
        n_pool=len(pool),
        n_sampled=len(sample),
        n_scored=scored,
        n_failed=failed,
        n_no_text=no_text,
        grounded=grounded,
        rate=rate,
        wilson_low=low,
        wilson_high=high,
        spent_cny=round(spent.cny, 6),
        path=str(path),
    )
    report = {
        **summary.model_dump(),
        "wilson": [low, high],
        "measures": MEASURES,
        "limits": list(LIMITS),
        "judge": {
            "profile": settings.agent_judge_profile,
            "schema": "SufficiencyJudgement",
            "window_hours": settings.online_eval_window_hours,
            "seed": seed,
        },
        "items": items,
    }
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", "utf-8", newline="")
    if rate is not None:
        gauge = create_online_eval_gauge(get_observability().meter)
        gauge.set(rate)
    logger.info("online sample: %s of %s grounded, report %s", grounded, scored, path)
    return summary


# ---- command line -------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None, *, settings: ServiceSettings | None = None) -> int:
    settings = settings if settings is not None else ServiceSettings()
    parser = argparse.ArgumentParser(prog="python -m rag.eval.online_sample", description=__doc__)
    parser.add_argument("--n", type=int, default=settings.online_eval_n, help="requests to judge")
    parser.add_argument("--window-hours", type=float, default=settings.online_eval_window_hours)
    parser.add_argument("--seed", type=int, default=None, help="repeatable sample")
    parser.add_argument("--records", default=settings.business_record_path)
    args = parser.parse_args(argv)
    settings = settings.model_copy(update={"online_eval_window_hours": args.window_hours})

    from rag.cost import BudgetExceeded
    from rag.llm.factory import make_chat_model
    from rag.observe import init_observability, shutdown_observability

    obs = init_observability(settings, "rag-online-eval")
    try:
        model = make_chat_model(settings.agent_judge_profile, settings)
        summary = run_online_sample(
            repo_path(args.records),
            n=args.n,
            seed=args.seed,
            now=datetime.now(UTC),
            model=model,
            chunks=EsChunkSource(settings),
            out_dir=repo_path(settings.online_eval_report_dir),
            settings=settings,
        )
        obs.flush(10.0)
    except BudgetExceeded as exc:
        print(f"not run: {exc}", file=sys.stderr)
        return 2
    finally:
        shutdown_observability()
    rate = "none" if summary.rate is None else f"{summary.rate:.3f}"
    interval = (
        ""
        if summary.wilson_low is None or summary.wilson_high is None
        else f" ({summary.wilson_low:.3f} to {summary.wilson_high:.3f})"
    )
    print(
        f"pool {summary.n_pool}, sampled {summary.n_sampled}, scored {summary.n_scored}, "
        f"failed {summary.n_failed}, no text {summary.n_no_text}; grounded rate {rate}{interval}; "
        f"spent {summary.spent_cny} CNY; report {summary.path}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
