"""Records of an answer eval run (spec P1-04, design s14-1, s14-4)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from rag.eval.retrieval.record import LatencySummary
from rag.eval.stats import Proportion
from rag.types import Answer

RunKind = Literal["hf", "security"]


class AnswerItem(BaseModel):
    """One question to answer, with the reference answer the judge compares against."""

    qid: str = Field(min_length=1)
    question: str = Field(min_length=1)
    reference: str = ""
    relevant_pages: list[str] = Field(default_factory=list)


class Usage(BaseModel):
    """Model calls and their tokens and CNY, summed from cost ledger lines."""

    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    """Reasoning tokens included (they are billed as output)."""
    reasoning_tokens: int = 0
    cny: float = 0.0
    usage_estimated: bool = False
    """Some line's tokens were estimated (a cancelled call, or no usage in the answer)."""

    @classmethod
    def from_rows(cls, rows: list[dict[str, Any]]) -> Usage:
        return cls(
            calls=len(rows),
            prompt_tokens=sum(int(r.get("prompt_tokens") or 0) for r in rows),
            completion_tokens=sum(int(r.get("completion_tokens") or 0) for r in rows),
            reasoning_tokens=sum(int(r.get("reasoning_tokens") or 0) for r in rows),
            cny=round(sum(float(r.get("cny") or 0.0) for r in rows), 10),
            usage_estimated=any(bool(r.get("usage_estimated")) for r in rows),
        )

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            calls=self.calls + other.calls,
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            reasoning_tokens=self.reasoning_tokens + other.reasoning_tokens,
            cny=round(self.cny + other.cny, 10),
            usage_estimated=self.usage_estimated or other.usage_estimated,
        )


def usage_by_kind(rows: list[dict[str, Any]]) -> dict[str, Usage]:
    """`chat` and `embedding` lines apart (a query embedding missed by the cache costs too)."""
    kinds = sorted({str(r.get("kind") or "unknown") for r in rows})
    return {k: Usage.from_rows([r for r in rows if str(r.get("kind")) == k]) for k in kinds}


class AnswerQuestionRecord(BaseModel):
    """One question: what was retrieved, the answer, its citation check, time and money."""

    qid: str
    question: str
    reference: str = ""
    relevant_pages: list[str] = Field(default_factory=list)
    answer: Answer | None = None
    """The full answer (sentences, citations, fragments given); None after a retrieval error."""
    text: str = ""
    """Answer sentences without markers (what the judge reads); the not-found sentence too."""
    status: str = ""
    found: bool = False
    answer_parse_failed: bool = False
    generation_error: str | None = None
    parse_attempts: int = 0
    sentences: int = 0
    unsupported_sentences: int = 0
    removed_markers: list[int] = Field(default_factory=list)
    context_chunk_ids: list[str] = Field(default_factory=list)
    """The numbered fragments, in order: marker n is `context_chunk_ids[n - 1]`."""
    cited_chunk_ids: list[str] = Field(default_factory=list)
    """Distinct chunk ids of valid citations, in first-cited order."""
    cited_pages: list[str] = Field(default_factory=list)
    cites_relevant_page: bool | None = None
    """Some citation points into a labelled page; None without labels or without citations."""
    retrieval_not_found: bool = False
    degraded: bool = False
    rerank_timeout: bool = False
    retrieval_ms: float = 0.0
    generation_ms: float = 0.0
    total_ms: float = 0.0
    usage: Usage = Field(default_factory=Usage)
    usage_by_kind: dict[str, Usage] = Field(default_factory=dict)
    error: str | None = None
    """A retrieval exception (`Type: message`); the question has no answer."""

    @classmethod
    def from_answer(
        cls,
        *,
        qid: str,
        question: str,
        reference: str,
        relevant_pages: list[str],
        answer: Answer | None,
        error: str | None,
        **extra: Any,
    ) -> AnswerQuestionRecord:
        fields: dict[str, Any] = {}
        if answer is not None:
            cited: list[str] = []
            for sentence in answer.sentences:
                for citation in sentence.citations:
                    if citation.chunk_id not in cited:
                        cited.append(citation.chunk_id)
            page_of = {c.chunk_id: c.doc_id for c in answer.chunks}
            pages = list(dict.fromkeys(page_of.get(c, c.split("#", 1)[0]) for c in cited))
            fields = {
                "text": answer.text,
                "status": str(answer.status),
                "found": answer.found,
                "answer_parse_failed": answer.answer_parse_failed,
                "generation_error": answer.generation_error,
                "parse_attempts": answer.citation_check.parse_attempts,
                "sentences": len(answer.sentences) if answer.found else 0,
                "unsupported_sentences": answer.citation_check.unsupported_sentences,
                "removed_markers": list(answer.citation_check.removed_markers),
                "context_chunk_ids": [c.chunk_id for c in answer.chunks],
                "cited_chunk_ids": cited,
                "cited_pages": pages,
                "cites_relevant_page": (
                    bool(set(pages) & set(relevant_pages)) if relevant_pages and pages else None
                ),
            }
        return cls(
            qid=qid,
            question=question,
            reference=reference,
            relevant_pages=relevant_pages,
            answer=answer,
            error=error,
            **fields,
            **extra,
        )


class FormatCompliance(BaseModel):
    """How well the answers follow the `[n]` format (design s11-1, s11-2 level 1).

    Denominators: `answered` = found answers; `model answers` = questions where the model
    produced text (at least one attempt, no gateway error); `n_questions` = all.
    """

    n_questions: int
    answered: Proportion
    """found=true of all questions."""
    found_false: Proportion
    """found=false of all questions (not-found sentence, empty retrieval, parse failure, error)."""
    fully_cited: Proportion
    """Answers whose every sentence keeps at least one valid `[n]`, of found answers."""
    fully_cited_all: Proportion
    """The same, of all questions."""
    with_unknown_markers: Proportion
    """Answers with at least one number that matched no fragment, of model answers."""
    unknown_markers_total: int
    with_unsupported: Proportion
    """Found answers with at least one sentence without a valid citation."""
    unsupported_sentences: Proportion
    """Sentences without a valid citation, of all sentences of found answers (sentences of one
    answer are not independent, so this interval is too narrow)."""
    parse_failed: Proportion
    """`answer_parse_failed` (no valid marker after the retry), of model answers."""
    retried: Proportion
    """Answers that needed the second attempt, of model answers."""
    generation_errors: int
    retrieval_not_found: int
    retrieval_errors: int


class AnswerRunRecord(BaseModel):
    run_id: str
    created_at: str
    kb: str
    label: str = ""
    kind: RunKind = "hf"
    git_commit: str | None = None
    git_dirty: bool | None = None
    lock_sha256: str | None = None
    models: dict[str, str] = Field(default_factory=dict)
    kb_config: dict[str, Any] = Field(default_factory=dict)
    kb_config_sha256: str = ""
    chat_model: str = ""
    chat_profile: str = "online"
    question_set: dict[str, Any] = Field(default_factory=dict)
    """path, sha256 and count of the questions (the HF set or the security set)."""
    questions: list[AnswerQuestionRecord]
    compliance: FormatCompliance
    usage: Usage = Field(default_factory=Usage)
    usage_by_kind: dict[str, Usage] = Field(default_factory=dict)
    latency: dict[str, LatencySummary] = Field(default_factory=dict)
    """`retrieval`, `generation`, `total`."""
