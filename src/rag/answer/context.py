# ruff: noqa: RUF002 (the docstrings show the full-width brackets that are escaped)
"""Context builder: retrieved chunks to numbered, delimited fragments (design s10-3, s11-3).

    ctx = build_context([result.chunks], budget_tokens=cfg.answer.context_tokens,
                        metadata_keys=cfg.metadata.llm)

Input is one ranked chunk list per sub-question (most relevant first; one list for a plain
question, several when P3 splits a question). The builder

1. removes repeated chunk ids, keeping the occurrence with the best rank;
2. gives every sub-question an equal share of the token budget, filled in rank order, then hands
   what is left to the remaining chunks of all sub-questions by score (rank when there is no
   score), so the first sub-question cannot take the whole budget (s10-3);
3. numbers the chosen chunks [1]..[n], most relevant first (rank 1 of every sub-question, then
   rank 2, ...), and renders each as a fragment: the number, then the text and the whitelisted
   metadata inside fixed delimiters that mark it as untrusted data, with chunks flagged by the
   injection detector labelled `suspicious_instruction="true"` (s11-3). Delimiter look-alikes in
   chunk text and metadata are escaped, so a chunk cannot close its fragment early.

Token counts are estimates of the rendered fragment (`estimate_tokens`, or a counter passed in).
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from pydantic import BaseModel, Field

from rag.types import Chunk

TokenCounter = Callable[[str], int]

FRAGMENT_TAG = "doc_fragment"
FRAGMENT_CLOSE = f"</{FRAGMENT_TAG}>"
DEFAULT_METADATA_KEYS: tuple[str, ...] = ("title_path", "source_rel")
TITLE_SEPARATOR = " > "

# `<doc_fragment` or `</doc_fragment`, any case, spaces allowed: what a chunk must not contain.
# Characters a model may not see: zero-width space, non-joiner, joiner, word joiner, BOM.
_ZERO_WIDTH = "".join(chr(c) for c in (0x200B, 0x200C, 0x200D, 0x2060, 0xFEFF))
_GAP = f"[{_ZERO_WIDTH}]*"
_SPACE_OR_GAP = rf"[\s{_ZERO_WIDTH}]*"
_FULL_WIDTH_OFFSET = 0xFEE0  # ASCII "!".."~" + offset = the full-width form


def _look_alike(char: str) -> str:
    """Regex class for one ASCII character of the tag name and its full-width forms."""
    forms = {char.lower(), char.upper()}
    forms |= {chr(ord(c) + _FULL_WIDTH_OFFSET) for c in tuple(forms)}
    return "[" + "".join(sorted(forms)) + "]"


# The `<` of anything that reads as `<doc_fragment` or `</doc_fragment`: any case, ASCII or
# full-width letters, `<` as ASCII, full-width or small form, spaces or zero-width characters in
# between. Only the `<` is replaced, so the text stays readable.
_LESS_THAN = "[<" + chr(0xFF1C) + chr(0xFE64) + "]"
_SLASH = "[/" + chr(0xFF0F) + "]"
_TAG_NAME = _GAP.join(_look_alike(char) for char in FRAGMENT_TAG)
_DELIMITER = re.compile(rf"{_LESS_THAN}(?={_SPACE_OR_GAP}{_SLASH}?{_SPACE_OR_GAP}{_TAG_NAME})")
# A line starting with a bracketed number looks like the `[n]` line before a fragment.
_OPEN_BRACKETS = r"\[" + chr(0x3010) + chr(0xFF3B)
_CLOSE_BRACKETS = r"\]" + chr(0x3011) + chr(0xFF3D)
_NUMBER_LINE = re.compile(
    rf"^([ \t{chr(0x3000)}]*)(?=[{_OPEN_BRACKETS}]\s*\d+\s*[{_CLOSE_BRACKETS}])", re.MULTILINE
)
# Characters counted as one token each: CJK ideographs, kana, Hangul, full-width forms.
_WIDE = re.compile(r"[　-ヿ㐀-䶿一-鿿가-힯豈-﫿＀-￯]")
_CHARS_PER_TOKEN = 3.0  # other text; English is nearer 4, code nearer 3: err on the high side


def estimate_tokens(text: str) -> int:
    """Rough token count for a budget: one per CJK character, one per three other characters.

    The generation model's tokenizer is not available locally; this errs on the high side for
    English prose, so a budget is not overrun (spec P1-02, docs/decisions.md).
    """
    wide = len(_WIDE.findall(text))
    return wide + math.ceil((len(text) - wide) / _CHARS_PER_TOKEN)


def escape_delimiters(text: str) -> str:
    """Neutralize what could pass for a fragment boundary in untrusted text (s11-3):

    - the `<` of anything that reads as `<doc_fragment` or `</doc_fragment` (any case, full-width
      letters or brackets, spaces or zero-width characters in between) becomes `&lt;`;
    - a line starting with a bracketed number (`[2]`, `【2】`, `［2］`), which looks like the
      number line before a fragment, gets a backslash in front of the bracket.
    """
    text = _DELIMITER.sub("&lt;", text)
    return _NUMBER_LINE.sub(lambda match: match.group(1) + "\\", text)


class Fragment(BaseModel):
    """One numbered fragment of the context."""

    number: int = Field(ge=1)
    chunk: Chunk
    sub_question: int = Field(ge=0)
    rank: int = Field(ge=0)
    """0-based position in its sub-question's list."""
    tokens: int = Field(ge=0)
    text: str
    """Rendered: `[n]`, the opening delimiter, metadata lines, the chunk text, the closing one."""


class AnswerContext(BaseModel):
    """The fragments given to the model; serializable (LangGraph state, P3)."""

    fragments: list[Fragment] = Field(default_factory=list)
    budget_tokens: int = Field(ge=0)
    tokens: int = Field(default=0, ge=0)
    duplicates: list[str] = Field(default_factory=list)
    """Chunk ids seen again after their first occurrence (one entry per extra occurrence)."""
    dropped: list[str] = Field(default_factory=list)
    """Chunk ids left out because they did not fit into the budget."""

    @property
    def text(self) -> str:
        return "\n\n".join(f.text for f in self.fragments)

    @property
    def chunks(self) -> list[Chunk]:
        """The fragments' chunks in number order: fragment n is `chunks[n - 1]`."""
        return [f.chunk for f in self.fragments]

    @property
    def suspicious(self) -> list[int]:
        """Numbers of the fragments flagged `suspicious_instruction`."""
        return [f.number for f in self.fragments if f.chunk.suspicious_instruction]

    def chunk_id(self, number: int) -> str | None:
        """Chunk id of fragment `number`, or None when there is no such fragment."""
        if 1 <= number <= len(self.fragments):
            return self.fragments[number - 1].chunk.chunk_id
        return None


def _metadata_value(chunk: Chunk, key: str) -> str:
    if key == "title_path":
        value: object = TITLE_SEPARATOR.join(chunk.title_path)
    elif key in Chunk.model_fields:
        value = getattr(chunk, key)
    else:
        value = chunk.metadata.get(key)
    if value is None:
        return ""
    if isinstance(value, list | tuple):
        value = TITLE_SEPARATOR.join(str(v) for v in value)
    # One line per key: a newline in a value cannot start a line of its own.
    return " ".join(str(value).split())


def render_fragment(number: int, chunk: Chunk, metadata_keys: Sequence[str]) -> str:
    """`[n]`, then metadata and text inside `<doc_fragment n="n" untrusted="true">` and
    `</doc_fragment>`; `suspicious_instruction="true"` is added for flagged chunks."""
    label = ' suspicious_instruction="true"' if chunk.suspicious_instruction else ""
    lines = [f"[{number}]", f'<{FRAGMENT_TAG} n="{number}" untrusted="true"{label}>']
    for key in metadata_keys:
        value = _metadata_value(chunk, key)
        if value:
            lines.append(f"{key}: {escape_delimiters(value)}")
    lines.append("")
    lines.append(escape_delimiters(chunk.text.strip("\n")))
    lines.append(FRAGMENT_CLOSE)
    return "\n".join(lines)


@dataclass
class _Candidate:
    chunk: Chunk
    sub_question: int
    rank: int
    tokens: int = 0

    def order(self) -> tuple[int, int]:
        """Context order: rank 1 of every sub-question first, then rank 2, ..."""
        return self.rank, self.sub_question

    def by_score(self) -> tuple[bool, float, int, int]:
        """Order for the leftover budget: scored chunks by score, then the rest by rank."""
        score = self.chunk.score
        return score is None, -(score or 0.0), self.rank, self.sub_question


def _candidates(
    chunk_lists: Sequence[Sequence[Chunk]],
) -> tuple[list[_Candidate], list[str]]:
    best: dict[str, _Candidate] = {}
    duplicates: list[str] = []
    for sub_question, chunks in enumerate(chunk_lists):
        for rank, chunk in enumerate(chunks):
            candidate = _Candidate(chunk, sub_question, rank)
            previous = best.get(chunk.chunk_id)
            if previous is None:
                best[chunk.chunk_id] = candidate
                continue
            duplicates.append(chunk.chunk_id)
            if candidate.order() < previous.order():
                best[chunk.chunk_id] = candidate
    return list(best.values()), duplicates


def build_context(
    chunk_lists: Sequence[Sequence[Chunk]],
    budget_tokens: int,
    *,
    metadata_keys: Sequence[str] = DEFAULT_METADATA_KEYS,
    count_tokens: TokenCounter = estimate_tokens,
) -> AnswerContext:
    """Numbered fragments from one ranked chunk list per sub-question (see the module doc).

    `metadata_keys` is the knowledge base's prompt whitelist (`metadata.llm`, s5-4): only these
    keys are shown, `title_path` joined with " > ".
    """
    candidates, duplicates = _candidates(chunk_lists)
    # Count with the widest number any fragment can get, so the final text never needs more.
    widest = max(len(candidates), 1)
    for candidate in candidates:
        rendered = render_fragment(widest, candidate.chunk, metadata_keys)
        candidate.tokens = count_tokens(rendered)

    chosen: list[_Candidate] = []
    used = 0
    sub_questions = max(len(chunk_lists), 1)
    share = budget_tokens // sub_questions
    for sub_question in range(sub_questions):
        own = sorted(
            (c for c in candidates if c.sub_question == sub_question), key=_Candidate.order
        )
        spent = 0
        for candidate in own:
            if spent + candidate.tokens > share:
                break
            chosen.append(candidate)
            spent += candidate.tokens
        used += spent

    picked = {id(c) for c in chosen}
    for candidate in sorted(
        (c for c in candidates if id(c) not in picked), key=_Candidate.by_score
    ):
        if used + candidate.tokens <= budget_tokens:
            chosen.append(candidate)
            picked.add(id(candidate))
            used += candidate.tokens

    fragments = [
        Fragment(
            number=number,
            chunk=c.chunk,
            sub_question=c.sub_question,
            rank=c.rank,
            tokens=c.tokens,
            text=render_fragment(number, c.chunk, metadata_keys),
        )
        for number, c in enumerate(sorted(chosen, key=_Candidate.order), start=1)
    ]
    return AnswerContext(
        fragments=fragments,
        budget_tokens=budget_tokens,
        tokens=used,
        duplicates=duplicates,
        dropped=[c.chunk.chunk_id for c in candidates if id(c) not in picked],
    )
