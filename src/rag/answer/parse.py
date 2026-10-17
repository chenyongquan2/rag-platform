# ruff: noqa: RUF001, RUF002 (Chinese markers and punctuation are what this module parses)
"""Parse a numbered plain-text answer and check its citations, level 1 (design s11-1, s11-2).

The model is asked to end every sentence with `[n]`. It does not always: it writes `【1】`,
`[1,2]`, `[1][2]`, a marker after the full stop, list items. `parse_answer` accepts these
variants and splits the text into sentences, each with the marker numbers it carries;
`check_citations` maps the numbers to the fragments of this answer's context, removes and counts
numbers that match none, and counts sentences left without a citation as unsupported.

Sentence ends: `。！？` always; `.!?` when followed by a space, a line end, the end of the text
or a marker (so `1.5`, `config.json` and `e.g.` do not end a sentence); a line break. Markers
right after a sentence end still belong to that sentence. Inline code, fenced code and indented
code lines are kept whole, so `x[0]` in code is not a marker. An ASCII bracket right after an
ASCII letter, digit, `_`, `(`, `)`, `[`, `{`, `}`, `=` or a code bracket (`logits[1]`, `f(x)[1]`,
`Size([1, 2])`, `x=[1, 2]`) is code too and stays in the text, unless Chinese punctuation follows
(`窗口是 127[2]。`); right after Chinese text, after another marker, or written with full-width
brackets it is a marker (`中文[1][2]`, `D［3］`).

The answer is "not found" when it is the fixed sentence, or when it has an uncited not-found
statement ("The documentation does not contain ...", "手册里没有找到...") and no other sentence
carries a marker ("I'm sorry. The documentation does not contain the answer."). A not-found
statement with a marker is an answer ("The documentation does not mention X [1].").
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from rag.answer.context import AnswerContext
from rag.answer.prompts import NOT_FOUND
from rag.types import AnswerSentence, Citation

_MAX_RANGE = 50  # [1-3] expands; a longer "range" is read as its two end numbers

_NUMBER = r"\d{1,4}(?:\s*[-–~～]\s*\d{1,4})?"
_PREFIX = r"(?:(?:fragments?|sources?|docs?|refs?|片段|来源|资料)\s*)?"
_MARKER = rf"[\[【［]\s*{_PREFIX}{_NUMBER}(?:\s*[,，、;；]\s*{_PREFIX}{_NUMBER})*\s*[\]】］]"
_MARKER_RE = re.compile(_MARKER, re.IGNORECASE)
_TOKEN = re.compile(
    r"(?P<indented>^(?: {4,}|\t)[^\n]*)"
    r"|(?P<fence>```.*?(?:```|\Z))"
    r"|(?P<code>`[^`\n]+`)"
    rf"|(?P<marker>{_MARKER})"
    r"|(?P<newline>\n)"
    r"|(?P<end>[。！？]+[”’）」』]*|[.!?]+[\"'”’)）」』]*(?=\s|\Z|[\[【［]))"
    r"|(?P<text>[^`\n\[【［。！？.!?]+|.)",
    re.DOTALL | re.IGNORECASE | re.MULTILINE,
)
# The character right before a bracket that makes the bracket code, not a marker.
_CODE_BEFORE_BRACKET = re.compile(r"[A-Za-z0-9_()\[\]{}=]")
# Markers, then Chinese punctuation: `127[2]。` in a Chinese answer is a citation, not code.
_CJK_PUNCT_AFTER = re.compile(rf"(?:\s*{_MARKER})*\s*[。！？，；：、]", re.IGNORECASE)
_LIST_MARK = re.compile(r"^[ \t]*(?:[-*+•]|\d{1,3}[.)、]|#{1,6})[ \t]+", re.MULTILINE)
_ABBREVIATION = re.compile(r"(?:^|\W)(?:e\.g|i\.e|vs|cf|approx|incl|resp|fig|eq)$", re.IGNORECASE)
_SPACE_BEFORE_PUNCT = re.compile(r"[ \t]+(?=[.!?。！？,，;；:：)）])")
_SPACES = re.compile(r"[ \t]{2,}")
_WORD_CHAR = re.compile(r"\w")
_NOT_FOUND_NOISE = re.compile(r"[\s\"'“”‘’「」『』`*_.。!！]")
# A sentence saying the fragments have no answer, on the normalized key (no spaces, lower case).
_NOT_FOUND_STATEMENT = re.compile(
    r"^(?:the)?(?:provided)?(?:documentation|docs|manual|fragments?|context|documents?)"
    r"(?:does|do)not(?:contain|include|provide|mention|say|cover)"
    r"|^(?:手册|文档|片段|资料)(?:里|中)?(?:没有|并没有|未)(?:找到|提到|提供|包含|说明|涉及)"
)


@dataclass(frozen=True)
class ParsedSentence:
    text: str
    """The sentence without its markers."""
    markers: tuple[int, ...]
    """Marker numbers in the order written, repeats removed (valid or not)."""


@dataclass(frozen=True)
class ParsedAnswer:
    not_found: bool
    """The answer is the fixed not-found sentence (either language)."""
    sentences: tuple[ParsedSentence, ...]


@dataclass(frozen=True)
class CheckedAnswer:
    """Citation check level 1 of one parsed answer against its context (s11-2)."""

    sentences: list[AnswerSentence]
    removed_markers: list[int]
    """Numbers that matched no fragment, removed (in order, once per occurrence)."""
    unsupported_sentences: int

    @property
    def has_valid_citation(self) -> bool:
        return any(s.citations for s in self.sentences)


def _not_found_key(text: str) -> str:
    return _NOT_FOUND_NOISE.sub("", _MARKER_RE.sub("", text)).casefold()


_NOT_FOUND_KEYS = frozenset(_not_found_key(sentence) for sentence in NOT_FOUND.values())


def _is_not_found_statement(text: str) -> bool:
    key = _not_found_key(text)
    return key in _NOT_FOUND_KEYS or bool(_NOT_FOUND_STATEMENT.match(key))


def _is_not_found_statement_sentence(sentence: ParsedSentence) -> bool:
    return not sentence.markers and _is_not_found_statement(sentence.text)


def _numbers(marker: str) -> list[int]:
    numbers: list[int] = []
    for start, end in re.findall(r"(\d+)(?:\s*[-–~～]\s*(\d+))?", marker):
        first = int(start)
        last = int(end) if end else first
        if first <= last and last - first <= _MAX_RANGE:
            numbers.extend(range(first, last + 1))
        else:
            numbers.extend([first, last])
    return numbers


def _clean(text: str) -> str:
    text = _SPACE_BEFORE_PUNCT.sub("", text)
    return _SPACES.sub(" ", text).strip()


class _Sentences:
    """Collects sentences while the tokens of one answer are read."""

    def __init__(self) -> None:
        self.sentences: list[ParsedSentence] = []
        self.parts: list[str] = []
        self.markers: list[int] = []
        self.ended = False

    def flush(self) -> None:
        text = _clean("".join(self.parts))
        markers = tuple(dict.fromkeys(self.markers))
        if _WORD_CHAR.search(text):
            self.sentences.append(ParsedSentence(text, markers))
        elif markers and self.sentences:  # markers with no words of their own: previous sentence
            last = self.sentences[-1]
            merged = tuple(dict.fromkeys((*last.markers, *markers)))
            self.sentences[-1] = ParsedSentence(last.text, merged)
        self.parts, self.markers, self.ended = [], [], False

    def add_text(self, text: str) -> None:
        if self.ended:
            if not text.strip():
                return
            self.flush()
        self.parts.append(text)


def parse_answer(text: str) -> ParsedAnswer:
    """Sentences and marker numbers of an answer; `not_found` for the fixed sentence."""
    if _not_found_key(text) in _NOT_FOUND_KEYS:
        return ParsedAnswer(not_found=True, sentences=())
    collector = _Sentences()
    source = _LIST_MARK.sub("", text.strip())
    previous_kind: str | None = None
    for match in _TOKEN.finditer(source):
        kind, value = match.lastgroup, match.group()
        if kind == "marker" and _is_code_bracket(source, match, previous_kind):
            kind = "text"
        previous_kind = kind
        if kind == "marker":
            collector.markers.extend(_numbers(value))
        elif kind == "newline":
            collector.ended = True
        elif kind == "end":
            collector.parts.append(value)
            sentence = "".join(collector.parts)[: -len(value)]
            if not (value.startswith(".") and _ABBREVIATION.search(sentence)):
                collector.ended = True
        else:
            collector.add_text(value)
    collector.flush()

    # Only an uncited sentence can be a not-found statement: "The documentation does not mention
    # X [1]." is an answer backed by fragment 1.
    statements = [s for s in collector.sentences if _is_not_found_statement_sentence(s)]
    kept = [s for s in collector.sentences if not _is_not_found_statement_sentence(s)]
    if statements and not any(s.markers for s in kept):
        return ParsedAnswer(not_found=True, sentences=())
    return ParsedAnswer(not_found=False, sentences=tuple(kept))


def _is_code_bracket(source: str, match: re.Match[str], previous_kind: str | None) -> bool:
    """An ASCII bracket right after code characters (`logits[1]`) is code, unless Chinese
    punctuation follows it (`127[2]。`). Full-width brackets and a bracket right after a marker
    are always markers."""
    start = match.start()
    if start == 0 or previous_kind == "marker" or not match.group().startswith("["):
        return False
    if not _CODE_BEFORE_BRACKET.match(source[start - 1]):
        return False
    return not _CJK_PUNCT_AFTER.match(source, match.end())


def check_citations(parsed: ParsedAnswer, context: AnswerContext) -> CheckedAnswer:
    """Map marker numbers to this context's chunks; drop and count numbers outside it."""
    sentences: list[AnswerSentence] = []
    removed: list[int] = []
    unsupported = 0
    for sentence in parsed.sentences:
        citations: list[Citation] = []
        for number in sentence.markers:
            chunk_id = context.chunk_id(number)
            if chunk_id is None:
                removed.append(number)
            else:
                citations.append(Citation(marker=number, chunk_id=chunk_id))
        if not citations:
            unsupported += 1
        sentences.append(AnswerSentence(text=sentence.text, citations=citations))
    return CheckedAnswer(sentences, removed, unsupported)
