# ruff: noqa: RUF001 (Chinese question marks and punctuation are what these rules match)
"""Rules of the route and rewrite steps: no model, no state (design s8-1, s13-3; spec P3-02).

Everything here is a pure function of the question (and whether the session has history), so the
cheap path of `ask_docs` never calls a model and every rule is tested on its own:

- `normalise`: Unicode NFKC and one space between words (full-width punctuation and digits
  become ASCII, the form the index and the reranker see).
- `protected_tokens`: identifiers, code spans, versions, flags, paths and quoted messages that a
  rewrite must never lose (s8-1: "protect proper nouns"). `missing_tokens` checks a rewrite.
- `is_question_form`, `looks_like_follow_up`: whether a rewrite is worth a model call.
- `clarity`, `multi_hop_hint`: signals for the route node. They answer "clear", "vague" or
  "undecided" (and "no", "maybe" or "yes") so the model is asked only about what rules cannot
  settle.

The rules are meant for English and Chinese (s8-4). Neither translates anything.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence
from typing import Literal

from rag.answer.prompts import AnswerLanguage

Clarity = Literal["clear", "vague", "undecided"]
MultiHop = Literal["no", "maybe", "yes"]

# ---- normalisation ---------------------------------------------------------------------------

_SPACES = re.compile(r"\s+")


def normalise(text: str) -> str:
    """NFKC, runs of whitespace (newlines too) collapsed to one space, ends trimmed."""
    return _SPACES.sub(" ", unicodedata.normalize("NFKC", text)).strip()


# ---- protected tokens ------------------------------------------------------------------------

# ASCII mode: `\w` and `\b` stop at Chinese characters, so `transformers怎么` has the word
# `transformers` and nothing else.
_FLAGS = re.ASCII
_CODE_SPAN = re.compile(r"`([^`\n]{1,120})`")
_QUOTED = re.compile(r'"([^"\n]{3,80})"|“([^”\n]{3,80})”')
_PLAIN: tuple[re.Pattern[str], ...] = tuple(
    re.compile(pattern, _FLAGS)
    for pattern in (
        # Foo::bar::baz
        r"\b[A-Za-z_]\w*(?:::[A-Za-z_]\w*)+",
        # torch.nn.Module, config.json (every part at least two characters: not "e.g.")
        r"\b[A-Za-z_]\w+(?:\.[A-Za-z_]\w+)+\b",
        # push_to_hub, MT_RET_ERR_PARTIAL, __init__
        r"(?<![A-Za-z0-9_])_*[A-Za-z0-9]+(?:_+[A-Za-z0-9]+)+_*(?![A-Za-z0-9_])",
        # TrainingArguments, fromPretrained, LoRA: a capital after a lower-case letter or digit
        r"\b[A-Za-z][A-Za-z0-9]*[a-z0-9][A-Z][A-Za-z0-9]*\b",
        # --push_to_hub, --max-new-tokens
        r"(?<![\w-])--[A-Za-z][\w-]*",
        # 4.36.0, v2.1, 3.10, 5.0rc1
        r"\bv?\d+(?:\.\d+){1,3}(?:[-.]?(?:rc|alpha|beta|dev|post|a|b)\d*)?\b",
        r"\bv\d+\b",
        r"\b0x[0-9A-Fa-f]+\b",
        # T5, MT5, GPT2, H100, E1234
        r"\b[A-Z]{1,5}\d+[A-Z0-9]*\b",
        # google/flan-t5-base, docs/source/en/index.md
        r"(?<![\w/])[\w.-]+(?:/[\w.-]+)+",
    )
)
_PATH_MARKS = re.compile(r"[\d._A-Z-]")


def _is_plain_slash(token: str) -> bool:
    """`and/or`, `yes/no`: slashes between plain lower-case words are not paths or model ids."""
    return "/" in token and not _PATH_MARKS.search(token)


def protected_tokens(text: str) -> list[str]:
    """The identifiers of `text` a rewrite must keep, in order of appearance, without repeats.

    Code spans (the text inside backticks) and double-quoted messages count as one token each;
    a token that is part of a longer one (`from_pretrained` inside `Auto.from_pretrained`) is
    dropped, because keeping the longer one keeps the shorter.
    """
    text = normalise(text)
    found: list[tuple[int, str]] = []
    for match in _CODE_SPAN.finditer(text):
        found.append((match.start(), match.group(1).strip()))
    for match in _QUOTED.finditer(text):
        found.append((match.start(), (match.group(1) or match.group(2)).strip()))
    for pattern in _PLAIN:
        for match in pattern.finditer(text):
            token = match.group(0)
            if not _is_plain_slash(token):
                found.append((match.start(), token))
    ordered: list[str] = []
    for _, token in sorted(found):
        if token and token not in ordered:
            ordered.append(token)
    return [t for t in ordered if not any(t != other and t in other for other in ordered)]


def missing_tokens(tokens: Sequence[str], *texts: str) -> list[str]:
    """Tokens that appear in none of `texts` (exact, case-sensitive, after normalisation)."""
    haystack = normalise(" ".join(texts))
    return [t for t in tokens if t not in haystack]


# ---- what kind of question is it -------------------------------------------------------------

_HAN = re.compile(r"[㐀-䶿一-鿿豈-﫿]")
_WORD = re.compile(r"[A-Za-z][A-Za-z']*")

_QUESTION_START = re.compile(
    r"^(?:how|what|why|when|where|which|who|whom|whose|can|could|does|do|did|is|are|was|were|"
    r"should|would|will|may|might|has|have|am)\b",
    re.IGNORECASE,
)
_QUESTION_ZH = re.compile(r"怎么|怎样|如何|什么|为什么|为何|吗|呢|哪|是否|能否|可否|多少|几个")

_FOLLOW_UP_START = re.compile(
    r"^(?:and|also|then|so|but|what about|how about|why not|what if|and what|"
    r"还有|那|然后|接着|另外|而且|如果)",
    re.IGNORECASE,
)

# "that" is also the relative pronoun ("a model that can solve ..."): it points at something
# only at the end of the question or before a noun like "one", "error", "function".
_DEICTIC_EN = re.compile(
    r"\b(?:it|its|this|these|those|the above)\b|\bsuch\b(?!\s+as\b)"
    r"|\bthat(?:\s+(?:one|error|issue|problem|function|class|method|model|file|code|option|"
    r"setting|parameter|argument)\b|\s*[?.!]*\s*$)",
    re.IGNORECASE,
)
_DEICTIC_ZH = re.compile(r"这个|这些|这种|那个|那些|那种|它们|它|他们|该|此|上面|上述|刚才|之前")

# Words that carry no topic: a question made only of these says nothing about the documents.
_FILLER_WORDS = (
    "a an the is are was were be been being am do does did done doing have has had how what why "
    "when where which who whom whose i you we he she it its this that these those they them my "
    "your our me us can could should would will shall may might must to of in on at for with "
    "from by about as into and or but if then so not no yes please help need want like get got "
    "make use using used work works working doesn't don't didn't isn't aren't can't won't "
    "error errors issue issues problem problems bug bugs fail fails failed failing broken wrong "
    "fix fixed thing things stuff something anything"
)
_FILLER = frozenset(_FILLER_WORDS.split())


_FILLER_ZH = re.compile(
    r"怎么办|怎么回事|怎么|怎样|如何|什么|为什么|为何|是否|能否|可以|请问|帮我|帮忙|"
    r"问题|报错|错误|出错|不行|不能用|失败|吗|呢|吧|啊"
)


def _content_units(text: str) -> int:
    """Topic words of `text`: English words that are not filler, and two Han characters count
    as one word (a Chinese word is about two characters), question and complaint words left out."""
    words = [w.lower() for w in _WORD.findall(text)]
    han = len(_HAN.findall(_FILLER_ZH.sub("", text)))
    return sum(1 for w in words if w not in _FILLER) + han // 2


def has_deictic_reference(text: str) -> bool:
    """The question points at something it does not name ("this error", "它")."""
    return bool(_DEICTIC_EN.search(text) or _DEICTIC_ZH.search(text))


def is_question_form(text: str) -> bool:
    """A question (not a statement or a bare keyword query): a question mark, an English
    question or auxiliary verb first, or a Chinese question word."""
    text = normalise(text)
    return bool(
        text.endswith(("?", "？")) or _QUESTION_START.match(text) or _QUESTION_ZH.search(text)
    )


def looks_like_follow_up(text: str) -> bool:
    """May this only make sense with the earlier turns: it starts like a continuation, points
    at something unnamed, or has at most three topic words."""
    text = normalise(text)
    return bool(
        _FOLLOW_UP_START.match(text) or has_deictic_reference(text) or _content_units(text) <= 3
    )


# A capitalised word that does not start the sentence: "Hugging Face", "Decision Transformer".
_NAME = re.compile(r"(?<=\s)[A-Z][A-Za-z0-9]+")


def clarity(text: str, *, has_history: bool) -> Clarity:
    """Can the question be searched as it stands?

    - `clear`: it names something (an identifier, a version, or a capitalised name such as
      "Hugging Face"), or it has enough topic words and points at nothing unnamed, or the
      session has history (a follow-up is rewritten with the earlier turns, not asked back).
    - `vague`: nothing to search for ("help", "it doesn't work", "怎么办").
    - `undecided`: it points at something unnamed ("this model") but has some topic words; the
      route asks the small model, when the knowledge base allows it.
    """
    text = normalise(text)
    if has_history or protected_tokens(text) or _NAME.search(text):
        return "clear"
    units = _content_units(text)
    if units == 0:
        return "vague"
    if has_deictic_reference(text):
        return "vague" if units <= 1 else "undecided"
    return "clear"


# A comparison needs one lookup per side. "compared to" is not one: "the speed-up of X compared
# to Y" is a single fact.
_COMPARE = re.compile(
    r"\b(?:differences?\s+between|compare|comparison|versus|vs\.?)\b|区别|对比|比较|分别|各自|不同",
    re.IGNORECASE,
)
_TWO_QUESTIONS = re.compile(
    r"\b(?:what|how|why|which|where|when)\b[^?？]*?"
    r"\b(?:and then|and also|and|then|after that|before that)\b[^?？]*?"
    r"\b(?:what|how|why|which|where|when)\b",
    re.IGNORECASE,
)
_TWO_QUESTIONS_ZH = re.compile(
    r"(?:怎么|如何|什么|为什么|哪).{1,40}(?:然后|之后|以及|并且|同时).{0,40}"
    r"(?:怎么|如何|什么|为什么|哪)"
)
# Two actions joined by "and": "fine-tune a model and push it", not "Q, K and V layers".
_ACTIONS = (
    "push|save|load|use|convert|train|fine-tune|finetune|evaluate|deploy|upload|share|export|run|"
    "install|create|build|test|add|set|configure|enable|then|also|how|what|why|which|where|when"
)
_JOINED = re.compile(rf"\band\s+(?:{_ACTIONS})\b|\bas well as\b|并且|而且|同时|然后|之后", re.I)


def multi_hop_hint(text: str) -> MultiHop:
    """Does the question need two or more separate lookups?

    - `yes`: a comparison, or two question clauses joined by "and", "then", "然后" and the like.
    - `maybe`: it joins two parts but is not clearly two questions.
    - `no`: a short or single-part question.

    `yes` and `maybe` only say that splitting is worth considering; the rewrite model may still
    return one query.
    """
    text = normalise(text)
    if _COMPARE.search(text) or _TWO_QUESTIONS.search(text) or _TWO_QUESTIONS_ZH.search(text):
        return "yes"
    if _JOINED.search(text) and _content_units(text) >= 6:
        return "maybe"
    return "no"


# ---- the question asked back -----------------------------------------------------------------

_CLARIFY_QUESTION: dict[AnswerLanguage, str] = {
    "en": (
        "Could you be more specific? Name the library, function, class or error message "
        "you mean, so that I can search the documentation for it."
    ),
    "zh": "能说得再具体一点吗？请写出你指的库、函数、类或者报错信息，我才能去文档里查。",
}


def default_clarification(language: AnswerLanguage) -> str:
    """The question asked back when rules (or the model, without a better one) find the
    question too vague to search."""
    return _CLARIFY_QUESTION[language]
