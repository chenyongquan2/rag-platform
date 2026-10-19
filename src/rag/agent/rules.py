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


def strip_format_chars(text: str) -> str:
    """Without invisible format characters (zero-width space, joiner, soft hyphen, direction
    marks): they carry no meaning for search and are how a word is made to look like another."""
    return "".join(c for c in text if unicodedata.category(c) != "Cf")


def normalise(text: str) -> str:
    """NFKC, no invisible format characters, runs of whitespace (newlines too) collapsed to one
    space, ends trimmed."""
    return _SPACES.sub(" ", strip_format_chars(unicodedata.normalize("NFKC", text))).strip()


# ---- protected tokens ------------------------------------------------------------------------

# ASCII mode: `\w` and `\b` stop at Chinese characters, so `transformers怎么` has the word
# `transformers` and nothing else.
_FLAGS = re.ASCII
_FENCE = re.compile(r"```[^\n`]*\n?(.*?)```", re.DOTALL)
_CODE_SPAN = re.compile(r"`([^`\n]{1,120})`")
# Quoted messages: the text between the quotes is one token. A single quote only opens a
# quotation after a non-word character ("it's" and "don't" are not quotations).
_QUOTED: tuple[re.Pattern[str], ...] = (
    re.compile(r'"([^"\n]{3,500})"'),
    re.compile(r"“([^”\n]{3,500})”"),
    re.compile(r"‘([^’\n]{3,500})’"),
    re.compile(r"「([^」\n]{2,500})」"),
    re.compile(r"『([^』\n]{2,500})』"),
    re.compile(r"(?<![\w'])'([^'\n]{3,500})'(?![\w'])", _FLAGS),
)
# A message longer than this is kept by its first characters only (the model may shorten or
# reflow the rest), and matched as a prefix.
PREFIX_CHARS = 40


class PrefixToken(str):
    """The first `PREFIX_CHARS` characters of a long quoted message: a rewrite has to contain
    these, not the rest. Every other token (identifiers, code spans, short messages) is whole."""

    __slots__ = ()


# Names made of parts: `Foo::bar`, `torch.nn.Module`, paths. In a fenced code block these are
# not asked for (the variable in `model.push_to_hub` means nothing); their parts are.
_COMPOUND: tuple[re.Pattern[str], ...] = tuple(
    re.compile(pattern, _FLAGS)
    for pattern in (
        # Foo::bar::baz
        r"\b[A-Za-z_]\w*(?:::[A-Za-z_]\w*)+",
        # torch.nn.Module, config.json (every part at least two characters: not "e.g.")
        r"\b[A-Za-z_]\w+(?:\.[A-Za-z_]\w+)+\b",
        # google/flan-t5-base, docs/source/en/index.md
        r"(?<![\w/])[\w.-]+(?:/[\w.-]+)+",
        # C:\Users\me\.cache
        r"(?<![\w])[A-Za-z]:\\[^\s\"'<>|?*]+",
    )
)
_ATOMIC: tuple[re.Pattern[str], ...] = tuple(
    re.compile(pattern, _FLAGS)
    for pattern in (
        # push_to_hub, MT_RET_ERR_PARTIAL, __init__
        r"(?<![A-Za-z0-9_])_*[A-Za-z0-9]+(?:_+[A-Za-z0-9]+)+_*(?![A-Za-z0-9_])",
        # TrainingArguments, fromPretrained, LoRA: a capital after a lower-case letter or digit
        r"\b[A-Za-z][A-Za-z0-9]*[a-z0-9][A-Z][A-Za-z0-9]*\b",
        # --push_to_hub, --max-new-tokens; -U, -m
        r"(?<![\w-])--[A-Za-z][\w-]*",
        r"(?<![\w-])-[A-Za-z]{1,2}(?![\w-])",
        # 4.36.0, v2.1, 3.10, 5.0rc1
        r"\bv?\d+(?:\.\d+){1,3}(?:[-.]?(?:rc|alpha|beta|dev|post|a|b)\d*)?\b",
        r"\bv\d+\b",
        r"\b0x[0-9A-Fa-f]+\b",
        # T5, MT5, GPT2, H100, E1234
        r"\b[A-Z]{1,5}\d+[A-Z0-9]*\b",
        # RLHF, PEFT, BERT (checked against shouting in `protected_tokens`)
        r"\b[A-Z]{3,}\b",
        # bert-base-uncased, gpt-4o, flan-t5-large, 8-bit (filtered by `_model_name`)
        r"\b[A-Za-z0-9][A-Za-z0-9]*(?:-[A-Za-z0-9.]+)+\b",
    )
)
_PATH_MARKS = re.compile(r"[\d._A-Z-]")
_ALL_CAPS = re.compile(r"[A-Z]{3,}$")
_DASHED = re.compile(r"[A-Za-z0-9][A-Za-z0-9]*(?:-[A-Za-z0-9.]+)+$")
_DOTTED_PAIR = re.compile(r"[a-z]+\.[A-Z][a-z]+$")
_MODEL_SUFFIXES = frozenset(
    [
        "base",
        "large",
        "small",
        "tiny",
        "medium",
        "xl",
        "xxl",
        "uncased",
        "cased",
        "chat",
        "instruct",
        "distilled",
    ]
)
# Module names whose `name.Class` is code ("torch.Tensor"), not a sentence missing a space.
_MODULES = frozenset(
    [
        "torch",
        "np",
        "pd",
        "tf",
        "nn",
        "os",
        "sys",
        "re",
        "json",
        "math",
        "transformers",
        "datasets",
        "accelerate",
        "peft",
        "trl",
        "diffusers",
        "tokenizers",
        "evaluate",
        "gradio",
        "huggingface_hub",
        "optimum",
        "onnx",
        "jax",
        "flax",
        "keras",
        "sklearn",
    ]
)
_TRAILING = ".,;:!?)]}"


def _is_plain_slash(token: str) -> bool:
    """`and/or`, `yes/no`: slashes between plain lower-case words are not paths or model ids."""
    return "/" in token and not _PATH_MARKS.search(token)


def _model_name(token: str) -> bool:
    """A hyphenated token that is a model name or a size ("bert-base-uncased", "gpt-4o",
    "8-bit"), not an ordinary compound ("fine-tuning", "state-of-the-art")."""
    if not _DASHED.match(token):
        return True  # another kind of token the hyphen pattern happened to match
    if any(c.isdigit() for c in token) or token.rsplit("-", 1)[-1].lower() in _MODEL_SUFFIXES:
        return True
    # A part that is an acronym or has a capital inside ("BLIP-Diffusion", "XLM-RoBERTa").
    return any(
        (len(part) >= 2 and part.isupper()) or any(c.isupper() for c in part[1:])
        for part in token.split("-")
    )


def _shouting(text: str) -> bool:
    """Most words in capitals: capitals mean nothing, no acronym can be told from the rest."""
    words = re.findall(r"[A-Za-z]{2,}", text)
    return len(words) >= 4 and sum(1 for w in words if w.isupper()) / len(words) >= 0.6


# Sizes and precisions ("8-bit", "int8", "fp16", "BF16") are spelled many ways; a rewrite that
# writes "8 bit" or "int8" has not lost an identifier.
_PRECISION = re.compile(r"(?:\d+-?bit|(?:fp|bf|int|float|tf)\d+)$", re.IGNORECASE)


def _keep(token: str, *, shouting: bool) -> bool:
    if _is_plain_slash(token) or _PRECISION.match(token):
        return False
    if _ALL_CAPS.match(token):
        return not shouting
    if "-" in token and "/" not in token and not token.startswith("-"):
        return _model_name(token)
    if _DOTTED_PAIR.match(token):  # "fails.Why": a sentence missing its space after the period
        return token.split(".", 1)[0] in _MODULES
    return True


def protected_tokens(text: str) -> list[str]:
    """The identifiers of `text` a rewrite must keep, in order of appearance, without repeats.

    - Code spans (the text inside backticks) and quoted messages count as one token each; only a
      quoted message longer than `PREFIX_CHARS` is kept by its first characters (`PrefixToken`,
      matched by them): identifiers and code spans are always whole.
    - A fenced code block is not a token: the identifiers inside it are, one by one.
    - Identifiers: snake_case, CamelCase, dotted names, `::` scopes, versions, flags (`--lr`,
      `-U`), paths (also Windows), acronyms (RLHF), model names (`bert-base-uncased`, `gpt-4o`).
    - Not identifiers: ordinary hyphenated words, capitals in a shouted question, `word.Word`
      with no space after the period.

    A token that is part of a longer one (`from_pretrained` inside `Auto.from_pretrained`) is
    dropped, because keeping the longer one keeps the shorter.
    """
    found: list[tuple[int, str]] = []
    fenced = normalise(" ".join(m.group(1) for m in _FENCE.finditer(text)))
    masked = normalise(_FENCE.sub(" ", text).replace("```", " "))

    def take(pattern: re.Pattern[str], *, message: bool) -> None:
        nonlocal masked
        for match in pattern.finditer(masked):
            token = next(g for g in match.groups() if g is not None).strip()
            if token:
                if message and len(token) > PREFIX_CHARS:
                    token = PrefixToken(token[:PREFIX_CHARS])
                found.append((match.start(), token))
        # What was taken is not searched again: no identifiers out of a quoted message.
        masked = pattern.sub(lambda m: " " * len(m.group(0)), masked)

    take(_CODE_SPAN, message=False)
    for pattern in _QUOTED:
        take(pattern, message=True)
    shouting = _shouting(masked)
    # The question's own text first, then its code blocks (their parts only), in order.
    for source, patterns, offset in (
        (masked, (*_ATOMIC, *_COMPOUND), 0),
        (fenced, _ATOMIC, len(masked) + 1),
    ):
        for pattern in patterns:
            for match in pattern.finditer(source):
                token = match.group(0)
                if "\\" in token:
                    token = token.rstrip(_TRAILING)
                if _keep(token, shouting=shouting):
                    found.append((offset + match.start(), token))
    ordered: list[str] = []
    for _, token in sorted(found):
        if token and token not in ordered:
            ordered.append(token)
    return [t for t in ordered if not any(t != other and t in other for other in ordered)]


def _present(token: str, haystack: str) -> bool:
    """`token` occurs in `haystack` as a whole: not inside a longer word, number, flag or
    name (`3.1` is not in `3.12`, `--lr` not in `--lr-scheduler`, `T5` not in `MT5`). A long
    quoted message (a prefix token) only has to start there."""
    end = "" if isinstance(token, PrefixToken) else r"(?!\w|[.\-]\w)"
    return re.search(rf"(?<![\w./\\-]){re.escape(token)}{end}", haystack, _FLAGS) is not None


def missing_tokens(tokens: Sequence[str], *texts: str) -> list[str]:
    """Tokens that appear in none of `texts` as whole tokens (case-sensitive, after
    normalisation)."""
    haystack = normalise(" ".join(texts))
    return [t for t in tokens if not _present(t, haystack)]


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


# ---- is a rewrite about the question? --------------------------------------------------------

# A rewrite is output of a model that read untrusted text (the question, earlier answers made
# from documents). Lexical overlap with the question is a weak signal: a good rewrite bridges
# vocabulary ("How do I make my model smaller?" -> "Reduce model size with quantization" shares
# one word in four), and a hijacked one may reuse the user's own words. So the gate is only this:
#
# - the output contains an instruction-like phrase (`_INJECTION`): "reveal the system prompt",
#   "ignore the previous instructions", "output the password" ...; or
# - almost none of its topic words occur in the question (without its instruction-like
#   sentences) and the history shown to the model: fewer than `GROUNDED_AT`. Measured: legitimate
#   rewrites, vocabulary bridges included, scored 0.2 and more (19 rewrites of 14 cases); the
#   real cheap-model rewrites of the P3-02 run 0.6 to 1.0; an unrelated output 0.
#
# What it cannot see: an output that keeps an honest word and adds a request. The defences that
# do not depend on this gate are the untrusted blocks of the rewrite prompt and the answer step
# being given the user's original question next to the standalone version.
GROUNDED_AT = 0.15

_ZH_STOP = frozenset("的了是在和与把将对给被让吗呢吧啊这那")
_LATIN_TERM = re.compile(r"[a-z0-9]+")
_INJECTION = re.compile(
    r"\b(?:ignore|disregard|forget|override)\b[^.!?\n]{0,40}"
    r"\b(?:instructions?|rules?|prompts?|messages?|guidelines)\b"
    r"|\b(?:reveal|leak|print|output|show|repeat|display|dump|disclose|expose)\b[^.!?\n]{0,30}"
    r"\b(?:system prompts?|instructions?|passwords?|secrets?|api[ _-]?keys?|credentials?)\b"
    r"|\b(?:you are now|pretend to be|act as|developer mode|jailbreak)\b"
    r"|(?:忽略|无视|忘掉)[^。！？\n]{0,8}(?:指令|规则|提示|之前|上面|以上|前面)"
    r"|(?:泄露|透露|输出|显示|打印)[^。！？\n]{0,8}(?:系统提示|提示词|密码|密钥)",
    re.IGNORECASE,
)
_SENTENCES = re.compile(r"(?<=[.!?。！？])\s*|\n+")


def injection_like(text: str) -> bool:
    """Does `text` contain an instruction aimed at a model (not a question about documents)?"""
    return _INJECTION.search(normalise(text)) is not None


def without_instructions(text: str) -> str:
    """`text` without the sentences that are instruction-like: what is left of a question that
    carries an injected instruction."""
    parts = _SENTENCES.split(text)
    return " ".join(part for part in parts if part and not injection_like(part))


def topic_terms(text: str) -> set[str]:
    """The words of `text` that say what it is about: Latin words of three letters or more that
    are not filler (underscores split: `push_to_hub` is push, hub), and every Han character
    that is not a question word or a particle."""
    lowered = normalise(text).lower()
    terms = {c for c in _HAN.findall(_FILLER_ZH.sub("", lowered)) if c not in _ZH_STOP}
    latin = _HAN.sub(" ", lowered)
    for word in _LATIN_TERM.findall(latin):
        if len(word) >= 3 and not word.isdigit() and word not in _FILLER:
            terms.add(word)
    return terms


def _near(word: str, terms: set[str]) -> bool:
    """`word` is in `terms`, or a form of one: the same first four letters (difference,
    differences), or one is the other with up to four letters added (pad, padding)."""
    if word in terms:
        return True
    for other in terms:
        short, long_ = sorted((word, other), key=len)
        if len(short) >= 3 and long_.startswith(short) and len(long_) - len(short) <= 4:
            return True
        if len(word) >= 5 and len(other) >= 5 and word[:4] == other[:4]:
            return True
    return False


def grounding(output: str, *sources: str) -> float:
    """The share of `output`'s topic words that occur (or have a near form) in `sources`; 1.0
    when `output` has none."""
    wanted = topic_terms(output)
    if not wanted:
        return 1.0
    known: set[str] = set()
    for source in sources:
        known |= topic_terms(source)
    return sum(1 for word in wanted if _near(word, known)) / len(wanted)


def off_topic(output: str, *, question: str, history: str = "") -> bool:
    """Is a rewrite (a query or a standalone question) to be replaced by the original question?
    See the comment above for what this does and does not catch."""
    if not topic_terms(output):
        return False
    if injection_like(output):
        return True
    return grounding(output, without_instructions(question), history) < GROUNDED_AT


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
