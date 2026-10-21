# ruff: noqa: RUF001 (the Chinese questions use full-width punctuation)
"""The cache test set: pairs of requests that must not share a cache entry (counter-examples) and
pairs that should (positive), made by fixed rules from the question sets (design s12-4; spec P5-01).

    pairs = build_pairs({"dev": [(qid, question), ...], "holdout": [...]})

Counter-example categories (each question gets at most `PER_QUESTION` of them, chosen by the hash
of its id, so the same input always gives the same file):

- `identifier_swap`: an identifier (snake_case, CamelCase, dotted name, flag, model name) is
  replaced by another one found in a different question of the same split;
- `identifier_case`: the same identifier written with another capital letter (`LoRA`, `lora`);
- `version_bump`, `number_swap`: a version, a number or a number word is changed;
- `negation_flip`: a negation is added after the first auxiliary verb, or the first one removed;
- `antonym_swap`: one word is replaced by its opposite (`push` and `pull`, `enable` and `disable`);
- `aspect_change`: the same identifier, another thing asked about it (parameters, return value,
  error, example): the entity guard cannot tell these apart (design s12-3), only distance can;
- `other_kb`, `other_acl`, `other_filters`, `other_top_k`, `other_language_setting`,
  `other_generation`: the same question under another knowledge base, ACL groups, filters, top_k,
  language setting or data generation (a pair with `b_ctx`);
- `other_language`: a question and its hand-written Chinese translation.

Positive categories (a variant the key's normaliser is meant to merge): `identical`, `whitespace`,
`punctuation`, `case` (only for questions without an identifier), `invisible` (zero-width
characters, full-width letters).

The dev part (public questions and synthetic dev) is where the threshold is chosen; the holdout
part is only read for the final report.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from rag.agent.rules import protected_tokens
from rag.cache.keys import cache_key, make_fields
from rag.cache.semantic import entity_signature, question_guard

PER_QUESTION = 3
"""Counter-examples made from one question at most (they are correlated, so more would only
make the interval look narrower than it is)."""

DEFAULT_CTX: dict[str, Any] = {
    "kb": "hf_docs",
    "acl": ["public"],
    "filters": {},
    "top_k": 8,
    "language_setting": "auto",
    "generation": 7,
}

# One word and its opposite, both directions; matched as whole words, first occurrence.
ANTONYMS = [
    ("push", "pull"),
    ("upload", "download"),
    ("load", "save"),
    ("enable", "disable"),
    ("create", "delete"),
    ("add", "remove"),
    ("encode", "decode"),
    ("start", "stop"),
    ("train", "evaluate"),
    ("input", "output"),
    ("maximum", "minimum"),
    ("increase", "decrease"),
    ("first", "last"),
    ("before", "after"),
    ("default", "custom"),
    ("public", "private"),
    ("install", "uninstall"),
    ("local", "remote"),
    ("encoder", "decoder"),
    ("large", "small"),
]
NUMBER_WORDS = {
    "one": "two",
    "two": "three",
    "three": "four",
    "four": "five",
    "five": "six",
    "six": "seven",
    "ten": "twelve",
}
ASPECTS = [
    "What parameters does {x} take?",
    "What does {x} return?",
    "What error does {x} raise?",
    "Show a code example that uses {x}.",
]
AUXILIARY = re.compile(r"\b(does|do|did|is|are|was|were|can|should|will|has|have)\b")
_VERSION = re.compile(r"^(v?)(\d+(?:\.\d+)+)(.*)$")
_LONE_NUMBER = re.compile(r"(?<![\w.\-])\d+(?![\w.\-])")

ZH_TRANSLATIONS: dict[str, str] = {
    "huggingface_doc_qa_eval:002": "如何在 Hugging Face Hub 上认领一篇论文的作者身份？",
    "huggingface_doc_qa_eval:003": "Datasets server API 中 /healthcheck 端点的用途是什么？",
    "huggingface_doc_qa_eval:006": "Diffusers 库的用途是什么？",
    "huggingface_doc_qa_eval:009": "`gradio.Blocks` API 的用途是什么？",
    "huggingface_doc_qa_eval:012": "`roberta-large-mnli` 检查点执行什么任务？",
    "huggingface_doc_qa_eval:013": "Hugging Face 用什么服务替代 Inference API 的付费档？",
    "huggingface_doc_qa_eval:015": "HuggingFace 团队的软件采用哪种许可证发布？",
    "huggingface_doc_qa_eval:023": "Gradio 可以用来做什么？",
    "huggingface_doc_qa_eval:025": "在 Hugging Face Endpoints 里哪里可以查看 Endpoint 的日志？",
    "huggingface_doc_qa_eval:027": "Hugging Face Hub 上 `create_repo` 函数默认创建哪种仓库类型？",
    "huggingface_doc_qa_eval:031": "Hugging Face 拥有哪种安全认证？",
    "huggingface_doc_qa_eval:034": "不使用 Git-LFS 同步到 HF Spaces 的文件大小上限是多少？",
    "huggingface_doc_qa_eval:035": "介绍 ByT5 模型的论文标题是什么？",
    "huggingface_doc_qa_eval:038": "🧨 Diffusers 教程的用途是什么？",
    "huggingface_doc_qa_eval:045": "Diffusers 库中 Safety Checker 的用途是什么？",
    "huggingface_doc_qa_eval:048": "Textual Inversion 的用途是什么？",
    "huggingface_doc_qa_eval:051": "如何在 Unity 项目中安装 Hugging Face Unity API？",
    "huggingface_doc_qa_eval:053": "Transformers 库的情感分析 pipeline 默认使用哪个检查点？",
    "huggingface_doc_qa_eval:055": "PyTorch 提供哪个命令行模块来在多张 GPU 上运行脚本？",
    "huggingface_doc_qa_eval:063": "在 Gradio 中用哪个参数保证同一行的元素高度相同？",
    "synthetic:10b77d46c158": "运行什么命令可以在本地预览 datasets 文档？",
    "synthetic:17f35ab9db62": "哪个 Datasets Server 端点返回数据集的 Parquet URL 的 JSON 列表？",
    "synthetic:1bdae637371d": "哪种协议用来加密发送到 Hugging Face Inference Endpoints 的数据？",
    "synthetic:28ffbdbbe610": "StableDiffusionPipeline 默认使用哪个调度器？",
    "synthetic:3312332c8a8f": "GLUE MRPC 数据集有多少个类别？",
    "synthetic:43449c1287f1": "MRPC 数据集属于哪个基准？",
    "synthetic:4a27748ad1a3": "运行什么命令可以获取 🤗 Transformers 的环境信息？",
    "synthetic:5af99050a0bb": "BigBirdPegasus 使用哪种分词器？",
    "synthetic:61c5805f57b4": "使用 Competition MATH 指标需要什么依赖？",
    "synthetic:679b9c890236": "哪个预分词器按空格和标点切分输入？",
}

# Metadata pairs: the question is the same, one thing of the request differs.
CONTEXT_CATEGORIES: dict[str, dict[str, Any]] = {
    "other_kb": {"kb": "mt5"},
    "other_acl": {"acl": ["public", "hr"]},
    "other_filters": {"filters": {"language": "en"}},
    "other_top_k": {"top_k": 5},
    "other_language_setting": {"language_setting": "zh"},
    "other_generation": {"generation": 8},
}
CONTEXT_PAIRS_PER_CATEGORY = {"dev": 8, "holdout": 5}


@dataclass
class Pair:
    pair_id: str
    split: str
    kind: str  # counter | positive
    category: str
    source_qid: str
    a: str
    b: str
    b_ctx: dict[str, Any] = field(default_factory=dict)
    """Differences of the second request's context from `DEFAULT_CTX` (the first has none)."""
    expect_share: bool = False
    """The two requests should share an L1 entry."""


def _h(*parts: str) -> int:
    return int(hashlib.sha256(":".join(parts).encode("utf-8")).hexdigest()[:12], 16)


def _first_replace(text: str, old: str, new: str) -> str:
    return text.replace(old, new, 1)


# ---- identifier helpers ----------------------------------------------------------------------


def _kind_of(token: str) -> str:
    if token.startswith("-"):
        return "flag"
    if _VERSION.match(token):
        return "version"
    if "/" in token or "\\" in token:
        return "path"
    if "." in token:
        return "dotted"
    if "_" in token:
        return "snake"
    if re.fullmatch(r"[A-Za-z]+-[A-Za-z0-9.\-]+", token):
        return "model"
    if re.search(r"[a-z][A-Z]", token):
        return "camel"
    return "other"


def identifiers(question: str) -> list[str]:
    return [str(t) for t in protected_tokens(question)]


_PRODUCTS = frozenset(
    {
        "PyTorch",
        "TensorFlow",
        "WordPiece",
        "HuggingFace",
        "GitHub",
        "JavaScript",
        "TypeScript",
        "OpenVINO",
        "DeepSpeed",
        "SpeechBrain",
        "ModelScope",
        "LoRA",
        "QLoRA",
        "BitsAndBytes",
    }
)
"""Product and method names that look like CamelCase identifiers but are not code: no aspect
question ("What parameters does PyTorch take?") is made about them."""


def _swappable(question: str) -> list[str]:
    """Identifiers of `question` that may be replaced: not versions, not paths."""
    return [
        t
        for t in identifiers(question)
        if _kind_of(t) in {"snake", "camel", "dotted", "flag", "model"}
    ]


# ---- the mutations ---------------------------------------------------------------------------


def identifier_swap(question: str, pool: Sequence[str], seed: str) -> str | None:
    found = _swappable(question)
    if not found:
        return None
    old = found[_h(seed, "which") % len(found)]
    candidates = [
        t for t in pool if t not in question and _kind_of(t) == _kind_of(old) and t != old
    ]
    if not candidates:
        return None
    new = candidates[_h(seed, "with") % len(candidates)]
    return _first_replace(question, old, new)


def identifier_case(question: str, seed: str) -> str | None:
    for token in _swappable(question):
        if token in _PRODUCTS:
            continue
        if _kind_of(token) in {"camel", "model", "snake", "dotted"} and token[0].isalpha():
            flipped = (token[0].lower() if token[0].isupper() else token[0].upper()) + token[1:]
            if flipped != token:
                return _first_replace(question, token, flipped)
    return None


def version_bump(question: str, seed: str) -> str | None:
    for token in identifiers(question):
        match = _VERSION.match(token)
        if match:
            prefix, numbers, rest = match.groups()
            parts = numbers.split(".")
            parts[-1] = str(int(parts[-1]) + 1)
            return _first_replace(question, token, f"{prefix}{'.'.join(parts)}{rest}")
    return None


def number_swap(question: str, seed: str) -> str | None:
    tokens = " ".join(identifiers(question))
    for match in _LONE_NUMBER.finditer(question):
        if match.group(0) in tokens.split():
            continue
        value = int(match.group(0))
        new = str(value + 1 if value < 100 else value * 2)
        return question[: match.start()] + new + question[match.end() :]
    words = "|".join(NUMBER_WORDS)
    for word in re.finditer(rf"\b({words})\b", question, re.IGNORECASE):
        replacement = NUMBER_WORDS[word.group(0).lower()]
        return question[: word.start()] + replacement + question[word.end() :]
    return None


_COPULAS = frozenset({"is", "are", "was", "were"})
_VERB = re.compile(
    r"(?<=\s)(?:use|have|support|return|take|accept|require|expect|load|create|provide|store|run"
    r"|handle|include|contain|need|work|get|apply|produce|generate|install|save|enable)\b"
)
_NEGATION_SHARE = 5
"""Negation is easy to make for almost every question, so it is made for 2 of every 5 of them:
otherwise it would be most of the set."""


def negation_flip(question: str, seed: str) -> str | None:
    """Remove the first negation of a question, or add one where it reads naturally: after
    `is/are/was/were`, or after a modal followed by `be` ("what can be used"). Other auxiliaries
    are followed by the subject ("does the dataset have"), where a plain `not` would read wrong,
    so nothing is made."""
    if entity_signature(question).negations:
        flipped = re.sub(r"\b(not|never) ", "", question, count=1)
        return None if flipped == question else flipped
    if _h(seed, "negation") % _NEGATION_SHARE >= 2:
        return None
    for match in AUXILIARY.finditer(question):
        word = match.group(0).lower()
        rest = question[match.end() :]
        following = rest.lstrip().split(" ", 1)[0].lower()
        if word in _COPULAS or following in {"be", "been"}:
            if word == "can":
                return question[: match.start()] + "cannot" + question[match.end() :]
            return question[: match.end()] + " not" + rest
        # "does the dataset have": `not` goes in front of the verb, after the subject
        found = _VERB.search(rest)
        if found is not None and rest[: found.start()].strip():
            return question[: match.end()] + rest[: found.start()] + "not " + rest[found.start() :]
    return None


def antonym_swap(question: str, seed: str) -> str | None:
    best: tuple[int, str, str] | None = None
    for left, right in ANTONYMS:
        for old, new in ((left, right), (right, left)):
            match = re.search(rf"\b{old}\b", question, re.IGNORECASE)
            if match and (best is None or match.start() < best[0]):
                best = (match.start(), match.group(0), new)
    if best is None:
        return None
    start, found, new = best
    if found[0].isupper():
        new = new.capitalize()
    return question[:start] + new + question[start + len(found) :]


def aspect_change(question: str, seed: str) -> str | None:
    found = [
        t
        for t in _swappable(question)
        if _kind_of(t) in {"snake", "camel", "dotted"} and t not in _PRODUCTS
    ]
    if not found:
        return None
    x = found[_h(seed, "x") % len(found)]
    candidates = [a.format(x=x) for a in ASPECTS]
    start = _h(seed, "aspect")
    for i in range(len(candidates)):
        new = candidates[(start + i) % len(candidates)]
        if new.rstrip("?.").lower() != question.rstrip("?.").lower():
            return new
    return None


TEXT_MUTATIONS: dict[str, Callable[[str, str], str | None]] = {
    "identifier_case": identifier_case,
    "version_bump": version_bump,
    "number_swap": number_swap,
    "negation_flip": negation_flip,
    "antonym_swap": antonym_swap,
    "aspect_change": aspect_change,
}


# ---- positive variants -----------------------------------------------------------------------


def whitespace_variant(question: str) -> str:
    words = question.split(" ")
    return "  " + "  ".join(words[:2]) + " \t" + " ".join(words[2:]) + " \n"


def punctuation_variant(question: str) -> str | None:
    if question.endswith("?"):
        return question[:-1] + "？"
    if question.endswith("."):
        return question[:-1]
    return question + "?"


def case_variant(question: str) -> str | None:
    if protected_tokens(question):
        return None  # an identifier is case sensitive: such a question is not merged by case
    return question.lower() if question != question.lower() else question.upper()


def invisible_variant(question: str) -> str:
    words = question.split(" ")
    return " ".join(words[:1]) + "​ " + " ".join(words[1:])


POSITIVE_VARIANTS: dict[str, Callable[[str], str | None]] = {
    "whitespace": whitespace_variant,
    "punctuation": punctuation_variant,
    "case": case_variant,
    "invisible": invisible_variant,
}
POSITIVES_PER_CLASS = 20
"""Pairs per positive class: 20 of each of identical, whitespace, punctuation, case, invisible."""


# ---- building --------------------------------------------------------------------------------


def build_pairs(sources: Mapping[str, Sequence[tuple[str, str]]]) -> list[Pair]:
    """Every pair for `sources` = `{split: [(qid, question), ...]}`, in a fixed order."""
    pairs: list[Pair] = []
    for split in sorted(sources):
        questions = list(sources[split])
        pool = sorted({t for _, q in questions for t in _swappable(q)})
        pairs.extend(_counter_text(split, questions, pool))
        pairs.extend(_counter_context(split, questions))
        pairs.extend(_counter_language(split, questions))
        pairs.extend(_positives(split, questions))
    return pairs


def _counter_text(
    split: str, questions: Sequence[tuple[str, str]], pool: Sequence[str]
) -> list[Pair]:
    out: list[Pair] = []
    for qid, question in questions:
        candidates: list[tuple[int, str, str]] = []
        swapped = identifier_swap(question, pool, qid)
        if swapped is not None:
            candidates.append((_h(qid, "identifier_swap"), "identifier_swap", swapped))
        for category, mutate in TEXT_MUTATIONS.items():
            mutated = mutate(question, f"{qid}:{category}")
            if mutated is not None and mutated != question:
                candidates.append((_h(qid, category), category, mutated))
        for _, category, mutated in sorted(candidates)[:PER_QUESTION]:
            out.append(
                Pair(
                    pair_id=f"{split}:{qid.split(':')[-1]}:{category}",
                    split=split,
                    kind="counter",
                    category=category,
                    source_qid=qid,
                    a=question,
                    b=mutated,
                )
            )
    return out


def _counter_context(split: str, questions: Sequence[tuple[str, str]]) -> list[Pair]:
    out: list[Pair] = []
    count = CONTEXT_PAIRS_PER_CATEGORY[split]
    for category, override in CONTEXT_CATEGORIES.items():
        chosen = sorted(questions, key=lambda item: _h(item[0], category))[:count]
        for qid, question in chosen:
            out.append(
                Pair(
                    f"{split}:{qid.split(':')[-1]}:{category}",
                    split,
                    "counter",
                    category,
                    qid,
                    question,
                    question,
                    b_ctx=dict(override),
                )
            )
    return out


def _counter_language(split: str, questions: Sequence[tuple[str, str]]) -> list[Pair]:
    return [
        Pair(
            f"{split}:{qid.split(':')[-1]}:other_language",
            split,
            "counter",
            "other_language",
            qid,
            question,
            ZH_TRANSLATIONS[qid],
        )
        for qid, question in questions
        if qid in ZH_TRANSLATIONS
    ]


def _positives(split: str, questions: Sequence[tuple[str, str]]) -> list[Pair]:
    out: list[Pair] = []
    share = 0.7 if split == "dev" else 0.3
    count = round(POSITIVES_PER_CLASS * share)
    for cls in ("identical", *POSITIVE_VARIANTS):
        made = 0
        for qid, question in sorted(questions, key=lambda item: _h(item[0], "pos", cls)):
            if made >= count:
                break
            if cls == "identical":
                variant: str | None = question
            else:
                variant = POSITIVE_VARIANTS[cls](question)
            if variant is None or (cls != "identical" and variant == question):
                continue
            out.append(
                Pair(
                    f"{split}:{qid.split(':')[-1]}:{cls}",
                    split,
                    "positive",
                    cls,
                    qid,
                    question,
                    variant,
                    expect_share=True,
                )
            )
            made += 1
    return out


# ---- files -----------------------------------------------------------------------------------


def write_pairs(pairs: Iterable[Pair], path: str | Path) -> int:
    lines = [json.dumps(asdict(p), ensure_ascii=False, sort_keys=True) for p in pairs]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8", newline="") as fh:
        fh.write("\n".join(lines) + "\n")
    return len(lines)


def read_pairs(path: str | Path) -> list[Pair]:
    out: list[Pair] = []
    with Path(path).open(encoding="utf-8", newline="") as fh:
        for line in fh:
            if line.strip():
                out.append(Pair(**json.loads(line)))
    return out


POLICY_CATEGORIES = frozenset({"identifier_case"})
"""Counter-examples that are a matter of policy, not of meaning: `LoRA` and `lora` are kept
apart because identifiers are case sensitive (an identifier written with another capital letter
may be a typo of the same question). They are in the L1 test and left out of the headline
number of the semantic cache."""

REVIEW_WIDE = frozenset({"identifier_swap", "negation_flip", "antonym_swap", "aspect_change"})


def review_sample(pairs: Sequence[Pair], n: int = 40) -> list[Pair]:
    """The pairs a person reads to check the generator: per category (in alphabetical order) the
    first 4 (the four text classes with many pairs) or 2 by the hash of the pair id, `n` in
    all. The same file always gives the same sample."""
    chosen: list[Pair] = []
    for category in sorted({p.category for p in pairs}):
        items = sorted(
            (p for p in pairs if p.category == category),
            key=lambda p: hashlib.sha256(f"review:{p.pair_id}".encode()).hexdigest(),
        )
        chosen.extend(items[: 4 if category in REVIEW_WIDE else 2])
    return chosen[:n]


# ---- evaluation ------------------------------------------------------------------------------


def key_of(question: str, ctx: Mapping[str, Any]) -> str:
    """The L1 key of `question` under the context `ctx` (as `DEFAULT_CTX`)."""
    fields = make_fields(
        "search",
        kb=ctx["kb"],
        question=question,
        filters=ctx["filters"],
        acl_groups=ctx["acl"],
        top_k=ctx["top_k"],
        data_generation=ctx["generation"],
        config_version="cfg",
        language_setting=ctx["language_setting"],
    )
    return cache_key(fields)


def contexts(pair: Pair) -> tuple[dict[str, Any], dict[str, Any]]:
    return dict(DEFAULT_CTX), {**DEFAULT_CTX, **pair.b_ctx}


def l1_shares(pair: Pair) -> bool:
    """True when L1 would give both requests of the pair one cache entry."""
    ctx_a, ctx_b = contexts(pair)
    return key_of(pair.a, ctx_a) == key_of(pair.b, ctx_b)


def evaluate_l1(pairs: Sequence[Pair]) -> dict[str, Any]:
    """False hits of the exact cache on the counter-examples (must be 0), and the share of
    positive pairs that do share an entry (must be all)."""
    counters = [p for p in pairs if p.kind == "counter"]
    positives = [p for p in pairs if p.kind == "positive"]
    false_hits = [p.pair_id for p in counters if l1_shares(p)]
    missed = [p.pair_id for p in positives if not l1_shares(p)]
    return {
        "counter_pairs": len(counters),
        "false_hits": false_hits,
        "positive_pairs": len(positives),
        "positives_not_shared": missed,
        "by_category": dict(Counter(p.category for p in counters)),
    }


def text_pairs(pairs: Sequence[Pair]) -> list[Pair]:
    """The counter-examples whose two requests differ in the question only (those the semantic
    cache's distance and entity guard have to separate): not the metadata pairs and not the
    translations (another language is another partition)."""
    return [
        p for p in pairs if p.kind == "counter" and not p.b_ctx and p.category != "other_language"
    ]


GUARDS = ("none", "entity", "full")
"""`none`: the distance alone; `entity`: the design's entity guard; `full`: the entity guard and
the lexical guard (`rag.cache.semantic.question_guard`)."""


def guard_accepts(
    pair: Pair,
    distance: float,
    threshold: float,
    strict_factor: float,
    guard: str = "full",
) -> bool:
    """Would the semantic cache have taken `pair.a` for `pair.b` (b new, a stored)?"""
    if guard == "none":
        return distance <= threshold
    verdict = question_guard(pair.b, pair.a, lexical=guard == "full")
    limit = threshold if verdict.entities else threshold * strict_factor
    return verdict.ok and distance <= limit
