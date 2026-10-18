"""Question generation and critique prompts (spec P1-05, design s14-2).

The texts follow the Hugging Face cookbook `rag_evaluation` notebook (Apache-2.0): a factoid
question answerable with one concise fact, phrased like a search-engine query, never mentioning
"the passage" or "the context"; then three critiques, each a 1-5 'total rating' with the
evaluation (rationale) written first:

- groundedness: can the question be answered unambiguously from the context;
- relevance: how useful the question is to ML developers building with the Hugging Face ecosystem;
- standalone: does the question make sense without the context.

Changes to the HF generation prompt (design s14-2: questions copied from the text are too easy for
keyword search): the question must be phrased in the user's own words, not copied from the
context (exact library, class, function, parameter and model names stay); the answer is one
short fact; English only. The critique prompts are HF's text unchanged.
"""

from __future__ import annotations

import re

from pydantic import BaseModel

GENERATION_PROMPT_VERSION = "p1-05-gen-v1"
CRITIQUE_PROMPT_VERSION = "p1-05-hf-critique-v1"
"""Bump when a prompt text or a parse rule changes."""

CRITERIA: tuple[str, ...] = ("groundedness", "relevance", "standalone")

_GENERATION = """Your task is to write a factoid question and an answer given a context.
Your factoid question should be answerable with a specific, concise piece of factual information from the context.
Your factoid question should be formulated in the same style as questions users could ask in a search engine.
This means that your factoid question MUST NOT mention something like "according to the passage" or "context".
Write the question the way a user who has not read the context would ask it: do not copy phrases or sentences from the context, use your own words (keep the exact names of libraries, classes, functions, parameters and models).
The answer must be one short fact (a few words or one sentence) stated in the context.
Write the question and the answer in English.

Provide your answer as follows:

Output:::
Factoid question: (your factoid question)
Answer: (your answer to the factoid question)

Now here is the context.

Context: {context}\n
Output:::"""  # noqa: E501 (prompt text, one instruction per line)

_GROUNDEDNESS = """You will be given a context and a question.
Your task is to provide a 'total rating' scoring how well one can answer the given question unambiguously with the given context.
Give your answer on a scale of 1 to 5, where 1 means that the question is not answerable at all given the context, and 5 means that the question is clearly and unambiguously answerable with the context.

Provide your answer as follows:

Answer:::
Evaluation: (your rationale for the rating, as a text)
Total rating: (your rating, as a number between 1 and 5)

You MUST provide values for 'Evaluation:' and 'Total rating:' in your answer.

Now here are the question and context.

Question: {question}\n
Context: {context}\n
Answer::: """  # noqa: E501

_RELEVANCE = """You will be given a question.
Your task is to provide a 'total rating' representing how useful this question can be to machine learning developers building NLP applications with the Hugging Face ecosystem.
Give your answer on a scale of 1 to 5, where 1 means that the question is not useful at all, and 5 means that the question is extremely useful.

Provide your answer as follows:

Answer:::
Evaluation: (your rationale for the rating, as a text)
Total rating: (your rating, as a number between 1 and 5)

You MUST provide values for 'Evaluation:' and 'Total rating:' in your answer.

Now here is the question.

Question: {question}\n
Answer::: """  # noqa: E501

_STANDALONE = """You will be given a question.
Your task is to provide a 'total rating' representing how context-independent this question is.
Give your answer on a scale of 1 to 5, where 1 means that the question depends on additional information to be understood, and 5 means that the question makes sense by itself.
For instance, if the question refers to a particular setting, like 'in the context' or 'in the document', the rating must be 1.
The questions can contain obscure technical nouns or acronyms like Gradio, Hub, Hugging Face or Space and still be a 5: it must simply be clear to an operator with access to documentation what the question is about.

For instance, "What is the name of the checkpoint from which the ViT model is imported?" should receive a 1, since there is an implicit mention of a context, thus the question is not independent from the context.

Provide your answer as follows:

Answer:::
Evaluation: (your rationale for the rating, as a text)
Total rating: (your rating, as a number between 1 and 5)

You MUST provide values for 'Evaluation:' and 'Total rating:' in your answer.

Now here is the question.

Question: {question}\n
Answer::: """  # noqa: E501

_CRITIQUES = {"groundedness": _GROUNDEDNESS, "relevance": _RELEVANCE, "standalone": _STANDALONE}


class GeneratedQA(BaseModel):
    question: str
    answer: str


def build_generation_prompt(context: str) -> str:
    return _GENERATION.format(context=context)


_LABEL = r"(?:^|\n)[ \t]*\**[ \t]*{name}[ \t]*\**[ \t]*:[ \t]*\**[ \t]*"
_QUESTION = re.compile(_LABEL.format(name="Factoid question"), re.IGNORECASE)
_ANSWER = re.compile(_LABEL.format(name="Answer"), re.IGNORECASE)


def _clean(text: str) -> str:
    return text.strip().strip("*").strip()


def parse_generation(text: str) -> GeneratedQA | None:
    """The question after the last `Factoid question:` and the answer after the `Answer:` that
    follows it; None when either is missing or empty."""
    text = text or ""
    questions = list(_QUESTION.finditer(text))
    if not questions:
        return None
    rest = text[questions[-1].end() :]
    answer = _ANSWER.search(rest)
    if answer is None:
        return None
    question = _clean(rest[: answer.start()])
    reply = _clean(rest[answer.end() :].split("Output:::")[0])
    if not question or not reply:
        return None
    return GeneratedQA(question=" ".join(question.split()), answer=reply)


def build_critique_prompt(criterion: str, question: str, context: str) -> str:
    if criterion not in _CRITIQUES:
        raise ValueError(f"unknown criterion {criterion!r}; choose from {CRITERIA}")
    return _CRITIQUES[criterion].format(question=question.strip(), context=context)


class CritiqueParse(BaseModel):
    score: float | None
    rationale: str


_RATING = re.compile(r"total rating\s*:\s*\**\s*(\d+(?:\.\d+)?)", re.IGNORECASE)
_EVALUATION = re.compile(r"evaluation\s*:\s*", re.IGNORECASE)


def parse_critique(text: str) -> CritiqueParse:
    """The number after the last `Total rating:` (1 to 5, else None) and the evaluation text."""
    text = text or ""
    ratings = list(_RATING.finditer(text))
    score: float | None = None
    head = text
    if ratings:
        value = float(ratings[-1].group(1))
        score = value if 1 <= value <= 5 else None
        head = text[: ratings[0].start()]
    evaluation = _EVALUATION.search(head)
    rationale = head[evaluation.end() :] if evaluation else head
    rationale = rationale.replace("Answer:::", "").strip()
    return CritiqueParse(score=score, rationale=rationale)
