"""Planted unsupported sentences: does the judge catch them? (spec P1-06, judge sanity check)

From the first pass, take sentences the judge called supported and change copies of their
answers so that the sentence is no longer supported:

- `swap_fragment`: the sentence keeps its text but cites one chunk of another answer, from a
  page none of its own citations came from.
- `change_number`: a stand-alone number of the sentence that its cited fragment also contains
  is replaced by a value no cited fragment contains; the citations stay.

Half of each kind (the swap gets the smaller half), seeded, one planted copy per original answer.
The originals are not changed. The other sentences of the copy stay as they were.
"""

from __future__ import annotations

import random
import re
from collections.abc import Sequence

from rag.eval.support.records import Planted, SupportItem, SupportLine
from rag.types import AnswerSentence, Chunk, Citation

# A number not glued to a letter, a digit, `_` or a decimal point (`x86_64`, `v2`, `1.5` whole).
_NUMBER = re.compile(r"(?<![\w.])\d+(?:\.\d+)?(?![\w]|\.\d)")


def _supported_sentences(line: SupportLine) -> list[int]:
    if line.result is None or line.item.planted is not None:
        return []
    return [s.index for s in line.result.sentences if s.status == "supported"]


def _numbers_in(text: str) -> set[str]:
    return set(_NUMBER.findall(text))


def _new_value(old: str, taken: set[str]) -> str | None:
    value = float(old)
    decimals = len(old.split(".")[1]) if "." in old else 0
    for candidate in (value * 2, value + 1, value * 10, value + 7, value + 13):
        text = f"{candidate:.{decimals}f}" if decimals else str(int(candidate))
        if text != old and text not in taken:
            return text
    return None


def _change_number(line: SupportLine, index: int) -> tuple[AnswerSentence, str] | None:
    sentence = line.item.sentences[index - 1]
    by_id = {c.chunk_id: c for c in line.item.chunks}
    cited = [by_id[c.chunk_id] for c in sentence.citations if c.chunk_id in by_id]
    in_fragments: set[str] = set()
    for chunk in cited:
        in_fragments |= _numbers_in(chunk.text)
    matches = [m for m in _NUMBER.finditer(sentence.text) if m.group() in in_fragments]
    if not matches:
        return None
    match = matches[-1]
    new = _new_value(match.group(), in_fragments)
    if new is None:
        return None
    text = sentence.text[: match.start()] + new + sentence.text[match.end() :]
    return sentence.model_copy(update={"text": text}), cited[0].chunk_id


def _copy(
    line: SupportLine, planted: Planted, sentence: AnswerSentence, extra: Chunk | None
) -> SupportItem:
    sentences = list(line.item.sentences)
    sentences[planted.sentence - 1] = sentence
    chunks = list(line.item.chunks) + ([extra] if extra is not None else [])
    return SupportItem(
        source="planted",
        system=line.system,
        qid=f"planted-{planted.kind}:{line.qid}",
        qtype=line.qtype,
        question=line.item.question,
        sentences=sentences,
        chunks=chunks,
        planted=planted,
    )


def plant(lines: Sequence[SupportLine], *, n: int = 10, seed: int = 0) -> list[SupportItem]:
    """`n` planted copies (fewer when there are not enough candidates)."""
    rng = random.Random(seed)
    candidates = [x for x in lines if _supported_sentences(x)]
    rng.shuffle(candidates)
    used: set[str] = set()
    out: list[SupportItem] = []

    n_swap = n // 2
    for line in candidates:
        if sum(1 for p in out if p.planted and p.planted.kind == "swap_fragment") >= n_swap:
            break
        index = rng.choice(_supported_sentences(line))
        sentence = line.item.sentences[index - 1]
        own_pages = {c.chunk_id.split("#", 1)[0] for c in sentence.citations}
        donors = [
            c
            for other in candidates
            if other.qid != line.qid
            for c in other.item.chunks
            if c.doc_id not in own_pages and c.chunk_id.split("#", 1)[0] not in own_pages
        ]
        if not donors:
            continue
        donor = rng.choice(sorted(donors, key=lambda c: c.chunk_id))
        planted = Planted(
            kind="swap_fragment",
            original_qid=line.qid,
            sentence=index,
            before=sentence.text,
            after=sentence.text,
            chunk_id=donor.chunk_id,
        )
        swapped = sentence.model_copy(
            update={"citations": [Citation(marker=1, chunk_id=donor.chunk_id)]}
        )
        out.append(_copy(line, planted, swapped, donor))
        used.add(line.qid)

    for line in candidates:
        if len(out) >= n:
            break
        if line.qid in used:
            continue
        for index in _supported_sentences(line):
            changed = _change_number(line, index)
            if changed is None:
                continue
            sentence, chunk_id = changed
            planted = Planted(
                kind="change_number",
                original_qid=line.qid,
                sentence=index,
                before=line.item.sentences[index - 1].text,
                after=sentence.text,
                chunk_id=chunk_id,
            )
            out.append(_copy(line, planted, sentence, None))
            used.add(line.qid)
            break
    return out
