"""Lightweight language detection per unit (design s5-4, s6-2): scripts first, then stopwords.

No model and no new dependency: the corpora here are mostly English with a few Chinese, Korean,
Japanese, Russian, Hindi, Spanish, Portuguese and German pages. Code blocks, inline code, URLs and
link targets are removed first, because code is English-looking whatever the page language is.
Every script is counted in "word units" (a Latin, Cyrillic, Hangul or Devanagari word; 1.5 Han
characters; 2 kana) and the script with the most units wins. Latin-script text is then told apart
by counting common function words; English wins ties.
"""

from __future__ import annotations

import re
from collections import Counter

from rag.ingest.markdown import split_fenced

UNDETERMINED = "und"
_MIN_UNITS = 3.0

_INLINE_CODE = re.compile(r"`[^`\n]*`")
_URL = re.compile(r"https?://\S+|www\.\S+|\S+@\S+\.\w+")
_LINK_TARGET = re.compile(r"\]\([^)]*\)")
_TAG = re.compile(r"</?[A-Za-z][^>\n]*>")

_HAN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
_KANA = re.compile(r"[\u3040-\u30ff\u31f0-\u31ff]")
_WORDS = {
    "latin": re.compile(r"[A-Za-z\u00c0-\u024f]+"),
    "ru": re.compile(r"[\u0400-\u04ff]+"),
    "ko": re.compile(r"[\uac00-\ud7af\u1100-\u11ff]+"),
    "hi": re.compile(r"[\u0900-\u097f]+"),
    "te": re.compile(r"[\u0c00-\u0c7f]+"),
    "ar": re.compile(r"[\u0600-\u06ff]+"),
    "he": re.compile(r"[\u0590-\u05ff]+"),
    "th": re.compile(r"[\u0e00-\u0e7f]+"),
}

_STOPWORDS = {
    "en": "the and of to is that for with this you are it be on as can from by an or will",
    "es": "el la los las de del que y en un una es por para con se lo como más",
    "pt": "os de do da dos das que em um uma para com não se por mais como é",
    "de": "der die das und ist nicht mit ein eine zu den von für auf sie es dem auch sich",
    "fr": "le la les de des du et est un une pour que dans en pas vous sur avec il ce",
    "it": "il lo gli le di che è un una per con non del della sono da nel si",
}
_STOP = {lang: frozenset(words.split()) for lang, words in _STOPWORDS.items()}


def _prose(text: str) -> str:
    parts = [seg.text for seg in split_fenced(text) if not seg.code]
    prose = "\n".join(parts)
    for pattern in (_INLINE_CODE, _URL, _LINK_TARGET, _TAG):
        prose = pattern.sub(" ", prose)
    return prose


def _latin_language(words: list[str]) -> str:
    counts = Counter(w.lower() for w in words)
    scores = {lang: sum(counts[w] for w in stop) for lang, stop in _STOP.items()}
    best = max(scores, key=lambda lang: (scores[lang], lang == "en"))
    return best if scores[best] > scores["en"] else "en"


def detect_language(text: str) -> str:
    """ISO 639-1 code of the main language of `text`, or `und` when there is too little prose."""
    prose = _prose(text)
    units: Counter[str] = Counter()
    found: dict[str, list[str]] = {}
    for name, pattern in _WORDS.items():
        found[name] = pattern.findall(prose)
        units[name] = len(found[name])
    han, kana = len(_HAN.findall(prose)), len(_KANA.findall(prose))
    if kana and kana >= 0.1 * (han + kana):
        units["ja"] = han / 1.5 + kana / 2
    else:
        units["zh"] = han / 1.5
    if sum(units.values()) < _MIN_UNITS:
        return UNDETERMINED
    best = max(units, key=lambda name: units[name])
    if best == "latin":
        return _latin_language(found["latin"])
    return best
