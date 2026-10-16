"""Token counting with the reranker's own tokenizer (design s7-1).

bge-reranker-base reads at most 512 tokens of "question + title path + chunk", so chunk sizes are
measured with its tokenizer, not LlamaIndex's default tiktoken. The tokenizer file is taken from
the local Hugging Face cache only (no network): the `tokenizers` library loads `tokenizer.json` in
about a second, where `transformers.AutoTokenizer` needs about a minute on this machine.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from pathlib import Path

TokenizerFn = Callable[[str], list[int]]


class TokenizerNotCachedError(RuntimeError):
    """The model's tokenizer.json is not in the local Hugging Face cache."""


@functools.lru_cache(maxsize=8)
def load_tokenizer(model: str, cache_dir: str | Path | None = None) -> TokenizerFn:
    """Return `text -> token ids` (no special tokens, no truncation) for `model`.

    `cache_dir` defaults to the Hugging Face hub cache (`HF_HUB_CACHE` / `HF_HOME`).
    """
    from huggingface_hub import try_to_load_from_cache
    from tokenizers import Tokenizer

    path = try_to_load_from_cache(model, "tokenizer.json", cache_dir=cache_dir)
    if not isinstance(path, str):
        where = cache_dir or "the default Hugging Face cache"
        raise TokenizerNotCachedError(
            f"tokenizer.json of {model!r} is not in {where}; download the model once "
            "(HF_ENDPOINT=https://hf-mirror.com) or point HF_HOME at a cache that has it"
        )
    tokenizer = Tokenizer.from_file(path)
    tokenizer.no_truncation()
    tokenizer.no_padding()

    def encode(text: str) -> list[int]:
        return tokenizer.encode(text, add_special_tokens=False).ids

    encode.__name__ = f"tokenizer[{model}]"
    return encode
