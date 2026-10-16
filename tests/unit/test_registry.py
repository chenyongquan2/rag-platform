from __future__ import annotations

from collections.abc import Callable

import pytest

from rag import registry as registry_module
from rag.registry import DuplicateNameError, Registry, UnknownNameError


class _Step:
    def __init__(self, max_tokens: int = 100) -> None:
        self.max_tokens = max_tokens


def test_register_and_create() -> None:
    reg: Registry[Callable[..., _Step]] = Registry("processing step")

    @reg.register("token_split")
    class TokenSplit(_Step):
        pass

    reg.add("plain", _Step)

    assert reg.get("token_split") is TokenSplit
    assert "token_split" in reg
    assert reg.names() == ["plain", "token_split"]
    built = reg.create("token_split", max_tokens=400)
    assert isinstance(built, TokenSplit)
    assert built.max_tokens == 400

    # Module-level registries exist for the three extension points (design s5-1).
    assert registry_module.connectors.kind == "connector"
    assert registry_module.readers.kind == "reader"
    assert registry_module.steps.kind == "processing step"


def test_unknown_name_error_lists_known_names() -> None:
    reg: Registry[Callable[..., _Step]] = Registry("connector")
    reg.add("local_dir", _Step)
    reg.add("hf_dataset", _Step)

    with pytest.raises(UnknownNameError) as exc_info:
        reg.get("confluence")
    message = str(exc_info.value)
    assert "connector" in message
    assert "'confluence'" in message
    assert "hf_dataset, local_dir" in message
    assert isinstance(exc_info.value, LookupError)

    with pytest.raises(UnknownNameError):
        reg.create("confluence")


def test_duplicate_name_rejected() -> None:
    reg: Registry[Callable[..., _Step]] = Registry("reader")
    reg.add(".md", _Step)
    with pytest.raises(DuplicateNameError, match=r"\.md"):
        reg.add(".md", _Step)
    with pytest.raises(ValueError, match="empty"):
        reg.add("", _Step)
