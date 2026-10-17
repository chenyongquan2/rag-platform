"""Spec P0-07 AC1: the filter both routes share (design s9-1, s16-1)."""

from __future__ import annotations

import pytest

from rag.retrieve.filters import CALLER_FILTER_KEYS, FilterError, build_es_filter
from rag.types import ErrorCategory


def test_filter_always_has_kb_and_acl() -> None:
    plain = build_es_filter(kb="hf_docs", acl_groups=["public"])
    assert plain == [
        {"term": {"kb": "hf_docs"}},
        {"terms": {"acl_groups": ["public"]}},
    ]

    narrowed = build_es_filter(
        kb="hf_docs",
        acl_groups=("team-a", "public"),
        filters={"doc_type": "mdx", "language": ["en", "zh"], "source_rel_prefix": "huggingface/"},
    )
    # kb and ACL come first and are always there; caller filters only add clauses.
    assert narrowed[:2] == [
        {"term": {"kb": "hf_docs"}},
        {"terms": {"acl_groups": ["team-a", "public"]}},
    ]
    assert narrowed[2:] == [
        {"term": {"doc_type": "mdx"}},
        {"terms": {"language": ["en", "zh"]}},
        {"prefix": {"source_rel": "huggingface/"}},
    ]
    assert set(CALLER_FILTER_KEYS) == {"doc_type", "language", "source_rel_prefix"}
    # None and empty filters are the same as no filters.
    assert build_es_filter(kb="k", acl_groups=["g"], filters={}) == build_es_filter(
        kb="k", acl_groups=["g"], filters=None
    )


@pytest.mark.parametrize(
    "filters",
    [
        {"acl_groups": ["admin"]},  # cannot widen or replace the server's ACL filter
        {"kb": "other"},  # cannot switch the knowledge base
        {"title_path": "x"},  # not whitelisted
        {"doc_type": "mdx", "unknown": 1},  # one bad key rejects the whole request
        {"doc_type": ""},  # empty value
        {"language": []},  # empty list
        {"doc_type": {"wildcard": "*"}},  # no query DSL through a value
        {"source_rel_prefix": ["a", "b"]},  # prefix takes one string
    ],
)
def test_filter_rejects_unknown_keys_and_cannot_drop_acl(filters: dict[str, object]) -> None:
    with pytest.raises(FilterError) as caught:
        build_es_filter(kb="hf_docs", acl_groups=["public"], filters=filters)
    assert caught.value.category is ErrorCategory.INVALID_FILTER


def test_filter_requires_acl_groups_and_kb() -> None:
    # No groups would mean "no ACL clause" if it were silently skipped: refuse instead.
    with pytest.raises(ValueError, match="acl_groups"):
        build_es_filter(kb="hf_docs", acl_groups=[])
    with pytest.raises(ValueError, match="acl_groups"):
        build_es_filter(kb="hf_docs", acl_groups=[""])
    with pytest.raises(ValueError, match="kb"):
        build_es_filter(kb="", acl_groups=["public"])
