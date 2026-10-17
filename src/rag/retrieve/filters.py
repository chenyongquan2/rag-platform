"""The filter both retrieval routes share (design s9-1, s16-1).

Every search is limited to one knowledge base and to the caller's ACL groups; the server adds
both from the request's knowledge base and the caller's identity. A caller may only narrow the
search further with the whitelisted keys below. Any other key (including `acl_groups` or `kb`)
is rejected with `invalid_filter`, so a caller can never replace, widen or drop the ACL clause.
The same list goes into the BM25 `bool.filter` and into `knn.filter`, so both routes see the
same documents and the kNN route filters during the search, not after it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from rag.types import ErrorCategory

CALLER_FILTER_KEYS = ("doc_type", "language", "source_rel_prefix")
"""Filter keys a caller may pass (s16-1). `doc_type` and `language` take one value or a list;
`source_rel_prefix` takes one path prefix."""


class FilterError(ValueError):
    """A caller filter outside the whitelist or with a value it does not accept."""

    category = ErrorCategory.INVALID_FILTER


def _keyword_clause(field: str, value: Any) -> dict[str, Any]:
    if isinstance(value, str):
        if not value:
            raise FilterError(f"filter {field!r} must not be empty")
        return {"term": {field: value}}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        values = list(value)
        if not values or not all(isinstance(v, str) and v for v in values):
            raise FilterError(f"filter {field!r} must be a non-empty string or list of strings")
        return {"terms": {field: values}}
    raise FilterError(f"filter {field!r} must be a string or a list of strings")


def build_es_filter(
    *, kb: str, acl_groups: Sequence[str], filters: Mapping[str, Any] | None = None
) -> list[dict[str, Any]]:
    """`[kb term, acl_groups terms, *caller clauses]` for `bool.filter` and `knn.filter`.

    `acl_groups` come from the caller's identity (server side), never from `filters`; an empty
    list is refused rather than turned into "no ACL clause". Raises `FilterError` for a caller
    filter outside `CALLER_FILTER_KEYS` or with a bad value.
    """
    if not kb:
        raise ValueError("kb must not be empty")
    groups = list(acl_groups)
    if not groups or not all(isinstance(g, str) and g for g in groups):
        raise ValueError("acl_groups must name at least one non-empty group of the caller")
    clauses: list[dict[str, Any]] = [
        {"term": {"kb": kb}},
        {"terms": {"acl_groups": groups}},
    ]
    filters = dict(filters or {})
    unknown = sorted(set(filters) - set(CALLER_FILTER_KEYS))
    if unknown:
        raise FilterError(f"unknown filter keys {unknown}; allowed: {list(CALLER_FILTER_KEYS)}")
    for key in CALLER_FILTER_KEYS:
        if key not in filters:
            continue
        value = filters[key]
        if key == "source_rel_prefix":
            if not isinstance(value, str) or not value:
                raise FilterError("filter 'source_rel_prefix' must be one non-empty string")
            clauses.append({"prefix": {"source_rel": value}})
        else:
            clauses.append(_keyword_clause(key, value))
    return clauses
