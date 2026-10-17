"""Create, list, switch and delete the versioned indices of one knowledge base (design s7-5).

Queries only use the alias `kb_<kb>`. A rebuild writes `kb_<kb>_v<N+1>`; when it is ready the
alias moves to it in one `_aliases` request, so a search sees the old or the new index, never
neither and never both. Two switches of the same knowledge base must not run at the same time
(the caller holds the knowledge base's ingest lock); `expected_current` also makes a switch that
was decided on a stale read fail as a whole instead of half-applying.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Final

from elasticsearch import BadRequestError, NotFoundError

if TYPE_CHECKING:
    from elasticsearch import Elasticsearch

    from rag.index.names import KbIndexNames


class IndexExistsError(RuntimeError):
    """The versioned index to create already exists."""


class AliasConflictError(RuntimeError):
    """The alias did not point where the caller expected; nothing was changed."""


class AliasedIndexDeletionError(RuntimeError):
    """Refused to delete an index that an alias still points to."""


class IndexStateError(RuntimeError):
    """The indices of a knowledge base are in a state the operation cannot handle safely."""


class _Unset:
    pass


_READ_NOW: Final = _Unset()


def _error(exc: BadRequestError | NotFoundError) -> dict[str, Any]:
    body = exc.body if isinstance(exc.body, dict) else {}
    error = body.get("error")
    return error if isinstance(error, dict) else {}


def _error_type(exc: BadRequestError | NotFoundError) -> str:
    return str(_error(exc).get("type", ""))


def create_kb_index(
    es: Elasticsearch, names: KbIndexNames, version: int, body: dict[str, Any]
) -> str:
    """Create `kb_<kb>_v<version>` with `body` (see `rag.index.mapping`); returns its name.

    Does not touch the alias. Raises IndexExistsError when the index exists.
    """
    index = names.index(version)
    try:
        es.indices.create(index=index, settings=body["settings"], mappings=body["mappings"])
    except BadRequestError as exc:
        if _error_type(exc) == "resource_already_exists_exception":
            raise IndexExistsError(f"index {index} already exists") from exc
        raise
    return index


def list_versions(es: Elasticsearch, names: KbIndexNames) -> dict[int, str]:
    """`{version: index}` of every existing version, in version order."""
    resolved = es.indices.resolve_index(name=names.pattern, expand_wildcards="all")
    versions: dict[int, str] = {}
    for item in resolved["indices"]:
        version = names.version_of(item["name"])
        if version is not None:
            versions[version] = item["name"]
    return dict(sorted(versions.items()))


def aliased_indices(es: Elasticsearch, names: KbIndexNames) -> list[str]:
    """Indices the alias points to: one normally, none before the first switch."""
    resp = es.options(ignore_status=404).indices.get_alias(name=names.alias)
    if resp.meta.status == 404:
        return []
    return sorted(resp.body)


def switch_alias(
    es: Elasticsearch,
    names: KbIndexNames,
    new_index: str,
    *,
    expected_current: str | _Unset | None = _READ_NOW,
) -> str | None:
    """Point the alias at `new_index` in one atomic request; return the index it pointed to.

    Without `expected_current` the current target is read first. Either way the request removes
    the alias from every expected index with `must_exist` - also when that index is `new_index`
    itself - so if the alias has moved in the meantime Elasticsearch rejects the whole request
    and AliasConflictError is raised instead of leaving the alias on two indices.
    `expected_current=None` means "no alias yet"; the request cannot check that, so it is
    checked by a read only and two first switches at once are left to the ingest lock.
    Raises IndexStateError when `new_index` is deleted between the existence check and the
    request. The caller adds 1 to the data generation after a switch (design s7-5).
    """
    if names.version_of(new_index) is None:
        raise ValueError(f"{new_index!r} is not a versioned index of knowledge base {names.kb!r}")
    if not es.indices.exists(index=new_index):
        raise ValueError(f"index {new_index} does not exist")
    if isinstance(expected_current, _Unset):
        expected = aliased_indices(es, names)
    elif expected_current is None:
        current = aliased_indices(es, names)
        if current:
            raise AliasConflictError(f"{names.alias} points to {current}, expected no alias")
        expected = []
    else:
        expected = [expected_current]
    actions: list[dict[str, Any]] = [
        {"remove": {"index": old, "alias": names.alias, "must_exist": True}} for old in expected
    ]
    actions.append({"add": {"index": new_index, "alias": names.alias}})
    try:
        es.indices.update_aliases(actions=actions)
    except NotFoundError as exc:
        error = _error(exc)
        missing = error.get("index")
        if error.get("type") == "index_not_found_exception" and missing == new_index:
            raise IndexStateError(
                f"index {new_index} was deleted during the switch; nothing was changed"
            ) from exc
        if error.get("type") == "index_not_found_exception" and missing in expected:
            raise AliasConflictError(
                f"expected {names.alias} on {missing}, but {missing} does not exist; "
                "nothing was changed"
            ) from exc
        if error.get("type") == "aliases_not_found_exception":
            raise AliasConflictError(
                f"{names.alias} no longer points to {', '.join(expected)}; nothing was changed"
            ) from exc
        raise
    return expected[0] if expected else None


def delete_kb_index(es: Elasticsearch, names: KbIndexNames, index: str) -> None:
    """Delete one old version. Refuses anything but an exact versioned index name of this
    knowledge base, and any index that an alias points to (switch the alias first)."""
    if names.version_of(index) is None:
        raise ValueError(
            f"{index!r} is not a versioned index of knowledge base {names.kb!r}; not deleted"
        )
    aliases = es.indices.get_alias(index=index)[index]["aliases"]
    if aliases:
        raise AliasedIndexDeletionError(
            f"{index} is behind alias {', '.join(sorted(aliases))}; switch the alias first"
        )
    es.indices.delete(index=index)


def init_kb(es: Elasticsearch, names: KbIndexNames, body: dict[str, Any]) -> tuple[str, bool]:
    """Create version 1 and point the alias at it, unless the alias exists already.

    Returns `(index the alias points to, created)`. Refuses when versions exist without an alias
    (pick one with `switch`) or when a plain index holds the alias name.
    """
    current = aliased_indices(es, names)
    if current:
        return current[0], False
    if es.indices.exists(index=names.alias):
        raise IndexStateError(
            f"an index named {names.alias} exists; the alias needs that name. Reindex it into "
            f"{names.index(1)} and delete it first"
        )
    versions = list_versions(es, names)
    if versions:
        raise IndexStateError(
            f"{names.alias} has versions {list(versions.values())} but no alias; point the alias "
            f"at one with: python -m rag.index.cli switch --kb {names.kb} --version N"
        )
    index = create_kb_index(es, names, 1, body)
    switch_alias(es, names, index, expected_current=None)
    return index, True


def ensure_index_fields(es: Elasticsearch, index: str, body: dict[str, Any]) -> list[str]:
    """Add the fields of `body` that `index` does not have yet; return their names.

    Adding fields to a mapping is allowed on a live index and changes no existing document, so an
    index created before a field was added to `rag.index.mapping` can still take new chunks (the
    mapping is strict). Fields that exist are left alone; changing their definition needs a new
    index version (s7-5).
    """
    current = es.indices.get_mapping(index=index)[index]["mappings"].get("properties", {})
    missing = {k: v for k, v in body["mappings"]["properties"].items() if k not in current}
    if missing:
        es.indices.put_mapping(index=index, properties=missing)
    return sorted(missing)


def index_embed_model(es: Elasticsearch, index: str) -> str | None:
    """The embedding model the index was written with (`_meta.embed_model`), None if not set."""
    meta = es.indices.get_mapping(index=index)[index]["mappings"].get("_meta") or {}
    value = meta.get("embed_model")
    return str(value) if value else None


def set_index_embed_model(es: Elasticsearch, index: str, model_id: str) -> None:
    """Record the embedding model in the index's `_meta` (other `_meta` keys are kept).

    Vectors of two models in one index cannot be compared with each other, and with the same
    dimensions nothing would report it (s7-5); the ingest job refuses a run with another model.
    """
    meta = dict(es.indices.get_mapping(index=index)[index]["mappings"].get("_meta") or {})
    meta["embed_model"] = model_id
    es.indices.put_mapping(index=index, meta=meta)
