"""Spec P2-03: delete the ablation index family of one prefix, and only that.

    uv run python scripts/p2_03/drop_variant.py p203bbq_ [--dry-run]

Deletes the indices `<prefix>kb_hf_docs_v<N>` (the alias goes with them), the docstore files
`data/docstore/<index>*.json` of this worktree and the Redis data generation key of the prefix.
The prefix must start with `p203` so that kb_hf_docs_v1, fake_kb_hf_docs_v1 and every other index
of the project are out of reach; nothing else is touched.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from rag.config import ServiceSettings
from rag.factories import make_es_client, make_redis_client
from rag.index.generation import DEFAULT_KEY_PREFIX

PREFIX_RE = re.compile(r"p203[a-z0-9]*_")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("prefix")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--kb", default="hf_docs")
    args = parser.parse_args()
    if not PREFIX_RE.fullmatch(args.prefix):
        print(f"refused: the prefix must look like p203xxx_, got {args.prefix!r}", file=sys.stderr)
        return 2
    settings = ServiceSettings()
    es = make_es_client(settings.es_url, request_timeout=60.0)
    pattern = re.compile(rf"{re.escape(args.prefix)}kb_{re.escape(args.kb)}_v[1-9][0-9]*")
    indices = sorted(
        n for n in es.indices.get(index=f"{args.prefix}kb_{args.kb}_v*") if pattern.fullmatch(n)
    )
    for name in indices:
        print(("would delete" if args.dry_run else "delete") + f" index {name}")
        if not args.dry_run:
            es.indices.delete(index=name)
    for name in indices:
        for path in sorted(Path("data/docstore").glob(f"{name}*.json")):
            print(("would remove " if args.dry_run else "remove ") + str(path))
            if not args.dry_run:
                path.unlink()
    key = f"{DEFAULT_KEY_PREFIX}{args.prefix}{args.kb}"
    if not args.dry_run:
        make_redis_client(settings.redis_url).delete(key)
    print(("would delete" if args.dry_run else "deleted") + f" Redis key {key}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
