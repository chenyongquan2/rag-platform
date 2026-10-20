"""`uv run python -m rag.data.chm unpack --chm FILE.chm --out data/raw/mt5/en` (spec P4-01).

Exit codes: 0 ok, 1 the unpack failed, 2 usage error (no such file).
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from rag.config import repo_path
from rag.data.chm.unpack import ChmUnpacker, UnpackError, unpack_chm


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m rag.data.chm", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    unpack = sub.add_parser(
        "unpack", help="unpack a .chm file into a folder (needs 7-Zip or chmlib)"
    )
    unpack.add_argument("--chm", required=True, help="path of the .chm file")
    unpack.add_argument(
        "--out", required=True, help="folder to unpack into (git-ignored data/raw/...)"
    )
    unpack.add_argument(
        "--overwrite", action="store_true", help="unpack into a folder that has files"
    )
    return parser


def main(argv: Sequence[str] | None = None, *, unpacker: ChmUnpacker | None = None) -> int:
    args = _parser().parse_args(argv)
    chm, out = Path(args.chm), repo_path(args.out)
    if not chm.is_file():
        print(f"error: no such file: {chm.name}", file=sys.stderr)
        return 2
    try:
        result = unpack_chm(chm, out, unpacker=unpacker, overwrite=args.overwrite)
    except UnpackError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"unpacked {chm.name} with {result.tool} into {out.as_posix()}")
    print(f"files: {result.files}  pages: {result.pages}  chm bytes: {result.chm_bytes:,}")
    print(f"sha256: {result.chm_sha256}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
