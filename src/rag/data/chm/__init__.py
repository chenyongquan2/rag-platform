"""CHM manuals as a data source (spec P4-01, design s5-1, s5-2).

Importing this package registers the `chm_dir` connector in `rag.registry.connectors`.
CLI: `uv run python -m rag.data.chm unpack --chm FILE.chm --out data/raw/mt5/en`.
"""

from rag.data.chm.connector import ChmDirConnector, VolumeStats
from rag.data.chm.encoding import DecodedPage, codec_family, declared_charset, decode_html
from rag.data.chm.source import (
    ChmError,
    ChmInfo,
    ChmSource,
    DirectoryChmSource,
    charset_hint_for_lcid,
    parse_system_file,
)
from rag.data.chm.toc import TocEntry, page_key_of, parse_index, parse_toc
from rag.data.chm.unpack import (
    ChmLibUnpacker,
    ChmUnpacker,
    HhDecompileUnpacker,
    SevenZipUnpacker,
    UnpackError,
    UnpackResult,
    find_unpacker,
    read_unpack_manifest,
    unpack_chm,
)

__all__ = [
    "ChmDirConnector",
    "ChmError",
    "ChmInfo",
    "ChmLibUnpacker",
    "ChmSource",
    "ChmUnpacker",
    "DecodedPage",
    "DirectoryChmSource",
    "HhDecompileUnpacker",
    "SevenZipUnpacker",
    "TocEntry",
    "UnpackError",
    "UnpackResult",
    "VolumeStats",
    "charset_hint_for_lcid",
    "codec_family",
    "declared_charset",
    "decode_html",
    "find_unpacker",
    "page_key_of",
    "parse_index",
    "parse_system_file",
    "parse_toc",
    "read_unpack_manifest",
    "unpack_chm",
]
