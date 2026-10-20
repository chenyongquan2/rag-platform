"""Data sources: connectors and dataset downloads (design s5-1).

Importing this package registers the built-in connectors in `rag.registry.connectors`.
"""

# Imported for their side effect: register the `hf_dataset` and `chm_dir` connectors.
from rag.data import chm as chm
from rag.data import hf as hf
from rag.data.base import ConnectorMode, SourceDocument, make_doc_id, normalize_rel_path

__all__ = ["ConnectorMode", "SourceDocument", "make_doc_id", "normalize_rel_path"]
