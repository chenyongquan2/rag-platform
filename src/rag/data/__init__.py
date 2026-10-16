"""Data sources: connectors and dataset downloads (design s5-1).

Importing this package registers the built-in connectors in `rag.registry.connectors`.
"""

# Imported for its side effect: registers the `hf_dataset` connector.
from rag.data import hf as hf
from rag.data.base import ConnectorMode, SourceDocument, make_doc_id

__all__ = ["ConnectorMode", "SourceDocument", "make_doc_id"]
