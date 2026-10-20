"""Reading, cleaning and chunking (design s5, s6, s7; spec P0-04).

Importing this package registers the readers (`rag.registry.readers`) and the cleaning rules and
chunk steps (`rag.registry.steps`) named in knowledge-base YAML.
"""

from rag.ingest import chm as chm
from rag.ingest import chunk as chunk
from rag.ingest import clean as clean
from rag.ingest import clean_chm as clean_chm
from rag.ingest import readers as readers
from rag.ingest.build import INGEST_CODE_VERSION, KbProcessor, ReadResult, pipeline_version
from rag.ingest.readers import reader_name_for

__all__ = [
    "INGEST_CODE_VERSION",
    "KbProcessor",
    "ReadResult",
    "pipeline_version",
    "reader_name_for",
]
