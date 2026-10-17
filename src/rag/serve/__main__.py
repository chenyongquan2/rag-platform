"""`uv run python -m rag.serve stdio` or `uv run python -m rag.serve http`."""

import sys

from rag.serve.cli import main

sys.exit(main())
