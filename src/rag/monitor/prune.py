"""Delete old trace files (spec P5-02, design s15-1).

The Collector's file exporter rotates its own file (`max_megabytes`, `max_days`, `max_backups` in
`deploy/otel-collector.yaml`). This is the second line: if the Collector was stopped for a while,
renamed backups from an older config are left behind, or the setting was changed, files that no
rule of the Collector covers still go. `python -m rag.monitor prune` runs it once.

Rules, in this order, on the files that match `pattern` in one directory (subfolders and other
files are never touched):

1. a file whose last change is older than `max_age_s` is removed, except the newest matching file
   (the Collector may have it open, and a quiet system writes nothing for days);
2. if the rest are bigger than `max_total_bytes` together, the oldest go until they fit; the
   newest file stays even when it alone is bigger than the cap.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def prune_files(
    directory: str | Path,
    pattern: str,
    *,
    max_age_s: float,
    max_total_bytes: int,
    now: float,
    dry_run: bool = False,
) -> list[Path]:
    """The files removed (or, with `dry_run`, the files that would be), oldest first."""
    root = Path(directory)
    if not root.is_dir():
        return []
    found = sorted(
        ((p.stat().st_mtime, p) for p in root.glob(pattern) if p.is_file()),
        key=lambda item: (item[0], item[1].name),
    )
    doomed: list[Path] = []
    kept: list[tuple[float, Path]] = []
    newest = found[-1][1] if found else None
    for mtime, path in found:
        if path != newest and now - mtime > max_age_s:
            doomed.append(path)
        else:
            kept.append((mtime, path))
    total = sum(path.stat().st_size for _, path in kept)
    # `kept` is oldest first and ends with the newest file, which is never removed.
    while len(kept) > 1 and total > max_total_bytes:
        _, path = kept.pop(0)
        total -= path.stat().st_size
        doomed.append(path)
    if not dry_run:
        for path in doomed:
            try:
                path.unlink()
            except OSError as exc:  # open by another process on Windows: try again next run
                logger.warning("could not remove %s: %s", path, exc)
    return doomed
