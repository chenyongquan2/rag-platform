"""Print the plain text of one section of the design doc.

Usage: python scripts/design_section.py s7-4 [s9-1 ...]
"""

import html
import re
import sys
from pathlib import Path

DOC = Path(__file__).resolve().parent.parent / "docs" / "design" / "rag-platform-design.html"


def section_text(src: str, sid: str) -> str:
    m = re.search(rf'<h([234])[^>]*id="{re.escape(sid)}"', src)
    if not m:
        return f"[section {sid} not found]"
    level = int(m.group(1))
    start = m.start()
    # stop at the next heading of the same or higher level
    nxt = re.compile(rf"<h([2-{level}])[^>]*id=")
    m2 = nxt.search(src, m.end())
    chunk = src[start : m2.start() if m2 else len(src)]
    chunk = re.sub(r"<svg.*?</svg>", "[figure]", chunk, flags=re.S)
    chunk = re.sub(r"<(br|p|li|tr|div|pre|h\d|summary|details|table)[^>]*>", "\n", chunk)
    chunk = re.sub(r"<t[dh][^>]*>", " | ", chunk)
    chunk = re.sub(r"<[^>]+>", "", chunk)
    return re.sub(r"\n\s*\n+", "\n", html.unescape(chunk)).strip()


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    src = DOC.read_text(encoding="utf-8")
    for sid in sys.argv[1:]:
        print(f"===== {sid} =====")
        print(section_text(src, sid))


if __name__ == "__main__":
    main()
