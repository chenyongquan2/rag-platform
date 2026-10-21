"""Draw the demo chart of the `pdf_demo` knowledge base into data/raw/pdf_demo/ (spec P6-01)."""

from __future__ import annotations

from pathlib import Path

from rag.testing.pdf import make_demo_chart

if __name__ == "__main__":
    out = make_demo_chart(Path("data/raw/pdf_demo/quarterly_revenue.png"))
    print(f"wrote {out} ({out.stat().st_size} bytes)")
