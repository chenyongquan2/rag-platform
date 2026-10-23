# ruff: noqa: RUF001 (Chinese report text on purpose)
"""Spec EVAL-3 AC12, AC13: the plain-Chinese section for docs/reports/eval-upgrade-report.md."""

from __future__ import annotations

import re
from pathlib import Path

from rag.eval.agentic.repeat.analysis import analyze
from rag.eval.agentic.repeat.report import ReportContext, render_section, write_section
from rag.testing.repeat import passes_from, qlist_for

QIDS = [f"single_hop:{i:02d}" for i in range(6)]
SPEC = {
    "standard": dict(zip(QIDS, ["ccc", "ccc", "ccw", "wcc", "www", "cww"], strict=True)),
    "agentic": dict(zip(QIDS, ["ccc", "ccc", "ccc", "ccx", "ccw", "www"], strict=True)),
}


def _section(stable: str | None = "stable_questions.json") -> str:
    qlist = qlist_for(QIDS, seed=20261023)
    result = analyze(passes_from(SPEC), qlist)
    ctx = ReportContext(
        qlist=qlist,
        base="D:/data/eval-upgrade/step3",
        list_path="evals/datasets/hf_docs/repeat_step3.json",
        pass1_old=["D:/data/p3-07/p3-07", "D:/data/p3-07/p3-07-retry"],
        pass1_new=["D:/data/eval-upgrade/step2-run1"],
        generated="2026-10-30",
    )
    return render_section(result, ctx, stable_path=stable)


def test_section_content() -> None:
    text = _section()
    # What is shown and how it was run.
    assert text.startswith("<!-- eval-upgrade:step3:begin -->")
    assert text.rstrip().endswith("<!-- eval-upgrade:step3:end -->")
    assert "## 3. 第 3 步" in text
    assert "### 3.1 这一节回答什么" in text and "### 3.2 怎么跑的" in text
    for needle in (
        "20261023",  # the seed
        "P3-07",  # pass 1 is the stored run
        "缓存里还没有这些题的检索结果",
        "p 值",
        "第 2 遍先跑标准 RAG",
        "第 3 遍先跑 ask_docs",
        "只检索、不调用模型",
        "McNemar",
        "95% 区间",
        "三遍都全对",
        "至少一遍全对",
        "相邻两遍",
        "稳定题",
        "stable_questions.json",
        "不能说",
        "150 道",
    ):
        assert needle in text, needle
    # The numbers: standard 2 of 6 (pass^3), graph 3 of 6; a gap of 1 against at most 2 changes.
    assert "6 题里 2 题" in text and "6 题里 3 题" in text
    assert "多 1 题" in text
    assert "没有超过" in text  # the gap is not larger than the noise
    # The deadline cut is listed and called not a refusal.
    assert "单跳题" in text
    assert "总时限截断" in text and "不是拒答" in text
    assert "single_hop:03" in text


def test_section_has_no_arrows_or_slash_codes() -> None:
    text = _section().replace("-->", "")  # the marker comments close with it
    for banned in ("->", "→", "w/o"):
        assert banned not in text, banned
    assert not re.search(r"\d/\d", text)  # no "12/20": write "20 题里 12 题"


def test_section_without_stable_file_and_with_incomplete_questions() -> None:
    qlist = qlist_for([*QIDS, "hard:00"])
    spec = {s: dict(rows) for s, rows in SPEC.items()}
    spec["standard"]["hard:00"] = "ccn"
    spec["agentic"]["hard:00"] = "ccc"
    result = analyze(passes_from(spec), qlist)
    ctx = ReportContext(
        qlist=qlist,
        base="b",
        list_path="l.json",
        pass1_old=["o"],
        pass1_new=[],
        generated="2026-10-30",
    )
    text = render_section(result, ctx)
    assert "没有计入的题" in text and "hard:00" in text
    assert "标准 RAG 第 3 遍没有记录" in text
    assert "没有提供" in text


def test_section_is_appended_replaced_and_created(tmp_path: Path) -> None:
    path = tmp_path / "docs" / "reports" / "eval-upgrade-report.md"
    first = _section()
    write_section(path, first)  # the file does not exist yet
    created = path.read_text(encoding="utf-8")
    assert created.startswith("# ") and first in created
    assert "\r" not in created

    # Another agent's section is in the file; ours is added after it and later replaced in place.
    other = "## 第 2 步：难题\n\n别的内容。\n"
    path.write_text(created.replace(first, "") + "\n" + other, encoding="utf-8", newline="\n")
    before = path.read_text(encoding="utf-8")
    write_section(path, first)
    appended = path.read_text(encoding="utf-8")
    assert appended.startswith(before.rstrip("\n"))
    assert appended.count("<!-- eval-upgrade:step3:begin -->") == 1
    assert other in appended

    newer = first.replace("第 3 步", "第 3 步（改过）")
    write_section(path, newer)
    replaced = path.read_text(encoding="utf-8")
    assert replaced.count("<!-- eval-upgrade:step3:begin -->") == 1
    assert "（改过）" in replaced and other in replaced
    assert replaced.replace(newer, first) == appended  # nothing else moved


def test_section_keeps_the_line_endings_of_a_crlf_report(tmp_path: Path) -> None:
    path = tmp_path / "report.md"
    path.write_bytes("# 报告\r\n\r\n## 第 2 步\r\n\r\n内容\r\n".encode())
    first = _section()
    write_section(path, first)
    write_section(path, first)  # replaced in place the second time
    data = path.read_bytes()
    assert data.count(b"\n") == data.count(b"\r\n")  # every line ends with CRLF
    assert data.count(b"eval-upgrade:step3:begin") == 1
    assert "## 第 2 步".encode() in data
    assert not (tmp_path / "report.md.tmp").exists()


STUB = """# 评测改造报告

| 节 | 内容 | 状态 |
|---|---|---|
| 2 | 第 2 步：难题和含糊题的结果 | 本次已写 |
| 3 | 第 3 步：重复跑，量噪声 | 还没做 |
| 4 | 第 4 步：回归题集和自动对比 | 还没做 |

## 2. 第 2 步：难题和含糊题的结果

第 2 步的内容。

## 3. 第 3 步：重复跑，量噪声

还没做。要先提高停止线才能做。

## 4. 第 4 步：回归题集和自动对比

还没做。
"""


def test_section_replaces_the_placeholder_left_for_step_3(tmp_path: Path) -> None:
    path = tmp_path / "report.md"
    path.write_text(STUB, encoding="utf-8", newline="\n")
    section = _section()
    write_section(path, section)
    text = path.read_text(encoding="utf-8")
    assert text.count("## 3. 第 3 步") == 1  # the placeholder is gone, not repeated
    assert "还没做。要先提高停止线才能做。" not in text
    assert "| 3 | 第 3 步：重复跑，量噪声 | 已写（`repeat-analyze` 生成） |" in text
    assert "| 4 | 第 4 步：回归题集和自动对比 | 还没做 |" in text  # step 4 untouched
    assert text.index("## 2. 第 2 步") < text.index("## 3. 第 3 步") < text.index("## 4. 第 4 步")
    assert "第 2 步的内容。" in text
    assert text.endswith("## 4. 第 4 步：回归题集和自动对比\n\n还没做。\n")
    # The second run finds the markers and replaces only the section.
    write_section(path, section)
    assert path.read_text(encoding="utf-8") == text


def test_a_written_step_3_section_without_markers_is_not_replaced(tmp_path: Path) -> None:
    path = tmp_path / "report.md"
    own = "## 3. 第 3 步：手写的\n\n这一节已经写好了。\n"
    path.write_text("# 报告\n\n" + own, encoding="utf-8", newline="\n")
    write_section(path, _section())
    text = path.read_text(encoding="utf-8")
    assert own in text and text.count("## 3. 第 3 步") == 2  # appended instead
