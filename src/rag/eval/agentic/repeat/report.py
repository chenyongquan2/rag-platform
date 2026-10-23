# ruff: noqa: RUF001, E501 (the text is Chinese on purpose; its lines are long in display width)
"""The step-3 section of `docs/reports/eval-upgrade-report.md`, in plain Chinese (spec EVAL-3).

    text = render_section(analysis, ReportContext(...), stable_path="stable_questions.json")
    write_section("docs/reports/eval-upgrade-report.md", text)

The section sits between two marker comments, so a second run replaces only this section and
never touches the sections other steps wrote. A missing file is created with a title line.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from rag.eval.agentic.repeat.analysis import (
    ALL_GROUP,
    GROUP_ORDER,
    GroupResult,
    Incident,
    RepeatAnalysis,
)
from rag.eval.agentic.repeat.passes import system_order
from rag.eval.agentic.repeat.sample import QuestionList
from rag.eval.answer.record import Proportion

BEGIN = "<!-- eval-upgrade:step3:begin -->"
END = "<!-- eval-upgrade:step3:end -->"
TITLE = "# RAG 平台评测改造报告\n"

GROUP_NAMES = {
    "single_hop": "单跳题",
    "two_hop": "两跳题",
    "colloquial": "口语化题",
    "hard": "难题",
    "ambiguous": "含糊题",
    "old": "旧题合计",
    "all": "全部题",
}
SYSTEM_NAMES = {"standard": "标准 RAG", "agentic": "ask_docs"}
AFTER_VERB = {"standard": "标准 RAG", "agentic": " ask_docs"}
"""Names after a Chinese verb: `ask_docs` is ASCII and needs a space before it."""
REASONS = {
    "deadline_truncated": "回答被总时限截断（不是拒答）",
    "generation_error": "回答模型出错，例如超时",
    "error": "这道题的运行出错",
}


@dataclass
class ReportContext:
    qlist: QuestionList
    base: str
    list_path: str
    pass1_old: Sequence[str] = field(default_factory=list)
    pass1_new: Sequence[str] = field(default_factory=list)
    generated: str = ""


def _kz(k: int) -> str:
    """Number of passes in words (`三`), for the sentences."""
    return {2: "两", 3: "三"}.get(k, str(k))


def _p(p: Proportion) -> str:
    if p.rate is None:
        return "0 题"
    return f"{p.count}/{p.n} = {p.rate:.1%}，区间 {p.low:.1%} 到 {p.high:.1%}"


def _count(p: Proportion) -> str:
    return f"{p.count}/{p.n}"


def _row(cells: Sequence[object]) -> str:
    return "| " + " | ".join(str(c) for c in cells) + " |"


def _table(header: Sequence[str], rows: Sequence[Sequence[object]]) -> list[str]:
    return [_row(header), _row(["---"] * len(header)), *(_row(r) for r in rows)]


def _groups(result: RepeatAnalysis) -> list[GroupResult]:
    return [result.groups[g] for g in GROUP_ORDER if g in result.groups]


def _verdict(g: GroupResult) -> str:
    p = g.paired
    if p.p_value <= 0.05:
        return (
            "ask_docs 更高（p 不超过 0.05）"
            if p.difference > 0
            else "标准 RAG 更高（p 不超过 0.05）"
        )
    if p.improved == p.worsened:
        return "没有差别"
    direction = "ask_docs 多" if p.improved > p.worsened else "标准 RAG 多"
    return f"看不出可靠的差别（只能说方向：{direction}）"


def _changes(g: GroupResult, system: str) -> str:
    cells = []
    for f in (f for f in g.flips if f.system == system):
        cells.append(
            f"第 {f.from_pass} 遍到第 {f.from_pass + 1} 遍变了 {f.changed} 题"
            f"（{f.to_right} 题变对，{f.to_wrong} 题变错）"
        )
    return "；".join(cells)


def _clean(g: GroupResult, system: str) -> str:
    return "、".join(f"{f.changed_clean} 题" for f in g.flips if f.system == system)


def _gap_sentence(g: GroupResult, k: int) -> str:
    name = GROUP_NAMES[g.group]
    kz = _kz(k)
    if g.gap > 0:
        gap = f"ask_docs {kz}遍都全对的题比标准 RAG 多 {g.gap} 题"
    elif g.gap < 0:
        gap = f"ask_docs {kz}遍都全对的题比标准 RAG 少 {-g.gap} 题"
    else:
        gap = f"两个系统{kz}遍都全对的题一样多"
    noise = f"同一个系统相邻两遍之间结果变了的题，最多 {g.max_flips} 题，平均 {g.mean_flips:.1f} 题"
    if g.gap_larger_than_noise:
        end = "两个系统的差别超过了各自的波动，这个差别不能只当作运气。"
    elif g.gap == 0 and g.max_flips == 0:
        end = "两个系统没有差别，各自两遍之间也没有变化。"
    else:
        end = "两个系统的差别没有超过各自的波动，现在不能说哪一个更稳。"
    return f"- {name}（{g.n} 题）：{gap}；{noise}。{end}"


def _incident_rows(incidents: Sequence[Incident]) -> list[list[object]]:
    return [
        [
            SYSTEM_NAMES[i.system],
            f"第 {i.pass_no} 遍",
            i.qid,
            i.qtype,
            REASONS.get(i.reason, i.reason),
        ]
        for i in sorted(incidents, key=lambda i: (i.system, i.pass_no, i.qid))
    ]


def render_section(
    result: RepeatAnalysis, ctx: ReportContext, *, stable_path: str | None = None
) -> str:
    k = result.passes
    kz = _kz(k)
    ql = ctx.qlist
    groups = _groups(result)
    out: list[str] = [BEGIN, "", f"## 第 3 步：每题重复运行 {k} 遍，量运行之间的波动", ""]
    out += [
        f"这一节由 `python -m rag.eval.agentic repeat-analyze` 在 {ctx.generated} 生成。"
        "重新运行命令会替换这一节，报告里其他的节不会被改动。",
        "",
        "### 这一节回答什么",
        "",
        "- **同一个系统、同一道题运行几遍，结果会变吗：** 看每个系统在相邻两遍之间有几道题的结果变了。",
        "- **标准 RAG 和 ask_docs 的差别，比它们各自两遍之间的变化大吗：** "
        "如果不大，一遍的数字里看到的差别，可能只是运行的运气。",
        "",
        "### 怎么跑的",
        "",
        f"- **题目：** {sum(len(v) for v in ql.qids.values())} 道，文件 `{ctx.list_path}`，"
        f"固定种子 {ql.seed}，{ql.date} 抽出。旧题 {len(ql.old_qids())} 道"
        f"（单跳 {ql.counts.get('single_hop', 0)}、两跳 {ql.counts.get('two_hop', 0)}、"
        f"口语化 {ql.counts.get('colloquial', 0)}），难题 {ql.counts.get('hard', 0)} 道，"
        f"含糊题 {ql.counts.get('ambiguous', 0)} 道。无答案题没有重复，"
        "因为两个系统每次都把它们全部拒答。",
        "- **第 1 遍不是这次重跑的，是已有的记录：** 旧题用 P3-07 的记录（目录 "
        + ("、".join(f"`{d}`" for d in ctx.pass1_old) or "没有提供")
        + "），那一遍开始时缓存是冷的；难题和含糊题用第 2 步的第一遍记录（目录 "
        + ("、".join(f"`{d}`" for d in ctx.pass1_new) or "没有提供，所以这两类题缺第 1 遍")
        + "）。",
    ]
    for number in range(2, k + 1):
        first, second = system_order(number)
        out.append(
            f"- **第 {number} 遍：** 开始前先做一轮只检索、不调用模型的预热；"
            f"第 {number} 遍先跑{AFTER_VERB[first]}，再跑{AFTER_VERB[second]}。"
            f"记录在 `{ctx.base}` 下的 `pass{number}` 目录。"
        )
    out += [
        "- **顺序为什么交替：** 先跑的系统遇到的缓存更冷，交替以后冷热的影响两个系统各占一半。",
        "- **怎么算对：** 有答案的题，裁判给 3 分算全对；含糊题用规则判断，系统反问算对。"
        "「旧题合计」是单跳、两跳、口语化三类的合计。"
        "回答被总时限截断、回答模型出错、运行出错，都不算全对，另外列在后面，也不当作拒答。",
        f"- **{kz}遍都全对（pass^{k}，读作 pass 的 {k} 次方）：** 同一道题{kz}遍都全对，衡量稳不稳。"
        f"**至少一遍全对（pass@{k}）：** {kz}遍里有一遍全对，衡量有没有能力做到。",
        f"- **进入数字的题：** {result.questions} 道，每道题两个系统都有 {k} 遍的结果。"
        f"另有 {len(result.incomplete)} 道缺记录或没判分，没有计入（见「没有计入的题」）。",
        "",
        f"### 每个系统{kz}遍都全对和至少一遍全对（pass^{k} 和 pass@{k}）",
        "",
        "表里的区间是 95% 区间，用 Wilson 法算。题数少，区间宽，看区间比看百分比可靠。",
        "",
    ]
    out += _table(
        [
            "题型",
            "题数",
            f"标准 RAG：{kz}遍都全对",
            "标准 RAG：至少一遍全对",
            f"ask_docs：{kz}遍都全对",
            "ask_docs：至少一遍全对",
        ],
        [
            [
                GROUP_NAMES[g.group],
                g.n,
                _p(g.systems["standard"].pass_all),
                _p(g.systems["standard"].pass_any),
                _p(g.systems["agentic"].pass_all),
                _p(g.systems["agentic"].pass_any),
            ]
            for g in groups
        ],
    )
    out += ["", "### 每一遍单独看，全对的题数", ""]
    out += [
        "只看一遍时会看到的数字。同一行的三个数字差得越多，说明单遍的结果越不可靠。",
        "",
    ]
    out += _table(
        ["题型", "系统", *[f"第 {p} 遍" for p in range(1, k + 1)]],
        [
            [GROUP_NAMES[g.group], SYSTEM_NAMES[s], *[_count(p) for p in g.systems[s].per_pass]]
            for g in groups
            for s in ("standard", "agentic")
        ],
    )
    out += ["", f"### 两个系统 pass^{k} 的配对检验（McNemar）", ""]
    out += [
        f"只看两个系统结果不同的题：一边{kz}遍都全对，另一边没有。检验的是这两种题的个数是否差得比运气大。",
        "",
    ]
    out += _table(
        [
            "题型",
            "题数",
            f"只有 ask_docs {kz}遍都全对",
            f"只有标准 RAG {kz}遍都全对",
            f"两边都{kz}遍全对",
            "p 值",
            "结论",
        ],
        [
            [
                GROUP_NAMES[g.group],
                g.paired.n,
                g.paired.improved,
                g.paired.worsened,
                g.paired.both,
                f"{g.paired.p_value:.3f}",
                _verdict(g),
            ]
            for g in groups
        ],
    )
    out += ["", "### 同一个系统相邻两遍之间结果变了的题数（波动）", ""]
    out += [
        "「变了」是这道题在一遍里全对、在下一遍里不全对，或者反过来。"
        "最后一列去掉了有回答被截断、模型出错或运行出错的记录，看剩下的变化。",
        "",
    ]
    out += _table(
        ["题型", "系统", "各次相邻两遍的变化", "去掉被截断和出错的题以后，各次变了几题"],
        [
            [GROUP_NAMES[g.group], SYSTEM_NAMES[s], _changes(g, s), _clean(g, s)]
            for g in groups
            for s in ("standard", "agentic")
        ],
    )
    out += ["", "### 两个系统的差别比自己的波动大吗", ""]
    out += [
        f"做法：把「ask_docs {kz}遍都全对的题数减去标准 RAG 的」，和「同一个系统相邻两遍之间变了的题数」"
        "的最大值比。这个尺子偏严：一道题变对、另一道题变错，在波动里算两道，在差别里互相抵消。",
        "",
    ]
    out += [_gap_sentence(g, k) for g in groups]
    out += ["", "### 回答被总时限截断、模型出错或运行出错的记录", ""]
    if result.incidents:
        out += [
            "这些记录都算没有答对。被总时限截断的回答不是系统选择拒答，所以不算拒答。"
            "第 1 遍的记录也按同一条规则检查过。",
            "",
        ]
        out += _table(["系统", "哪一遍", "题号", "题型", "原因"], _incident_rows(result.incidents))
    else:
        out.append("没有这样的记录。")
    out += ["", "### 没有计入的题", ""]
    if result.incomplete:
        out += _table(
            ["题号", "原因"],
            [
                [
                    i.qid,
                    f"{SYSTEM_NAMES[i.system]} 第 {i.pass_no} 遍没有记录"
                    if i.kind == "missing"
                    else f"{SYSTEM_NAMES[i.system]} 第 {i.pass_no} 遍的回答没有判分",
                ]
                for i in result.incomplete
            ],
        )
    else:
        out.append("没有。")
    out += ["", "### 每遍实际花费（元）", ""]
    out += [
        "只算抽到的这些题的回答和判分。第 1 遍是旧记录里这些题的花费。",
        "",
    ]
    out += _table(
        ["系统", *[f"第 {p} 遍" for p in range(1, k + 1)]],
        [
            [SYSTEM_NAMES[s], *[f"{c:.4f}" for c in result.pass_cny.get(s, [])]]
            for s in ("standard", "agentic")
        ],
    )
    out += ["", f"### 稳定题：{kz}遍都全对的题", ""]
    out += [
        "第 4 步的回归题集从这里选。题数按题型和系统列出；清单文件："
        + (f"`{stable_path}`" if stable_path else "这次没有写文件")
        + "。",
        "",
    ]
    types = [
        t
        for t in ("single_hop", "two_hop", "colloquial", "hard", "ambiguous")
        if t in result.groups
    ]
    out += _table(
        ["题型", "标准 RAG", "ask_docs", "两个系统都稳定"],
        [
            [
                GROUP_NAMES[t],
                len(result.stable["standard"].get(t, [])),
                len(result.stable["agentic"].get(t, [])),
                len(result.stable["both"].get(t, [])),
            ]
            for t in types
        ],
    )
    out += ["", "### 现在能说和不能说", ""]
    total = result.groups.get(ALL_GROUP)
    if total is not None:
        out += [
            f"- **能说：** 在这 {total.n} 道抽样题上，每题运行了 {k} 遍。"
            f"ask_docs 有 {_count(total.systems['agentic'].pass_all)} 道{kz}遍都全对，"
            f"标准 RAG 有 {_count(total.systems['standard'].pass_all)} 道；"
            f"同一个系统相邻两遍之间最多有 {total.max_flips} 道题结果变了。"
            + (
                "两个系统的差别超过了这个波动。"
                if total.gap_larger_than_noise
                else f"所以两个系统相差 {total.max_flips} 道以内，只能算波动。"
            ),
        ]
    out += [
        "- **不能说：** 150 道题全部跑了三遍；无答案题的稳定性（这次没有重复）；"
        "难题和含糊题代表真实用户的问题（它们是照已有失败的样子写的，只能说「在这类题上」）。",
        "- **条件不完全一样：** 第 1 遍是另一天跑的旧记录，缓存是冷的，后面的遍有预热，"
        "网关忙闲也不同；这些差别都包含在上面的波动里，分不开。",
        "- **样本小：** 每个题型只有几道到几十道题，区间很宽；差别「没有超过波动」"
        "不等于「两个系统没有差别」，只是这批题测不出来。",
        "",
        END,
    ]
    return "\n".join(out) + "\n"


def write_section(path: str | Path, section: str) -> None:
    """Put `section` into the report: replace the previous copy between the markers, else append
    after the existing text; create the file (with a title) when it does not exist."""
    file = Path(path)
    file.parent.mkdir(parents=True, exist_ok=True)
    if not file.is_file():
        file.write_text(TITLE + "\n" + section, encoding="utf-8", newline="\n")
        return
    text = file.read_bytes().decode("utf-8")
    start = text.find(BEGIN)
    end = text.find(END, start) if start >= 0 else -1
    if start >= 0 and end >= 0:
        tail = end + len(END)
        if text[tail : tail + 1] == "\n":
            tail += 1
        new = text[:start] + section + text[tail:]
    else:
        new = text.rstrip("\n") + "\n\n" + section
    file.write_bytes(new.encode("utf-8"))
