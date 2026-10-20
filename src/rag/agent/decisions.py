"""What the route, rewrite and judge hooks hand back to the graph (spec P3-01, P3-02, P3-03).

Kept apart from `rag.agent.hooks` so the hooks that make these values (`rag.agent.routing`,
`rag.agent.rewrite`, `rag.agent.judge`) and the module that lists the hooks do not import each
other.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from rag.agent.state import SearchQuery
from rag.types import Sufficiency

Verdict = Literal["sufficient", "insufficient"]
"""What a P3-01 judge hook returned: still accepted by the node (enough / partly enough with
every sub-question missing)."""


@dataclass(frozen=True)
class JudgeOutcome:
    """What the sufficiency judge decided after a round (design s13-4).

    `verdict`: `enough` (every open sub-question has grounds), `partly_enough` (some
    chunks passed the threshold, but a sub-question is still without grounds) or `not_enough`
    (nothing passed the threshold in any round). `missing` are the sub-questions still without
    grounds, as the first round's queries; the next round asks about these only. `fallback`
    names why the model's judgement was not used (the rules decided alone); `used_model` is set
    when the model's judgement was used.
    """

    verdict: Sufficiency
    missing: tuple[str, ...] = ()
    fallback: str | None = None
    used_model: bool = False


@dataclass(frozen=True)
class RouteDecision:
    """What the route step decided.

    `retrieve`, `kb`, `needs_clarification` and `reason` are decisions (a hook may set them
    directly). `clarity` and `multi_hop` are the rules' signals (`rag.agent.rules`): the route
    node reads them together with the knowledge base's `agent` settings, asks the small model
    when a signal is `undecided` or `maybe` and the settings allow it (`route_fallback`), and
    acts on the result. `two_hop` says the question needs two or more searches;
    `clarify_question` is the question to ask back (empty: the default one); `fallback` is the
    reason the small model was needed but failed (the rules' signals then stand).
    """

    retrieve: bool = True
    kb: str = ""
    needs_clarification: bool = False
    reason: str = "rules"
    clarity: str = "clear"
    multi_hop: str = "no"
    two_hop: bool = False
    clarify_question: str | None = None
    fallback: str | None = None


@dataclass(frozen=True)
class RewriteOutcome:
    """What a rewrite changed.

    `standalone_question` is the question made understandable on its own (it goes to the answer);
    `sub_questions` are the search queries, one search each (empty: keep what the state has);
    `fallback` names why the model's rewrite was not used (the original question is searched);
    `used_model` is set when a model call was made. From round two on (`queries`, spec P3-03):
    the searches of the round, each for one sub-question, with `technique` (`multi_query` or
    `hyde`) saying how they were made; `exhausted` says no new query could be made, so the run
    stops instead of searching what it already searched.
    """

    standalone_question: str | None = None
    sub_questions: tuple[str, ...] = ()
    fallback: str | None = None
    used_model: bool = False
    queries: tuple[SearchQuery, ...] = ()
    technique: str | None = None
    exhausted: bool = False
