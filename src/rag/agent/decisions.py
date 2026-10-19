"""What the route and rewrite hooks hand back to the graph (spec P3-01, P3-02).

Kept apart from `rag.agent.hooks` so the hooks that make these values (`rag.agent.routing`,
`rag.agent.rewrite`) and the module that lists the hooks do not import each other.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Verdict = Literal["sufficient", "insufficient"]


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
    `used_model` is set when a model call was made.
    """

    standalone_question: str | None = None
    sub_questions: tuple[str, ...] = ()
    fallback: str | None = None
    used_model: bool = False
