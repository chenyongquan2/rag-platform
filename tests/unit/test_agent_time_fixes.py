"""Spec P3-06 AC1-AC9: the time and evidence fixes of the ask_docs graph.

What P3-05 found with real models, reproduced with scripted retrieval and a clock the test moves:
a later round that gets no time for its search, a search that comes back without rerank scores,
a retrieval error in round two, and the round gate. Fake models only.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from typing import Any

import pytest

from rag.agent import AgentHooks, AgentRuntime, RouteDecision, ask_docs
from rag.agent.decisions import JudgeOutcome
from rag.agent.nodes import plan_next
from rag.agent.state import RetrievalRound
from rag.observe import shutdown_observability
from rag.retrieve import RetrievalError
from rag.testing import FakeChatModel, StreamingFakeChatModel
from rag.testing.agent import (
    FakeClock,
    ScriptedSearchBackend,
    agent_settings,
    judge_reply,
    kb_config,
    make_context,
    multi_query_reply,
    scored_result,
    structured_reply,
)
from rag.testing.serve import fake_chunk
from rag.types import AnswerStatus, AskResult, ErrorCategory, RetrievalResult

KB = "hf_docs"


@pytest.fixture(autouse=True)
def _no_active_observability() -> Iterator[None]:
    shutdown_observability()
    yield
    shutdown_observability()


def _two_hop(question: str, *, kb: str, history: Any) -> RouteDecision:
    return RouteDecision(retrieve=True, kb=kb, two_hop=True)


def _ask(
    question: str,
    backend: ScriptedSearchBackend,
    *,
    helper: Any = None,
    judge: Any = None,
    answer: Any = None,
    agent: dict[str, Any] | None = None,
    settings: Any = None,
    hooks: AgentHooks | None = None,
    clock: Any = None,
    max_rounds: int = 3,
) -> AskResult:
    settings = settings or agent_settings()
    kwargs_clock = {"clock": clock} if clock is not None else {}
    runtime = AgentRuntime.from_settings(settings, hooks=hooks or AgentHooks(), **kwargs_clock)
    cfg = kb_config(KB, max_rounds=max_rounds, **{"judge": True, "rewrite": True, **(agent or {})})
    return asyncio.run(
        ask_docs(
            question,
            cfg=cfg,
            retriever=backend,
            settings=settings,
            runtime=runtime,
            model=answer or StreamingFakeChatModel(default_response="Use it [1]."),
            helper_model=helper if helper is not None else FakeChatModel(responses=[]),
            judge_model=judge if judge is not None else FakeChatModel(responses=[]),
        )
    )


def structured_plan(first: str, second: str) -> Any:
    return structured_reply(
        "RewritePlan", standalone_question=f"{first} and {second}", search_queries=[first, second]
    )


def _asked(backend: ScriptedSearchBackend) -> list[str]:
    return [c["question"] for c in backend.calls]


def _unscored(kb: str, pages: list[int]) -> RetrievalResult:
    """What a search returns when the reranker ran out of time before its first batch: the
    fused candidates in fusion order, no rerank score, `rerank_timeout` set, nothing cut by the
    threshold (s16-8)."""
    base = scored_result(kb, [0.5] * len(pages), pages=pages)
    return base.model_copy(
        update={
            "chunks": [c.model_copy(update={"rerank_score": None}) for c in base.chunks],
            "below_threshold": [],
            "not_found": False,
            "rerank_timeout": True,
            "rerank": [],
        }
    )


# ---- AC1: plan_next -------------------------------------------------------------------------


def _ctx_with(
    remaining_ms: int, *, judge: bool = True, rewrite: bool = True, spent_tokens: int = 0
) -> Any:
    """A context whose deadline has `remaining_ms` left of 30 s; reserve 6 s, round minimum 8 s."""
    clock = FakeClock()
    settings = agent_settings(agent_answer_reserve_ms=6_000, agent_round_min_ms=8_000)
    ctx = make_context(
        cfg=kb_config(KB, judge=judge, rewrite=rewrite, max_rounds=3),
        settings=settings,
        total_ms=30_000,
        clock=clock,
    )
    clock.advance_ms(30_000 - remaining_ms)
    ctx.budget.add(spent_tokens)
    return ctx


PARTLY = JudgeOutcome("partly_enough", ("a",))
NOT_ENOUGH = JudgeOutcome("not_enough", ("a",))


@pytest.mark.parametrize(
    ("verdict", "round_no", "remaining_ms", "round_ms", "expected", "why"),
    [
        (JudgeOutcome("enough"), 1, 25_000, 5_000, (False, None), "enough ends the loop"),
        (PARTLY, 1, 25_000, 5_000, (True, None), "time for another round"),
        (PARTLY, 1, 14_000, 5_000, (False, "deadline"), "exactly reserve + minimum is not enough"),
        (PARTLY, 1, 14_001, 5_000, (True, None), "one ms more is"),
        (PARTLY, 1, 14_001, None, (True, None), "no previous round time: the floor decides"),
        (PARTLY, 1, 16_000, 12_000, (False, "deadline"), "0.9 x 12 s = 10.8 s > the 8 s floor"),
        (PARTLY, 1, 16_900, 12_000, (True, None), "6 s + 10.8 s = 16.8 s fits in 16.9 s"),
        (PARTLY, 1, 16_900, 2_000, (True, None), "a short round keeps the 8 s floor"),
        (PARTLY, 3, 29_000, 1_000, (False, "max_rounds"), "last round used, part still missing"),
        (NOT_ENOUGH, 3, 29_000, 1_000, (False, None), "three rounds of nothing: plain not found"),
    ],
)
def test_plan_next_table(
    verdict: JudgeOutcome,
    round_no: int,
    remaining_ms: int,
    round_ms: float | None,
    expected: tuple[bool, str | None],
    why: str,
) -> None:
    ctx = _ctx_with(remaining_ms)
    assert plan_next({"round": round_no}, verdict, ctx, round_ms=round_ms) == expected, why


def test_plan_next_without_a_model_before_the_answer_and_at_the_token_cap() -> None:
    # A judged KB that allows no rewrite cannot make a new phrasing.
    assert plan_next({"round": 1}, PARTLY, _ctx_with(25_000, rewrite=False), round_ms=1_000) == (
        False,
        "no_new_query",
    )
    # No tokens left for another rewrite and judgement after keeping the answer's.
    ctx = _ctx_with(25_000, spent_tokens=19_000)
    assert plan_next({"round": 1}, PARTLY, ctx, round_ms=1_000) == (False, "token_cap")


# ---- AC7: the round minimum follows the previous round --------------------------------------


def test_round_minimum_follows_previous_round() -> None:
    """26 s in all, 6 s kept for the answer, a floor of 8 s per round. A first round of 4 s leaves
    22 s: another round needs 6 + max(8, 3.6) = 14 s. A first round of 11 s leaves 15 s, which the
    fixed floor would accept (6 + 8 = 14 s), but a round that took 11 s needs 6 + 9.9 s."""

    def run(first_round_ms: int) -> AskResult:
        clock = FakeClock()
        backend = ScriptedSearchBackend(
            KB,
            default=scored_result(KB, [0.5], pages=[1]),
            on_call=lambda q: clock.advance_ms(first_round_ms) if q == "gamma delta" else None,
        )
        helper = FakeChatModel(responses=[multi_query_reply((1, ["gamma epsilon"]))])
        judge = FakeChatModel(responses=[judge_reply(False), judge_reply(True)])
        settings = agent_settings(
            agent_deadline_ms=26_000, agent_answer_reserve_ms=6_000, agent_round_min_ms=8_000
        )
        return _ask(
            "gamma delta",
            backend,
            helper=helper,
            judge=judge,
            settings=settings,
            clock=clock,
            agent={"hyde": False},
        )

    assert run(4_000).rounds == 2
    late = run(11_000)
    assert (late.rounds, late.stop_reason, late.coverage_unconfirmed) == (1, "deadline", True)


# ---- AC2, AC3: a search of a later round needs a minimum of time -----------------------------


def test_round_two_without_search_time_stops_with_deadline() -> None:
    clock = FakeClock()
    backend = ScriptedSearchBackend(
        KB,
        default=scored_result(KB, [0.5], pages=[1]),
        on_call=lambda q: clock.advance_ms(8_000),
    )
    helper = FakeChatModel(responses=[multi_query_reply((1, ["gamma epsilon"]))])
    judge = FakeChatModel(responses=[judge_reply(False)])  # a second call would raise
    answer = StreamingFakeChatModel(default_response="Use it [1].")
    # 20 s in all, 4 s kept for the answer; round one takes 8 s. The round gate passes (12 s left
    # > 4 + 7.2 s), but a search then has 8 s and the minimum is 9 s: no search is made.
    settings = agent_settings(
        agent_deadline_ms=20_000,
        agent_answer_reserve_ms=4_000,
        agent_round_min_ms=1_000,
        agent_search_min_ms=9_000,
    )
    result = _ask(
        "gamma delta",
        backend,
        helper=helper,
        judge=judge,
        answer=answer,
        settings=settings,
        clock=clock,
        agent={"hyde": False},
    )
    assert _asked(backend) == ["gamma delta"]  # round two made no search
    assert len(judge.calls) == 1  # and the judge was not asked about what was not searched
    assert (result.rounds, result.stop_reason, result.coverage_unconfirmed) == (1, "deadline", True)
    # Still answered from round one, with the flag.
    assert result.status == AnswerStatus.ANSWERED and result.found
    assert len(answer.calls) == 1
    assert "agent_deadline" in result.error_categories


def test_searches_that_do_not_fit_are_dropped_round_robin() -> None:
    clock = FakeClock()

    def on_call(question: str) -> None:
        if question.startswith("alt"):  # every search of round two takes 5 s
            clock.advance_ms(5_000)

    backend = ScriptedSearchBackend(
        KB, default=scored_result(KB, [0.5], pages=[1]), on_call=on_call
    )
    helper = FakeChatModel(
        responses=[
            structured_plan("alpha setting", "beta setting"),
            multi_query_reply(
                (1, ["alt alpha one", "alt alpha two"]), (2, ["alt beta one", "alt beta two"])
            ),
        ]
    )
    judge = FakeChatModel(responses=[judge_reply(False, False), judge_reply(True, True)])
    # 17 s total, 4 s reserve, minimum 4 s per search: 13 s of search time. The first search
    # starts with 13 s, the second with 8 s, the third with 3 s: it does not fit.
    settings = agent_settings(
        agent_deadline_ms=17_000,
        agent_answer_reserve_ms=4_000,
        agent_round_min_ms=1_000,
        agent_search_min_ms=4_000,
    )
    result = _ask(
        "alpha and beta",
        backend,
        helper=helper,
        judge=judge,
        settings=settings,
        clock=clock,
        hooks=AgentHooks(decide_route=_two_hop),
        agent={"hyde": False},
    )
    # Taken round-robin over the sub-questions: one phrasing of each, not two of the first.
    assert _asked(backend) == [
        "alpha setting",
        "beta setting",
        "alt alpha one",
        "alt beta one",
    ]
    assert result.rounds == 2 and result.status == AnswerStatus.ANSWERED


# ---- AC4: no scores, no evidence ----------------------------------------------------------------


def test_unscored_round_two_is_not_found_and_not_answered() -> None:
    backend = ScriptedSearchBackend(
        KB,
        by_question={
            "gamma delta": scored_result(KB, [0.1], pages=[7]),
            "gamma epsilon": _unscored(KB, [8, 9]),
        },
    )
    helper = FakeChatModel(responses=[multi_query_reply((1, ["gamma epsilon"]))])
    judge = FakeChatModel(responses=[])  # nothing is above the threshold: no judge call at all
    answer = FakeChatModel(responses=[])  # any call raises: nothing was found, nothing is asked
    result = _ask("gamma delta", backend, helper=helper, judge=judge, answer=answer, max_rounds=2)
    assert _asked(backend) == ["gamma delta", "gamma epsilon"]
    assert result.status == AnswerStatus.NOT_FOUND and not result.found
    assert answer.calls == [] and judge.calls == []
    assert result.sources == []  # the unscored chunks are no source
    assert result.rerank_timeout  # the flag stays: the user can see the search was cut short
    assert result.sufficiency == "not_enough"


def test_a_partly_scored_round_two_keeps_what_the_reranker_judged() -> None:
    base = scored_result(KB, [0.9, 0.2, 0.5], pages=[8, 9, 10])
    partial = base.model_copy(
        update={
            "chunks": [
                base.chunks[0],  # scored 0.9
                fake_chunk(KB, 9, rerank_score=0.2),  # scored below the threshold 0.3
                fake_chunk(KB, 10),  # never scored
            ],
            "rerank_timeout": True,
            "not_found": False,
            "below_threshold": [],
        }
    )
    found = RetrievalRound.from_result(2, "q", partial, scored_only=True)
    # 0.9 is evidence; 0.2 is below the threshold; the unscored chunk is nothing.
    assert [c.chunk_id for c in found.chunks] == ["hf_docs:p8.md#0"]
    assert [c.chunk_id for c in found.below_threshold] == ["hf_docs:p9.md#0"]
    assert not found.not_found and found.rerank_timeout

    # A complete rerank is taken as it is, and so is a search without a rerank timeout.
    whole = RetrievalRound.from_result(1, "q", base)
    assert [c.chunk_id for c in whole.chunks] == [c.chunk_id for c in base.chunks]


def test_a_round_of_scored_chunks_below_the_threshold_and_unscored_ones_is_not_found() -> None:
    base = scored_result(KB, [0.2, 0.1], pages=[8, 9])
    partial = base.model_copy(
        update={
            "chunks": [
                fake_chunk(KB, 8, rerank_score=0.2),
                fake_chunk(KB, 9, rerank_score=0.1),
                fake_chunk(KB, 10),
            ],
            "rerank_timeout": True,
            "not_found": False,
            "below_threshold": [],
        }
    )
    found = RetrievalRound.from_result(2, "q", partial, scored_only=True)
    assert found.chunks == [] and found.not_found is True and found.rerank_timeout
    assert [c.chunk_id for c in found.below_threshold] == ["hf_docs:p8.md#0", "hf_docs:p9.md#0"]


def test_a_score_equal_to_the_threshold_is_evidence() -> None:
    base = scored_result(KB, [0.3], pages=[8])
    partial = base.model_copy(
        update={
            "chunks": [fake_chunk(KB, 8, rerank_score=0.3), fake_chunk(KB, 9)],
            "rerank_timeout": True,
            "not_found": False,
        }
    )
    found = RetrievalRound.from_result(2, "q", partial, scored_only=True)
    assert [c.chunk_id for c in found.chunks] == ["hf_docs:p8.md#0"]
    assert found.not_found is False


def test_round_one_keeps_the_fused_chunks_after_a_rerank_timeout() -> None:
    """Review probe (P3-06): in round one nothing else was found, so the unscored fused chunks
    stay for the answer (design s16-8) and the round says `rerank_timeout`; the judge does not
    count them as grounds (they are no evidence that the threshold passed)."""
    partial = _unscored(KB, [8, 9])
    found = RetrievalRound.from_result(1, "q", partial)
    assert [c.chunk_id for c in found.chunks] == ["hf_docs:p8.md#0", "hf_docs:p9.md#0"]
    assert found.not_found is False and found.rerank_timeout


def test_a_completely_unscored_round_one_is_still_answered_and_flagged() -> None:
    backend = ScriptedSearchBackend(KB, default=_unscored(KB, [8, 9]))
    judge = FakeChatModel(responses=[])  # the judge model is not asked about unscored chunks
    answer = StreamingFakeChatModel(default_response="Use it [1].")
    result = _ask("gamma delta", backend, judge=judge, answer=answer, max_rounds=1)
    assert result.status == AnswerStatus.ANSWERED and result.found
    assert len(answer.calls) == 1 and judge.calls == []
    assert [c.chunk_id for c in result.sources] == ["hf_docs:p8.md#0", "hf_docs:p9.md#0"]
    assert result.degraded is False  # nothing fell back
    assert result.rerank_timeout and "rerank_timeout" in result.error_categories
    assert result.sufficiency == "not_enough"  # no chunk is known to pass the threshold
    assert result.cache_allowed is False


# ---- AC6: a retrieval error in a later round ----------------------------------------------


def _raise_on(question: str, error: Exception) -> Any:
    def on_call(q: str) -> None:
        if q == question:
            raise error

    return on_call


def test_retrieval_error_in_round_two_still_answers() -> None:
    error = RetrievalError(
        "deadline passed before the search was sent", category=ErrorCategory.RETRIEVAL_TIMEOUT
    )
    backend = ScriptedSearchBackend(
        KB,
        default=scored_result(KB, [0.5], pages=[1]),
        on_call=_raise_on("gamma epsilon", error),
    )
    helper = FakeChatModel(responses=[multi_query_reply((1, ["gamma epsilon"]))])
    judge = FakeChatModel(responses=[judge_reply(False)])
    answer = StreamingFakeChatModel(default_response="Use it [1].")
    result = _ask("gamma delta", backend, helper=helper, judge=judge, answer=answer)
    assert _asked(backend) == ["gamma delta", "gamma epsilon"]
    assert result.status == AnswerStatus.ANSWERED and result.found and len(answer.calls) == 1
    # An accurate reason: the search failed, the time did not run out (spec P3-06 review).
    assert (result.stop_reason, result.coverage_unconfirmed) == ("retrieval_error", True)
    assert result.degraded and result.fallbacks == ["retrieve:retrieval_timeout"]
    assert result.retrieval_error == ErrorCategory.RETRIEVAL_TIMEOUT
    assert "retrieval_timeout" in result.error_categories
    assert "agent_deadline" not in result.error_categories
    assert result.cache_allowed is False


def test_retrieval_error_in_round_one_raises() -> None:
    error = RetrievalError("no route answered", category=ErrorCategory.RETRIEVAL_FAILED)
    backend = ScriptedSearchBackend(KB, raises=error)
    with pytest.raises(RetrievalError):
        _ask("gamma delta", backend)


def test_a_failed_search_after_a_partly_found_round_stops_the_run_and_answers() -> None:
    """Two phrasings in round two: the first is found, the second fails. What the first found
    is kept (round two counts), the run stops with `retrieval_error` (no judge call) and answers."""
    error = RetrievalError("es is down", category=ErrorCategory.RETRIEVAL_FAILED)
    backend = ScriptedSearchBackend(
        KB,
        by_question={
            "gamma delta": scored_result(KB, [0.5], pages=[1]),
            "gamma epsilon": scored_result(KB, [0.6], pages=[2]),
        },
        on_call=_raise_on("gamma zeta", error),
    )
    helper = FakeChatModel(responses=[multi_query_reply((1, ["gamma epsilon", "gamma zeta"]))])
    judge = FakeChatModel(responses=[judge_reply(False)])  # a second call would raise
    answer = StreamingFakeChatModel(default_response="Use it [1].")
    result = _ask("gamma delta", backend, helper=helper, judge=judge, answer=answer)
    assert _asked(backend) == ["gamma delta", "gamma epsilon", "gamma zeta"]
    assert (result.rounds, result.stop_reason) == (2, "retrieval_error")
    assert result.retrieval_error == ErrorCategory.RETRIEVAL_FAILED
    assert result.status == AnswerStatus.ANSWERED and len(judge.calls) == 1
    assert {c.chunk_id for c in result.sources} >= {"hf_docs:p2.md#0"}


def test_a_search_may_start_at_exactly_the_minimum_time() -> None:
    clock = FakeClock()
    settings = agent_settings(
        agent_deadline_ms=30_000, agent_answer_reserve_ms=7_000, agent_search_min_ms=6_000
    )
    ctx = make_context(settings=settings, total_ms=30_000, clock=clock)
    clock.advance_ms(30_000 - 13_000)  # 13 s left: 7 s for the answer, exactly 6 s to search
    assert ctx.search_time_ms() == pytest.approx(6_000) and ctx.can_search()
    clock.advance_ms(1)
    assert not ctx.can_search()


def test_the_response_exposes_the_retrieval_error() -> None:
    """What the report and the MCP description say is there: the category and the stop reason
    reach the caller in `AskResponse`."""
    from rag.serve.schema import AskResponse

    result = AskResult(
        request_id="r",
        kb=KB,
        degraded=True,
        retrieval_error=ErrorCategory.RETRIEVAL_TIMEOUT,
        stop_reason="retrieval_error",
        coverage_unconfirmed=True,
        notice="n",
    )
    response = AskResponse.from_result(result)
    assert (response.retrieval_error, response.stop_reason) == (
        "retrieval_timeout",
        "retrieval_error",
    )
    assert "retrieval_error" in AskResponse.model_json_schema()["properties"]
    assert AskResponse.from_result(AskResult(request_id="r", kb=KB)).retrieval_error is None


def test_the_graph_caps_the_answer_call_and_other_calls_keep_their_profile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Review item S5 (P3-06): the runtime builds the answer model with `agent_answer_max_tokens`
    (one answer ran to 4096 tokens and past the deadline); the helper and judge models are built
    from their profiles as before."""
    import rag.llm

    built: list[tuple[str, int | None]] = []

    def fake_factory(profile: str, settings: Any = None, **kwargs: Any) -> Any:
        built.append((profile, kwargs.get("max_tokens")))
        return FakeChatModel(responses=[])

    monkeypatch.setattr(rag.llm, "make_chat_model", fake_factory)
    settings = agent_settings(agent_answer_max_tokens=900)
    runtime = AgentRuntime.from_settings(settings)
    runtime.default_model()
    runtime.default_helper_model()
    assert built == [("online", 900), ("cheap", None)]


# ---- round2_on_not_enough (config key, default = the design's behaviour) ----------------------


def test_round2_on_not_enough_is_on_by_default_and_plan_next_follows_it() -> None:
    ctx_on = _ctx_with(25_000)
    assert ctx_on.cfg.agent.round2_on_not_enough is True
    assert plan_next({"round": 1}, NOT_ENOUGH, ctx_on, round_ms=1_000) == (True, None)
    # Off: a plain "not enough" goes to "not found" (no stop reason); "partly enough", which
    # names a missing sub-question, still gets its round.
    ctx = make_context(
        cfg=kb_config(KB, judge=True, rewrite=True, round2_on_not_enough=False),
        settings=agent_settings(agent_answer_reserve_ms=6_000, agent_round_min_ms=8_000),
    )
    assert plan_next({"round": 1}, NOT_ENOUGH, ctx, round_ms=1_000) == (False, None)
    assert plan_next({"round": 1}, PARTLY, ctx, round_ms=1_000) == (True, None)


def test_with_round2_on_not_enough_off_a_plain_miss_is_answered_as_not_found() -> None:
    nothing = scored_result(KB, [0.1], pages=[7])
    cases = (({}, 2, 1), ({"round2_on_not_enough": False}, 1, 0))
    for agent, expected_rounds, helper_calls in cases:
        backend = ScriptedSearchBackend(KB, default=nothing)
        helper = FakeChatModel(responses=[multi_query_reply((1, ["gamma epsilon"]))])
        result = _ask(
            "gamma delta",
            backend,
            helper=helper,
            answer=FakeChatModel(responses=[]),  # nothing found: no answer call
            agent={"hyde": False, **agent},
            max_rounds=2,
        )
        assert result.rounds == expected_rounds and len(helper.calls) == helper_calls
        assert result.status == AnswerStatus.NOT_FOUND and result.stop_reason is None
        assert [c.chunk_id for c in result.nearest] == ["hf_docs:p7.md#0"]


def test_only_searches_of_round_two_and_later_ask_for_the_rerank_reserve() -> None:
    """Review must-fix 1 (P3-06): round one is the plain search `search_docs` makes (reserve 0);
    later rounds pass `agent_rerank_reserve_ms`."""
    backend = ScriptedSearchBackend(KB, default=scored_result(KB, [0.5], pages=[1]))
    helper = FakeChatModel(responses=[multi_query_reply((1, ["gamma epsilon"]))])
    judge = FakeChatModel(responses=[judge_reply(False), judge_reply(True)])
    settings = agent_settings(agent_rerank_reserve_ms=2_500)
    _ask(
        "gamma delta", backend, helper=helper, judge=judge, settings=settings, agent={"hyde": False}
    )
    assert [(c["question"], c["rerank_reserve_ms"]) for c in backend.calls] == [
        ("gamma delta", 0),
        ("gamma epsilon", 2_500),
    ]
