"""Spec P3-02 AC12-AC17: asking the caller back with `interrupt`, and continuing with a token.

Everything goes through the public `ask_docs`; the checkpoint, the history and the token store
are the real ones (memory, or SQLite files for the restart test).
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from rag.agent import (
    AgentDeadlineExceeded,
    AgentRuntime,
    ResumeTokenInvalid,
    ask_docs,
)
from rag.observe import shutdown_observability
from rag.testing import FakeChatModel, StreamingFakeChatModel
from rag.testing.agent import (
    ScriptedSearchBackend,
    agent_settings,
    kb_config,
    structured_reply,
)
from rag.types import AskResult

KB = "hf_docs"
VAGUE = "How do I fix this error?"  # no topic words, nothing named: the rules ask back
ANSWER = "transformers raises KeyError 'llama' when loading the model"


@pytest.fixture(autouse=True)
def _no_active_observability() -> Iterator[None]:
    shutdown_observability()
    yield
    shutdown_observability()


class Rig:
    """One runtime, one backend, one answer model: what a service process has."""

    def __init__(self, settings: Any = None, *, kb: str = KB, **agent: Any) -> None:
        self.settings = settings or agent_settings()
        self.runtime = AgentRuntime.from_settings(self.settings)
        self.backend = ScriptedSearchBackend(kb)
        self.answers = StreamingFakeChatModel(default_response="Use the new loader [1].")
        self.helper = FakeChatModel(responses=[])
        self.agent: dict[str, Any] = {"clarify": True, **agent}
        self.kb = kb

    def ask(self, question: str = VAGUE, **kwargs: Any) -> AskResult:
        return asyncio.run(self.aask(question, **kwargs))

    async def aask(self, question: str = VAGUE, **kwargs: Any) -> AskResult:
        kwargs.setdefault("runtime", self.runtime)
        return await ask_docs(
            question,
            cfg=kb_config(self.kb, **self.agent),
            retriever=self.backend,
            settings=self.settings,
            model=self.answers,
            helper_model=self.helper,
            **kwargs,
        )

    def resume(self, token: str, answer: str = ANSWER, **kwargs: Any) -> AskResult:
        return asyncio.run(self.aresume(token, answer, **kwargs))

    async def aresume(self, token: str, answer: str = ANSWER, **kwargs: Any) -> AskResult:
        kwargs.setdefault("session_id", "s1")
        return await self.aask("(ignored)", resume_token=token, answer=answer, **kwargs)

    def history(self, session: str = "s1", caller: str = "local") -> list[Any]:
        key = self.runtime.session_key(caller, self.kb, session)
        return asyncio.run(self.runtime.store.load(key))


def test_clarification_round_trip() -> None:
    rig = Rig()
    first = rig.ask(session_id="s1", request_id="r1")
    assert first.status == "need_clarification"
    assert "Could you be more specific?" in first.answer  # the question to ask
    assert first.resume_token and first.thread_id.startswith("s1:")
    assert first.resume_token not in first.thread_id  # the token is not the thread
    assert first.cache_allowed is False and first.rounds == 0
    assert rig.backend.calls == [] and rig.answers.calls == []  # nothing searched or answered
    assert rig.history() == []  # nothing is remembered while the question is open

    done = rig.resume(first.resume_token, request_id="r2")
    assert done.status == "answered" and done.found and done.clarified
    assert done.resume_token is None and done.request_id == "r2"
    assert done.cache_allowed is False  # a clarified answer is not for the exact cache
    merged = f"{VAGUE} ({ANSWER})"
    assert [c["question"] for c in rig.backend.calls] == [merged]
    assert merged in str(rig.answers.calls[0][-1].content)  # the answer is for the whole question
    assert done.rounds == 1 and done.thread_id == first.thread_id  # the same checkpoint thread
    assert done.request_id == "r2" and rig.backend.calls[0]["request_id"] == "r2"


def test_clarification_with_rewrite_uses_the_answer_in_the_standalone_question() -> None:
    rig = Rig(rewrite=True)
    rig.helper = FakeChatModel(
        responses=[
            structured_reply(
                "RewritePlan",
                standalone_question="How do I fix KeyError 'llama' in transformers?",
                search_queries=["Fix KeyError 'llama' in transformers by upgrading transformers"],
            )
        ]
    )
    first = rig.ask(session_id="s1")
    done = rig.resume(first.resume_token or "")
    assert done.standalone_question == "How do I fix KeyError 'llama' in transformers?"
    assert rig.backend.calls[0]["question"].startswith("Fix KeyError 'llama'")
    prompt = "\n".join(str(m.content) for m in rig.helper.calls[0])
    assert VAGUE in prompt and ANSWER in prompt  # the model sees both
    assert "Keep exactly as written: KeyError" in prompt  # the identifier of the answer


def test_second_clarification_is_not_asked() -> None:
    rig = Rig()
    first = rig.ask(session_id="s1")
    # The answer is as vague as the question: it is searched anyway, the caller is not asked
    # again. (A new call with a new question may ask again: that is another request.)
    done = rig.resume(first.resume_token or "", answer="I do not know")
    assert done.status == "answered" and done.resume_token is None
    assert len(rig.backend.calls) == 1
    again = rig.ask(session_id="s2")  # a session with no history
    assert again.status == "need_clarification" and again.resume_token != first.resume_token


def test_resume_token_is_checked() -> None:
    rig = Rig()
    first = rig.ask(session_id="s1")
    token = first.resume_token or ""

    def rejected(**kwargs: Any) -> None:
        with pytest.raises(ResumeTokenInvalid):
            rig.resume(**{"token": token, **kwargs})  # type: ignore[arg-type]

    rejected(caller_id="mallory")  # another caller
    rejected(session_id="s2")  # another session
    rejected(session_id=None)  # no session at all
    with pytest.raises(ResumeTokenInvalid):  # another knowledge base
        asyncio.run(
            ask_docs(
                "(ignored)",
                cfg=kb_config("other_docs", **rig.agent),
                retriever=ScriptedSearchBackend("other_docs"),
                settings=rig.settings,
                runtime=rig.runtime,
                model=rig.answers,
                helper_model=rig.helper,
                session_id="s1",
                resume_token=token,
                answer=ANSWER,
            )
        )
    for unknown in ("not-a-token", "x" * 32, ""):
        with pytest.raises(ResumeTokenInvalid):
            rig.resume(unknown)
    # None of that searched, answered, wrote history, or used the token up.
    assert rig.backend.calls == [] and rig.answers.calls == [] and rig.history() == []
    done = rig.resume(token)
    assert done.status == "answered"

    # Used once: the same token again is refused, and so is the first token of a new question.
    rejected()
    # A token that has expired (a clock far ahead) is refused.
    second = rig.ask(session_id="s1")
    rig.runtime.resumes._now = lambda: time.time() + 10_000  # type: ignore[attr-defined]
    with pytest.raises(ResumeTokenInvalid):
        rig.resume(second.resume_token or "")


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_abandoned_questions_are_cleaned_up(backend: str, tmp_path: Path) -> None:
    settings = None
    if backend == "sqlite":
        settings = agent_settings(
            checkpoint_url=f"sqlite://{(tmp_path / 'cp.db').as_posix()}",
            session_store_url=f"sqlite://{(tmp_path / 'sessions.db').as_posix()}",
        )
    rig = Rig(settings)
    abandoned = rig.ask(session_id="s1")
    assert asyncio.run(rig.runtime.state_values(abandoned.thread_id))  # waiting in the store
    rig.runtime.resumes._now = lambda: time.time() + 10_000  # type: ignore[attr-defined]
    rig.ask(session_id="s2")  # the next question asked back drops what expired
    assert asyncio.run(rig.runtime.state_values(abandoned.thread_id)) == {}
    with pytest.raises(ResumeTokenInvalid):
        rig.resume(abandoned.resume_token or "")


def test_foreign_caller_does_not_consume_the_token() -> None:
    rig = Rig()
    token = rig.ask(session_id="s1", caller_id="alice").resume_token or ""
    for _ in range(3):
        with pytest.raises(ResumeTokenInvalid):
            rig.resume(token, caller_id="mallory")
    done = rig.resume(token, caller_id="alice")  # the owner can still answer
    assert done.status == "answered"
    assert rig.history(caller="alice")[0].question == VAGUE and rig.history(caller="mallory") == []


def test_token_of_a_finished_thread_is_rejected() -> None:
    rig = Rig()
    finished = rig.ask("push_to_hub parameters", session_id="s1")  # a run that completed
    waiting_elsewhere = rig.ask(session_id="s2")  # a run waiting in another session
    answered_before = len(rig.answers.calls)

    async def forge(thread: str, session: str) -> str:
        return await rig.runtime.resumes.create(
            caller_id="local", kb=KB, session_id=session, thread_id=thread
        )

    # The registry itself says yes (caller, KB and session match), the thread is not waiting.
    token = asyncio.run(forge(finished.thread_id, "s1"))
    with pytest.raises(ResumeTokenInvalid, match="not waiting"):
        rig.resume(token)
    # A thread that waits but belongs to another session is refused as well.
    token = asyncio.run(forge(waiting_elsewhere.thread_id, "s1"))
    with pytest.raises(ResumeTokenInvalid, match="another session"):
        rig.resume(token)
    # And a thread that does not exist.
    token = asyncio.run(forge("s1:nothing", "s1"))
    with pytest.raises(ResumeTokenInvalid):
        rig.resume(token)
    assert len(rig.answers.calls) == answered_before


def test_two_resumes_at_once_continue_the_run_only_once() -> None:
    rig = Rig()
    token = rig.ask(session_id="s1").resume_token or ""

    async def both() -> list[Any]:
        return await asyncio.gather(rig.aresume(token), rig.aresume(token), return_exceptions=True)

    outcomes = asyncio.run(both())
    assert sorted(type(o).__name__ for o in outcomes) == ["AskResult", "ResumeTokenInvalid"]
    assert len(rig.backend.calls) == 1


def test_lock_is_not_held_across_the_interrupt() -> None:
    rig = Rig()
    first = rig.ask(session_id="s1")
    assert len(rig.runtime.locks) == 0  # released when the call returned

    async def meanwhile() -> AskResult:
        # Another request of the same session runs at once, while the question is open.
        return await asyncio.wait_for(rig.aask("push_to_hub parameters", session_id="s1"), 5)

    other = asyncio.run(meanwhile())
    assert other.status == "answered" and other.used_history is False
    done = rig.resume(first.resume_token or "")
    assert done.status == "answered"
    # History is in the order things finished: the other request, then the clarified one.
    assert [t.question for t in rig.history()] == ["push_to_hub parameters", VAGUE, ANSWER]
    assert len(rig.runtime.locks) == 0


def test_resume_waits_for_the_session_lock() -> None:
    rig = Rig()
    token = rig.ask(session_id="s1").resume_token or ""
    key = rig.runtime.session_key("local", KB, "s1")

    async def scenario() -> None:
        async with rig.runtime.locks.hold(key):  # another request of the session is running
            task = asyncio.create_task(rig.aresume(token))
            await asyncio.sleep(0.1)
            assert not task.done()  # it waits for its turn
            # A resume with a short deadline gives up (the wait counts against it) ...
            with pytest.raises(AgentDeadlineExceeded):
                await rig.aresume(token, deadline_ms=100)
        done = await asyncio.wait_for(task, 5)
        assert done.status == "answered"

    asyncio.run(scenario())
    # ... and giving up did not use the token: the waiting one got it.
    assert len(rig.backend.calls) == 1


def test_a_resume_that_gave_up_waiting_keeps_the_token() -> None:
    rig = Rig()
    token = rig.ask(session_id="s1").resume_token or ""
    key = rig.runtime.session_key("local", KB, "s1")

    async def scenario() -> AskResult:
        async with rig.runtime.locks.hold(key):
            with pytest.raises(AgentDeadlineExceeded):
                await rig.aresume(token, deadline_ms=100)
        return await rig.aresume(token)

    assert asyncio.run(scenario()).status == "answered"


def test_history_after_resume_has_the_clarifying_exchange() -> None:
    rig = Rig()
    first = rig.ask(session_id="s1", request_id="r1")
    assert rig.history() == []
    done = rig.resume(first.resume_token or "", request_id="r2")
    turns = rig.history()
    assert [(t.request_id, t.question, t.answer) for t in turns] == [
        ("r1", VAGUE, first.answer),  # the question, and what was asked back
        ("r2", ANSWER, done.answer),  # the reply, and the answer
    ]
    # The next follow-up sees the conversation as it happened.
    follow_up = rig.ask("push_to_hub parameters", session_id="s1")
    assert follow_up.used_history

    # No session: nothing is stored, and the run still works.
    bare = rig.ask(session_id=None)
    token = bare.resume_token or ""
    assert rig.resume(token, session_id=None).status == "answered"


def test_resume_after_restart_with_sqlite(tmp_path: Path) -> None:
    settings = agent_settings(
        checkpoint_url=f"sqlite://{(tmp_path / 'cp.db').as_posix()}",
        session_store_url=f"sqlite://{(tmp_path / 'sessions.db').as_posix()}",
    )
    before = Rig(settings)
    first = before.ask(session_id="s1")
    token = first.resume_token or ""

    after = Rig(settings)  # a new process: new runtime objects over the same files
    done = after.resume(token)
    assert done.status == "answered" and done.clarified
    assert [t.question for t in after.history()] == [VAGUE, ANSWER]

    again = Rig(settings)  # and a third one still knows the token is used
    with pytest.raises(ResumeTokenInvalid):
        again.resume(token)
    # A token only lives in its hash: the file never contains the token itself.
    assert token.encode() not in (tmp_path / "sessions.db").read_bytes()
    assert token.encode() not in b"".join(p.read_bytes() for p in tmp_path.glob("cp.db*"))


def test_token_budget_spans_both_calls() -> None:
    # Cap 7,000, 6,000 kept for the answer. The route's small-model call spends 1,100 tokens
    # before the question is asked back. The continuing call must start from those 1,100: its
    # rewrite (needs 6,000 left) is refused. With a fresh budget it would have been sent.
    settings = agent_settings(agent_token_cap=7_000, agent_answer_reserve_tokens=6_000)
    rig = Rig(settings, route_model=True, rewrite=True)
    rig.helper = FakeChatModel(
        responses=[
            structured_reply(
                "RouteJudgement",
                usage=(900, 200),
                needs_clarification=True,
                clarifying_question="Which model?",
                two_hop=False,
            )
        ]
    )
    first = rig.ask("How do I save this model to the hub?", session_id="s1")
    assert first.status == "need_clarification" and first.answer == "Which model?"
    assert first.tokens_used == 1_100
    done = rig.resume(first.resume_token or "", answer="bert-base-uncased")
    assert len(rig.helper.calls) == 1  # the rewrite call was refused before it was sent
    assert done.fallbacks == ["rewrite:token_cap"] and done.tokens_used >= 1_100
    assert done.status == "answered"
