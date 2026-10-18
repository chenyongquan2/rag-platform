"""Answer eval: standard RAG answers, format compliance, LLM judge, human agreement (spec P1-04).

    run = await run_answer_eval(items, RetrieverContexts(pipeline, kb="hf_docs", top_k=8),
                                kb_config=cfg, model=chat, chat_model="deepseek-v4-flash")
    judged = await judge_answers(run, model=judge_chat, config=JudgeConfig())

uv run python -m rag.eval.answer run --kb hf_docs [--limit N] [--label L]
uv run python -m rag.eval.answer judge --answers RUN [--scales 1-5,1-3] [--repeat 2]
uv run python -m rag.eval.answer stability JUDGE_A JUDGE_B [--scale 1-5]
uv run python -m rag.eval.answer human-sheet --answers RUN [--n 50] [--seed 0]
uv run python -m rag.eval.answer human-agreement --sheet CSV --key CSV --judge RUN
uv run python -m rag.eval.answer security [--label L]
uv run python -m rag.eval.answer project --answers PILOT [--judge PILOT_JUDGE] [--questions 65]
"""

from rag.eval.answer.compliance import format_compliance
from rag.eval.answer.cost import CostProjection, project_cost
from rag.eval.answer.record import (
    AnswerItem,
    AnswerQuestionRecord,
    AnswerRunRecord,
    FormatCompliance,
    Proportion,
    Usage,
)
from rag.eval.answer.runner import (
    RetrievedContext,
    RetrieverContexts,
    items_from_question_set,
    load_answer_run,
    run_answer_eval,
    write_answer_run,
)

__all__ = [
    "AnswerItem",
    "AnswerQuestionRecord",
    "AnswerRunRecord",
    "CostProjection",
    "FormatCompliance",
    "Proportion",
    "RetrievedContext",
    "RetrieverContexts",
    "Usage",
    "format_compliance",
    "items_from_question_set",
    "load_answer_run",
    "project_cost",
    "run_answer_eval",
    "write_answer_run",
]
