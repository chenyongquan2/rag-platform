"""Spec P7-01 (2): one tiny call per candidate chat model through the project's client.

Checks that the gateway serves the model and what usage it reports (prompt, completion and total
tokens: a model whose total is larger than prompt plus completion bills hidden reasoning that the
ledger cannot see). Each call goes to the cost ledger. MODEL_PRICES must list the models.

  uv run python scripts/p7_01/ping_models.py MODEL [MODEL ...]
"""

from __future__ import annotations

import sys
import time

from rag.config import ServiceSettings
from rag.llm import make_chat_model


def main(models: list[str]) -> int:
    base = ServiceSettings()
    for model in models:
        profile = base.chat_online.model_copy(update={"model": model, "read_timeout_s": 60})
        settings = base.model_copy(update={"chat_online": profile})
        chat = make_chat_model("online", settings)
        start = time.perf_counter()
        try:
            msg = chat.invoke("Answer in one short sentence: what is the capital of France?")
        except Exception as exc:  # report and go on to the next model
            print(f"{model}: {type(exc).__name__}: {exc}")
            continue
        ms = (time.perf_counter() - start) * 1000
        meta = getattr(msg, "response_metadata", {}) or {}
        usage = meta.get("token_usage") or getattr(msg, "usage_metadata", None)
        print(f"{model}: {ms:.0f} ms; usage {usage}; text {str(msg.content)[:80]!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
