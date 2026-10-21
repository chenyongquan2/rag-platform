"""Spec P8-01: every new setting is in .env.example, the snapshot change is documented and not
applied, and the README names the new commands."""

from __future__ import annotations

import re
from pathlib import Path

import yaml

from rag.config import ServiceSettings

REPO = Path(__file__).resolve().parents[2]
PREFIXES = ("auth_", "rate_limit_", "budget_daily", "budget_ask_", "chat_breaker_")


def test_env_example_names_every_p801_setting() -> None:
    text = (REPO / ".env.example").read_text("utf-8")
    names = [n.upper() for n in ServiceSettings.model_fields if n.startswith(PREFIXES)]
    assert len(names) >= 20
    for name in [*names, "ES_SNAPSHOT_DIR"]:
        assert re.search(rf"^# {name}=", text, re.MULTILINE), name


def test_snapshot_repository_is_an_override_not_applied_to_the_main_compose_file() -> None:
    override = yaml.safe_load(
        (REPO / "deploy" / "elasticsearch" / "snapshot.override.yml").read_text("utf-8")
    )
    es = override["services"]["elasticsearch"]
    assert es["environment"]["path.repo"] == "/usr/share/elasticsearch/snapshots"
    assert any(v.endswith(":/usr/share/elasticsearch/snapshots") for v in es["volumes"])
    compose = (REPO / "docker-compose.yml").read_text("utf-8")
    assert "path.repo" not in compose


def test_readme_documents_the_guards_and_the_tools() -> None:
    text = (REPO / "README.md").read_text("utf-8")
    for needle in (
        "AUTH_MODE=oauth",
        "rag.eval.access",
        "rag.ops.backup export",
        "rag.ops.backup rebind-docstore",
        "rag.ops.loadtest",
        "snapshot.override.yml",
        "ModelGatewayBreakerOpen",
    ):
        assert needle in text, needle
