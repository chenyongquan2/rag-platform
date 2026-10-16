"""Spec P0-05: cost ledger, budget check and the `python -m rag.cost summary` CLI."""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pytest

from rag.config import ModelPrice, ServiceSettings
from rag.config.paths import main_checkout, shared_data_dir
from rag.cost import (
    BudgetExceeded,
    CostLedger,
    UnknownModelPriceError,
    check_budget,
    total_spent,
)
from rag.cost.cli import main as cost_main

REPO_ROOT = Path(__file__).resolve().parents[2]
PRICES = {
    "emb": ModelPrice(input_cny_per_m=0.5),
    "chat": ModelPrice(input_cny_per_m=2.0, output_cny_per_m=8.0),
}


def _ledger(tmp_path: Path, stop: float = 28.0) -> CostLedger:
    return CostLedger(tmp_path / "cost" / "ledger.jsonl", prices=PRICES, budget_stop_cny=stop)


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_record_appends_json_line_with_cny(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)
    first = ledger.record(
        kind="embedding", model="emb", prompt_tokens=2_000_000, profile="ingest", inputs=10
    )
    ledger.record(kind="chat", model="chat", prompt_tokens=1_000, completion_tokens=500)

    rows = _rows(ledger.path)
    assert len(rows) == 2
    assert rows[0] == first
    assert rows[0]["kind"] == "embedding"
    assert rows[0]["model"] == "emb"
    assert rows[0]["prompt_tokens"] == 2_000_000
    assert rows[0]["completion_tokens"] == 0
    assert rows[0]["total_tokens"] == 2_000_000
    assert rows[0]["cny"] == pytest.approx(1.0)
    assert rows[0]["profile"] == "ingest"
    assert rows[0]["inputs"] == 10
    assert datetime.fromisoformat(rows[0]["time"]).tzinfo is not None
    assert rows[1]["cny"] == pytest.approx(1_000 * 2.0 / 1e6 + 500 * 8.0 / 1e6)
    assert rows[1]["total_tokens"] == 1_500

    # The default price table has the embedding model (0.5 CNY per million) and the chat model.
    prices = ServiceSettings(_env_file=None).model_prices
    assert prices["qwen/text-embedding-v4"].input_cny_per_m == 0.5
    assert "deepseek-v4-flash" in prices


def test_unknown_model_price_is_rejected(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)
    with pytest.raises(UnknownModelPriceError, match="no-such-model"):
        ledger.record(kind="embedding", model="no-such-model", prompt_tokens=1)
    with pytest.raises(UnknownModelPriceError, match="model_prices"):
        ledger.estimate("no-such-model", 10)
    assert not ledger.path.exists()


def test_total_spent_sums_ledger(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)
    assert ledger.total_spent() == 0.0  # no file yet
    for tokens in (1_000_000, 2_000_000, 3_000_000):
        ledger.record(kind="embedding", model="emb", prompt_tokens=tokens)
    with ledger.path.open("a", encoding="utf-8") as fh:
        fh.write('{"time": "2026-10-17T00:00:00+00:00", "cny": 5')  # crash in mid-write
    assert ledger.total_spent() == pytest.approx(3.0)
    assert len(ledger.entries()) == 3

    settings = ServiceSettings(_env_file=None, cost_ledger_path=str(ledger.path))
    assert total_spent(settings) == pytest.approx(3.0)


def test_check_budget_raises_over_stop(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path, stop=1.0)
    ledger.record(kind="embedding", model="emb", prompt_tokens=1_500_000)  # 0.75 CNY

    assert ledger.check_budget(0.25) == pytest.approx(0.75)  # exactly at the stop is allowed
    with pytest.raises(BudgetExceeded, match=r"0\.75.*0\.3.*1\.0"):
        ledger.check_budget(0.3)
    assert ledger.estimate("chat", 1_000_000, 100_000) == pytest.approx(2.8)

    settings = ServiceSettings(
        _env_file=None, cost_ledger_path=str(ledger.path), budget_cny=1.0, budget_stop_cny=1.0
    )
    assert check_budget(0.1, settings) == pytest.approx(0.75)
    with pytest.raises(BudgetExceeded):
        check_budget(0.3, settings)


def test_cli_summary_prints_totals(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    ledger = _ledger(tmp_path)
    ledger.record(kind="embedding", model="emb", prompt_tokens=1_000_000, profile="ingest")
    ledger.record(kind="embedding", model="emb", prompt_tokens=1_000_000, profile="query")
    ledger.record(kind="chat", model="chat", prompt_tokens=100, completion_tokens=50)

    assert cost_main(["summary", "--ledger", str(ledger.path)]) == 0
    out = capsys.readouterr().out
    assert "calls: 3" in out
    assert "spent: 1.000600 CNY" in out
    assert "stop at 28.00" in out
    assert "embedding" in out and "emb" in out and "calls=2" in out and "tokens=2000000" in out
    assert "chat" in out and "tokens=150" in out

    # The module runs as `python -m rag.cost summary`.
    proc = subprocess.run(
        [sys.executable, "-m", "rag.cost", "summary", "--ledger", str(ledger.path)],
        capture_output=True,
        text=True,
        check=False,
        cwd=tmp_path,
    )
    assert proc.returncode == 0, proc.stderr
    assert "calls: 3" in proc.stdout


def test_ledger_is_shared_outside_checkouts_and_git_ignored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Default: one file next to the main checkout, the same for every worktree of the repo.
    monkeypatch.delenv("COST_LEDGER_PATH", raising=False)
    default = Path(ServiceSettings(_env_file=None).cost_ledger_path)
    assert default.is_absolute()
    assert default == shared_data_dir() / "cost" / "ledger.jsonl"
    assert REPO_ROOT not in default.parents

    # A linked worktree (`.git` is a file) resolves to its main checkout, wherever it lives.
    main = tmp_path / "workspace" / "repo"
    (main / ".git" / "worktrees" / "wt1").mkdir(parents=True)
    worktree = main / ".claude" / "worktrees" / "wt1"
    worktree.mkdir(parents=True)
    gitdir = (main / ".git" / "worktrees" / "wt1").as_posix()
    (worktree / ".git").write_text(f"gitdir: {gitdir}\n", encoding="utf-8")
    assert main_checkout(worktree) == main
    assert main_checkout(main) == main
    assert shared_data_dir(worktree) == tmp_path / "workspace" / "rag-platform-data"
    plain = tmp_path / "installed"
    plain.mkdir()
    assert main_checkout(plain) == plain  # no git metadata

    # A ledger inside a checkout is never committed.
    ignored = subprocess.run(
        ["git", "check-ignore", "-q", "data/cost/ledger.jsonl"], check=False, cwd=REPO_ROOT
    ).returncode
    assert ignored == 0
    tracked = subprocess.run(
        ["git", "ls-files", "data/cost"], capture_output=True, text=True, check=True, cwd=REPO_ROOT
    ).stdout
    assert tracked == ""
