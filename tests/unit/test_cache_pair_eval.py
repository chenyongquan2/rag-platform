"""Spec P5-01 AC8: evaluation of the entity guard and the distance threshold on the pairs."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from rag.cache.pair_eval import (
    choose_threshold,
    collect_texts,
    cosine_distance,
    estimate_cost,
    evaluate_split,
    false_accepts,
    paraphrase_pairs,
    sweep,
)
from rag.cache.pairs import Pair
from rag.config import repo_path
from rag.eval.stats import wilson_interval

DIM = 64


def _pair(n: int, category: str, a: str, b: str, split: str = "dev") -> Pair:
    return Pair(f"{split}:{n}:{category}", split, "counter", category, f"q:{n}", a, b)


def _vectors(pairs: list[Pair], distances: dict[str, float]) -> dict[str, list[float]]:
    """`a` is basis vector i, `b` is at the cosine distance `distances[category]` from it."""
    vectors: dict[str, list[float]] = {}
    for i, p in enumerate(pairs):
        cos = 1.0 - distances[p.category]
        a = [0.0] * DIM
        a[i] = 1.0
        b = [0.0] * DIM
        b[i] = cos
        b[DIM - 1] = math.sqrt(max(0.0, 1.0 - cos * cos))
        vectors[p.a] = a
        vectors[p.b] = b
    return vectors


PAIRS = [
    _pair(1, "identifier_swap", "What does push_to_hub do?", "What does create_repo do?"),
    _pair(2, "identifier_swap", "What does Trainer.train do?", "What does Trainer.save do?"),
    _pair(3, "number_swap", "the top 5 results", "the top 10 results"),
    _pair(4, "negation_flip", "which models support fp16", "which models do not support fp16"),
    _pair(5, "antonym_swap", "how do i push a model", "how do i pull a model"),
    _pair(6, "aspect_change", "What is push_to_hub for?", "What does push_to_hub return?"),
    _pair(7, "identifier_case", "What does LoRA do?", "What does lora do?"),
]
DISTANCES = {
    "identifier_swap": 0.04,
    "number_swap": 0.03,
    "negation_flip": 0.05,
    "antonym_swap": 0.09,
    "aspect_change": 0.12,
    "identifier_case": 0.01,
}
VECTORS = _vectors(PAIRS, DISTANCES)


def test_cosine_distance() -> None:
    assert cosine_distance([1.0, 0.0], [1.0, 0.0]) == pytest.approx(0.0)
    assert cosine_distance([1.0, 0.0], [0.0, 1.0]) == pytest.approx(1.0)
    assert cosine_distance([1.0, 1.0], [2.0, 2.0]) == pytest.approx(0.0)
    assert cosine_distance([0.0, 0.0], [1.0, 0.0]) == 1.0


def test_texts_to_embed_are_the_distinct_texts_of_the_text_pairs_in_order() -> None:
    meta = Pair("m", "dev", "counter", "other_kb", "q", "same", "same", b_ctx={"kb": "mt5"})
    texts = collect_texts([*PAIRS, meta], extra=[("p1", "p2"), ("p1", "p3")])
    assert texts[:2] == [PAIRS[0].a, PAIRS[0].b]
    assert "same" not in texts and texts[-3:] == ["p1", "p2", "p3"]
    assert len(texts) == len(set(texts))


def test_cost_estimate_is_pessimistic_and_proportional() -> None:
    small = estimate_cost(["a" * 300])
    assert small["tokens"] == 100 and small["texts"] == 1
    assert small["cny"] == pytest.approx(100 / 1e6 * 0.5)
    assert estimate_cost(["a" * 300] * 10)["cny"] == pytest.approx(small["cny"] * 10)


def test_the_guard_removes_the_identifier_number_and_negation_false_accepts() -> None:
    entity = {
        p.category for p in false_accepts(PAIRS, VECTORS, 0.15, guard="entity", strict_factor=1.0)
    }
    none = {p.category for p in false_accepts(PAIRS, VECTORS, 0.15, guard="none")}
    full = {
        p.category for p in false_accepts(PAIRS, VECTORS, 0.15, guard="full", strict_factor=1.0)
    }
    assert entity == {"antonym_swap", "aspect_change", "identifier_case"}
    assert {"identifier_swap", "number_swap", "negation_flip"} <= none
    # the lexical guard sees the one-word changes (push and pull, an added word) that the
    # entities do not; the capital letter of an identifier is not a difference for either
    assert full == {"identifier_case"}
    # a threshold under the antonym distance keeps those out for the entity guard as well
    small = false_accepts(PAIRS, VECTORS, 0.06, guard="entity", strict_factor=1.0)
    assert {p.category for p in small} == {"identifier_case"}


def test_a_question_without_entities_needs_the_stricter_distance() -> None:
    p = _pair(8, "antonym_swap", "how do i push a model", "how do i pull a model")
    vectors = _vectors([p], {"antonym_swap": 0.07})
    assert false_accepts([p], vectors, 0.10, guard="entity", strict_factor=1.0) == [p]
    assert false_accepts([p], vectors, 0.10, guard="entity", strict_factor=0.5) == []  # 0.07 > 0.05
    assert false_accepts([p], vectors, 0.10, guard="full", strict_factor=1.0) == []  # one word


def test_sweep_and_the_choice_of_the_threshold() -> None:
    paraphrases = [("how do i upload a file", "what do i call to send my file online")]
    vectors = dict(VECTORS)
    vectors["how do i upload a file"] = [1.0] + [0.0] * (DIM - 1)
    vectors["what do i call to send my file online"] = [0.9, math.sqrt(1 - 0.81)] + [0.0] * (
        DIM - 2
    )  # distance 0.1
    rows = sweep(
        PAIRS, vectors, paraphrases, thresholds=[0.02, 0.04, 0.06, 0.1, 0.2], strict_factor=1.0
    )
    by_t = {r["threshold"]: r for r in rows}
    assert by_t[0.06]["false_accepts_entity"] == 0 and by_t[0.06]["false_accepts_none"] >= 4
    assert by_t[0.1]["false_accepts_entity"] >= 1
    assert by_t[0.1]["false_accepts_full"] <= by_t[0.1]["false_accepts_entity"]
    assert by_t[0.1]["paraphrases_accepted_full"] == 1
    assert by_t[0.06]["paraphrases_accepted_full"] == 0
    assert by_t[0.02]["of"] == 6  # the policy class is not counted
    assert choose_threshold(rows, "entity") == 0.06


def test_choose_threshold_falls_back_to_the_smallest_of_the_grid() -> None:
    rows = [
        {"threshold": 0.02, "false_accepts_full": 3},
        {"threshold": 0.04, "false_accepts_full": 5},
    ]
    assert choose_threshold(rows) == 0.02
    rows = [
        {"threshold": 0.02, "false_accepts_full": 0},
        {"threshold": 0.04, "false_accepts_full": 0},
        {"threshold": 0.06, "false_accepts_full": 1},
    ]
    assert choose_threshold(rows) == 0.04


def test_evaluate_split_reports_rates_with_wilson_intervals_by_category() -> None:
    result = evaluate_split(PAIRS, VECTORS, 0.10, strict_factor=1.0)
    assert result["pairs"] == 6  # identifier_case is a policy class
    assert result["entity"]["n"] == 1  # the antonym pair (the aspect pair is at 0.12)
    assert result["entity"]["wilson95"] == pytest.approx(wilson_interval(result["entity"]["n"], 6))
    assert result["none"]["n"] == 5
    antonym = result["by_category"]["antonym_swap"]
    assert antonym["entity"]["n"] == 1 and antonym["full"]["n"] == 0 and antonym["pairs"] == 1
    assert result["by_category"]["identifier_swap"]["entity"]["n"] == 0
    assert result["by_category"]["identifier_swap"]["none"]["n"] == 2
    assert result["by_category"]["identifier_swap"]["distance_median"] == pytest.approx(0.04)
    assert "identifier_case" not in result["by_category"]


def test_the_real_paraphrase_pairs_are_the_colloquial_rewrites() -> None:
    pairs = paraphrase_pairs(repo_path("evals/datasets/hf_docs"))
    assert len(pairs) == 15
    assert all(a != b and a and b for a, b in pairs)


def _hash_vectors(texts: list[str]) -> dict[str, list[float]]:
    import hashlib

    return {t: [b / 255.0 + 0.01 for b in hashlib.sha256(t.encode()).digest()[:16]] for t in texts}


def test_the_command_evaluates_dev_and_touches_the_holdout_only_with_final(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from rag.cache.cli import main
    from rag.testing import observe_settings

    out = tmp_path / "out"
    code = main(
        ["pairs", "evaluate", "--out", str(out)],
        settings=observe_settings(),
        embedder=_hash_vectors,
    )
    assert code == 0
    first = json.loads((out / "pairs_eval.json").read_text("utf-8"))
    assert first["l1"]["false_hits"] == [] and first["l1"]["positives_not_shared"] == []
    assert "dev" in first and "holdout" not in first
    assert first["threshold"] in [r["threshold"] for r in first["dev_sweep"]]
    assert first["embedding"]["estimate"]["texts"] > 400
    code = main(
        ["pairs", "evaluate", "--final", "--out", str(out)],
        settings=observe_settings(),
        embedder=_hash_vectors,
    )
    final = json.loads((out / "pairs_eval_final.json").read_text("utf-8"))
    assert code == 0 and final["holdout"]["threshold"] == first["threshold"]
    assert final["holdout"]["pairs"] >= 100


def test_estimate_only_makes_no_request(capsys: pytest.CaptureFixture[str]) -> None:
    from rag.cache.cli import main
    from rag.testing import observe_settings

    def never(texts: list[str]) -> dict[str, list[float]]:
        raise AssertionError("no embedding on --estimate-only")

    assert (
        main(["pairs", "evaluate", "--estimate-only"], settings=observe_settings(), embedder=never)
        == 0
    )
    assert "embedding estimate" in capsys.readouterr().out


def test_the_final_file_is_not_overwritten_without_force(tmp_path: Path) -> None:
    from rag.cache.cli import main
    from rag.testing import observe_settings

    out = tmp_path / "out"
    args = ["pairs", "evaluate", "--final", "--out", str(out)]
    assert main(args, settings=observe_settings(), embedder=_hash_vectors) == 0
    final = out / "pairs_eval_final.json"
    first = final.read_text("utf-8")
    final.write_text(first.replace('"threshold"', '"threshold_kept"', 1), encoding="utf-8")
    assert main(args, settings=observe_settings(), embedder=_hash_vectors) == 4
    assert "threshold_kept" in final.read_text("utf-8")  # untouched
    assert main([*args, "--force"], settings=observe_settings(), embedder=_hash_vectors) == 0
    assert "threshold_kept" not in final.read_text("utf-8")


def test_a_slice_is_evaluated_on_its_own_at_a_given_threshold(tmp_path: Path) -> None:
    from rag.cache.cli import main
    from rag.cache.pairs import read_pairs
    from rag.config import repo_path
    from rag.testing import observe_settings

    out = tmp_path / "out"
    code = main(
        [
            "pairs",
            "evaluate",
            "--slice",
            "v2",
            "--tag",
            "before",
            "--threshold",
            "0.06",
            "--final",
            "--out",
            str(out),
        ],
        settings=observe_settings(),
        embedder=_hash_vectors,
    )
    assert code == 0
    result = json.loads((out / "pairs_eval_v2_before_final.json").read_text("utf-8"))
    assert result["threshold"] == 0.06  # given, not chosen
    pairs = read_pairs(repo_path("evals/datasets/hf_docs/cache_pairs.jsonl"))
    v2 = [p for p in pairs if p.slice == "v2" and p.split == "dev"]
    assert result["dev"]["pairs"] == len(v2)
    assert set(result["dev"]["by_category"]) <= {
        "word_order_swap",
        "wh_swap",
        "modal_swap",
        "zh_antonym_swap",
    }
    assert result["summary"]["counter"] == sum(p.slice == "v2" for p in pairs)
