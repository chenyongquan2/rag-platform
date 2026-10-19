# P2-02 Rerank threshold sweep and full-read timing

Status: in progress
Phase: P2
Depends on: P0-08 (rerank scores, threshold), P0-10 (Wilson interval), P0-13 / P0-13b (threshold 0.3, dev-set confirmation), P1-05 (dev and holdout sets)
Design sections: s10-2, s14-4, s15-3, s19 (P2 row)
Calls real models: no (question vectors come from the L0 cache; the 20 extra no-answer questions need about 20 query embeddings, estimated below 0.001 CNY)

## Goal
A script that sets the rerank score threshold from recorded rerank scores and reports, for every candidate
threshold, the false-refusal rate (answerable question, every candidate below the threshold) and the
wrong-answer-accepted rate (no-answer question that is not refused), each with a Wilson 95% interval; plus a
measurement of the full read-and-clean time of the 2,647-page corpus (design s19 P2: over about 10 minutes
means file-level skip later). The result and the threshold decision go into `docs/reports/P2-report.md`.

## Interface
- `rag.eval.threshold.records`: `Candidate`, `ScoreRecord`, `ScoreFile` (pydantic); `load_scores(path)`, `write_scores(file, path)`.
- `rag.eval.threshold.sweep` (pure maths, no I/O):
  `top_score(rec)`, `is_refused(rec, t)`, `group_rates(records, t) -> GroupRates`, `sweep(groups, thresholds) -> list[SweepRow]`,
  `recommend(rows, *, max_false_refusal=0.05, bound="observed") -> float | None`, `parse_thresholds(text) -> list[float]`,
  `render_sweep(...) -> str`.
- CLI `uv run python -m rag.eval.threshold`:
  - `collect --kb hf_docs --questions P --kind answerable|no_answer --out FILE [--rerank-n N] [--deadline-ms MS]`: runs the real chain
    with threshold 0 and stores every reranked candidate's score. Seam `build_retriever` is the one of `rag.eval.retrieval.cli`.
  - `sweep --dev FILE [--check NAME=FILE]... [--no-answer NAME=FILE]... [--thresholds LIST|START:STOP:STEP]
    [--max-false-refusal 0.05] [--bound observed|wilson_upper] [--out FILE]`.
- `scripts/p2_02/read_timing.py`: times connector listing, reader and cleaning of the whole corpus (no embedding, no Elasticsearch).
- Data: `evals/datasets/hf_docs/no_answer_extra.jsonl` (20 hand-written no-answer questions, labelled in `meta`; the original 10 are untouched).
- No config default changes unless the report's data supports it (then `docs/decisions.md` records why).

## Acceptance criteria
Each line maps to one test in `tests/unit/test_threshold_sweep.py` (S) or `tests/unit/test_threshold_cli.py` (C).
- [ ] AC1 A question is refused when its top rerank score is below the threshold; a score equal to the threshold passes; no candidates is refused — `S::test_refusal_is_top_score_below_threshold`
- [ ] AC2 False-refusal and wrong-answer-accepted rates come with the Wilson interval of the exact counts — `S::test_rates_have_wilson_intervals`
- [ ] AC3 All-refused and none-refused thresholds, and an empty group, give the right counts and a (0, 1) interval for n = 0 — `S::test_all_refused_none_refused_and_empty_group`
- [ ] AC4 A correct page counts as lost only when it was scored and every scored copy is below the threshold — `S::test_lost_page_needs_a_scored_correct_page`
- [ ] AC5 `recommend` returns the largest threshold whose dev false-refusal rate is within the cap (observed rate, or Wilson upper bound); None when even the lowest exceeds it — `S::test_recommend_largest_threshold_within_cap`, `S::test_recommend_wilson_upper_mode`, `S::test_recommend_none_when_no_threshold_fits`
- [ ] AC6 The threshold list parses (list, `start:stop:step` without float drift) and bad input is refused — `S::test_parse_thresholds`
- [ ] AC7 `collect` runs the chain with threshold 0, records all reranked candidates with the question kind, and refuses a run that had a rerank timeout — `C::test_collect_records_all_candidates_with_threshold_zero`, `C::test_collect_rejects_rerank_timeout`
- [ ] AC8 `sweep` prints the table, the recommendation and the rule it used; the check and no-answer sets are reported but never change the recommendation — `C::test_sweep_prints_table_and_recommendation`, `C::test_check_and_no_answer_sets_do_not_move_the_recommendation`
- [ ] AC9 Bad arguments exit with code 2 and one line on stderr — `C::test_sweep_usage_errors_exit_2`
- [ ] AC10 (by running) dev, public-65, 10 + 20 no-answer sets scored; holdout scored once at the chosen threshold; read timing measured; report section 「P2-02 阈值和读取耗时」 in `docs/reports/P2-report.md`.

## Out of scope
Changing the reranker or the fusion settings (P0-13b); the answer-step refusal (`found=false`, P1-02); a gateway
reranker; file-level skip itself (only needed if the measured time is over about 10 minutes); the rest of the
P2 row (alias switch, five change types, ablations: P2-01, P2-03, P2-04).

## Handoff (filled at the end)
- Files:
- Decisions:
- Open issues:
- Review rounds:
