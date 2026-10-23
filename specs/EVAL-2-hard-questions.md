# EVAL-2 Hard answerable questions and ambiguous questions for the Agentic RAG comparison

Status: review
Phase: eval upgrade (step 2 of `rag-platform-eval-upgrade-plan.md`, section 8)
Depends on: P3-05 (question sets, runner, report), P3-03 (`rag.eval.absence`, two-hop checks), P3-02 (clarification)
Design sections: s14-2, s13-5
Calls real models: no (a model is only needed by the later run, which is not part of this spec)

## Goal
A new question file `evals/datasets/hf_docs/hard.jsonl` with 25 hand-written hard answerable questions (each needs
two or three pages) and 10 ambiguous questions (the right reply is a clarifying question), a checking tool that
re-proves from the corpus that no single page answers a hard question, a rule check for the ambiguous ones, both
registered as question types the agentic runner can select by itself, and an export of 10 questions for the
owner to answer blind.

## What the questions are modelled on
The 8 answerable questions where round 1 missed the right page (two_hop 04, 07, 08, 11, 14, 15, 18 and
`qa_eval:015`) show five failure types; the new questions copy the types, not the questions:
bridge (the second page is named only on the first), comparison (one fact on each of two pages), chain of three
pages, list across pages (a set built from lists on two pages), and wording that does not use the document's words.
They are written by an AI by hand, not drawn from user traffic: results only show behaviour "on this kind of
question".

## Interface
- `evals/datasets/hf_docs/hard.jsonl` (qids `hf_docs_hard:NN`, `hf_docs_ambiguous:NN`), `hard_manifest.json`
  (version, sha256, counts), a NOTICE entry.
- `rag.eval.agentic.sets`: types `hard` and `ambiguous` (`HARD_TYPES`); `DEFAULT_TYPES` (the original four) is what
  `load_items` loads without `types`.
- `rag.eval.hard`: `validate_questions`, `corpus_report`, `asked_back`, `clarification_summary`, `export_self_check`;
  `python -m rag.eval.hard check | self-check | clarify-report`.

## Acceptance criteria
Tests in `tests/unit/test_eval_hard_data.py` (D, data), `test_eval_hard_tool.py` (T, tool).
- [ ] AC1 The file loads with the repo loader; 25 hard and 10 ambiguous questions; ids unique, repeat nothing of the other sets — `D::test_set_loads_with_the_expected_counts_and_ids`, `::test_nothing_repeats_the_other_sets`
- [ ] AC2 Every hard question: reference answer, `requires: all`, at least two distinct gold pages, one fact (verbatim evidence, key pattern) per gold page, a failure type it is modelled on; shape mix bridge >= 6, comparison >= 8, list_all >= 5, chain3 >= 2 — `D::test_every_hard_question_has_answer_pages_facts_and_a_template`, `::test_shape_mix`
- [ ] AC3 Single-page impossibility from the corpus: each evidence text is in its page; each key pattern matches its own page and no page outside the declared copies (`also_on`); the loose patterns (inflection-aware anchor and fact pairs) match the gold page and no page matches the loose patterns of all facts of a question; a co-mention search over every page, gold pages included (every term of every fact, or every subject phrase, on one page) finds pages that the author must read and list in `read_and_cleared` with a reason — `D::test_evidence_is_verbatim_in_the_gold_pages`, `::test_no_single_page_states_all_facts`; the checker itself: `T::test_corpus_report_accepts_a_question_no_single_page_answers`, `::test_corpus_report_finds_each_kind_of_problem`, `::test_declared_copies_are_allowed_and_must_be_complete`, `::test_pages_that_mention_every_term_must_be_read_and_cleared`, `::test_subjects_narrow_the_co_mention_search`, `::test_a_gold_page_that_holds_the_other_facts_is_a_co_mention`, `::test_a_page_that_names_every_subject_is_a_co_mention_even_in_other_words`, `::test_validate_questions_flags_malformed_ones`
- [ ] AC4 At least 8 questions are marked `paraphrase`; their `avoided_terms` occur in the evidence and not in the question — `D::test_paraphrase_questions_avoid_the_documents_words`
- [ ] AC5 Every ambiguous question has no gold page, at least two readings (page, verbatim evidence, answer), the answers differ, the reference says what to ask, and `check` is the rule `asked_back` — `D::test_ambiguous_questions_have_readings_with_evidence`, `::test_ambiguous_readings_are_in_the_corpus`
- [ ] AC6 `hard` and `ambiguous` are selectable by `load_items(types=[...])`; the default selection stays the 150 questions; an ambiguous item is not answerable — `T::test_hard_types_load_by_themselves`, `::test_default_selection_is_unchanged`
- [ ] AC7 `asked_back`: a graph clarification counts; a not-found, an error and an ordinary answer do not; a plain-text reply that ends by asking counts (the standard system has no clarification path) — `T::test_asked_back_rule`, `::test_clarification_summary_with_wilson_intervals`
- [ ] AC8 The report and the cost projection work with hard and ambiguous records (the ambiguous ones are not judged, not counted as answerable, and get their own table) — `T::test_report_with_hard_and_ambiguous_records`, `::test_render_report_without_hard_records_has_no_ambiguous_section`
- [ ] AC9 `check` exits 0 on the shipped file and 1 on a file whose evidence is not in the page, 2 without the corpus — `T::test_check_command_passes_and_fails`; `clarify-report` prints the rule check of an agentic run's records — `T::test_clarify_report_command`
- [ ] AC10 `self-check` writes a sheet (question, empty answer column) and a key (reference, pages) for 10 hard questions drawn with a seed; same seed, same questions; the sheet shows neither reference nor pages — `T::test_self_check_sheet_hides_the_reference`, `::test_self_check_command_writes_the_files`
- [ ] AC11 The manifest names the version and the sha256 of the file; the NOTICE mentions the file — `D::test_manifest_and_notice`

## Out of scope
Running either system on the questions, the judge, the 60-second rerun (plan steps 2 and 2b: the owner of the run),
the owner's agreement figure (written into the manifest after the owner has answered), expanding the checks to the
other question files.

## Rules kept
Questions are not changed after any system has seen them; a question may be dropped only because its reference
answer turns out to be ambiguous, and every drop is recorded in the manifest (`dropped`).

## Handoff
- Files: `evals/datasets/hf_docs/{hard.jsonl,hard_manifest.json,NOTICE.md}`; new package `src/rag/eval/hard/` (`spec`, `corpus`, `clarify`, `selfcheck`, `cli`); `src/rag/eval/agentic/{sets,analysis,report,runs,cli}.py` (types `hard` and `ambiguous`, `DEFAULT_TYPES`, `NOT_ANSWERABLE`, the ambiguous report section, projection and refusal pool fixes); `tests/unit/test_eval_hard_{data,tool}.py`, one line changed in `test_eval_agentic_sets.py` (iterate `DEFAULT_TYPES`; `QTYPES` gained two types).
- Decisions: both types live in one file and are told apart by `meta.type`; the default selection stays the 150 questions, the new types are selected with `--types hard,ambiguous`; the single-page proof is the key pattern unique to its page, loose patterns, and a co-mention search (every term of every fact, or every subject phrase, on one page; gold pages included) whose hits were read and listed in `read_and_cleared`; ambiguous questions are scored by the rule `asked_back` (graph clarification, or a short plain reply that ends in a question and has no concrete value before it), no judge; questions replaced in review round 1 are listed in the manifest (`dropped`), version stays 1 because no system ran on the file.
- Open issues: the three data-proof tests and the `check` command need the downloaded corpus (skipped without it); the owner's blind self-check (10 questions, sheet in `C:/workspace/rag-platform-data/human/hard_self_check/`) is not answered yet, so its agreement is not in the manifest; a word search cannot show that a page never says a fact in other words; the questions are AI-written on the model of known failures, so results only show behaviour on this kind of question.
- Review rounds: see the progress row.
