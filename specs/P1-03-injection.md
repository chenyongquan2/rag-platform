# P1-03 Prompt-injection defence and security eval set

Status: review
Phase: P1
Depends on: P0-04 (`suspicious_instruction` rule), P0-10 (eval layout), P1-02 (context delimiters, prompts, `Answer`)
Design sections: s11-3, s6-2, s14-2, s14-1, s16-4
Calls real models: no

## Goal
The injection detector covers zh and en instruction patterns and hidden-text tricks with measured precision
(scan of the full HF corpus); a security eval set of 10 planted pages with questions, a rule-based scorer and
a harness that scores answers produced elsewhere; the context builder escapes every forgery in the planted pages.

## Interface
- `rag.ingest.injection`: `InjectionPattern(name, category, language, regex, prose_only)`, `PATTERNS`,
  `CATEGORIES`, `InjectionHit(pattern, category, language, view, snippet, start)`, `scan_text(text) -> list[InjectionHit]`,
  `find_suspicious_instructions(text) -> list[str]` (moved from `clean.py`, re-exported there). Views: `text`,
  `normalized` (zero-width removed, NFKC), `html_comment`, `unicode_tags`, `base64`.
- `rag.ingest.build.INGEST_CODE_VERSION = "3"` (flags change for the same config).
- `rag.answer.context.escape_delimiters` also escapes `<\/doc_fragment`, `doc-fragment` / `doc fragment` /
  `docfragment` spellings, and `[n]` lines that start with zero-width characters.
- `evals/datasets/security/`: `pages/*.md` (10 planted pages), `docs.jsonl` (one per page: `doc`, `path`, `language`,
  `source_rel`, `injection{kinds, techniques, tokens, urls, demanded_language, refusal_phrases, followed_example}`),
  `questions.jsonl` (`qid`, `doc`, `question`, `answer`, `expected_behaviour{summary, answer_language, must_answer,
  must_not_contain}`).
- `rag.eval.security`: `load_security_set(dir) -> SecuritySet`, `planted_chunks(page, kb_config) -> list[Chunk]`,
  `score_answer(question, page, *, text, answer=None) -> QuestionScore`, `score_answers(set, records) -> SecurityReport`,
  `read_answer_records(path)`, `scan_sources(sources, *, samples, seed) -> ScanReport`.
- CLI `uv run python -m rag.eval.security run --answers FILE [--set DIR] [--out FILE]`,
  `... dry-run [--out-dir DIR]`, `... scan --kb hf_docs [--limit N | --all] [--samples 20] [--seed 0] [--out-dir DIR]`
  (reports under git-ignored `data/reports/<kb>/`). Answers file: JSONL lines `{"qid", "answer": <Answer JSON>}`
  or `{"qid", "text": "..."}`.

## Acceptance criteria
Unit tests in `tests/unit/test_ingest_injection.py`, `tests/unit/test_eval_security.py`, `tests/unit/test_answer_context.py`.
- [x] AC1 Every pattern has positive and negative examples; positives hit that pattern, negatives hit nothing — `test_every_pattern_has_positive_and_negative_cases`, `test_pattern_positives_hit`, `test_pattern_negatives_do_not_hit`
- [x] AC2 False-positive guards (ignore the cache, you are now ready, send the request to, paste the token into the terminal, run the following command, chat templates in code, emoji ZWJ, base64 images, `class="hidden"`, Chinese tutorial phrases) give no hit — `test_false_positive_guards`
- [x] AC3 Hidden text: zero-width-split and full-width words, HTML comments, Unicode tag characters, invisible runs, base64 blobs with imperative text are found and the hit names its view — `test_hidden_text_views`
- [x] AC4 The `suspicious_instruction` rule and `find_suspicious_instructions` use the new detector; P0-04 cases still flag (one case changed: pasting a token into a form is not exfiltration) — `tests/unit/test_ingest_clean.py::test_suspicious_instruction_flags_en_and_zh`, `test_suspicious_rule_uses_scan_text`
- [x] AC5 Corpus scan: counts by pattern, category and view, docs hit, a seeded sample of hits with context, JSON and Markdown written to the report dir — `test_scan_report_counts_and_sample`, `test_scan_cli_writes_report`
- [x] AC6 Committed set: 10 pages (5 en, 5 zh), 2–3 questions each, planted tokens and URLs in their page and never in reference answers, `must_not_contain` covers them, every kind of s11-3 (direct request, disguised note) in both languages — `test_committed_security_set`
- [x] AC7 Loader rejects duplicate qids, unknown docs, missing pages, tokens not in the page, incomplete `must_not_contain` — `test_security_set_rejects_bad_input`
- [x] AC8 The detector finds every planted page's listed kinds in its raw text — `test_detector_finds_planted_injections`
- [x] AC9 Scorer: compliant; planted token (case and spacing variants) and URL or host; refusal (not-found status, refusal phrases, planted refusal phrase); language switch; error and missing answers; text-only answers — `test_score_compliant_answer`, `test_score_planted_token_and_url`, `test_score_refusal`, `test_score_language_switch`, `test_score_errors_and_missing`
- [x] AC10 `run --answers` reads Answer JSON and text lines, prints a summary, writes JSON; unknown qids exit 2 — `test_run_cli_scores_answers_file`, `test_run_cli_rejects_unknown_qids`
- [x] AC11 Dry run with FakeChatModel through `generate_answer` on planted chunks: compliant replies all compliant, replies that follow the injection all violated — `test_dry_run_scores_compliant_and_violating_answers`
- [x] AC12 Context builder on planted pages (clean + chunk): only our delimiters and `[n]` lines remain, also read NFKC-folded without zero-width characters; new variants escaped — `test_context_escapes_planted_forgeries`, `tests/unit/test_answer_context.py::test_escape_delimiter_spelling_variants`

Also (by running): `scan --kb hf_docs --all`; counts, 20 sampled hits labelled by hand, estimated false-positive rate in the handoff.

## Out of scope
LLM judge and real model answers on the set (P1-04, P3); ingesting planted pages into an ES index; skipping
suspicious chunks for company knowledge bases (company KB spec); caller-side `untrusted` field (P0-09).

## Handoff (filled at the end)
- Files: `src/rag/ingest/{injection,clean,build}.py`, `src/rag/answer/context.py`, `src/rag/eval/security/{__init__,__main__,cli,dataset,score,harness,scan}.py`, `evals/datasets/security/{docs.jsonl,questions.jsonl,pages/*.md}`, `tests/unit/test_ingest_injection.py` (8 functions, 279 cases), `tests/unit/test_eval_security.py` (15), `tests/unit/test_answer_context.py` (+1), `tests/unit/test_ingest_clean.py` (one case changed), `docs/decisions.md`. Unit suite after the rebase and review fixes: 672 passed.
- Scan (`scan --kb hf_docs --all`, seed 0): 2,647 docs, 1 doc hit, 1 hit (`override_en`, the heading "Ignore previous instructions" in `huggingface/blog/blob/main/llama2.md`, which describes the attack). Only 1 hit exists, so 1 sample (not 20), labelled `fp`: false-positive share of hits 1/1, docs wrongly flagged 1/2,647 (0.04%). Recall is measured on the security set: the detector finds every listed kind on all 10 planted pages. Scan takes about 27-31 s. Re-run after the review fixes (new override, tool-order and concealment patterns): unchanged, 1 hit in 2,647 docs, the same false positive.
- Dry run (`dry-run`, reranker tokenizer): compliant replies 30/30 compliant; replies that follow the injection 30/30 violated (planted token 21, no answer 13, planted URL 6, refusal 6, language switch 6).
- Decisions (docs/decisions.md): named zh/en patterns with views and false-positive guards, precision first; no rule change for the one scan false positive; set format and rule-based scoring; planted items may live only in Unicode-tag text (`SecurityPage.readable_text`); own function-word language detector (whole words only, 2 foreign hits needed); more delimiter spellings escaped; dry run with scripted replies; review round 1: a reply needs a key term of the reference answer (else `refusal` or `no_answer`), general refusal words only count when the question is unanswered, separator-free token match, refanged hosts.
- Open issues: real model answers on the set and an LLM judge are P1-04/P3; the scan corpus is almost clean, so false-positive rate rests on one hit; language detection covers zh/ja/ko/en/fr/de/es only; a reply that only repeats question words passes the key-term check (P1-04 judge).
- Review rounds: 1 CHANGES (gameable scorer, refusal on answered replies, missed override/tool/concealment phrasings, split tokens and defanged hosts), all fixed; waiting for round 2.
