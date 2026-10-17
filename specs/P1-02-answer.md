# P1-02 Answer generation with numbered citations

Status: in progress
Phase: P1
Depends on: P0-01 (types, fakes), P0-04 (`suspicious_instruction` flag), P0-11 (spans, attributes), P1-01 (chat factory, `online` profile, ledger)
Design sections: s11-1, s11-2 (level 1), s11-3, s10-3, s8-4, s16-8
Calls real models: yes, one `llm` test (3 chunks, about 3k prompt tokens; estimated at most 0.02 CNY)

## Goal
`rag.answer` turns a question plus ordered chunks into an `Answer`: a context builder (dedupe, budget per
sub-question, numbered fragments in fixed untrusted-data delimiters), zh/en prompts, generation on the
`online` profile as a stream of text deltas (plus a non-streaming call on top), a tolerant `[n]` parser and
citation check level 1, one retry before `answer_parse_failed`. No retrieval coupling: the reranker
(P0-08) is replaced by given chunk lists.

## Interface
- `rag.config.AnswerConfig(context_tokens=4000, language="auto"|"zh"|"en", parse_retries=1)`;
  `KnowledgeBaseConfig.answer`; `kb/public/hf_docs.yaml` sets it explicitly.
- `rag.types`: `CitationCheck(removed_markers, unsupported_sentences, parse_attempts)`;
  `Answer.citation_check`, `Answer.generation_error: ErrorCategory | None` (counted in `error_categories`).
- `rag.answer.context`: `estimate_tokens(text)`, `escape_delimiters(text)`, `Fragment`, `AnswerContext`
  (`chunks`, `chunk_id(n)`), `build_context(chunk_lists, *, budget_tokens, metadata_keys, count_tokens)`.
- `rag.answer.prompts`: `AnswerLanguage`, `detect_question_language(q)`, `answer_language(q, setting)`,
  `NOT_FOUND[lang]`, `SYSTEM_PROMPTS[lang]`, `build_messages(question, context, language)`,
  `retry_messages(messages, previous, language)`.
- `rag.answer.parse`: `parse_answer(text) -> ParsedAnswer`, `check_citations(parsed, context)`.
- `rag.answer.generate`: events `AnswerDelta(text, attempt)`, `AnswerRetry(attempt, reason)`,
  `AnswerDone(answer)`; `stream_answer(question, context, *, kb, config, model=None, request_id=None,
  settings=None) -> AsyncIterator[AnswerEvent]`; `generate_answer(...) -> Answer`.
- `rag.observe.attributes`: `rag.context.*`, `rag.answer.*`, `gen_ai.usage.*` names.
- `rag.testing.StreamingFakeChatModel` (scripted replies sent in pieces).

## Acceptance criteria
Unit tests in `tests/unit/test_answer_{context,parse,generate}.py`.
- [ ] AC1 Dedupe by chunk id (best rank wins), fragments numbered [1]..[n] most relevant first, with title path and source — `test_context_dedupes_and_numbers_fragments`
- [ ] AC2 Token budget on the rendered fragment; what does not fit is dropped and reported — `test_context_respects_token_budget`
- [ ] AC3 Budget split: equal share per sub-question first, the rest by score — `test_context_budget_split_per_sub_question`
- [ ] AC4 Fixed delimiters marked as untrusted data; delimiter look-alikes in text and metadata escaped; suspicious chunks labelled — `test_context_delimiters_escape_and_suspicious_label`
- [ ] AC5 `AnswerConfig` defaults; `hf_docs.yaml` sets every key — `test_answer_config_defaults_and_hf_docs`
- [ ] AC6 Answer language follows the question (identifiers do not count); the KB setting overrides — `test_answer_language_detection_and_override`
- [ ] AC7 zh and en templates carry every rule; the system message is a constant; an injection chunk stays inside one fragment — `test_prompt_templates_zh_en`, `test_injected_chunk_cannot_change_instructions`
- [ ] AC8 Parser variants: `[1]`, `【1】`, `［1］`, `[1,2]`, `[1，2]`, `[1][2]`, `[1-3]`, marker after the period, trailing spaces, Chinese punctuation, list markers, abbreviations, brackets in code untouched — `test_parse_citation_variants`
- [ ] AC9 Unknown numbers removed and counted; sentences without a valid marker counted as unsupported — `test_citation_check_unknown_and_unsupported`
- [ ] AC10 Fixed not-found sentence (either language, small variants) gives `found=false`, `not_found`, no retry; an empty context answers not-found without a model call — `test_not_found_sentence`, `test_empty_context_skips_model`
- [ ] AC11 No valid marker: one retry with a reminder; recovers, or `answer_parse_failed` with chunks only — `test_parse_retry_recovers`, `test_parse_failure_retries_once_then_flags`
- [ ] AC12 The stream yields deltas, a retry event and the final answer; `generate_answer` returns the same answer; default model is the `online` profile — `test_stream_yields_deltas_and_final_answer`, `test_default_model_is_online_profile`
- [ ] AC13 Gateway timeout/error/budget: chunks only with `generation_error`; other exceptions propagate — `test_model_failure_returns_chunks_only`
- [ ] AC14 `generate` span carries context and answer attributes and the category; the call's ledger rows give tokens and CNY on the span — `test_generate_span_attributes`, `test_generate_cost_from_ledger`
- [ ] AC15 (`llm`) A real HF question from 3 given chunks: answer found, cites the right fragment; format compliance printed — `tests/llm/test_answer_gateway.py::test_real_answer_cites_fragments`

## Out of scope
Retrieval, reranking, threshold refusal and context expansion (P0-07/P0-08); citation check level 2 and
answer eval (P1-04); `ask_docs`, SSE events and business records (P0-09/P0-12, P3); sub-question planning (P3).

## Handoff (filled at the end)
- Files:
- Decisions:
- Open issues:
- Review rounds:
