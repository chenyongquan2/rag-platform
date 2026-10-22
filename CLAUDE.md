# rag-platform — rules for every session and subagent

Multi-knowledge-base RAG platform. The design is fixed in `docs/design/rag-platform-design.html` (Chinese; section index in `docs/design/README.md`; print one section with `python scripts/design_section.py s7-4`). The work plan is `docs/plan.md`; progress is `docs/progress.md`; decisions taken without the owner are in `docs/decisions.md`.

## Workflow (one sub-requirement = one spec = one clean context)

1. Each sub-requirement is done by **one fresh subagent or fresh session** that sees only this file, its spec and the design sections it names. Never carry another spec's chat history.
2. **Spec first.** Copy `specs/_template.md` to `specs/<ID>-<slug>.md`, fill it (English, at most one page; link design sections by id, do not copy them). Acceptance criteria map one-to-one to test names.
3. **TDD.** Write the failing tests for the acceptance criteria first, run them, see them fail; then implement until they pass; then refactor.
4. **Independent review.** A separate reviewer subagent checks the branch diff against the spec, the design sections and this file, runs the tests, and returns `APPROVE` or `CHANGES` with numbered issues. On `CHANGES` the author fixes and the reviewer looks again (at most 3 rounds; anything still open goes to `docs/progress.md` as a known issue).
5. **Merge.** Work on branch `spec/<ID>-<slug>`; after `APPROVE`, rebase on `main`, fast-forward merge, push `main`, delete the branch. Update the spec status and add a 5-line handoff note at the bottom of the spec (files touched, decisions, open issues).
6. If the design is wrong or a library does not behave as the design assumes: decide by the comparison criteria in the design, record it in `docs/decisions.md` (Chinese, one entry per decision: what, why, alternatives), and update the copy of the design in this repo only if the change is substantive. Stop and list in `docs/decisions.md` under 「待确认」 anything that spends money beyond the budget, makes the repo public, deletes data, or changes scope.

## Code

- Python 3.12, managed with `uv` (`uv sync`, `uv run pytest`). Package in `src/rag/`, tests in `tests/`.
- Code, comments, docstrings, README, specs, commit messages: **English**. Owner-facing docs (`docs/plan.md`, `docs/progress.md`, `docs/decisions.md`, the design doc): **Chinese**.
- Layers pass only our own data classes (`Chunk`, `RetrievalResult`, `Answer`, …); framework types stay inside their layer.
- Every parameter is explicit in config; never rely on framework defaults (LlamaIndex defaults to OpenAI, top_k=2, English prompts, metadata in embed text). `Settings.llm` is a stub that raises when called.
- Every external dependency is created from a connection string by a factory (design 16.7). Secrets only from environment variables / `.env` (git-ignored). Never print or commit secrets.
- Error categories (`degraded`, `rerank_timeout`, `answer_parse_failed`, …) per design 16.8.

## Tests

- Markers: `unit` (default, no network, no Docker, seconds), `integration` (needs the local Docker services: Elasticsearch, Redis), `llm` (calls the real model gateway; costs money; **never runs by default**).
- While developing, run only the tests you touch (`uv run pytest tests/unit/test_x.py`). The pre-commit hook runs only the tests affected by your change (pytest-testmon; the first run in a fresh checkout builds a baseline, about 4 minutes). Run the whole suite in parallel before you report or merge: `uv run pytest -n 4` (about 1.5 minutes, versus about 4 minutes serial).
- Default `uv run pytest` runs only `unit`. `uv run pytest -m integration` needs `docker compose up -d`. `uv run pytest -m llm` only when a spec says so.
- Do not edit an existing test to make it pass unless the spec changed; say why in the commit body.
- Unit tests use deterministic fakes (`rag.testing.FakeEmbedding`, `FakeChatModel`), never LlamaIndex `MockLLM` (it silently echoes the prompt).

## Models and money

- Gateway: OpenAI-compatible, base URLs and keys in `.env` (`LLM_BASE_URL`, `LLM_API_KEY`, `EMBED_BASE_URL`, `EMBED_API_KEY`).
- Embedding model id is `qwen/text-embedding-v4` (plain `text-embedding-v4` returns 503). Chat model: `deepseek-v4-flash` (the cheapest; the gateway routes it to `deepseek-v4.1-flash`, a reasoning model — budget for reasoning tokens).
- Every real model call appends a line to the cost ledger (`data/cost/ledger.jsonl`). **Budget: 39 CNY total; the owner raised the stop line to 33 CNY on 2026-10-22 for P7 (scaled down), the 150-question re-run and the citation comparison, and to 37 CNY on 2026-10-23 for the eval upgrade (hard questions, repeated runs, regression set).** Before any job that calls real models, sum the ledger; if spent + estimated cost > 37 CNY, do not run it — switch to the fake models and leave the real run for the owner.
- If the gateway is unreachable or errors persistently, switch that work to fakes, record it in `docs/progress.md`, and move on to specs that need no model.

## Git

- Conventional Commits in English: `feat(retrieve): ...`, `fix(ingest): ...`, `test: ...`, `docs: ...`, `chore: ...`. Body mentions the spec id.
- **No `Co-Authored-By` trailer** and no generated-by lines in commits.
- Repo is private on GitHub (`chenyongquan2/rag-platform`). Do not make it public.

## Runtime (P0)

- Infrastructure runs in Docker Compose (Elasticsearch pinned minor version, Redis, OTel Collector, Jaeger). The Python app runs on the host via `uv run` during P0.
- WSL memory stays at the default (~16 GB). Keep Elasticsearch heap at 2 GB.
- **Never run `docker compose down` (with or without `-v`/`--profile`) in this repo, and never delete a Docker volume.** `down` removes every container of the project and `-v` deletes `rag-platform_es-data`, i.e. all indexes (this happened once, 10-21, during P5-02; the indexes had to be rebuilt). To stop optional services, name them: `docker compose --profile monitoring stop prometheus grafana` (a bare `docker compose --profile <name> stop` also stops Elasticsearch, Redis, the Collector and Jaeger). Never stop `elasticsearch` or `redis-cache` while other worktrees use them. Several worktrees share these services; never recreate Elasticsearch or Redis.
- Hugging Face data comes from `https://hf-mirror.com` (huggingface.co is blocked). Reranker `BAAI/bge-reranker-base` is in the local HF cache; load with `HF_HUB_OFFLINE=1`.
