"""Knowledge-base config (design s5-1): one YAML file per knowledge base under `kb/`.

Every field has a default, so a new knowledge base only has to name its data source.
Unknown keys are rejected so a typo in YAML fails loudly instead of silently using a default.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class StepConfig(_Strict):
    """One named entry of a clean or step chain, looked up in `rag.registry.steps`.

    YAML accepts `name` or `{name: {param: value}}`; both normalize to this model.
    """

    name: str = Field(min_length=1)
    params: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _normalize(cls, value: Any) -> Any:
        if isinstance(value, str):
            return {"name": value, "params": {}}
        if isinstance(value, dict) and "name" in value and set(value) <= {"name", "params"}:
            return value
        if isinstance(value, dict):
            if len(value) != 1:
                raise ValueError(f"a step must be `name` or `{{name: {{params}}}}`, got {value!r}")
            ((name, params),) = value.items()
            return {"name": name, "params": params or {}}
        return value


def _steps(*entries: str | dict[str, Any]) -> list[StepConfig]:
    return [StepConfig.model_validate(e) for e in entries]


class PipelineConfig(_Strict):
    """How files with one extension are read and processed.

    `clean` runs before the LlamaIndex IngestionPipeline; `steps` run inside it (design s5-1, s7-4).
    `reader` is a name in `rag.registry.readers`; `None` means look it up by the file extension.
    """

    reader: str | None = None
    clean: list[StepConfig] = Field(default_factory=list)
    steps: list[StepConfig] = Field(
        default_factory=lambda: _steps(
            {"token_split": {"max_tokens": 400}}, "merge_tiny", "assign_chunk_ids"
        )
    )


def _default_pipelines() -> dict[str, PipelineConfig]:
    return {
        ".md": PipelineConfig(
            clean=_steps("md_clean"),
            steps=_steps(
                "markdown_split",
                {"token_split": {"max_tokens": 400}},
                "merge_tiny",
                "assign_chunk_ids",
            ),
        )
    }


class SourceConfig(BaseModel):
    """Data source. `connector` is a name in `rag.registry.connectors`.

    Connector-specific keys (`path`, `glob`, `dataset`, `limit`, ...) are kept in `model_extra`
    and passed to the connector factory.
    """

    model_config = ConfigDict(extra="allow")

    connector: str = "local_dir"


class MetadataConfig(_Strict):
    """Metadata whitelists (design s5-4): only these keys go into embed text / LLM prompt text."""

    embed: list[str] = Field(default_factory=lambda: ["title_path"])
    llm: list[str] = Field(default_factory=lambda: ["title_path", "source_rel"])


class EsConfig(_Strict):
    """Elasticsearch settings of one knowledge base.

    - `analyzer`: analyzer of `content` and `title_path` (design s7-3).
    - `vector_index`: type of the dense_vector field (spec P2-03 quantization ablation):
      `int8_hnsw` (default), `bbq_hnsw` or `flat`; `bbq_disk` needs Enterprise and is not offered.
      The HNSW and rescore parameters are our defaults for the type (`rag.index.mapping`).
    """

    analyzer: str = "standard"
    vector_index: Literal["int8_hnsw", "bbq_hnsw", "flat"] = "int8_hnsw"


RRF_ROUTES = ("bm25", "vector")
"""Route names of the hybrid retriever (`rag.retrieve.hybrid.ROUTES`; config cannot import it)."""

# Elasticsearch rejects a kNN search with num_candidates above this.
MAX_NUM_CANDIDATES = 10_000


class RetrievalConfig(_Strict):
    """Retrieval parameters of one knowledge base (design s9-1, s9-3, s10).

    - `top_k_per_route`: hits each route (BM25, vector) returns before fusion.
    - `num_candidates`: HNSW candidates per shard of the kNN route (tuned on the question set).
    - `title_boost`: weight of `title_path` against `content` in the BM25 `multi_match`; 1.0 since
      P0-13b (2.0 let short section titles that match only question words win, docs/decisions.md).
    - `rrf_k`: the constant of reciprocal rank fusion, score = sum over routes w / (k + rank).
    - `rrf_weights`: the weight w of each route (`bm25`, `vector`) in that sum; equal by default
      (s9-3), >= 0, not all 0 (spec P0-13).
    - `rerank_n`: fused candidates sent to the reranker (s10-1); later ones are dropped.
    - `final_k`: chunks returned when the caller gives no top_k.
    - `rerank_with_title`: the reranker reads the title path before the body (s10-1); false sends
      the body alone (spec P2-03 ablation).
    - `threshold`: rerank score (sigmoid, 0 to 1) a chunk needs to be returned (s10-2).
    - `expand`: context added around each returned chunk (s10-3, s7-2): `none`, `neighbors`
      (same page, chunk_index within `expand_window`) or `parent` (its heading section).
    - `expand_max_tokens`: reranker tokens of returned chunks plus added context; context that
      does not fit is left out, returned chunks always stay.
    """

    top_k_per_route: int = Field(default=20, gt=0)
    num_candidates: int = Field(default=100, gt=0, le=MAX_NUM_CANDIDATES)
    title_boost: float = Field(default=1.0, gt=0)
    rrf_k: int = Field(default=60, ge=0)
    rrf_weights: dict[str, float] = Field(
        default_factory=lambda: {route: 1.0 for route in RRF_ROUTES}
    )
    rerank_n: int = Field(default=30, gt=0)
    final_k: int = Field(default=8, gt=0)
    rerank_with_title: bool = True
    threshold: float = Field(default=0.3, ge=0, le=1)
    expand: Literal["none", "neighbors", "parent"] = "none"
    expand_window: int = Field(default=1, ge=1)
    expand_max_tokens: int = Field(default=4000, gt=0)

    @model_validator(mode="after")
    def _rrf_weights_per_route(self) -> Self:
        if set(self.rrf_weights) != set(RRF_ROUTES):
            raise ValueError(
                f"rrf_weights must give one weight for each route {list(RRF_ROUTES)}, "
                f"got {sorted(self.rrf_weights)}"
            )
        if any(w < 0 for w in self.rrf_weights.values()):
            raise ValueError(f"rrf_weights must be >= 0, got {self.rrf_weights}")
        if not any(w > 0 for w in self.rrf_weights.values()):
            raise ValueError("rrf_weights must not all be 0")
        return self

    @model_validator(mode="after")
    def _final_k_within_rerank_n(self) -> Self:
        if self.final_k > self.rerank_n:
            raise ValueError(f"final_k ({self.final_k}) must not exceed rerank_n ({self.rerank_n})")
        return self

    @model_validator(mode="after")
    def _num_candidates_cover_k(self) -> Self:
        if self.num_candidates < self.top_k_per_route:
            raise ValueError(
                f"num_candidates ({self.num_candidates}) must be at least top_k_per_route "
                f"({self.top_k_per_route}): Elasticsearch requires num_candidates >= k"
            )
        return self


class EvalConfig(_Strict):
    """Question sets. `dir`: `None` means `evals/datasets/<kb>` (design s16-2). `questions`: the
    file name in `dir` of the labelled set the eval uses by default; `None` = the only *.jsonl
    there (the directory may also hold other sets, e.g. no-answer questions, spec P0-13)."""

    dir: str | None = None
    questions: str | None = None


class AccessConfig(_Strict):
    """Callers allowed to use this knowledge base; `*` means every authenticated caller.

    The model default is empty (nobody). `load_kb_config` gives configs under `kb/public/`
    `["*"]` and requires every other config to set `access` explicitly.
    """

    callers: list[str] = Field(default_factory=list)


class AnswerConfig(_Strict):
    """Answer generation (design s10-3, s11-1, s8-4; spec P1-02).

    - `context_tokens`: estimated tokens of all numbered fragments together (s10-3).
    - `language`: `auto` answers in the question's language; `zh` / `en` fix it for this
      knowledge base (s8-4).
    - `parse_retries`: extra model calls when an answer cites no fragment at all (s11-1).
    """

    context_tokens: int = Field(default=4000, gt=0)
    language: Literal["auto", "zh", "en"] = "auto"
    parse_retries: int = Field(default=1, ge=0)


class AgentConfig(_Strict):
    """The `ask_docs` graph for this knowledge base (design s13-3, s15-3; spec P3-01, P3-02).

    - `max_rounds`: retrieval rounds of one request at most (the sufficiency judge of P3-03 asks
      for another round while this many have not been used). Deadline, recursion limit and
      token cap are service settings (`agent_*`), they bound a request whatever the KB.
    - `rewrite`: the rewrite step may call the small model (standalone question from the
      history, declarative query, sub-questions). Off: the question is searched as asked and no
      model is called before the answer. Where the rules settle the question (a statement with
      no history) no model is called either way.
    - `declarative`: write the search query as a statement, not a question (s8-2).
    - `max_sub_questions`: most searches a split question may make (1: never split).
    - `clarify`: ask the caller back when the question is too unclear to search
      (`need_clarification`, at most once per request). Off: such a question is searched as is.
    - `route_model`: the route step may ask the small model about what the rules cannot decide
      (only questions the rules call undecided, and only for the features switched on above).
    - `judge` (spec P3-03, s13-4): after every round, decide whether what was found is enough
      (the threshold first, the small model second) and go on to another round when it is not.
      Off: one round, as before. Another round needs a new phrasing, which needs `rewrite`;
      with `rewrite` off the run stops after the judgement (`stop_reason: no_new_query`).
    - `multi_queries`: phrasings of each missing sub-question in round two (multi-query).
    - `hyde`: round three searches with a hypothetical answer passage (HyDE, s8-2); off: round
      three asks for new phrasings again.
    - `judge_skip_score` (spec P3-06): a sub-question that has a chunk with a rerank score at or
      above this is taken as grounded without asking the judge model; when no sub-question is left
      to ask, no model call is made. None: the model is asked about every sub-question that has
      chunks above the threshold (the P3-03 behaviour).
    - `round2_on_not_enough` (spec P3-06): with the judge on, a plain `not_enough` (no chunk
      passed the threshold in any round) goes on to another round (true, design s13-4). False:
      it goes straight to "not found" with the nearest chunks, and a further round runs only for
      `partly_enough`, which names a missing sub-question. Data behind the switch: decisions.md.
    """

    max_rounds: int = Field(default=3, ge=1)
    rewrite: bool = False
    declarative: bool = True
    max_sub_questions: int = Field(default=3, ge=1, le=8)
    clarify: bool = False
    route_model: bool = False
    judge: bool = False
    multi_queries: int = Field(default=3, ge=1, le=6)
    hyde: bool = True
    judge_skip_score: float | None = Field(default=None, ge=0.0, le=1.0)
    round2_on_not_enough: bool = True


class KnowledgeBaseConfig(_Strict):
    kb: str = Field(default="default", pattern=r"^[a-z0-9][a-z0-9_-]*$")
    description: str = ""
    """What the knowledge base contains, for callers choosing one: shown in the MCP tool
    descriptions and by `list_knowledge_bases` (spec P0-09). Not part of the config version."""
    source: SourceConfig = Field(default_factory=SourceConfig)
    language: str = "auto"
    pipelines: dict[str, PipelineConfig] = Field(default_factory=_default_pipelines)
    metadata: MetadataConfig = Field(default_factory=MetadataConfig)
    es: EsConfig = Field(default_factory=EsConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    eval: EvalConfig = Field(default_factory=EvalConfig)
    access: AccessConfig = Field(default_factory=AccessConfig)
    answer: AnswerConfig = Field(default_factory=AnswerConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)

    @model_validator(mode="after")
    def _skip_score_is_not_below_the_threshold(self) -> Self:
        skip = self.agent.judge_skip_score
        if skip is not None and skip < self.retrieval.threshold:
            raise ValueError(
                f"agent.judge_skip_score ({skip}) must not be below retrieval.threshold "
                f"({self.retrieval.threshold}): a chunk that does not pass the threshold is "
                "not a sure one"
            )
        return self

    @model_validator(mode="after")
    def _default_eval_dir(self) -> Self:
        if self.eval.dir is None:
            self.eval.dir = f"evals/datasets/{self.kb}"
        return self


def is_public_kb_file(path: str | Path, kb_root: str | Path) -> bool:
    """True only when `path`, relative to `kb_root`, starts with `public/`.

    `kb/private/public/x.yaml` is not public, and neither is a file outside `kb_root`.
    """
    try:
        rel = Path(path).resolve().relative_to(Path(kb_root).resolve())
    except ValueError:
        return False
    return len(rel.parts) > 1 and rel.parts[0] == "public"


def load_kb_config(path: str | Path, *, kb_root: str | Path = "kb") -> KnowledgeBaseConfig:
    """Load one knowledge-base YAML file. `kb` defaults to the file name without extension.

    A config under `<kb_root>/public/` without `access` is open to every caller; any other
    config must set `access.callers` explicitly, so a private knowledge base is never open by
    omission. `kb_root` is the knowledge-base config directory (`ServiceSettings.kb_dir`).
    """
    path = Path(path)
    data = yaml.safe_load(path.read_text("utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: knowledge-base config must be a mapping")
    data.setdefault("kb", path.stem)
    if "access" not in data:
        if not is_public_kb_file(path, kb_root):
            raise ValueError(
                f"{path}: knowledge bases outside {kb_root}/public/ must set access.callers "
                "explicitly"
            )
        data["access"] = {"callers": ["*"]}
    return KnowledgeBaseConfig.model_validate(data)


def load_kb_configs(kb_dir: str | Path) -> dict[str, KnowledgeBaseConfig]:
    """Load every `*.yaml` / `*.yml` under `kb_dir` (public and private); names must be unique."""
    kb_dir = Path(kb_dir)
    configs: dict[str, KnowledgeBaseConfig] = {}
    origins: dict[str, Path] = {}
    files = sorted([*kb_dir.rglob("*.yaml"), *kb_dir.rglob("*.yml")])
    for path in files:
        cfg = load_kb_config(path, kb_root=kb_dir)
        if cfg.kb in configs:
            raise ValueError(f"duplicate knowledge base {cfg.kb!r} in {origins[cfg.kb]} and {path}")
        configs[cfg.kb] = cfg
        origins[cfg.kb] = path
    return configs
