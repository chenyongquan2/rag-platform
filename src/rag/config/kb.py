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
    analyzer: str = "standard"


# Elasticsearch rejects a kNN search with num_candidates above this.
MAX_NUM_CANDIDATES = 10_000


class RetrievalConfig(_Strict):
    """Retrieval parameters of one knowledge base (design s9-1, s9-3, s10).

    - `top_k_per_route`: hits each route (BM25, vector) returns before fusion.
    - `num_candidates`: HNSW candidates per shard of the kNN route (tuned on the question set).
    - `title_boost`: weight of `title_path` against `content` in the BM25 `multi_match`.
    - `rrf_k`: the constant of reciprocal rank fusion, score = sum over routes 1 / (k + rank).
    """

    top_k_per_route: int = Field(default=20, gt=0)
    num_candidates: int = Field(default=100, gt=0, le=MAX_NUM_CANDIDATES)
    title_boost: float = Field(default=2.0, gt=0)
    rrf_k: int = Field(default=60, ge=0)
    rerank_n: int = Field(default=30, gt=0)
    final_k: int = Field(default=8, gt=0)
    threshold: float = 0.3

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
    """Question sets. `None` means `evals/datasets/<kb>` (design s16-2)."""

    dir: str | None = None


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


class KnowledgeBaseConfig(_Strict):
    kb: str = Field(default="default", pattern=r"^[a-z0-9][a-z0-9_-]*$")
    source: SourceConfig = Field(default_factory=SourceConfig)
    language: str = "auto"
    pipelines: dict[str, PipelineConfig] = Field(default_factory=_default_pipelines)
    metadata: MetadataConfig = Field(default_factory=MetadataConfig)
    es: EsConfig = Field(default_factory=EsConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    eval: EvalConfig = Field(default_factory=EvalConfig)
    access: AccessConfig = Field(default_factory=AccessConfig)
    answer: AnswerConfig = Field(default_factory=AnswerConfig)

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
