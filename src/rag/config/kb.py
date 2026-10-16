"""Knowledge-base config (design s5-1): one YAML file per knowledge base under `kb/`.

Every field has a default, so a new knowledge base only has to name its data source.
Unknown keys are rejected so a typo in YAML fails loudly instead of silently using a default.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Self

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


class RetrievalConfig(_Strict):
    top_k_per_route: int = Field(default=20, gt=0)
    rerank_n: int = Field(default=30, gt=0)
    final_k: int = Field(default=8, gt=0)
    threshold: float = 0.3

    @model_validator(mode="after")
    def _final_k_within_rerank_n(self) -> Self:
        if self.final_k > self.rerank_n:
            raise ValueError(f"final_k ({self.final_k}) must not exceed rerank_n ({self.rerank_n})")
        return self


class EvalConfig(_Strict):
    """Question sets. `None` means `evals/datasets/<kb>` (design s16-2)."""

    dir: str | None = None


class AccessConfig(_Strict):
    """Callers allowed to use this knowledge base; `*` means every authenticated caller."""

    callers: list[str] = Field(default_factory=lambda: ["*"])


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

    @model_validator(mode="after")
    def _default_eval_dir(self) -> Self:
        if self.eval.dir is None:
            self.eval.dir = f"evals/datasets/{self.kb}"
        return self


def load_kb_config(path: str | Path) -> KnowledgeBaseConfig:
    """Load one knowledge-base YAML file. `kb` defaults to the file name without extension."""
    path = Path(path)
    data = yaml.safe_load(path.read_text("utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: knowledge-base config must be a mapping")
    data.setdefault("kb", path.stem)
    return KnowledgeBaseConfig.model_validate(data)


def load_kb_configs(kb_dir: str | Path) -> dict[str, KnowledgeBaseConfig]:
    """Load every `*.yaml` / `*.yml` under `kb_dir` (public and private); names must be unique."""
    kb_dir = Path(kb_dir)
    configs: dict[str, KnowledgeBaseConfig] = {}
    origins: dict[str, Path] = {}
    files = sorted([*kb_dir.rglob("*.yaml"), *kb_dir.rglob("*.yml")])
    for path in files:
        cfg = load_kb_config(path)
        if cfg.kb in configs:
            raise ValueError(f"duplicate knowledge base {cfg.kb!r} in {origins[cfg.kb]} and {path}")
        configs[cfg.kb] = cfg
        origins[cfg.kb] = path
    return configs
