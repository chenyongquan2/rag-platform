"""Build readers, cleaning rules and chunk steps of a knowledge base from its YAML (design s5-1).

`KbProcessor` runs the three stages the ingestion job (P0-06) and the quality report share:
`read` (connector documents -> Documents), `clean` (rules, before the IngestionPipeline, s7-4) and
`chunk` (steps; P0-06 hands `steps_for(ext)` to the IngestionPipeline instead).
Extensions whose pipelines are configured the same share one reader, one rule chain (so `dedupe`
sees all of them) and one `pipeline_version`.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from llama_index.core.schema import BaseNode, Document, TransformComponent

from rag.config import KnowledgeBaseConfig, PipelineConfig, ServiceSettings, StepConfig
from rag.data import SourceDocument
from rag.ingest.clean import CleanRule, RuleStats
from rag.ingest.language import UNDETERMINED
from rag.ingest.meta import apply_whitelist
from rag.ingest.readers import reader_name_for
from rag.ingest.tokens import TokenizerFn, load_tokenizer
from rag.registry import readers, steps

INGEST_CODE_VERSION = "1"
"""Raise when a reader, rule or step changes its output for the same config, so every page is
processed again (the version is part of the Document metadata and therefore of its hash)."""


def pipeline_version(
    cfg: KnowledgeBaseConfig, ext: str, *, tokenizer_model: str | None = None
) -> str:
    """`v<code>-<hash>` of everything that decides a page's chunks: reader, rules, steps and their
    parameters, metadata whitelist, tokenizer. Changing any of them changes every Document hash,
    so the IngestionPipeline processes the pages again although their text did not change (s5-4).
    """
    pipe = _pipeline(cfg, ext)
    payload = {
        "code": INGEST_CODE_VERSION,
        "reader": reader_name_for(ext, pipe),
        "clean": [s.model_dump() for s in pipe.clean],
        "steps": [s.model_dump() for s in pipe.steps],
        "metadata": cfg.metadata.model_dump(),
        "language": cfg.language,
        "tokenizer": tokenizer_model or ServiceSettings().reranker_model,
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    return f"v{INGEST_CODE_VERSION}-{digest[:10]}"


def _pipeline(cfg: KnowledgeBaseConfig, ext: str) -> PipelineConfig:
    for key, pipe in cfg.pipelines.items():
        if key.lower() == ext.lower():
            return pipe
    raise KeyError(f"knowledge base {cfg.kb!r} has no pipeline for {ext!r}")


@dataclass
class ReadResult:
    documents: list[Document]
    skipped: dict[str, str] = field(default_factory=dict)
    """doc_id -> `no_pipeline` | `read_error`."""
    errors: dict[str, str] = field(default_factory=dict)


@dataclass
class _Group:
    version: str
    pipeline: PipelineConfig
    reader: Any
    rules: list[CleanRule]
    steps: list[TransformComponent] | None = None


def _create(step: StepConfig, context: dict[str, Any]) -> list[Any]:
    factory = steps.get(step.name)
    wanted: dict[str, str] = getattr(factory, "context_params", {})
    params = {p: context[k] for p, k in wanted.items() if k in context and p not in step.params}
    params.update(step.params)
    made = factory(**params)
    return list(made) if isinstance(made, list) else [made]


class KbProcessor:
    """Readers, rules and steps of one knowledge base, built once from its config."""

    def __init__(
        self,
        cfg: KnowledgeBaseConfig,
        *,
        settings: ServiceSettings | None = None,
        tokenizer: TokenizerFn | None = None,
    ) -> None:
        self.cfg = cfg
        self.settings = settings if settings is not None else ServiceSettings()
        self._tokenizer = tokenizer
        self.kb_language = UNDETERMINED if cfg.language == "auto" else cfg.language
        self._groups: dict[str, _Group] = {}
        self._ext_version: dict[str, str] = {}
        for ext, pipe in cfg.pipelines.items():
            version = pipeline_version(cfg, ext, tokenizer_model=self.settings.reranker_model)
            self._ext_version[ext.lower()] = version
            if version in self._groups:
                continue
            reader = readers.create(
                reader_name_for(ext, pipe),
                kb=cfg.kb,
                pipeline_version=version,
                language=self.kb_language,
                embed_keys=cfg.metadata.embed,
                llm_keys=cfg.metadata.llm,
            )
            context = {"kb_language": self.kb_language}
            rules = [r for s in pipe.clean for r in _create(s, context)]
            for rule in rules:
                if not isinstance(rule, CleanRule):
                    raise TypeError(f"{type(rule).__name__} in `clean` is not a cleaning rule")
            self._groups[version] = _Group(version, pipe, reader, rules)

    @property
    def tokenizer(self) -> TokenizerFn:
        if self._tokenizer is None:
            self._tokenizer = load_tokenizer(self.settings.reranker_model)
        return self._tokenizer

    def version_for(self, ext: str) -> str | None:
        return self._ext_version.get(ext.lower())

    def read(self, sources: Iterable[SourceDocument]) -> ReadResult:
        result = ReadResult(documents=[])
        for src in sources:
            version = self.version_for(src.ext)
            if version is None:
                result.skipped[src.doc_id] = "no_pipeline"
                continue
            try:
                result.documents.extend(self._groups[version].reader.load_data(src))
            except Exception as exc:  # one bad file must not stop the run (s7-4); it is reported
                result.skipped[src.doc_id] = "read_error"
                result.errors[src.doc_id] = f"{type(exc).__name__}: {exc}"
        return result

    def _by_group(self, nodes: Iterable[BaseNode]) -> dict[str, list[BaseNode]]:
        grouped: dict[str, list[BaseNode]] = {}
        for node in nodes:
            version = node.metadata.get("pipeline_version")
            if version not in self._groups:
                raise ValueError(f"{node.node_id}: unknown pipeline_version {version!r}")
            grouped.setdefault(version, []).append(node)
        return grouped

    def clean(self, documents: Iterable[Document]) -> list[Document]:
        """Run each group's rules in order; returns the kept Documents in input order."""
        kept: set[str] = set()
        documents = list(documents)
        for version, docs in self._by_group(documents).items():
            for rule in self._groups[version].rules:
                docs = rule(docs)
            for doc in docs:
                apply_whitelist(doc, self.cfg.metadata.embed, self.cfg.metadata.llm)
                kept.add(doc.node_id)
        return [d for d in documents if d.node_id in kept]

    def steps_for(self, ext: str) -> list[TransformComponent]:
        version = self.version_for(ext)
        if version is None:
            raise KeyError(f"knowledge base {self.cfg.kb!r} has no pipeline for {ext!r}")
        return self._steps(self._groups[version])

    def _steps(self, group: _Group) -> list[TransformComponent]:
        if group.steps is None:
            context: dict[str, Any] = {
                "kb_language": self.kb_language,
                "tokenizer": self.tokenizer,
                "tokenizer_model": self.settings.reranker_model,
                "embed_keys": list(self.cfg.metadata.embed),
                "llm_keys": list(self.cfg.metadata.llm),
            }
            for step in group.pipeline.steps:
                if step.name == "token_split" and "max_tokens" in step.params:
                    context["max_tokens"] = step.params["max_tokens"]
            group.steps = [t for s in group.pipeline.steps for t in _create(s, context)]
        return group.steps

    def chunk(self, documents: Iterable[Document]) -> list[BaseNode]:
        """Run each group's steps (what the IngestionPipeline does in P0-06), in input order."""
        out: list[BaseNode] = []
        for version, docs in self._by_group(documents).items():
            nodes: list[BaseNode] = list(docs)
            for transform in self._steps(self._groups[version]):
                nodes = list(transform(nodes))
            out.extend(nodes)
        return out

    def rule_stats(self) -> list[tuple[str, RuleStats]]:
        """(rule name, stats) of every cleaning rule; `chunk_rule_stats` has the chunk steps."""
        found: list[tuple[str, RuleStats]] = []
        for group in self._groups.values():
            for rule in group.rules:
                found.append((rule.name, rule.stats))
        return found

    def chunk_rule_stats(self) -> list[tuple[str, RuleStats]]:
        return [
            (t.name, t.stats)
            for g in self._groups.values()
            for t in (g.steps or [])
            if isinstance(t, CleanRule)
        ]

    def step_param(self, version: str, name: str, param: str) -> Any:
        """A parameter of a built step (`max_tokens` of `token_split`, ...), or None."""
        group = self._groups[version]
        for transform in self._steps(group):
            if type(transform).class_name() == name:
                return getattr(transform, param, None)
        return None
