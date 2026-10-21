"""The search config version of the existing knowledge bases did not change (spec P6-01 review).

The expected values were computed with the search code of `main` before this spec (commit
00cd020, which already leaves the cache section out) and are written down here, not rebuilt from
the dump this branch makes: a field this spec adds to the config (`reader_params`, the two
`replace_table_summaries*` switches) that showed up in the dump would change the hash, every
result's `config_version` and every cache key made from it. If a later spec changes these values
on purpose, recompute them with the code of that commit and say so in its commit message.
"""

from __future__ import annotations

import pytest

from rag.config import ServiceSettings, load_kb_config, repo_path
from rag.retrieve.pipeline import search_config_version

KB_ROOT = repo_path("kb")
EMBED_MODEL = "qwen/text-embedding-v4"


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("kb/public/hf_docs.yaml", "3d91c2151f167e04"),
        ("kb/private/mt5.yaml", "5986075fd45914f0"),
    ],
)
def test_search_config_version_of_the_existing_knowledge_bases_is_unchanged(
    path: str, expected: str
) -> None:
    settings = ServiceSettings(_env_file=None, embed_backend="fake")  # type: ignore[call-arg]
    cfg = load_kb_config(repo_path(path), kb_root=KB_ROOT)
    assert search_config_version(cfg, settings, EMBED_MODEL) == expected


def test_a_pipeline_with_reader_params_has_another_search_config_version() -> None:
    """The other way round: reader parameters do change what is indexed, so for a knowledge base
    that sets them they stay in the version."""
    settings = ServiceSettings(_env_file=None, embed_backend="fake")  # type: ignore[call-arg]
    cfg = load_kb_config(repo_path("kb/private/pdf_demo.yaml"), kb_root=KB_ROOT)
    pipe = cfg.pipelines[".pdf"]
    changed = cfg.model_copy(
        update={
            "pipelines": {
                **cfg.pipelines,
                ".pdf": pipe.model_copy(
                    update={"reader_params": {**pipe.reader_params, "min_image_px": 128}}
                ),
            }
        }
    )
    assert search_config_version(changed, settings, EMBED_MODEL) != search_config_version(
        cfg, settings, EMBED_MODEL
    )
