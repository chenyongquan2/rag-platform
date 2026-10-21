"""How the cache served a request, as callers see it (the `cache` field of a response)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class CacheInfo(BaseModel):
    """How the result cache served this request (s12, spec P5-01); the response field `cache`
    is null when the service runs without a cache layer."""

    model_config = ConfigDict(extra="forbid")

    layer: str = Field(description="The cache layer that was asked: l1 (exact).")
    hit: bool = Field(description="The result came from the cache or from an identical request.")
    outcome: str = Field(
        description="hit; shared (waited for an identical request that was running); miss "
        "(computed, written when complete); bypass (cache not used: breaker open, or a call "
        "that must not read it); error (the cache store failed, the result was computed)."
    )
    age_s: float | None = Field(
        default=None, description="Seconds since the cached result was made (hit only)."
    )
