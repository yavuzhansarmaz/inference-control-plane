"""Public domain models. Missing metrics are None, never fabricated zero."""
from __future__ import annotations

from pydantic import BaseModel, Field


def as_ratio(value: float | None) -> float | None:
    """Normalize a percent-form value (>1) to a 0..1 ratio."""
    return value / 100.0 if value is not None and value > 1.0 else value


class EndpointConfig(BaseModel):
    provider: str = "vllm"
    base_url: str
    metrics_url: str | None = None
    model: str | None = None
    gpu_hourly_usd: float | None = Field(default=None, ge=0)

class MetricSnapshot(BaseModel):
    provider: str
    status: str  # ok | degraded | unreachable
    collected_at: str
    source: str
    errors: list[str] = Field(default_factory=list)
    requests_running: float | None = None
    requests_waiting: float | None = None
    kv_cache_usage_ratio: float | None = None  # 0..1 when known
    gpu_utilization_ratio: float | None = None
    prompt_tokens_total: float | None = None
    generation_tokens_total: float | None = None
    request_success_total: float | None = None
    request_failure_total: float | None = None
    error_rate: float | None = None
    ttft_p95_seconds: float | None = None
    e2e_latency_p95_seconds: float | None = None
    tokens_per_second: float | None = None  # only if caller supplies a window rate

class EndpointCapabilities(BaseModel):
    """Best-effort, GET-only capability probe. Failures are notes, never exceptions."""

    health_ok: bool = False
    models: list[str] = Field(default_factory=list)
    metrics_ok: bool = False
    errors: list[str] = Field(default_factory=list)
    checked_at: str = ""


class DerivedRates(BaseModel):
    """Rates derived from two counter snapshots; unknown rates stay None."""

    elapsed_seconds: float | None = None
    tokens_per_second: float | None = None
    generation_tokens_per_second: float | None = None
    prompt_tokens_per_second: float | None = None
    requests_per_second: float | None = None


class EffectiveCost(BaseModel):
    """Cost estimate from a user-supplied GPU price; kind is always user_assumption."""

    kind: str = "user_assumption"
    gpu_hourly_usd: float
    hourly_cost_usd: float
    utilization_ratio: float | None = None
    effective_cost_per_utilized_hour_usd: float | None = None
    cost_per_million_generation_tokens_usd: float | None = None
    notes: str = ""


class Recommendation(BaseModel):
    code: str
    title: str
    evidence: str
    assumptions: list[str] = Field(default_factory=list)
    confidence: str = "low"

class ComparisonRow(BaseModel):
    metric: str
    current: float | None = None
    managed: float | None = None
    unit: str
    kind: str  # measured | estimated | user_assumption
    notes: str = ""
