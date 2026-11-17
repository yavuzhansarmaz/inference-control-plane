"""Derived counter rates and effective-cost estimates from snapshots."""

from __future__ import annotations

from datetime import datetime

from inference_control_plane.models import (
    DerivedRates,
    EffectiveCost,
    MetricSnapshot,
    as_ratio,
)


def _parse_iso(value: str) -> datetime | None:
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return None


def _counter_rate(prev: float | None, curr: float | None, elapsed: float) -> float | None:
    if prev is None or curr is None:
        return None
    delta = curr - prev
    if delta < 0:  # counter reset / rewound (endpoint restarted)
        return None
    return delta / elapsed


def derive_rates(prev: MetricSnapshot, curr: MetricSnapshot) -> DerivedRates:
    """Derive per-second rates from two snapshots; unknown rates stay None."""
    prev_at, curr_at = _parse_iso(prev.collected_at), _parse_iso(curr.collected_at)
    if prev_at is None or curr_at is None:
        return DerivedRates()
    try:
        elapsed = (curr_at - prev_at).total_seconds()
    except TypeError:  # naive vs aware mix
        return DerivedRates()
    if elapsed <= 0:
        return DerivedRates(elapsed_seconds=elapsed if elapsed == 0 else None)
    generation = _counter_rate(prev.generation_tokens_total, curr.generation_tokens_total, elapsed)
    # Requests: prefer success+failure totals when both snapshots report them.
    def _req_total(s: MetricSnapshot) -> float | None:
        if s.request_success_total is None and s.request_failure_total is None:
            return None
        return (s.request_success_total or 0.0) + (s.request_failure_total or 0.0)

    return DerivedRates(
        elapsed_seconds=elapsed,
        tokens_per_second=generation,
        generation_tokens_per_second=generation,
        prompt_tokens_per_second=_counter_rate(
            prev.prompt_tokens_total, curr.prompt_tokens_total, elapsed
        ),
        requests_per_second=_counter_rate(_req_total(prev), _req_total(curr), elapsed),
    )


def estimate_effective_cost(
    snapshot: MetricSnapshot,
    gpu_hourly_usd: float | None = None,
    *,
    tokens_per_second: float | None = None,
) -> EffectiveCost | None:
    """Estimate cost from a user GPU price; None when no price is given."""
    if gpu_hourly_usd is None or gpu_hourly_usd < 0:
        return None
    utilization = as_ratio(snapshot.gpu_utilization_ratio)
    effective = (
        gpu_hourly_usd / utilization
        if utilization is not None and utilization > 0
        else None
    )
    tps = tokens_per_second if tokens_per_second is not None else snapshot.tokens_per_second
    cost_per_million = None
    if tps is not None and tps > 0:
        cost_per_million = gpu_hourly_usd / (tps * 3600.0) * 1_000_000.0
    notes = (
        "Estimate only: based on the user-supplied GPU hourly price "
        f"(${gpu_hourly_usd:.4g}/h), not a measured bill."
    )
    if utilization is None:
        notes += " GPU utilization was not reported, so no effective (utilization-adjusted) cost is given."
    elif effective is None:
        notes += " GPU utilization is zero, so no effective (utilization-adjusted) cost is given."
    return EffectiveCost(
        gpu_hourly_usd=gpu_hourly_usd,
        hourly_cost_usd=gpu_hourly_usd,
        utilization_ratio=utilization,
        effective_cost_per_utilized_hour_usd=effective,
        cost_per_million_generation_tokens_usd=cost_per_million,
        notes=notes,
    )
