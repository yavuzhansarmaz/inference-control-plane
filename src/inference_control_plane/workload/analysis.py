"""Generic workload aggregation and transparent managed-target projections."""

from __future__ import annotations

import math
import uuid
from collections import defaultdict
from statistics import mean
from typing import Any


def _round(value: float | None, digits: int = 2) -> float | None:
    return round(value, digits) if value is not None and math.isfinite(value) else None


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return ordered[index]


def _improvement(current: float | None, projected: float | None) -> float | None:
    if current is None or projected is None or current <= 0:
        return None
    return _round(((current - projected) / current) * 100, 1)


def _confidence(requests: int, latency_count: int, cost_count: int) -> str:
    if requests >= 500 and latency_count >= requests * 0.8 and cost_count >= requests * 0.8:
        return "high"
    if requests >= 100 and latency_count >= requests * 0.5:
        return "medium"
    return "low"


def _workload_aggregate(
    key: str,
    rows: list[dict[str, Any]],
    *,
    window_days: int,
    managed_rate_usd_per_million: float,
    ttft_ms: float,
    output_tokens_per_second: float,
) -> dict[str, Any]:
    """Generic per-workload aggregation and managed-target cost/latency projection."""
    request_count = len(rows)
    input_tokens = sum(max(0, int(row.get("input_tokens") or 0)) for row in rows)
    output_tokens = sum(max(0, int(row.get("output_tokens") or 0)) for row in rows)
    total_tokens = sum(max(0, int(row.get("total_tokens") or 0)) for row in rows)
    if total_tokens <= 0:
        total_tokens = input_tokens + output_tokens
    latencies = [float(row["latency_ms"]) for row in rows if row.get("latency_ms") is not None]
    costs = [float(row["cost_usd"]) for row in rows if row.get("cost_usd") is not None]
    qualities = [
        float(row["quality_score"]) for row in rows if row.get("quality_score") is not None
    ]
    successful = sum(1 for row in rows if row.get("success", True))
    success_rate = successful / max(request_count, 1)
    models = sorted({str(row.get("model") or "unknown-model") for row in rows})
    workload_name = str(rows[0].get("workload_name") or key)

    requests_per_day = request_count / max(window_days, 1)
    monthly_scale = 30 / max(window_days, 1)
    monthly_requests = round(request_count * monthly_scale)
    monthly_tokens = round(total_tokens * monthly_scale)
    monthly_current_cost = sum(costs) * monthly_scale if costs else None
    monthly_projected_cost = monthly_tokens / 1_000_000 * managed_rate_usd_per_million

    avg_output_tokens = output_tokens / max(request_count, 1)
    projected_latency_ms = ttft_ms + avg_output_tokens / max(output_tokens_per_second, 1) * 1000
    current_latency = _percentile(latencies, 0.95)
    current_quality = mean(qualities) * 100 if qualities else None
    quality_source = "provider_scores" if qualities else "not_reported"

    return {
        "id": str(uuid.uuid4()),
        "workload_key": key,
        "workload_name": workload_name,
        "confidence": _confidence(request_count, len(latencies), len(costs)),
        "current": {
            "requests": request_count,
            "requests_per_day": _round(requests_per_day, 1),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "average_latency_ms": _round(mean(latencies), 1) if latencies else None,
            "p95_latency_ms": _round(current_latency, 1),
            "total_cost_usd": _round(sum(costs), 4) if costs else None,
            "success_rate": _round(success_rate, 4),
            "quality_index": _round(current_quality, 1),
            "quality_source": quality_source,
            "models": models,
        },
        "monthly_requests": monthly_requests,
        "monthly_tokens": monthly_tokens,
        "cost": {
            "current": _round(monthly_current_cost, 2),
            "projected": _round(monthly_projected_cost, 2),
            "improvement_percent": _improvement(monthly_current_cost, monthly_projected_cost),
            "unit": "USD/month",
            "source": "observed_cost_scaled_vs_managed_rate",
        },
        "latency": {
            "current": _round(current_latency, 1),
            "projected": _round(projected_latency_ms, 1),
            "improvement_percent": _improvement(current_latency, projected_latency_ms),
            "unit": "ms p95",
            "source": "observed_p95_vs_throughput_model",
        },
        "quality": {
            "current": _round(current_quality, 1),
            "projected": None,
            "improvement_percent": None,
            "unit": "/100",
            "source": quality_source,
        },
    }


def analyze_events(
    events: list[dict[str, Any]],
    *,
    window_days: int,
    managed_rate_usd_per_million: float,
    ttft_ms: float,
    output_tokens_per_second: float,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[str]]:
    """Aggregate normalized events per workload. No product or migration scoring."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        grouped[str(event.get("workload_key") or "unclassified")].append(event)

    opportunities = [
        _workload_aggregate(
            key,
            rows,
            window_days=window_days,
            managed_rate_usd_per_million=managed_rate_usd_per_million,
            ttft_ms=ttft_ms,
            output_tokens_per_second=output_tokens_per_second,
        )
        for key, rows in grouped.items()
    ]
    opportunities.sort(key=lambda item: item["current"]["requests"], reverse=True)
    known_costs = [
        float(event["cost_usd"]) for event in events if event.get("cost_usd") is not None
    ]
    summary = {
        "total_requests": len(events),
        "total_tokens": sum(max(0, int(event.get("total_tokens") or 0)) for event in events),
        "total_cost_usd": _round(sum(known_costs), 4) if known_costs else None,
        "workload_count": len(opportunities),
    }
    assumptions = [
        f"Managed inference is modeled at ${managed_rate_usd_per_million:.2f} per 1M total tokens.",
        f"Latency uses {ttft_ms:.0f} ms time-to-first-token and {output_tokens_per_second:.0f} output tokens/second.",
        "Monthly projections scale the selected historical window to 30 days.",
        "Projections are estimates, not production benchmarks; validate under representative load before acting on them.",
    ]
    return summary, opportunities, assumptions
