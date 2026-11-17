"""Multi-endpoint (fleet) aggregation and recommendations.

Aggregation rules: ratios are averaged over endpoints that report them and
are never summed. Counts (requests running/waiting) are summed only over
endpoints reporting that metric; the ``*_reporting`` coverage counts on
:class:`FleetSummary` name how many endpoints contributed. Missing values
stay ``None``.
"""

from __future__ import annotations

from collections.abc import Mapping
from statistics import mean
from typing import Any

from inference_control_plane.models import (
    FleetEndpoint,
    FleetSummary,
    MetricSnapshot,
    Recommendation,
)
from inference_control_plane.thresholds import (
    ERROR_RATE_THRESHOLD,
    KV_IDLE_RATIO,
    KV_SATURATED_RATIO,
    WAITING_QUEUE_THRESHOLD,
)


def coerce_fleet(source: Any) -> list[FleetEndpoint]:
    """Accept a mapping name->snapshot, FleetEndpoints, or (name, snapshot) pairs."""
    if source is None:
        return []
    if isinstance(source, FleetSummary):
        return []  # a summary has no per-endpoint snapshots; see recommend_from_fleet
    if isinstance(source, Mapping):
        return [
            FleetEndpoint(name=str(name), snapshot=snap)
            for name, snap in source.items()
            if isinstance(snap, MetricSnapshot)
        ]
    endpoints: list[FleetEndpoint] = []
    for item in source:
        if isinstance(item, FleetEndpoint):
            endpoints.append(item)
        elif isinstance(item, (tuple, list)) and len(item) == 2:
            name, snap = item
            if isinstance(snap, MetricSnapshot):
                endpoints.append(FleetEndpoint(name=str(name), snapshot=snap))
        elif isinstance(item, dict) and isinstance(item.get("snapshot"), MetricSnapshot):
            endpoints.append(
                FleetEndpoint(name=str(item.get("name") or "endpoint"), snapshot=item["snapshot"])
            )
    return endpoints


def _worst(endpoints: list[FleetEndpoint], field: str) -> tuple[str | None, float | None]:
    best_name, best_value = None, None
    for ep in endpoints:
        value = getattr(ep.snapshot, field)
        if value is not None and (best_value is None or value > best_value):
            best_name, best_value = ep.name, value
    return best_name, best_value


def _mean(endpoints: list[FleetEndpoint], field: str) -> float | None:
    values = [
        getattr(ep.snapshot, field)
        for ep in endpoints
        if getattr(ep.snapshot, field) is not None
    ]
    return mean(values) if values else None


def _total(endpoints: list[FleetEndpoint], field: str) -> tuple[float | None, int]:
    values = [
        getattr(ep.snapshot, field)
        for ep in endpoints
        if getattr(ep.snapshot, field) is not None
    ]
    return (sum(values) if values else None), len(values)


def summarize_fleet(source: Any) -> FleetSummary:
    """Summarize named snapshots into a :class:`FleetSummary`."""
    if isinstance(source, FleetSummary):
        return source
    endpoints = coerce_fleet(source)
    status_counts: dict[str, int] = {}
    for ep in endpoints:
        status_counts[ep.snapshot.status] = status_counts.get(ep.snapshot.status, 0) + 1
    kv_name, kv_value = _worst(endpoints, "kv_cache_usage_ratio")
    wait_name, wait_value = _worst(endpoints, "requests_waiting")
    err_name, err_value = _worst(endpoints, "error_rate")
    total_running, running_n = _total(endpoints, "requests_running")
    total_waiting, waiting_n = _total(endpoints, "requests_waiting")
    return FleetSummary(
        endpoint_count=len(endpoints),
        endpoints_reporting=sum(1 for ep in endpoints if ep.snapshot.status != "unreachable"),
        status_counts=status_counts,
        endpoint_names=[ep.name for ep in endpoints],
        worst_kv_cache_name=kv_name,
        worst_kv_cache_usage_ratio=kv_value,
        worst_requests_waiting_name=wait_name,
        worst_requests_waiting=wait_value,
        worst_error_rate_name=err_name,
        worst_error_rate=err_value,
        mean_kv_cache_usage_ratio=_mean(endpoints, "kv_cache_usage_ratio"),
        mean_error_rate=_mean(endpoints, "error_rate"),
        mean_requests_waiting=_mean(endpoints, "requests_waiting"),
        total_requests_running=total_running,
        total_requests_waiting=total_waiting,
        requests_running_reporting=running_n,
        requests_waiting_reporting=waiting_n,
        kv_cache_reporting=sum(
            1 for ep in endpoints if ep.snapshot.kv_cache_usage_ratio is not None
        ),
        saturated_endpoints=[
            ep.name
            for ep in endpoints
            if ep.snapshot.kv_cache_usage_ratio is not None
            and ep.snapshot.kv_cache_usage_ratio >= KV_SATURATED_RATIO
        ],
        idle_endpoints=[
            ep.name
            for ep in endpoints
            if ep.snapshot.status != "unreachable"
            and (
                (ep.snapshot.kv_cache_usage_ratio is not None
                 and ep.snapshot.kv_cache_usage_ratio <= KV_IDLE_RATIO)
                or ep.snapshot.requests_running == 0
            )
        ],
    )


def recommend_from_fleet(summary_or_snapshots: Any) -> list[Recommendation]:
    """Fleet-level recommendations, evidence-based with endpoint names."""
    summary = summarize_fleet(summary_or_snapshots)
    if summary.endpoint_count == 0:
        return [
            Recommendation(
                code="insufficient_evidence",
                title="No specific change recommended from the available measurements",
                evidence="Fleet contains no endpoints (0 samples).",
                assumptions=["Provide at least one named endpoint snapshot."],
                confidence="low",
            )
        ]
    if summary.endpoints_reporting == 0:
        return [
            Recommendation(
                code="endpoint_unreachable",
                title="Restore endpoint observability before optimizing",
                evidence=(
                    f"All {summary.endpoint_count} endpoints are unreachable: "
                    f"{', '.join(summary.endpoint_names) or 'none'}."
                ),
                assumptions=["No performance conclusion is drawn from failed collections."],
                confidence="high",
            )
        ]
    recs: list[Recommendation] = []
    saturated = set(summary.saturated_endpoints)
    idle_others = [name for name in summary.idle_endpoints if name not in saturated]
    if saturated and idle_others:
        recs.append(
            Recommendation(
                code="fleet_load_imbalance",
                title="Load is imbalanced across the fleet; review routing/placement",
                evidence=(
                    f"Saturated (KV >=95%): {', '.join(sorted(saturated))}; "
                    f"idle: {', '.join(idle_others)}; worst KV "
                    f"{summary.worst_kv_cache_usage_ratio:.0%} on "
                    f"{summary.worst_kv_cache_name}."
                ),
                assumptions=["This library does not route traffic; the host decides placement."],
                confidence="medium",
            )
        )
    elif saturated:
        recs.append(
            Recommendation(
                code="kv_cache_saturated",
                title="KV cache is saturated; review concurrency / max_num_seqs",
                evidence=(
                    f"Saturated endpoints: {', '.join(sorted(saturated))}; worst KV "
                    f"{summary.worst_kv_cache_usage_ratio:.0%} on {summary.worst_kv_cache_name} "
                    f"across {summary.endpoint_count} endpoints."
                ),
                confidence="medium",
            )
        )
    if summary.worst_error_rate is not None and summary.worst_error_rate >= ERROR_RATE_THRESHOLD:
        recs.append(
            Recommendation(
                code="elevated_error_rate",
                title="Elevated error rate; inspect endpoint errors before tuning performance",
                evidence=(
                    f"Worst error rate {summary.worst_error_rate:.1%} on "
                    f"{summary.worst_error_rate_name}; fleet mean "
                    f"{summary.mean_error_rate:.1%}."
                    if summary.mean_error_rate is not None
                    else f"Worst error rate {summary.worst_error_rate:.1%} on "
                    f"{summary.worst_error_rate_name}."
                ),
                confidence="medium",
            )
        )
    if (
        summary.worst_requests_waiting is not None
        and summary.worst_requests_waiting >= WAITING_QUEUE_THRESHOLD
        and not saturated
    ):
        recs.append(
            Recommendation(
                code="queue_backlog",
                title="Requests are queueing; review batching/concurrency or add capacity",
                evidence=(
                    f"Worst queue {summary.worst_requests_waiting:.0f} waiting on "
                    f"{summary.worst_requests_waiting_name}; total waiting "
                    f"{summary.total_requests_waiting} across "
                    f"{summary.requests_waiting_reporting} reporting endpoints."
                ),
                confidence="medium",
            )
        )
    if not recs:
        recs.append(
            Recommendation(
                code="insufficient_evidence",
                title="No specific change recommended from the available measurements",
                evidence=(
                    f"{summary.endpoints_reporting}/{summary.endpoint_count} endpoints "
                    "reporting; no documented fleet threshold crossed."
                ),
                assumptions=["Collect a longer representative window before deciding."],
                confidence="low",
            )
        )
    return recs
