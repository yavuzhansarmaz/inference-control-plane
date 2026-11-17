"""Current-vs-managed snapshot comparison rows."""

from __future__ import annotations

from inference_control_plane.models import ComparisonRow, MetricSnapshot


def compare_snapshots(
    current: MetricSnapshot,
    *,
    managed_ttft_p95_seconds: float | None = None,
    managed_e2e_p95_seconds: float | None = None,
    managed_kind: str = "estimated",
    managed_snapshot: MetricSnapshot | None = None,
) -> list[ComparisonRow]:
    """Compare snapshots; managed side is estimated unless a measured snapshot is given."""
    if managed_snapshot is not None:
        if managed_ttft_p95_seconds is None:
            managed_ttft_p95_seconds = managed_snapshot.ttft_p95_seconds
        if managed_e2e_p95_seconds is None:
            managed_e2e_p95_seconds = managed_snapshot.e2e_latency_p95_seconds
    rows = [
        ComparisonRow(
            metric="ttft_p95_seconds", current=current.ttft_p95_seconds,
            managed=managed_ttft_p95_seconds, unit="s", kind=managed_kind,
            notes="Current side is measured from vLLM histograms when present.",
        ),
        ComparisonRow(
            metric="e2e_latency_p95_seconds", current=current.e2e_latency_p95_seconds,
            managed=managed_e2e_p95_seconds, unit="s", kind=managed_kind,
            notes="Do not claim improvement unless both sides are measured on the same workload.",
        ),
        ComparisonRow(
            metric="error_rate", current=current.error_rate, managed=None,
            unit="ratio", kind="measured" if current.error_rate is not None else "estimated",
        ),
    ]
    if managed_snapshot is not None:
        shared_kind = "measured" if managed_snapshot.status == "ok" else managed_kind
        rows.extend([
            ComparisonRow(
                metric="requests_running", current=current.requests_running,
                managed=managed_snapshot.requests_running, unit="requests", kind=shared_kind,
                notes="Managed side counts unfinished requests on ready serving-pool replicas.",
            ),
            ComparisonRow(
                metric="kv_cache_usage_ratio", current=current.kv_cache_usage_ratio,
                managed=managed_snapshot.kv_cache_usage_ratio, unit="ratio", kind=shared_kind,
                notes="Managed side is the peak KV cache utilization across ready replicas.",
            ),
            ComparisonRow(
                metric="gpu_utilization_ratio", current=current.gpu_utilization_ratio,
                managed=managed_snapshot.gpu_utilization_ratio, unit="ratio", kind=shared_kind,
                notes="Managed side is the peak GPU SM utilization across ready replicas.",
            ),
        ])
    return rows
