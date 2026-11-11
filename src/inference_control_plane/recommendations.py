"""Evidence-based optimization recommendations (review-only, no auto-apply)."""

from __future__ import annotations

from inference_control_plane.models import ComparisonRow, MetricSnapshot, Recommendation


def recommend_from_snapshot(snapshot: MetricSnapshot) -> list[Recommendation]:
    recs: list[Recommendation] = []
    if snapshot.status == "unreachable":
        return [
            Recommendation(
                code="endpoint_unreachable",
                title="Restore endpoint observability before optimizing",
                evidence=f"Collection failed: {'; '.join(snapshot.errors) or 'unreachable'}",
                assumptions=["No performance conclusion is drawn from a failed collection."],
                confidence="high",
            )
        ]
    if snapshot.kv_cache_usage_ratio is not None:
        kv = snapshot.kv_cache_usage_ratio
        if kv >= 0.95:
            recs.append(Recommendation(
                code="kv_cache_saturated",
                title="KV cache is saturated; review concurrency / max_num_seqs",
                evidence=f"KV cache usage is {kv:.0%} of capacity.",
                assumptions=["Sustained saturation, not a single spike, should be confirmed."],
                confidence="medium",
            ))
        elif kv <= 0.25 and (snapshot.requests_running or 0) <= 1:
            recs.append(Recommendation(
                code="gpu_underutilized",
                title="Capacity appears underutilized; consider a smaller GPU or consolidation",
                evidence=f"KV cache usage is {kv:.0%} with {snapshot.requests_running} running request(s).",
                assumptions=["Based on one snapshot; confirm over a representative window."],
                confidence="low",
            ))
    if snapshot.requests_waiting is not None and snapshot.requests_waiting >= 5:
        recs.append(Recommendation(
            code="queue_backlog",
            title="Requests are queueing; review batching/concurrency or add capacity",
            evidence=f"{snapshot.requests_waiting:.0f} requests are waiting.",
            confidence="medium",
        ))
    if snapshot.error_rate is not None and snapshot.error_rate >= 0.05:
        recs.append(Recommendation(
            code="elevated_error_rate",
            title="Elevated error rate; inspect endpoint errors before tuning performance",
            evidence=f"Error rate is {snapshot.error_rate:.1%} of completed requests.",
            confidence="medium",
        ))
    if not recs:
        recs.append(Recommendation(
            code="insufficient_evidence",
            title="No specific change recommended from the available measurements",
            evidence="Available metrics do not cross any documented threshold.",
            assumptions=["Collect a longer representative window before deciding."],
            confidence="low",
        ))
    return recs


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
