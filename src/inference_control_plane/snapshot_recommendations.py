"""Snapshot-level optimization recommendations (review-only, no auto-apply)."""

from __future__ import annotations

from inference_control_plane.models import MetricSnapshot, Recommendation
from inference_control_plane.thresholds import (
    ERROR_RATE_THRESHOLD,
    KV_IDLE_RATIO,
    KV_SATURATED_RATIO,
    WAITING_QUEUE_THRESHOLD,
)


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
        if kv >= KV_SATURATED_RATIO:
            recs.append(Recommendation(
                code="kv_cache_saturated",
                title="KV cache is saturated; review concurrency / max_num_seqs",
                evidence=f"KV cache usage is {kv:.0%} of capacity.",
                assumptions=["Sustained saturation, not a single spike, should be confirmed."],
                confidence="medium",
            ))
        elif kv <= KV_IDLE_RATIO and (snapshot.requests_running or 0) <= 1:
            recs.append(Recommendation(
                code="gpu_underutilized",
                title="Capacity appears underutilized; consider a smaller GPU or consolidation",
                evidence=f"KV cache usage is {kv:.0%} with {snapshot.requests_running} running request(s).",
                assumptions=["Based on one snapshot; confirm over a representative window."],
                confidence="low",
            ))
    if snapshot.requests_waiting is not None and snapshot.requests_waiting >= WAITING_QUEUE_THRESHOLD:
        recs.append(Recommendation(
            code="queue_backlog",
            title="Requests are queueing; review batching/concurrency or add capacity",
            evidence=f"{snapshot.requests_waiting:.0f} requests are waiting.",
            confidence="medium",
        ))
    if snapshot.error_rate is not None and snapshot.error_rate >= ERROR_RATE_THRESHOLD:
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
