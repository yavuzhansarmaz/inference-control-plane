"""Adapt a serving-pool status payload to a MetricSnapshot; never invent values."""

from __future__ import annotations

from typing import Any

from inference_control_plane.models import MetricSnapshot, as_ratio
from inference_control_plane.timeutil import now_iso


def _peak(records: list[dict[str, Any]], field: str) -> float | None:
    values = [
        float(record["resources"][field])
        for record in records
        if isinstance((record.get("resources") or {}).get(field), (int, float))
    ]
    return max(values) if values else None


def snapshot_from_pool_status(pool: dict[str, Any], *, source: str = "managed-serving-pool") -> MetricSnapshot:
    replicas = [
        record
        for record in pool.get("replicas") or []
        if isinstance(record, dict) and record.get("status") == "ready"
    ]
    unfinished = [
        float(record["engine_unfinished_requests"])
        for record in replicas
        if isinstance(record.get("engine_unfinished_requests"), (int, float))
    ]
    kv = as_ratio(_peak(replicas, "kv_cache_utilization"))
    return MetricSnapshot(
        provider="managed",
        status="ok" if replicas else "degraded",
        collected_at=now_iso(),
        source=source,
        errors=[] if replicas else ["No ready replicas reported by the serving pool"],
        requests_running=sum(unfinished) if unfinished else None,
        kv_cache_usage_ratio=kv,
        gpu_utilization_ratio=_peak(replicas, "gpu_sm_utilization"),
    )
