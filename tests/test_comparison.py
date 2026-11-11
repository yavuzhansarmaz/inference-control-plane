from inference_control_plane import compare_snapshots, snapshot_from_pool_status
from inference_control_plane.models import MetricSnapshot


def _current():
    return MetricSnapshot(
        provider="vllm", status="ok", collected_at="now", source="x",
        requests_running=2.0, kv_cache_usage_ratio=0.5,
        gpu_utilization_ratio=0.6, ttft_p95_seconds=0.3, error_rate=0.01,
    )


def test_compare_with_managed_snapshot_is_measured():
    pool = {"replicas": [{
        "status": "ready", "engine_unfinished_requests": 3,
        "resources": {"kv_cache_utilization": 0.4, "gpu_sm_utilization": 0.7},
    }]}
    managed = snapshot_from_pool_status(pool)
    rows = {r.metric: r for r in compare_snapshots(
        _current(), managed_snapshot=managed, managed_kind="measured")}
    assert rows["requests_running"].managed == 3.0
    assert rows["requests_running"].kind == "measured"
    assert rows["kv_cache_usage_ratio"].managed == 0.4
    assert rows["gpu_utilization_ratio"].managed == 0.7


def test_compare_without_managed_snapshot_unchanged():
    rows = compare_snapshots(_current(), managed_ttft_p95_seconds=0.2)
    assert len(rows) == 3
    assert rows[0].kind == "estimated"
