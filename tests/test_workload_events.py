from inference_control_plane.workload import WorkloadEvent
from inference_control_plane.workload.analysis import analyze_events

FIXTURE = [{
    "external_id": str(i), "trace_id": "t1", "workload_key": "chat-m1",
    "workload_name": "chat", "model": "m1", "provider_name": "p",
    "input_tokens": 100, "output_tokens": 50, "total_tokens": 150,
    "latency_ms": 900.0, "ttft_ms": 100.0, "cost_usd": 0.5,
    "success": i % 10 != 0, "quality_score": 0.8,
    "observed_at": "2026-01-01T00:00:00+00:00", "source_system": "fixture-a",
} for i in range(20)]

KW = dict(window_days=7, managed_rate_usd_per_million=1.0, ttft_ms=100,
          output_tokens_per_second=10)


def _strip_ids(opportunities):
    return [{k: v for k, v in item.items() if k != "id"} for item in opportunities]


def test_round_trip_preserves_analysis():
    events = [WorkloadEvent.from_dict(row) for row in FIXTURE]
    assert events[0].to_dict()["source_system"] == "fixture-a"  # extra kept
    assert events[0].to_dict()["external_id"] == "0"
    s1, o1, a1 = analyze_events(FIXTURE, **KW)
    s2, o2, a2 = analyze_events(events, **KW)
    s3, o3, a3 = analyze_events([e.to_dict() for e in events], **KW)
    assert s1 == s2 == s3
    assert _strip_ids(o1) == _strip_ids(o2) == _strip_ids(o3)
    assert a1 == a2 == a3


def test_defaults_and_mixed_input():
    event = WorkloadEvent.from_dict({"external_id": "x"})
    assert event.success is True and event.latency_ms is None and event.total_tokens == 0
    assert WorkloadEvent.from_dict(event) is event
    summary, _, _ = analyze_events([FIXTURE[0], WorkloadEvent.from_dict(FIXTURE[1])], **KW)
    assert summary["total_requests"] == 2

