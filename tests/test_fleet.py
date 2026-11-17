from inference_control_plane import recommend_from_fleet, summarize_fleet
from inference_control_plane.fleet import FleetEndpoint
from inference_control_plane.models import MetricSnapshot


def snap(status="ok", **kw):
    return MetricSnapshot(provider="vllm", status=status, collected_at="2026-01-01T00:00:00+00:00",
                          source="x", **kw)


def test_empty_fleet():
    s = summarize_fleet({})
    assert s.endpoint_count == 0 and s.worst_kv_cache_name is None
    assert s.total_requests_running is None and s.mean_kv_cache_usage_ratio is None
    assert recommend_from_fleet({})[0].code == "insufficient_evidence"
    assert recommend_from_fleet([])[0].code == "insufficient_evidence"


def test_all_unreachable():
    fleet = {"a": snap("unreachable"), "b": snap("unreachable")}
    s = summarize_fleet(fleet)
    assert s.endpoints_reporting == 0 and s.status_counts == {"unreachable": 2}
    recs = recommend_from_fleet(fleet)
    assert recs[0].code == "endpoint_unreachable" and "a" in recs[0].evidence


def test_mixed_worst_and_sums_coverage():
    fleet = {
        "hot": snap(kv_cache_usage_ratio=0.99, requests_waiting=12.0, requests_running=4.0,
                    error_rate=0.1),
        "cold": snap(kv_cache_usage_ratio=0.1, requests_waiting=0.0, requests_running=0.0,
                     error_rate=0.0),
        "silent": snap(),  # reports nothing
    }
    s = summarize_fleet(fleet)
    assert s.worst_kv_cache_name == "hot" and s.worst_requests_waiting_name == "hot"
    assert s.worst_error_rate_name == "hot"
    assert s.mean_kv_cache_usage_ratio == (0.99 + 0.1) / 2  # mean, never sum of ratios
    assert s.total_requests_waiting == 12.0 and s.requests_waiting_reporting == 2
    assert s.saturated_endpoints == ["hot"] and "cold" in s.idle_endpoints
    codes = {r.code for r in recommend_from_fleet(fleet)}
    assert "fleet_load_imbalance" in codes and "elevated_error_rate" in codes
    # Summary input gives same recommendations.
    assert {r.code for r in recommend_from_fleet(s)} == codes


def test_none_fields_and_pairs_input():
    fleet = [FleetEndpoint(name="a", snapshot=snap()), ("b", snap(requests_waiting=9.0))]
    s = summarize_fleet(fleet)
    assert s.endpoint_count == 2 and s.worst_kv_cache_name is None
    assert s.mean_error_rate is None and s.total_requests_running is None
    recs = recommend_from_fleet(fleet)
    assert any(r.code == "queue_backlog" and "b" in r.evidence for r in recs)


def test_healthy_fleet_insufficient():
    fleet = {"a": snap(kv_cache_usage_ratio=0.5, requests_waiting=1.0, error_rate=0.0)}
    assert recommend_from_fleet(fleet)[0].code == "insufficient_evidence"
