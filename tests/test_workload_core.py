from inference_control_plane.workload.analysis import analyze_events
from inference_control_plane.workload.event_utils import prepare_events
from inference_control_plane.workload.otlp import normalize_otlp_export


def test_analyze_empty():
    s, o, a = analyze_events([], window_days=7, managed_rate_usd_per_million=1.0, ttft_ms=100, output_tokens_per_second=10)
    assert s["total_requests"] == 0 and o == []

def test_prepare_strips_content():
    out = prepare_events("org", "conn", [{"external_id": "1", "prompt": "secret", "latency_ms": 5}])
    assert "prompt" not in out[0] and out[0]["org_id"] == "org"

def test_otlp_empty():
    assert normalize_otlp_export({}) == [] or isinstance(normalize_otlp_export({}), list)


def test_sample_events_deterministic_and_stripped():
    from datetime import datetime, timezone

    from inference_control_plane.workload.sample_data import build_sample_events

    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    events = build_sample_events("langfuse", now=now)
    assert events and events == build_sample_events("langfuse", now=now)
    assert all(e["is_sample"] and e["workload_key"] for e in events)


def test_otlp_normalizes_span_attributes():
    payload = {"resourceSpans": [{"resource": {}, "scopeSpans": [{"spans": [{
        "spanId": "s1", "traceId": "t1", "name": "gen",
        "startTimeUnixNano": "1000000000", "endTimeUnixNano": "2000000000",
        "attributes": [
            {"key": "gen_ai.response.model", "value": {"stringValue": "m1"}},
            {"key": "gen_ai.usage.input_tokens", "value": {"intValue": "10"}},
            {"key": "gen_ai.usage.output_tokens", "value": {"intValue": "5"}},
        ],
        "status": {"code": 1},
    }]}]}]}
    events = normalize_otlp_export(payload)
    assert len(events) == 1
    assert events[0]["model"] == "m1" and events[0]["total_tokens"] == 15
    assert events[0]["latency_ms"] == 1000.0 and events[0]["success"] is True


def test_analyze_events_aggregates_without_product_scoring():
    events = [{
        "external_id": str(i), "workload_key": "chat-m1", "workload_name": "chat",
        "model": "m1", "provider_name": "p", "input_tokens": 100, "output_tokens": 50,
        "total_tokens": 150, "latency_ms": 9000.0, "ttft_ms": 2000.0,
        "cost_usd": 1.0, "success": i % 10 != 0, "quality_score": 0.8,
        "observed_at": "2026-01-01T00:00:00+00:00",
    } for i in range(20)]
    summary, opportunities, _ = analyze_events(
        events, window_days=7, managed_rate_usd_per_million=1.0,
        ttft_ms=100, output_tokens_per_second=10,
    )
    assert summary["total_requests"] == 20 and opportunities
    item = opportunities[0]
    assert item["current"]["requests"] == 20 and item["current"]["quality_index"] == 80.0
    assert "candidate_score" not in item and "migration_candidate" not in item
    assert "migration_candidate_count" not in summary


def test_provider_normalize_helpers():
    from inference_control_plane.workload.helicone import HeliconeAdapter
    from inference_control_plane.workload.langfuse import LangfuseAdapter

    h = HeliconeAdapter("https://x", {})._normalize({
        "request_id": "r1", "model": "m1", "prompt_tokens": 3, "completion_tokens": 2,
        "request_created_at": "2026-01-01T00:00:00Z", "response_status": 200,
    })
    assert h["total_tokens"] == 5 and h["success"] is True
    lf = LangfuseAdapter("https://x", {})._normalize({
        "id": "g1", "type": "GENERATION", "providedModelName": "m1",
        "inputUsage": 3, "outputUsage": 2, "startTime": "2026-01-01T00:00:00Z",
    })
    assert lf["total_tokens"] == 5 and lf["provider_name"] == "langfuse"
