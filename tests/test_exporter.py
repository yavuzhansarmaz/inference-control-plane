import json
from pathlib import Path

from inference_control_plane import render_prometheus, summarize_fleet
from inference_control_plane.models import MetricSnapshot


def snap(status="ok", source="https://a.example/metrics", **kw):
    return MetricSnapshot(provider="vllm", status=status,
                          collected_at="2026-01-01T00:00:00+00:00", source=source, **kw)


def test_render_named_and_none_skipped():
    out = render_prometheus({"ep-a": snap(kv_cache_usage_ratio=0.99, requests_waiting=3.0)})
    assert 'inference_control_plane_kv_cache_usage_ratio{endpoint="ep-a"} 0.99' in out
    assert "error_rate" not in out  # None skipped, never emitted as 0
    assert "tokens_per_second" not in out
    assert 'inference_control_plane_snapshot_status{endpoint="ep-a",status="ok"} 1' in out


def test_render_single_uses_source_and_status():
    out = render_prometheus(snap(status="unreachable"))
    assert 'status="unreachable"' in out
    assert 'endpoint="https://a.example/metrics"' in out


def test_label_escaping():
    out = render_prometheus({'we"ird\\na\nme': snap(requests_running=1.0)})
    assert 'endpoint="we\\"ird\\\\na\\nme"' in out
    assert "\nme" not in out.split("requests_running")[1].split("\n")[0]


def test_custom_prefix_and_empty():
    assert render_prometheus({}, prefix="custom") == ""
    out = render_prometheus({"a": snap(requests_running=2.0)}, prefix="custom")
    assert "custom_requests_running" in out and "inference_control_plane" not in out


def test_fleet_summary_gauges():
    fleet = {"hot": snap(kv_cache_usage_ratio=0.99), "cold": snap(kv_cache_usage_ratio=0.1)}
    out = render_prometheus(summarize_fleet(fleet))
    assert "inference_control_plane_fleet_endpoints 2.0" in out
    assert 'fleet_worst_kv_cache_usage_ratio{endpoint="hot"}' in out
    # Raw-snapshot input also gains fleet gauges.
    assert "fleet_endpoints" in render_prometheus(fleet)


def test_grafana_dashboard_valid_json():
    path = Path(__file__).resolve().parents[1] / "docs" / "grafana-dashboard.json"
    data = json.loads(path.read_text())
    titles = {p["title"] for p in data["panels"]}
    assert {"KV cache usage ratio", "Queue waiting", "Error rate", "TTFT p95",
            "Endpoint status"} <= titles
    exprs = " ".join(t["expr"] for p in data["panels"] for t in p["targets"])
    for metric in ("kv_cache_usage_ratio", "requests_waiting", "error_rate",
                   "ttft_p95_seconds", "snapshot_status"):
        assert f"inference_control_plane_{metric}" in exprs
