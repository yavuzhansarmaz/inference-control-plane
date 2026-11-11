import httpx
import pytest

from inference_control_plane import EndpointConfig, VllmClient, snapshot_from_metrics
from inference_control_plane.recommendations import compare_snapshots, recommend_from_snapshot

SAMPLE = """
# HELP vllm:num_requests_running x
vllm:num_requests_running 3.0
vllm:num_requests_waiting 7.0
vllm:gpu_cache_usage_perc 0.97
vllm:prompt_tokens_total 1000.0
vllm:generation_tokens_total 500.0
vllm:request_success_total 90.0
vllm:request_failure_total 10.0
vllm:time_to_first_token_seconds_bucket{le="0.1"} 50.0
vllm:time_to_first_token_seconds_bucket{le="0.5"} 100.0
vllm:time_to_first_token_seconds_bucket{le="+Inf"} 100.0
"""

def test_parse_and_snapshot_missing_is_none():
    s = snapshot_from_metrics(SAMPLE, source="http://x/metrics")
    assert s.requests_running == 3.0 and s.error_rate == 0.1
    assert s.kv_cache_usage_ratio == 0.97 and s.ttft_p95_seconds is not None
    assert s.tokens_per_second is None  # never fabricated

def test_recommendations_evidence():
    codes = {r.code for r in recommend_from_snapshot(snapshot_from_metrics(SAMPLE, source="x"))}
    assert "kv_cache_saturated" in codes and "queue_backlog" in codes

@pytest.mark.asyncio
async def test_collect_mocked_ok_and_unreachable():
    def ok(request): return httpx.Response(200, text=SAMPLE)
    c = VllmClient(EndpointConfig(base_url="https://example.com"), transport=httpx.MockTransport(ok))
    assert (await c.collect()).status == "ok"
    def bad(request): raise httpx.ConnectTimeout("t", request=request)
    c2 = VllmClient(EndpointConfig(base_url="https://example.com"), transport=httpx.MockTransport(bad))
    assert (await c2.collect()).status == "unreachable"

def test_comparison_labels_estimated():
    rows = compare_snapshots(snapshot_from_metrics(SAMPLE, source="x"), managed_ttft_p95_seconds=0.2)
    assert rows[0].kind == "estimated"

def _cap_handler(*, health=200, models=200, metrics=200):
    def handler(request):
        path = request.url.path
        if path == "/health":
            return httpx.Response(health, text="ok" if health == 200 else "err")
        if path == "/v1/models":
            if models != 200:
                return httpx.Response(models, text="nope")
            return httpx.Response(200, json={"object": "list", "data": [{"id": "model-a"}, {"id": "model-b"}]})
        if path == "/metrics":
            return httpx.Response(metrics, text=SAMPLE if metrics == 200 else "err")
        return httpx.Response(404, text="nf")
    return handler


@pytest.mark.asyncio
async def test_capabilities_health_and_models_ok():
    c = VllmClient(EndpointConfig(base_url="https://example.com"), transport=httpx.MockTransport(_cap_handler()))
    caps = await c.capabilities()
    assert caps.health_ok is True and caps.models == ["model-a", "model-b"]
    assert caps.metrics_ok is True and caps.errors == [] and caps.checked_at


@pytest.mark.asyncio
async def test_capabilities_health_500_never_raises():
    c = VllmClient(EndpointConfig(base_url="https://example.com"), transport=httpx.MockTransport(_cap_handler(health=500)))
    caps = await c.capabilities()
    assert caps.health_ok is False and caps.models == ["model-a", "model-b"]
    assert any("health" in e and "500" in e for e in caps.errors)


@pytest.mark.asyncio
async def test_capabilities_models_404_empty_with_note():
    c = VllmClient(EndpointConfig(base_url="https://example.com"), transport=httpx.MockTransport(_cap_handler(models=404)))
    caps = await c.capabilities()
    assert caps.health_ok is True and caps.models == []
    assert any("models" in e and "404" in e for e in caps.errors)


@pytest.mark.asyncio
async def test_capabilities_unreachable_never_raises():
    def bad(request): raise httpx.ConnectTimeout("t", request=request)
    c = VllmClient(EndpointConfig(base_url="https://example.com"), transport=httpx.MockTransport(bad))
    caps = await c.capabilities()
    assert caps.health_ok is False and caps.models == [] and caps.metrics_ok is False
    assert len(caps.errors) == 3


@pytest.mark.asyncio
async def test_capabilities_is_get_only():
    seen = []
    def handler(request):
        seen.append(request.method)
        return _cap_handler()(request)
    c = VllmClient(EndpointConfig(base_url="https://example.com"), transport=httpx.MockTransport(handler))
    await c.capabilities()
    assert seen and all(m == "GET" for m in seen)


def _snap(at, gen=None, prompt=None, succ=None, fail=None, gpu=None, tps=None):
    from inference_control_plane.models import MetricSnapshot
    return MetricSnapshot(provider="vllm", status="ok", collected_at=at, source="x",
                          generation_tokens_total=gen, prompt_tokens_total=prompt,
                          request_success_total=succ, request_failure_total=fail,
                          gpu_utilization_ratio=gpu, tokens_per_second=tps)


def test_derive_rates_basic():
    from inference_control_plane import derive_rates
    prev = _snap("2026-01-01T00:00:00+00:00", gen=1000.0, prompt=2000.0, succ=90.0, fail=10.0)
    curr = _snap("2026-01-01T00:00:10+00:00", gen=1500.0, prompt=2500.0, succ=140.0, fail=10.0)
    r = derive_rates(prev, curr)
    assert r.elapsed_seconds == 10.0 and r.tokens_per_second == 50.0
    assert r.generation_tokens_per_second == 50.0 and r.prompt_tokens_per_second == 50.0
    assert r.requests_per_second == 5.0


def test_derive_rates_rewound_counter_is_none():
    from inference_control_plane import derive_rates
    prev = _snap("2026-01-01T00:00:00+00:00", gen=1000.0)
    curr = _snap("2026-01-01T00:00:10+00:00", gen=10.0)
    assert derive_rates(prev, curr).tokens_per_second is None


def test_derive_rates_missing_and_zero_window():
    from inference_control_plane import derive_rates
    prev = _snap("2026-01-01T00:00:00+00:00", gen=None)
    curr = _snap("2026-01-01T00:00:10+00:00", gen=10.0)
    assert derive_rates(prev, curr).tokens_per_second is None
    same = _snap("2026-01-01T00:00:00+00:00", gen=10.0)
    assert derive_rates(same, same).tokens_per_second is None
    bad = _snap("not-a-date", gen=10.0)
    assert derive_rates(bad, curr).tokens_per_second is None


def test_estimate_effective_cost_labelled_assumption():
    from inference_control_plane import estimate_effective_cost
    snap = _snap("2026-01-01T00:00:00+00:00", gpu=0.5, tps=100.0)
    cost = estimate_effective_cost(snap, 2.0)
    assert cost is not None and cost.kind == "user_assumption"
    assert cost.hourly_cost_usd == 2.0 and cost.effective_cost_per_utilized_hour_usd == 4.0
    assert cost.cost_per_million_generation_tokens_usd is not None
    assert "user-supplied" in cost.notes and "Estimate" in cost.notes


def test_estimate_effective_cost_no_price_or_no_util_is_none_parts():
    from inference_control_plane import estimate_effective_cost
    snap = _snap("2026-01-01T00:00:00+00:00", gpu=None)
    assert estimate_effective_cost(snap, None) is None
    assert estimate_effective_cost(snap) is None
    cost = estimate_effective_cost(snap, 1.5)
    assert cost is not None and cost.effective_cost_per_utilized_hour_usd is None
    assert cost.cost_per_million_generation_tokens_usd is None


def test_estimate_effective_cost_derived_tps_override():
    from inference_control_plane import derive_rates, estimate_effective_cost
    prev = _snap("2026-01-01T00:00:00+00:00", gen=0.0, gpu=0.25)
    curr = _snap("2026-01-01T00:01:00+00:00", gen=6000.0, gpu=0.25)
    rates = derive_rates(prev, curr)
    cost = estimate_effective_cost(curr, 3.6, tokens_per_second=rates.tokens_per_second)
    assert cost is not None and cost.cost_per_million_generation_tokens_usd == 10.0


def test_pool_status_adapter_missing_stays_none():
    from inference_control_plane.pool_status import snapshot_from_pool_status
    pool = {"replicas": [{"status": "ready", "engine_unfinished_requests": 4, "resources": {"kv_cache_utilization": 0.4, "gpu_sm_utilization": None, "gpu_memory_utilization": 0.7, "preemptions": 0.0}}]}
    s = snapshot_from_pool_status(pool)
    assert s.provider == "managed" and s.requests_running == 4.0 and s.kv_cache_usage_ratio == 0.4
    assert s.gpu_utilization_ratio is None
    empty = snapshot_from_pool_status({"replicas": []})
    assert empty.status == "degraded" and empty.requests_running is None
