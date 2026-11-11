"""End-to-end tests against a real (in-process) fake vLLM HTTP server."""

import json

import httpx
import pytest

from inference_control_plane import EndpointConfig, VllmClient, compare_snapshots, derive_rates
from inference_control_plane.pool_status import snapshot_from_pool_status

METRICS_V1 = """# HELP vllm:num_requests_running x
vllm:num_requests_running 2.0
vllm:num_requests_waiting 1.0
vllm:gpu_cache_usage_perc 0.42
vllm:prompt_tokens_total 1000.0
vllm:generation_tokens_total 500.0
vllm:request_success_total 90.0
vllm:request_failure_total 10.0
vllm:time_to_first_token_seconds_bucket{le="0.1"} 50.0
vllm:time_to_first_token_seconds_bucket{le="0.5"} 100.0
vllm:time_to_first_token_seconds_bucket{le="+Inf"} 100.0
"""

METRICS_V2 = METRICS_V1.replace(
    "vllm:generation_tokens_total 500.0", "vllm:generation_tokens_total 1100.0"
).replace("vllm:prompt_tokens_total 1000.0", "vllm:prompt_tokens_total 1600.0")


class FakeVllmServer:
    """Minimal ASGI app mimicking a vLLM server's read-only HTTP surface."""

    def __init__(self):
        self.metrics_text = METRICS_V1
        self.methods: list[str] = []
        self.paths: list[str] = []

    async def __call__(self, scope, receive, send):
        assert scope["type"] == "http"
        self.methods.append(scope["method"])
        self.paths.append(scope["path"])
        path = scope["path"]
        if path == "/health":
            status, body, ctype = 200, b"ok", "text/plain"
        elif path == "/v1/models":
            payload = {"object": "list", "data": [{"id": "example-model-a"}, {"id": "example-model-b"}]}
            status, body, ctype = 200, json.dumps(payload).encode(), "application/json"
        elif path == "/metrics":
            status, body, ctype = 200, self.metrics_text.encode(), "text/plain"
        else:
            status, body, ctype = 404, b"not found", "text/plain"
        await send({
            "type": "http.response.start",
            "status": status,
            "headers": [(b"content-type", ctype.encode())],
        })
        await send({"type": "http.response.body", "body": body})


def _client(server: FakeVllmServer) -> VllmClient:
    transport = httpx.ASGITransport(app=server)
    return VllmClient(
        EndpointConfig(base_url="https://vllm.example.com"),
        transport=transport,
    )


@pytest.mark.asyncio
async def test_fake_server_collect_capabilities_and_compare():
    server = FakeVllmServer()
    client = _client(server)

    caps = await client.capabilities()
    assert caps.health_ok is True
    assert caps.models == ["example-model-a", "example-model-b"]
    assert caps.metrics_ok is True
    assert caps.errors == []

    snap = await client.collect()
    assert snap.status == "ok"
    assert snap.requests_running == 2.0
    assert snap.requests_waiting == 1.0
    assert snap.kv_cache_usage_ratio == 0.42
    assert snap.error_rate == 0.1
    assert snap.ttft_p95_seconds is not None

    # Comparison flow: current (fake vLLM) vs a generic serving-pool payload.
    pool = {"replicas": [{
        "status": "ready", "engine_unfinished_requests": 1,
        "resources": {"kv_cache_utilization": 0.30, "gpu_sm_utilization": 0.55},
    }]}
    other = snapshot_from_pool_status(pool)
    rows = {r.metric: r for r in compare_snapshots(snap, managed_snapshot=other, managed_kind="measured")}
    assert rows["requests_running"].current == 2.0
    assert rows["requests_running"].managed == 1.0
    assert rows["kv_cache_usage_ratio"].kind == "measured"

    # The fake server only ever received GET requests.
    assert server.methods and all(m == "GET" for m in server.methods)
    assert "/health" in server.paths and "/v1/models" in server.paths and "/metrics" in server.paths


@pytest.mark.asyncio
async def test_fake_server_rates_between_two_collects():
    server = FakeVllmServer()
    client = _client(server)
    first = await client.collect()
    server.metrics_text = METRICS_V2
    second = await client.collect()
    # collected_at timestamps are real now_iso() values; rates need a positive
    # window, so only assert counters moved and a same-window rate is None-safe.
    assert second.generation_tokens_total == 1100.0
    assert first.generation_tokens_total == 500.0
    rates = derive_rates(first, second)
    assert rates.tokens_per_second is None or rates.tokens_per_second >= 0
