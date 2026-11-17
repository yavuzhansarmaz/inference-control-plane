import httpx
import pytest

from inference_control_plane import EndpointConfig, SglangClient, snapshot_from_sglang_metrics

SAMPLE = """
# HELP sglang:num_running_reqs x
sglang:num_running_reqs 4.0
sglang:num_queue_reqs 6.0
sglang:token_usage 0.96
sglang:gen_throughput 123.5
sglang:prompt_tokens_total 1000.0
sglang:generation_tokens_total 800.0
sglang:request_success_total 95.0
sglang:request_failure_total 5.0
"""


def test_snapshot_from_sglang_metrics():
    s = snapshot_from_sglang_metrics(SAMPLE, source="http://x/metrics")
    assert s.provider == "sglang" and s.status == "ok"
    assert s.requests_running == 4.0 and s.requests_waiting == 6.0
    assert s.kv_cache_usage_ratio == 0.96 and s.error_rate == 0.05
    assert s.tokens_per_second == 123.5  # genuine gen_throughput rate gauge


def test_missing_and_underscore_aliases():
    s = snapshot_from_sglang_metrics("sglang_num_running_reqs 2.0\n", source="x")
    assert s.requests_running == 2.0
    assert s.tokens_per_second is None and s.kv_cache_usage_ratio is None


@pytest.mark.asyncio
async def test_collect_mocked_ok_unreachable_degraded_and_get_only():
    seen = []

    def ok(request):
        seen.append(request.method)
        return httpx.Response(200, text=SAMPLE)

    c = SglangClient(EndpointConfig(base_url="https://example.com"), transport=httpx.MockTransport(ok))
    snap = await c.collect()
    assert snap.status == "ok" and snap.provider == "sglang" and seen == ["GET"]

    def bad(request):
        raise httpx.ConnectTimeout("t", request=request)

    c2 = SglangClient(EndpointConfig(base_url="https://example.com"), transport=httpx.MockTransport(bad))
    assert (await c2.collect()).status == "unreachable"

    def junk(request):
        return httpx.Response(200, text="<html>not metrics</html>")

    c3 = SglangClient(EndpointConfig(base_url="https://example.com"), transport=httpx.MockTransport(junk))
    degraded = await c3.collect()
    assert degraded.status == "degraded" and degraded.requests_running is None
