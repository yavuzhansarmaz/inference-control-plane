"""Security and failure-mode tests: URL validation, VllmClient resilience, secret hygiene."""

import logging

import httpx
import pytest

from inference_control_plane import EndpointConfig, VllmClient, snapshot_from_metrics
from inference_control_plane.workload.adapter_base import (
    ProviderAdapterError,
    validate_provider_url,
)

SECRET = "sk-test-SECRET-do-not-leak-12345"

# validate_provider_url: private endpoints


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "url",
    [
        "https://10.0.0.5:8000",
        "https://192.168.1.20",
        "https://172.16.0.9/metrics",
        "https://169.254.169.254/latest/meta-data",  # link-local / cloud metadata
        "https://127.0.0.1:8000",
    ],
)
async def test_private_ip_rejected_by_default(url):
    with pytest.raises(ProviderAdapterError, match="Private observability endpoints"):
        await validate_provider_url(url)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "url",
    [
        "https://10.0.0.5:8000",
        "https://192.168.1.20",
        "https://127.0.0.1:8000",
    ],
)
async def test_private_ip_accepted_when_explicitly_allowed(url):
    assert await validate_provider_url(url, allow_private_endpoints=True) == url.rstrip("/")


@pytest.mark.asyncio
async def test_unresolvable_hostname_rejected():
    with pytest.raises(ProviderAdapterError, match="could not be resolved"):
        await validate_provider_url("https://nonexistent.invalid.example")


# validate_provider_url: credentials and scheme


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "url",
    [
        "https://user:pass@example.com",
        "https://user@example.com",
        f"https://:{SECRET}@example.com",
    ],
)
async def test_url_with_credentials_rejected(url):
    with pytest.raises(ProviderAdapterError, match="must not contain credentials"):
        await validate_provider_url(url)


@pytest.mark.asyncio
async def test_url_with_credentials_rejected_even_when_private_allowed():
    # Credential check runs before the allow_private_endpoints shortcut.
    with pytest.raises(ProviderAdapterError, match="must not contain credentials"):
        await validate_provider_url(
            "https://user:pass@10.0.0.5", allow_private_endpoints=True
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "url",
    [
        "http://example.com",
        "http://example.com:8000/metrics",
        "http://10.0.0.5:8000",
        "http://203.0.113.10",  # public IP but plain http
    ],
)
async def test_plain_http_non_localhost_rejected(url):
    with pytest.raises(ProviderAdapterError, match="must use HTTPS"):
        await validate_provider_url(url, allow_private_endpoints=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("url", ["http://localhost:8000", "http://127.0.0.1:8000"])
async def test_plain_http_localhost_allowed_only_with_private_opt_in(url):
    # Scheme check permits localhost, but the private-IP check still applies.
    with pytest.raises(ProviderAdapterError, match="Private observability endpoints"):
        await validate_provider_url(url)
    assert await validate_provider_url(url, allow_private_endpoints=True) == url


@pytest.mark.asyncio
@pytest.mark.parametrize("url", ["", "not-a-url", "ftp://example.com", "https://"])
async def test_malformed_url_rejected(url):
    with pytest.raises(ProviderAdapterError, match="absolute http"):
        await validate_provider_url(url)


# VllmClient: failure modes never raise, missing metrics stay None


def _client(handler, **kwargs) -> VllmClient:
    return VllmClient(
        EndpointConfig(base_url="https://example.com"),
        transport=httpx.MockTransport(handler),
        **kwargs,
    )


@pytest.mark.asyncio
async def test_collect_timeout_is_unreachable_not_raised():
    def handler(request):
        raise httpx.ReadTimeout("slow", request=request)

    snap = await _client(handler, timeout_seconds=0.1).collect()
    assert snap.status == "unreachable"
    assert snap.errors and "ReadTimeout" in snap.errors[0]
    assert snap.requests_running is None


@pytest.mark.asyncio
async def test_collect_connect_error_is_unreachable():
    def handler(request):
        raise httpx.ConnectError("refused", request=request)

    snap = await _client(handler).collect()
    assert snap.status == "unreachable"


@pytest.mark.asyncio
async def test_collect_http_500_is_unreachable():
    snap = await _client(lambda r: httpx.Response(500, text="boom")).collect()
    assert snap.status == "unreachable"


@pytest.mark.asyncio
async def test_collect_non_prometheus_body_is_degraded():
    snap = await _client(lambda r: httpx.Response(200, text="<html>hello</html>")).collect()
    assert snap.status == "degraded"
    assert "Prometheus" in snap.errors[0]
    assert snap.requests_running is None and snap.kv_cache_usage_ratio is None


@pytest.mark.asyncio
async def test_collect_partial_metrics_missing_stay_none():
    body = "# HELP vllm:num_requests_running x\nvllm:num_requests_running 1.0\n"
    snap = await _client(lambda r: httpx.Response(200, text=body)).collect()
    assert snap.status == "ok"
    assert snap.requests_running == 1.0
    # Everything not present in the exposition stays None, never fabricated 0.
    assert snap.requests_waiting is None
    assert snap.kv_cache_usage_ratio is None
    assert snap.gpu_utilization_ratio is None
    assert snap.ttft_p95_seconds is None
    assert snap.e2e_latency_p95_seconds is None
    assert snap.error_rate is None
    assert snap.tokens_per_second is None


def test_snapshot_from_empty_metrics_all_none():
    snap = snapshot_from_metrics("# HELP nothing here\n", source="x")
    assert snap.status == "ok"
    assert snap.requests_running is None and snap.error_rate is None


@pytest.mark.asyncio
async def test_test_raises_only_on_unreachable():
    def handler(request):
        raise httpx.ConnectError("refused", request=request)

    with pytest.raises(ProviderAdapterError):
        await _client(handler).test()


# Secret hygiene: api_key never appears in repr / snapshot / logs


@pytest.mark.asyncio
async def test_api_key_not_in_repr_or_snapshot(caplog):
    seen_headers = {}

    def handler(request):
        seen_headers.update(request.headers)
        return httpx.Response(200, text="vllm:num_requests_running 1.0\n")

    client = VllmClient(
        EndpointConfig(base_url="https://example.com"),
        api_key=SECRET,
        transport=httpx.MockTransport(handler),
    )
    assert SECRET not in repr(client)
    assert SECRET not in str(client)
    with caplog.at_level(logging.DEBUG):
        snap = await client.collect()
    # The key IS sent to the endpoint (that's its job) ...
    assert seen_headers.get("authorization") == f"Bearer {SECRET}"
    # ... but never leaks into the snapshot, its dump, or log output.
    assert SECRET not in str(snap)
    assert SECRET not in str(snap.model_dump())
    assert SECRET not in caplog.text


# VllmClient: SSRF validation is enforced on the fetch path, not opt-in


@pytest.mark.asyncio
async def test_collect_rejects_private_base_without_fetching():
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(200, text="vllm:num_requests_running 1.0\n")

    client = VllmClient(
        EndpointConfig(base_url="https://169.254.169.254"),
        transport=httpx.MockTransport(handler),
    )
    snap = await client.collect()
    assert snap.status == "unreachable"
    assert any("Private" in e for e in snap.errors)
    assert calls == []  # fail closed: no request was ever sent


@pytest.mark.asyncio
async def test_collect_rejects_private_metrics_url_without_fetching():
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(200, text="vllm:num_requests_running 1.0\n")

    client = VllmClient(
        EndpointConfig(
            base_url="https://example.com",
            metrics_url="https://10.0.0.5:8000/metrics",
        ),
        transport=httpx.MockTransport(handler),
    )
    snap = await client.collect()
    assert snap.status == "unreachable"
    assert any("Private" in e for e in snap.errors)
    assert calls == []


@pytest.mark.asyncio
async def test_capabilities_rejects_private_metrics_url_without_fetching():
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(200, text="ok")

    client = VllmClient(
        EndpointConfig(
            base_url="https://example.com",
            metrics_url="https://169.254.169.254/latest/meta-data",
        ),
        transport=httpx.MockTransport(handler),
    )
    caps = await client.capabilities()
    assert caps.health_ok is False and caps.metrics_ok is False and caps.models == []
    assert any("Private" in e for e in caps.errors)
    assert calls == []


@pytest.mark.asyncio
async def test_validate_also_validates_metrics_url():
    client = VllmClient(
        EndpointConfig(
            base_url="https://example.com",
            metrics_url="https://192.168.1.20/metrics",
        ),
        transport=httpx.MockTransport(lambda r: httpx.Response(200)),
    )
    with pytest.raises(ProviderAdapterError, match="Private"):
        await client.validate()


@pytest.mark.asyncio
async def test_collect_allows_private_endpoint_with_explicit_opt_in():
    client = VllmClient(
        EndpointConfig(base_url="http://127.0.0.1:8000"),
        allow_private_endpoints=True,
        transport=httpx.MockTransport(
            lambda r: httpx.Response(200, text="vllm:num_requests_running 1.0\n")
        ),
    )
    snap = await client.collect()
    assert snap.status == "ok" and snap.requests_running == 1.0


# adapter_for: allow_private_endpoints reaches the provider adapters


@pytest.mark.asyncio
async def test_adapter_for_private_helicone_allowed_only_with_opt_in(monkeypatch):
    from inference_control_plane.workload.adapters import adapter_for

    def fake_client():
        return httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda r: httpx.Response(200, json={"data": []})
            )
        )

    monkeypatch.setattr(
        "inference_control_plane.workload.helicone.HeliconeAdapter.client",
        staticmethod(fake_client),
    )
    creds = {"api_key": "k"}
    blocked = adapter_for("helicone", "https://10.0.0.5:8080", creds)
    with pytest.raises(ProviderAdapterError, match="Private"):
        await blocked.pull("2026-01-01T00:00:00Z", "2026-01-02T00:00:00Z", limit=1)

    allowed = adapter_for(
        "helicone", "https://10.0.0.5:8080", creds, allow_private_endpoints=True
    )
    assert await allowed.pull(
        "2026-01-01T00:00:00Z", "2026-01-02T00:00:00Z", limit=1
    ) == []


@pytest.mark.asyncio
async def test_api_key_not_in_error_snapshot():
    def handler(request):
        # Even if the transport error echoes the credential, collect() must
        # sanitize it to the exception type name only.
        raise httpx.ConnectError(f"failed with {SECRET}", request=request)

    client = VllmClient(
        EndpointConfig(base_url="https://example.com"),
        api_key=SECRET,
        transport=httpx.MockTransport(handler),
    )
    snap = await client.collect()
    assert snap.status == "unreachable"
    assert SECRET not in str(snap.model_dump())
    assert SECRET not in repr(client)
