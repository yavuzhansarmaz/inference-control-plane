"""Read-only vLLM monitoring via Prometheus /metrics and /health (GET only)."""

from __future__ import annotations

import re
from datetime import datetime

import httpx

from inference_control_plane.models import (
    DerivedRates,
    EffectiveCost,
    EndpointCapabilities,
    EndpointConfig,
    MetricSnapshot,
    as_ratio,
)
from inference_control_plane.timeutil import now_iso
from inference_control_plane.workload.adapter_base import (
    ProviderAdapterError,
    _first,
    validate_provider_url,
)

_LINE = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{[^}]*\})?\s+([-+eE0-9.]+|NaN|\+Inf|-Inf)$")


def parse_prometheus(text: str) -> dict[str, float]:
    """Sum samples per metric name; histogram buckets keep their ``_bucket`` name."""
    totals: dict[str, float] = {}
    buckets: dict[str, list[tuple[float, float]]] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            # Bucket bounds live in the label set, handled below.
            if line.startswith("#") or not line:
                continue
        match = _LINE.match(line)
        if not match:
            continue
        name, value = match.group(1), match.group(2)
        try:
            number = float(value)
        except ValueError:
            continue
        if number != number:  # NaN
            continue
        totals[name] = totals.get(name, 0.0) + number
        if name.endswith("_bucket"):
            bound_match = re.search(r'le="([^"]+)"', line)
            if bound_match:
                try:
                    bound = float(bound_match.group(1))
                except ValueError:
                    bound = float("inf")
                buckets.setdefault(name, []).append((bound, number))
    # Store bucket-derived p95 under a synthetic key for the collector.
    for name, pairs in buckets.items():
        p95 = _histogram_p95(pairs)
        if p95 is not None:
            totals[f"{name}::p95"] = p95
    return totals


def _histogram_p95(pairs: list[tuple[float, float]]) -> float | None:
    finite = sorted((b, c) for b, c in pairs if b != float("inf"))
    if not finite:
        return None
    total = max(c for _, c in pairs)
    if total <= 0:
        return None
    target = 0.95 * total
    previous_bound, previous_count = 0.0, 0.0
    for bound, count in finite:
        if count >= target:
            if count == previous_count:
                return bound
            fraction = (target - previous_count) / (count - previous_count)
            return previous_bound + fraction * (bound - previous_bound)
        previous_bound, previous_count = bound, count
    return finite[-1][0]


def _first_metric(metrics: dict[str, float], *names: str) -> float | None:
    return _first(metrics, *names)


def snapshot_from_metrics(text: str, *, source: str) -> MetricSnapshot:
    metrics = parse_prometheus(text)
    success = _first_metric(
        metrics, "vllm:request_success_total", "vllm_request_success_total"
    )
    failure = _first_metric(
        metrics,
        "vllm:request_failure_total",
        "vllm_request_failure_total",
        "vllm:request_error_total",
    )
    error_rate = None
    if success is not None or failure is not None:
        total = (success or 0.0) + (failure or 0.0)
        error_rate = ((failure or 0.0) / total) if total > 0 else None
    kv = as_ratio(_first_metric(
        metrics, "vllm:gpu_cache_usage_perc", "vllm:kv_cache_usage_perc",
        "vllm_gpu_cache_usage_perc", "vllm_kv_cache_usage_perc",
    ))
    return MetricSnapshot(
        provider="vllm",
        status="ok",
        collected_at=now_iso(),
        source=source,
        requests_running=_first_metric(
            metrics, "vllm:num_requests_running", "vllm_num_requests_running"
        ),
        requests_waiting=_first_metric(
            metrics, "vllm:num_requests_waiting", "vllm_num_requests_waiting"
        ),
        kv_cache_usage_ratio=kv,
        gpu_utilization_ratio=_first_metric(
            metrics, "DCGM_FI_DEV_GPU_UTIL", "vllm:gpu_utilization"
        ),
        prompt_tokens_total=_first_metric(
            metrics, "vllm:prompt_tokens_total", "vllm_prompt_tokens_total"
        ),
        generation_tokens_total=_first_metric(
            metrics, "vllm:generation_tokens_total", "vllm_generation_tokens_total"
        ),
        request_success_total=success,
        request_failure_total=failure,
        error_rate=error_rate,
        ttft_p95_seconds=_first_metric(
            metrics,
            "vllm:time_to_first_token_seconds_bucket::p95",
            "vllm_time_to_first_token_seconds_bucket::p95",
        ),
        e2e_latency_p95_seconds=_first_metric(
            metrics,
            "vllm:e2e_request_latency_seconds_bucket::p95",
            "vllm_e2e_request_latency_seconds_bucket::p95",
        ),
    )


def _parse_iso(value: str) -> datetime | None:
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return None


def _counter_rate(prev: float | None, curr: float | None, elapsed: float) -> float | None:
    if prev is None or curr is None:
        return None
    delta = curr - prev
    if delta < 0:  # counter reset / rewound (endpoint restarted)
        return None
    return delta / elapsed


def derive_rates(prev: MetricSnapshot, curr: MetricSnapshot) -> DerivedRates:
    """Derive per-second rates from two snapshots; unknown rates stay None."""
    prev_at, curr_at = _parse_iso(prev.collected_at), _parse_iso(curr.collected_at)
    if prev_at is None or curr_at is None:
        return DerivedRates()
    try:
        elapsed = (curr_at - prev_at).total_seconds()
    except TypeError:  # naive vs aware mix
        return DerivedRates()
    if elapsed <= 0:
        return DerivedRates(elapsed_seconds=elapsed if elapsed == 0 else None)
    generation = _counter_rate(prev.generation_tokens_total, curr.generation_tokens_total, elapsed)
    # Requests: prefer success+failure totals when both snapshots report them.
    def _req_total(s: MetricSnapshot) -> float | None:
        if s.request_success_total is None and s.request_failure_total is None:
            return None
        return (s.request_success_total or 0.0) + (s.request_failure_total or 0.0)

    return DerivedRates(
        elapsed_seconds=elapsed,
        tokens_per_second=generation,
        generation_tokens_per_second=generation,
        prompt_tokens_per_second=_counter_rate(
            prev.prompt_tokens_total, curr.prompt_tokens_total, elapsed
        ),
        requests_per_second=_counter_rate(_req_total(prev), _req_total(curr), elapsed),
    )


def estimate_effective_cost(
    snapshot: MetricSnapshot,
    gpu_hourly_usd: float | None = None,
    *,
    tokens_per_second: float | None = None,
) -> EffectiveCost | None:
    """Estimate cost from a user GPU price; None when no price is given."""
    if gpu_hourly_usd is None or gpu_hourly_usd < 0:
        return None
    utilization = as_ratio(snapshot.gpu_utilization_ratio)
    effective = (
        gpu_hourly_usd / utilization
        if utilization is not None and utilization > 0
        else None
    )
    tps = tokens_per_second if tokens_per_second is not None else snapshot.tokens_per_second
    cost_per_million = None
    if tps is not None and tps > 0:
        cost_per_million = gpu_hourly_usd / (tps * 3600.0) * 1_000_000.0
    notes = (
        "Estimate only: based on the user-supplied GPU hourly price "
        f"(${gpu_hourly_usd:.4g}/h), not a measured bill."
    )
    if utilization is None:
        notes += " GPU utilization was not reported, so no effective (utilization-adjusted) cost is given."
    elif effective is None:
        notes += " GPU utilization is zero, so no effective (utilization-adjusted) cost is given."
    return EffectiveCost(
        gpu_hourly_usd=gpu_hourly_usd,
        hourly_cost_usd=gpu_hourly_usd,
        utilization_ratio=utilization,
        effective_cost_per_utilized_hour_usd=effective,
        cost_per_million_generation_tokens_usd=cost_per_million,
        notes=notes,
    )


class VllmClient:
    """Bounded, read-only collector. Never raises for endpoint failures."""

    def __init__(
        self,
        config: EndpointConfig,
        *,
        api_key: str | None = None,
        allow_private_endpoints: bool = False,
        timeout_seconds: float = 5.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        self.config = config
        self._api_key = api_key
        self.allow_private_endpoints = allow_private_endpoints
        self.timeout_seconds = timeout_seconds
        self._transport = transport
        self._validated: tuple[str, str] | None = None

    def __repr__(self) -> str:  # never leak the credential
        return f"VllmClient(base_url={self.config.base_url!r})"

    async def _validated_urls(self) -> tuple[str, str]:
        """SSRF-check base and metrics URLs before any request is sent."""
        if self._validated is not None:
            return self._validated
        # Metrics URL first: a public base must not launder a private metrics URL.
        allow = self.allow_private_endpoints
        # An injected transport is a trusted test seam: hostnames are not
        # DNS-resolved then, but IP-literal/scheme/credential checks remain.
        resolve_dns = self._transport is None
        metrics_url = self.config.metrics_url
        if metrics_url:
            metrics_url = await validate_provider_url(
                metrics_url, allow_private_endpoints=allow, resolve_dns=resolve_dns
            )
        base_url = await validate_provider_url(
            self.config.base_url, allow_private_endpoints=allow, resolve_dns=resolve_dns
        )
        self._validated = (base_url, metrics_url or f"{base_url}/metrics")
        return self._validated

    async def validate(self) -> str:
        base_url, _ = await self._validated_urls()
        return base_url

    async def collect(self) -> MetricSnapshot:
        try:
            _, metrics_url = await self._validated_urls()
        except ProviderAdapterError as exc:
            return MetricSnapshot(
                provider="vllm",
                status="unreachable",
                collected_at=now_iso(),
                source=self.config.metrics_url or self.config.base_url,
                errors=[str(exc)],
            )
        headers = {"Authorization": f"Bearer {self._api_key}"} if self._api_key else {}
        try:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(self.timeout_seconds, connect=3.0),
                follow_redirects=False,
                transport=self._transport,
            ) as client:
                response = await client.get(metrics_url, headers=headers)
                response.raise_for_status()
                text = response.text
        except Exception as exc:  # unreachable / auth / timeout / malformed
            return MetricSnapshot(
                provider="vllm",
                status="unreachable",
                collected_at=now_iso(),
                source=metrics_url,
                errors=[f"{type(exc).__name__}: collection failed"][:1],
            )
        if "vllm" not in text and "# HELP" not in text:
            return MetricSnapshot(
                provider="vllm",
                status="degraded",
                collected_at=now_iso(),
                source=metrics_url,
                errors=["Metrics response did not look like Prometheus exposition"],
            )
        return snapshot_from_metrics(text, source=metrics_url)

    async def capabilities(self) -> EndpointCapabilities:
        """Probe /health, /v1/models and /metrics with GET only. Never raises."""
        try:
            base, metrics_url = await self._validated_urls()
        except ProviderAdapterError as exc:
            return EndpointCapabilities(
                health_ok=False,
                models=[],
                metrics_ok=False,
                errors=[str(exc)],
                checked_at=now_iso(),
            )
        headers = {"Authorization": f"Bearer {self._api_key}"} if self._api_key else {}
        errors: list[str] = []
        health_ok = False
        models: list[str] = []
        metrics_ok = False
        try:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(self.timeout_seconds, connect=3.0),
                follow_redirects=False,
                transport=self._transport,
            ) as client:
                try:
                    resp = await client.get(f"{base}/health", headers=headers)
                    health_ok = 200 <= resp.status_code < 300
                    if not health_ok:
                        errors.append(f"health: HTTP {resp.status_code}")
                except Exception as exc:
                    errors.append(f"health: {type(exc).__name__}: health check failed")
                try:
                    resp = await client.get(f"{base}/v1/models", headers=headers)
                    if 200 <= resp.status_code < 300:
                        try:
                            payload = resp.json()
                        except Exception:
                            payload = None
                            errors.append("models: response was not valid JSON")
                        if isinstance(payload, dict):
                            data = payload.get("data")
                            if isinstance(data, list):
                                for item in data:
                                    if isinstance(item, dict) and item.get("id"):
                                        models.append(str(item["id"]))
                                    elif isinstance(item, str) and item:
                                        models.append(item)
                            else:
                                errors.append("models: response had no model list")
                    else:
                        errors.append(f"models: HTTP {resp.status_code}")
                except Exception as exc:
                    errors.append(f"models: {type(exc).__name__}: model list failed")
                try:
                    resp = await client.get(metrics_url, headers=headers)
                    if 200 <= resp.status_code < 300:
                        text = resp.text
                        metrics_ok = "vllm" in text or "# HELP" in text
                        if not metrics_ok:
                            errors.append("metrics: response did not look like Prometheus exposition")
                    else:
                        errors.append(f"metrics: HTTP {resp.status_code}")
                except Exception as exc:
                    errors.append(f"metrics: {type(exc).__name__}: metrics check failed")
        except Exception as exc:  # client construction etc. — still never raise
            errors.append(f"capabilities: {type(exc).__name__}: probe failed")
        return EndpointCapabilities(
            health_ok=health_ok,
            models=models,
            metrics_ok=metrics_ok,
            errors=errors,
            checked_at=now_iso(),
        )

    async def test(self) -> None:
        snapshot = await self.collect()
        if snapshot.status == "unreachable":
            raise ProviderAdapterError(snapshot.errors[0] if snapshot.errors else "unreachable")
