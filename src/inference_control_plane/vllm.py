"""Read-only vLLM monitoring via Prometheus /metrics and /health (GET only)."""

from __future__ import annotations

import httpx

from inference_control_plane.metrics_client import _MetricsClientBase
from inference_control_plane.models import (
    EndpointCapabilities,
    MetricSnapshot,
    as_ratio,
)
from inference_control_plane.prometheus import error_rate_from_totals, parse_prometheus
from inference_control_plane.rates import derive_rates, estimate_effective_cost
from inference_control_plane.timeutil import now_iso
from inference_control_plane.workload.adapter_base import (
    ProviderAdapterError,
    _first,
)

__all__ = [
    "VllmClient",
    "derive_rates",
    "estimate_effective_cost",
    "parse_prometheus",
    "snapshot_from_metrics",
]


def snapshot_from_metrics(text: str, *, source: str) -> MetricSnapshot:
    metrics = parse_prometheus(text)
    success = _first(
        metrics, "vllm:request_success_total", "vllm_request_success_total"
    )
    failure = _first(
        metrics,
        "vllm:request_failure_total",
        "vllm_request_failure_total",
        "vllm:request_error_total",
    )
    error_rate = error_rate_from_totals(success, failure)
    kv = as_ratio(_first(
        metrics, "vllm:gpu_cache_usage_perc", "vllm:kv_cache_usage_perc",
        "vllm_gpu_cache_usage_perc", "vllm_kv_cache_usage_perc",
    ))
    return MetricSnapshot(
        provider="vllm",
        status="ok",
        collected_at=now_iso(),
        source=source,
        requests_running=_first(
            metrics, "vllm:num_requests_running", "vllm_num_requests_running"
        ),
        requests_waiting=_first(
            metrics, "vllm:num_requests_waiting", "vllm_num_requests_waiting"
        ),
        kv_cache_usage_ratio=kv,
        gpu_utilization_ratio=_first(
            metrics, "DCGM_FI_DEV_GPU_UTIL", "vllm:gpu_utilization"
        ),
        prompt_tokens_total=_first(
            metrics, "vllm:prompt_tokens_total", "vllm_prompt_tokens_total"
        ),
        generation_tokens_total=_first(
            metrics, "vllm:generation_tokens_total", "vllm_generation_tokens_total"
        ),
        request_success_total=success,
        request_failure_total=failure,
        error_rate=error_rate,
        ttft_p95_seconds=_first(
            metrics,
            "vllm:time_to_first_token_seconds_bucket::p95",
            "vllm_time_to_first_token_seconds_bucket::p95",
        ),
        e2e_latency_p95_seconds=_first(
            metrics,
            "vllm:e2e_request_latency_seconds_bucket::p95",
            "vllm_e2e_request_latency_seconds_bucket::p95",
        ),
    )


class VllmClient(_MetricsClientBase):
    """Bounded, read-only vLLM collector. Never raises for endpoint failures."""

    provider = "vllm"
    metrics_marker = "vllm"
    _build_snapshot = staticmethod(snapshot_from_metrics)

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
        headers = self._headers()
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

