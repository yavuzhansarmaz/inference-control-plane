"""Read-only SGLang monitoring via Prometheus /metrics (GET only).

Mirrors :mod:`inference_control_plane.vllm`: never raises for endpoint
failures, missing metrics stay ``None``, SSRF validation is shared.

``tokens_per_second`` is set only from ``sglang:gen_throughput``: that metric
is a genuine instantaneous rate gauge (generated tokens per second reported
by the engine), not a cumulative counter, so reading it directly does not
fabricate a rate. When it is absent the field stays ``None``.
"""

from __future__ import annotations

import httpx

from inference_control_plane.metrics_client import _MetricsClientBase
from inference_control_plane.models import (
    EndpointConfig,
    MetricSnapshot,
    as_ratio,
)
from inference_control_plane.prometheus import error_rate_from_totals, parse_prometheus
from inference_control_plane.timeutil import now_iso
from inference_control_plane.workload.adapter_base import (
    _first,
)


def snapshot_from_sglang_metrics(text: str, *, source: str) -> MetricSnapshot:
    """Map SGLang Prometheus exposition to a MetricSnapshot (aliases tolerated)."""
    metrics = parse_prometheus(text)
    success = _first(
        metrics,
        "sglang:request_success_total",
        "sglang_request_success_total",
        "sglang:num_success_reqs",
    )
    failure = _first(
        metrics,
        "sglang:request_failure_total",
        "sglang_request_failure_total",
        "sglang:request_error_total",
        "sglang:num_failed_reqs",
    )
    error_rate = error_rate_from_totals(success, failure)
    kv = as_ratio(
        _first(
            metrics,
            "sglang:token_usage",
            "sglang_token_usage",
            "sglang:kv_cache_usage_perc",
            "sglang_kv_cache_usage_perc",
        )
    )
    return MetricSnapshot(
        provider="sglang",
        status="ok",
        collected_at=now_iso(),
        source=source,
        requests_running=_first(
            metrics, "sglang:num_running_reqs", "sglang_num_running_reqs"
        ),
        requests_waiting=_first(
            metrics,
            "sglang:num_queue_reqs",
            "sglang_num_queue_reqs",
            "sglang:num_waiting_reqs",
            "sglang_num_waiting_reqs",
        ),
        kv_cache_usage_ratio=kv,
        gpu_utilization_ratio=_first(metrics, "DCGM_FI_DEV_GPU_UTIL"),
        prompt_tokens_total=_first(
            metrics, "sglang:prompt_tokens_total", "sglang_prompt_tokens_total"
        ),
        generation_tokens_total=_first(
            metrics, "sglang:generation_tokens_total", "sglang_generation_tokens_total"
        ),
        request_success_total=success,
        request_failure_total=failure,
        error_rate=error_rate,
        ttft_p95_seconds=_first(
            metrics,
            "sglang:time_to_first_token_seconds_bucket::p95",
            "sglang_time_to_first_token_seconds_bucket::p95",
        ),
        e2e_latency_p95_seconds=_first(
            metrics,
            "sglang:e2e_request_latency_seconds_bucket::p95",
            "sglang_e2e_request_latency_seconds_bucket::p95",
        ),
        # gen_throughput is an instantaneous tokens/s rate gauge (see module docstring).
        tokens_per_second=_first(
            metrics, "sglang:gen_throughput", "sglang_gen_throughput"
        ),
    )


class SglangClient(_MetricsClientBase):
    """Bounded, read-only SGLang collector. Never raises for endpoint failures."""

    provider = "sglang"
    metrics_marker = "sglang"
    _build_snapshot = staticmethod(snapshot_from_sglang_metrics)

    def __init__(
        self,
        config: EndpointConfig,
        *,
        api_key: str | None = None,
        allow_private_endpoints: bool = False,
        timeout_seconds: float = 5.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        if config.provider == "vllm":
            config = config.model_copy(update={"provider": "sglang"})
        super().__init__(
            config,
            api_key=api_key,
            allow_private_endpoints=allow_private_endpoints,
            timeout_seconds=timeout_seconds,
            transport=transport,
        )
