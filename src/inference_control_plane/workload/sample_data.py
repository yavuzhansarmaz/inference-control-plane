"""Deterministic synthetic inference history for product evaluation."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from inference_control_plane.workload.adapter_base import _workload_key

SAMPLE_DATA_VERSION = "v1"

_PROVIDER_PROFILES = {
    "langfuse": {
        "provider_name": "example-provider-a",
        "models": {
            "support-answer": "example-chat-small",
            "invoice-extraction": "example-chat-large",
            "embedding-search": "example-embedding",
        },
        "cost_rate": 10.0,
        "latency_factor": 1.0,
        "quality_offset": 0.02,
    },
    "helicone": {
        "provider_name": "example-provider-b",
        "models": {
            "support-answer": "example-chat-small",
            "invoice-extraction": "example-chat-large",
            "embedding-search": "example-embedding",
        },
        "cost_rate": 12.0,
        "latency_factor": 1.08,
        "quality_offset": 0.04,
    },
    "opentelemetry": {
        "provider_name": "example-provider-c",
        "models": {
            "support-answer": "example-chat-small",
            "invoice-extraction": "example-chat-large",
            "embedding-search": "example-embedding",
        },
        "cost_rate": 11.0,
        "latency_factor": 1.04,
        "quality_offset": 0.0,
    },
}

_SCENARIOS = (
    {
        "name": "support-answer",
        "count": 320,
        "input_tokens": 950,
        "output_tokens": 180,
        "latency_ms": 7200,
        "ttft_ms": 1100,
        "quality": 0.82,
        "failure_every": 80,
    },
    {
        "name": "invoice-extraction",
        "count": 120,
        "input_tokens": 6500,
        "output_tokens": 240,
        "latency_ms": 9800,
        "ttft_ms": 1450,
        "quality": 0.76,
        "failure_every": 50,
    },
    {
        "name": "embedding-search",
        "count": 40,
        "input_tokens": 420,
        "output_tokens": 0,
        "latency_ms": 180,
        "ttft_ms": 0,
        "quality": 0.88,
        "failure_every": 100,
    },
)


def build_sample_events(
    provider: str,
    *,
    now: datetime | None = None,
) -> list[dict[str, Any]]:
    """Return stable provider-flavoured events spread across the last 28 days."""
    profile = _PROVIDER_PROFILES.get(provider)
    if profile is None:
        raise ValueError(f"Unsupported sample-data provider: {provider}")

    anchor = now or datetime.now(timezone.utc)
    events: list[dict[str, Any]] = []
    for scenario_index, scenario in enumerate(_SCENARIOS):
        name = str(scenario["name"])
        model = profile["models"][name]
        for index in range(int(scenario["count"])):
            input_tokens = int(scenario["input_tokens"]) + ((index * 37) % 121) - 60
            base_output_tokens = int(scenario["output_tokens"])
            output_tokens = (
                max(0, base_output_tokens + ((index * 17) % 31) - 15) if base_output_tokens else 0
            )
            total_tokens = input_tokens + output_tokens
            latency_variation = 0.88 + ((index * 13) % 25) / 100
            latency_ms = (
                float(scenario["latency_ms"]) * float(profile["latency_factor"]) * latency_variation
            )
            ttft_ms = (
                float(scenario["ttft_ms"])
                * float(profile["latency_factor"])
                * (0.9 + (index % 9) / 50)
            )
            success = (index + 1) % int(scenario["failure_every"]) != 0
            observed_at = anchor - timedelta(
                days=(index * 7 + scenario_index * 3) % 28,
                minutes=(index * 19 + scenario_index * 11) % 1440,
            )
            quality = min(
                0.98,
                max(
                    0.0,
                    float(scenario["quality"])
                    + float(profile["quality_offset"])
                    + ((index % 7) - 3) / 100,
                ),
            )
            event = {
                "external_id": (f"sample-{SAMPLE_DATA_VERSION}:{provider}:{name}:{index:04d}"),
                "workload_key": _workload_key(name, model),
                "workload_name": name,
                "model": model,
                "provider_name": profile["provider_name"],
                "observed_at": observed_at.isoformat(),
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
                "latency_ms": round(latency_ms, 2),
                "ttft_ms": round(ttft_ms, 2),
                "cost_usd": round(
                    total_tokens / 1_000_000 * float(profile["cost_rate"]),
                    6,
                ),
                "quality_score": round(quality, 4),
                "success": success,
                "status_code": 200 if success else 500,
                "is_sample": True,
                "sample_data_version": SAMPLE_DATA_VERSION,
            }
            events.append(event)
    return events
