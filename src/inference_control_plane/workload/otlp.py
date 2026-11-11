"""OpenTelemetry OTLP/HTTP JSON normalization."""

from datetime import datetime, timezone
from typing import Any

from inference_control_plane.workload.adapter_base import (
    MAX_PROVIDER_EVENTS,
    ProviderAdapterError,
    _first,
    _integer,
    _number,
    _quality_score,
    _safe_name,
    _workload_key,
)


def _otel_value(value: Any) -> Any:
    if not isinstance(value, dict):
        return value
    for key in ("stringValue", "intValue", "doubleValue", "boolValue"):
        if key in value:
            return value[key]
    if "arrayValue" in value:
        values = (value.get("arrayValue") or {}).get("values") or []
        return [_otel_value(item) for item in values]
    if "kvlistValue" in value:
        return _otel_attributes((value.get("kvlistValue") or {}).get("values") or [])
    return value


def _otel_attributes(items: Any) -> dict[str, Any]:
    if not isinstance(items, list):
        return {}
    return {
        str(item.get("key")): _otel_value(item.get("value"))
        for item in items
        if isinstance(item, dict) and item.get("key")
    }


def normalize_otlp_export(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Normalize an OTLP/HTTP JSON ExportTraceServiceRequest."""
    events: list[dict[str, Any]] = []
    resource_spans = payload.get("resourceSpans") or payload.get("resource_spans") or []
    if not isinstance(resource_spans, list):
        raise ProviderAdapterError("OTLP payload must contain resourceSpans")
    for resource_group in resource_spans:
        if not isinstance(resource_group, dict):
            continue
        resource = resource_group.get("resource") or {}
        resource_attrs = _otel_attributes(resource.get("attributes") or [])
        scope_groups = resource_group.get("scopeSpans") or resource_group.get("scope_spans") or []
        for scope_group in scope_groups:
            spans = (scope_group or {}).get("spans") or []
            for span in spans:
                if not isinstance(span, dict):
                    continue
                attrs = {**resource_attrs, **_otel_attributes(span.get("attributes") or [])}
                model = _safe_name(
                    _first(
                        attrs,
                        "gen_ai.response.model",
                        "gen_ai.request.model",
                        "llm.model_name",
                        "openinference.llm.model_name",
                    ),
                    "unknown-model",
                )
                name = _safe_name(
                    _first(
                        attrs,
                        "gen_ai.prompt.name",
                        "langfuse.observation.name",
                        "llm.prompt_template",
                        default=span.get("name"),
                    ),
                    "unnamed-span",
                )
                input_tokens = _integer(
                    _first(
                        attrs,
                        "gen_ai.usage.input_tokens",
                        "llm.token_count.prompt",
                        "openinference.llm.token_count.prompt",
                    )
                )
                output_tokens = _integer(
                    _first(
                        attrs,
                        "gen_ai.usage.output_tokens",
                        "llm.token_count.completion",
                        "openinference.llm.token_count.completion",
                    )
                )
                start_ns = _integer(span.get("startTimeUnixNano"))
                end_ns = _integer(span.get("endTimeUnixNano"))
                latency_ms = (end_ns - start_ns) / 1_000_000 if end_ns > start_ns else None
                observed = ""
                if start_ns:
                    observed = datetime.fromtimestamp(
                        start_ns / 1_000_000_000,
                        tz=timezone.utc,
                    ).isoformat()
                cost = _number(
                    _first(attrs, "gen_ai.usage.cost", "llm.usage.total_cost", "cost_usd"),
                    -1,
                )
                status = span.get("status") if isinstance(span.get("status"), dict) else {}
                events.append(
                    {
                        "external_id": str(span.get("spanId") or f"otel-{len(events)}-{start_ns}"),
                        "trace_id": str(span.get("traceId") or ""),
                        "observed_at": observed,
                        "workload_key": _workload_key(name, model),
                        "workload_name": name,
                        "model": model,
                        "provider_name": _safe_name(
                            _first(attrs, "gen_ai.provider.name", "llm.system"),
                            "opentelemetry",
                        ),
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "total_tokens": input_tokens + output_tokens,
                        "latency_ms": latency_ms,
                        "ttft_ms": _number(_first(attrs, "gen_ai.server.time_to_first_token"), 0)
                        or None,
                        "cost_usd": cost if cost >= 0 else None,
                        "success": _integer(status.get("code"), 0) != 2,
                        "quality_score": _quality_score(
                            _first(attrs, "gen_ai.evaluation.score", "llm.quality_score")
                        ),
                    }
                )
                if len(events) >= MAX_PROVIDER_EVENTS:
                    return events
    return events
