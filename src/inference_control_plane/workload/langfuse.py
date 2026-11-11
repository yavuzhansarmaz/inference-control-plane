"""Langfuse read-only observations adapter."""

from typing import Any

import httpx

from inference_control_plane.workload.adapter_base import (
    PAGE_SIZE,
    ProviderAdapter,
    ProviderAdapterError,
    _first,
    _integer,
    _number,
    _quality_score,
    _safe_name,
    _workload_key,
    validate_provider_url,
)

LANGFUSE_OBSERVATIONS_PATH = "/api/public/v2/observations"
LANGFUSE_OBSERVATION_FIELDS = "core,basic,time,model,usage,metrics,trace_context"


class LangfuseAdapter(ProviderAdapter):
    async def pull(self, start_iso: str, end_iso: str, *, limit: int) -> list[dict[str, Any]]:
        public_key = str(self.credentials.get("public_key") or "")
        secret_key = str(self.credentials.get("secret_key") or "")
        if not public_key or not secret_key:
            raise ProviderAdapterError("Langfuse public and secret keys are required")
        base_url = await validate_provider_url(
            self.base_url, allow_private_endpoints=self.allow_private_endpoints
        )
        params: dict[str, Any] = {
            "fromStartTime": start_iso,
            "toStartTime": end_iso,
            "fields": LANGFUSE_OBSERVATION_FIELDS,
            "limit": min(PAGE_SIZE, limit),
        }
        events: list[dict[str, Any]] = []
        async with self.client() as client:
            while len(events) < limit:
                try:
                    response = await client.get(
                        f"{base_url}{LANGFUSE_OBSERVATIONS_PATH}",
                        params=params,
                        auth=httpx.BasicAuth(public_key, secret_key),
                        headers={"Accept": "application/json"},
                    )
                    response.raise_for_status()
                    body = response.json()
                except (httpx.HTTPError, ValueError) as exc:
                    raise ProviderAdapterError(
                        f"Langfuse connection failed: {type(exc).__name__}"
                    ) from exc
                rows = body.get("data") if isinstance(body, dict) else None
                if not isinstance(rows, list):
                    raise ProviderAdapterError("Langfuse returned an invalid observations payload")
                for row in rows:
                    if isinstance(row, dict) and str(row.get("type") or "").upper() in {
                        "GENERATION",
                        "COMPLETION",
                    }:
                        events.append(self._normalize(row))
                        if len(events) >= limit:
                            break
                cursor = (body.get("meta") or {}).get("cursor") if isinstance(body, dict) else None
                if not cursor or not rows:
                    break
                params["cursor"] = cursor
        return events

    def _normalize(self, row: dict[str, Any]) -> dict[str, Any]:
        usage = row.get("usageDetails") if isinstance(row.get("usageDetails"), dict) else {}
        cost_details = row.get("costDetails") if isinstance(row.get("costDetails"), dict) else {}
        input_tokens = _integer(_first(row, "inputUsage", default=usage.get("input", 0)))
        output_tokens = _integer(_first(row, "outputUsage", default=usage.get("output", 0)))
        total_tokens = _integer(
            _first(row, "totalUsage", default=usage.get("total", input_tokens + output_tokens))
        )
        model = _safe_name(
            _first(row, "providedModelName", "modelId", "internalModelId"),
            "unknown-model",
        )
        name = _safe_name(
            _first(row, "promptName", "traceName", "name"),
            "unnamed-generation",
        )
        latency_seconds = _number(row.get("latency"), 0)
        total_cost = _number(
            _first(row, "totalCost", default=cost_details.get("total")),
            -1,
        )
        return {
            "external_id": _safe_name(
                row.get("id"), f"langfuse-{len(name)}-{row.get('startTime')}"
            ),
            "trace_id": str(row.get("traceId") or ""),
            "observed_at": str(row.get("startTime") or row.get("createdAt") or ""),
            "workload_key": _workload_key(name, model),
            "workload_name": name,
            "model": model,
            "provider_name": "langfuse",
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens or input_tokens + output_tokens,
            "latency_ms": latency_seconds * 1000 if latency_seconds > 0 else None,
            "ttft_ms": _number(row.get("timeToFirstToken"), 0) * 1000 or None,
            "cost_usd": total_cost if total_cost >= 0 else None,
            "success": str(row.get("level") or "DEFAULT").upper() not in {"ERROR", "FATAL"},
            "quality_score": _quality_score(row.get("scores")),
        }
