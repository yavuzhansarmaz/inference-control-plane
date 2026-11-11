"""Helicone read-only request-history adapter."""

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

HELICONE_REQUESTS_PATH = "/v1/request/query-clickhouse"


class HeliconeAdapter(ProviderAdapter):
    @staticmethod
    def _query_payload(
        start_iso: str,
        end_iso: str,
        *,
        offset: int,
        limit: int,
    ) -> dict[str, Any]:
        # Helicone's filter language accepts one comparison per leaf. Date ranges
        # are represented as two leaves joined by an explicit boolean operator.
        return {
            "filter": {
                "left": {"request_response_rmt": {"request_created_at": {"gte": start_iso}}},
                "operator": "and",
                "right": {"request_response_rmt": {"request_created_at": {"lte": end_iso}}},
            },
            "offset": offset,
            "limit": limit,
            "sort": {"created_at": "asc"},
            "includeInputs": False,
        }

    async def pull(self, start_iso: str, end_iso: str, *, limit: int) -> list[dict[str, Any]]:
        api_key = str(self.credentials.get("api_key") or "")
        if not api_key:
            raise ProviderAdapterError("Helicone API key is required")
        base_url = await validate_provider_url(
            self.base_url, allow_private_endpoints=self.allow_private_endpoints
        )
        events: list[dict[str, Any]] = []
        offset = 0
        async with self.client() as client:
            while len(events) < limit:
                page_limit = min(PAGE_SIZE, limit - len(events))
                payload = self._query_payload(
                    start_iso,
                    end_iso,
                    offset=offset,
                    limit=page_limit,
                )
                try:
                    response = await client.post(
                        f"{base_url}{HELICONE_REQUESTS_PATH}",
                        json=payload,
                        headers={
                            "Authorization": f"Bearer {api_key}",
                            "Accept": "application/json",
                        },
                    )
                    response.raise_for_status()
                    body = response.json()
                except (httpx.HTTPError, ValueError) as exc:
                    raise ProviderAdapterError(
                        f"Helicone connection failed: {type(exc).__name__}"
                    ) from exc
                rows = body.get("data") if isinstance(body, dict) else None
                if not isinstance(rows, list):
                    raise ProviderAdapterError("Helicone returned an invalid requests payload")
                events.extend(self._normalize(row) for row in rows if isinstance(row, dict))
                if len(rows) < page_limit:
                    break
                offset += len(rows)
        return events[:limit]

    def _normalize(self, row: dict[str, Any]) -> dict[str, Any]:
        properties = row.get("request_properties") or row.get("properties") or {}
        if not isinstance(properties, dict):
            properties = {}
        model = _safe_name(
            _first(row, "model", "response_model", "request_model", "model_override"),
            "unknown-model",
        )
        name = _safe_name(
            _first(
                row,
                "prompt_id",
                default=_first(
                    properties,
                    "Helicone-Prompt-Id",
                    "Helicone-Session-Name",
                    default=row.get("request_path"),
                ),
            ),
            "unnamed-request",
        )
        input_tokens = _integer(_first(row, "prompt_tokens", "input_tokens"))
        output_tokens = _integer(_first(row, "completion_tokens", "output_tokens"))
        total_tokens = _integer(row.get("total_tokens")) or input_tokens + output_tokens
        cost = _number(_first(row, "costUSD", "cost_usd"), -1)
        if cost < 0:
            raw_cost = _number(row.get("cost"), -1)
            cost = raw_cost if 0 <= raw_cost < 1000 else -1
        latency = _number(_first(row, "latency", "delay_ms"), 0)
        feedback = row.get("feedback_rating")
        quality = _quality_score(row.get("scores"))
        if quality is None and isinstance(feedback, bool):
            quality = 1.0 if feedback else 0.0
        return {
            "external_id": _safe_name(
                row.get("request_id"), f"helicone-{row.get('request_created_at')}-{name}"
            ),
            "trace_id": str(
                _first(properties, "Helicone-Session-Id", default=row.get("session_id")) or ""
            ),
            "observed_at": str(row.get("request_created_at") or row.get("created_at") or ""),
            "workload_key": _workload_key(name, model),
            "workload_name": name,
            "model": model,
            "provider_name": str(row.get("provider") or "helicone").lower(),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "latency_ms": latency or None,
            "ttft_ms": _number(row.get("time_to_first_token"), 0) or None,
            "cost_usd": cost if cost >= 0 else None,
            "success": _integer(_first(row, "response_status", "status", default=200)) < 400,
            "quality_score": quality,
        }
