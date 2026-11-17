"""Shared scaffold for read-only Prometheus metrics clients (vLLM, SGLang).

Owns SSRF validation, bearer headers, and the never-raise collect/test flow.
Subclasses supply only their provider name, the marker substring expected in
a metrics response, and their snapshot builder.
"""

from __future__ import annotations

import httpx

from inference_control_plane.models import EndpointConfig, MetricSnapshot
from inference_control_plane.timeutil import now_iso
from inference_control_plane.workload.adapter_base import (
    ProviderAdapterError,
    validate_provider_url,
)


class _MetricsClientBase:
    """Bounded, read-only collector. Never raises for endpoint failures."""

    provider: str = ""
    metrics_marker: str = ""

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
        return f"{type(self).__name__}(base_url={self.config.base_url!r})"

    @staticmethod
    def _build_snapshot(text: str, *, source: str) -> MetricSnapshot:
        raise NotImplementedError

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._api_key}"} if self._api_key else {}

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

    def _unreachable(self, source: str, error: str) -> MetricSnapshot:
        return MetricSnapshot(
            provider=self.provider,
            status="unreachable",
            collected_at=now_iso(),
            source=source,
            errors=[error],
        )

    async def collect(self) -> MetricSnapshot:
        try:
            _, metrics_url = await self._validated_urls()
        except ProviderAdapterError as exc:
            return self._unreachable(
                self.config.metrics_url or self.config.base_url, str(exc)
            )
        try:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(self.timeout_seconds, connect=3.0),
                follow_redirects=False,
                transport=self._transport,
            ) as client:
                response = await client.get(metrics_url, headers=self._headers())
                response.raise_for_status()
                text = response.text
        except Exception as exc:  # unreachable / auth / timeout / malformed
            return self._unreachable(
                metrics_url, f"{type(exc).__name__}: collection failed"
            )
        if self.metrics_marker not in text and "# HELP" not in text:
            return MetricSnapshot(
                provider=self.provider,
                status="degraded",
                collected_at=now_iso(),
                source=metrics_url,
                errors=["Metrics response did not look like Prometheus exposition"],
            )
        return self._build_snapshot(text, source=metrics_url)

    async def test(self) -> None:
        snapshot = await self.collect()
        if snapshot.status == "unreachable":
            raise ProviderAdapterError(snapshot.errors[0] if snapshot.errors else "unreachable")
