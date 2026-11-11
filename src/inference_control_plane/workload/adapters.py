"""Public adapter registry for Workload Optimizer provider integrations."""

from typing import Any

from inference_control_plane.workload.adapter_base import (
    MAX_PROVIDER_EVENTS,
    ProviderAdapter,
    ProviderAdapterError,
    validate_provider_url,
)
from inference_control_plane.workload.helicone import HeliconeAdapter
from inference_control_plane.workload.langfuse import LangfuseAdapter
from inference_control_plane.workload.otlp import normalize_otlp_export


def adapter_for(
    provider: str,
    base_url: str,
    credentials: dict[str, Any],
    *,
    allow_private_endpoints: bool = False,
) -> ProviderAdapter:
    if provider == "langfuse":
        return LangfuseAdapter(
            base_url, credentials, allow_private_endpoints=allow_private_endpoints
        )
    if provider == "helicone":
        return HeliconeAdapter(
            base_url, credentials, allow_private_endpoints=allow_private_endpoints
        )
    raise ProviderAdapterError(f"Provider {provider!r} uses import instead of remote sync")


__all__ = [
    "MAX_PROVIDER_EVENTS",
    "ProviderAdapterError",
    "adapter_for",
    "normalize_otlp_export",
    "validate_provider_url",
]
