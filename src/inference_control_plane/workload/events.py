"""Typed workload event model mirroring the adapters' normalized dicts.

Extra fields producers emit (for example ``source_system``, ``batch_id``) are kept:
``model_config`` allows extras and :meth:`WorkloadEvent.to_dict` includes
them, so a dict -> event -> dict round-trip preserves analyzed output.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict


class WorkloadEvent(BaseModel):
    model_config = ConfigDict(extra="allow")

    external_id: str = ""
    trace_id: str = ""
    observed_at: str = ""
    workload_key: str = ""
    workload_name: str = ""
    model: str = ""
    provider_name: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    latency_ms: float | None = None
    ttft_ms: float | None = None
    cost_usd: float | None = None
    success: bool = True
    quality_score: float | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any] | WorkloadEvent) -> WorkloadEvent:
        """Accept-and-validate helper: accepts a dict or an existing event."""
        if isinstance(data, WorkloadEvent):
            return data
        return cls.model_validate(dict(data))

    def to_dict(self) -> dict[str, Any]:
        """Plain dict including any extra fields the producer emitted."""
        return self.model_dump()


def normalize_events(events: Any) -> list[dict[str, Any]]:
    """Normalize ``list[dict | WorkloadEvent]`` to plain dicts at entry points."""
    normalized: list[dict[str, Any]] = []
    for event in events or []:
        if isinstance(event, WorkloadEvent):
            normalized.append(event.to_dict())
        elif isinstance(event, dict):
            normalized.append(dict(event))
        else:
            normalized.append(WorkloadEvent.from_dict(event).to_dict())
    return normalized
