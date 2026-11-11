"""Shared preparation for normalized workload events before persistence."""

from typing import Any

from inference_control_plane.timeutil import now_iso
from inference_control_plane.workload.adapter_base import MAX_PROVIDER_EVENTS

CONTENT_FIELDS = (
    "training_pair",
    "training_pair_encrypted",
    "input",
    "output",
    "prompt",
    "completion",
    "messages",
    "request_body",
    "response_body",
    "system_prompt",
)


def prepare_events(
    org_id: str,
    connection_id: str,
    events: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Scope metric-only events to a connection before persistence."""
    prepared: list[dict[str, Any]] = []
    now = now_iso()
    for source_event in events[:MAX_PROVIDER_EVENTS]:
        event = dict(source_event)
        for field in CONTENT_FIELDS:
            event.pop(field, None)
        event.update(
            {
                "org_id": org_id,
                "connection_id": connection_id,
                "observed_at": event.get("observed_at") or now,
                "updated_at": now,
            }
        )
        prepared.append(event)
    return prepared
