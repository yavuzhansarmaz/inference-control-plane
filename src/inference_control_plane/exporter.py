"""Prometheus text exporter (pure function; no server, no new dependencies).

Missing (``None``) metrics are skipped entirely — a missing value is never
rendered as ``0``. Fleet summaries contribute worst-endpoint gauges labelled
with the worst endpoint's name where applicable.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from inference_control_plane.fleet import summarize_fleet
from inference_control_plane.models import FleetSummary, MetricSnapshot

_GAUGES: list[tuple[str, str, str]] = [
    ("requests_running", "requests_running", "Requests currently running."),
    ("requests_waiting", "requests_waiting", "Requests currently waiting."),
    ("kv_cache_usage_ratio", "kv_cache_usage_ratio", "KV cache usage ratio (0..1)."),
    ("gpu_utilization_ratio", "gpu_utilization_ratio", "GPU utilization ratio (0..1)."),
    ("error_rate", "error_rate", "Error rate ratio (0..1)."),
    ("ttft_p95_seconds", "ttft_p95_seconds", "Time-to-first-token p95 in seconds."),
    ("e2e_latency_p95_seconds", "e2e_latency_p95_seconds", "End-to-end latency p95 in seconds."),
    ("tokens_per_second", "tokens_per_second", "Tokens per second when a window rate is known."),
]


def _escape_label(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def _format(value: float) -> str:
    return repr(float(value))


def _named_snapshots(snapshots: Any) -> list[tuple[str, MetricSnapshot]]:
    if isinstance(snapshots, MetricSnapshot):
        return [(snapshots.source, snapshots)]
    if isinstance(snapshots, FleetSummary):
        return []
    if isinstance(snapshots, Mapping):
        return [
            (str(name), snap)
            for name, snap in snapshots.items()
            if isinstance(snap, MetricSnapshot)
        ]
    named: list[tuple[str, MetricSnapshot]] = []
    for item in snapshots or []:
        if isinstance(item, MetricSnapshot):
            named.append((item.source, item))
        elif hasattr(item, "name") and hasattr(item, "snapshot"):
            named.append((str(item.name), item.snapshot))
        elif isinstance(item, (tuple, list)) and len(item) == 2:
            name, snap = item
            if isinstance(snap, MetricSnapshot):
                named.append((str(name), snap))
    return named


def render_prometheus(snapshots: Any, *, prefix: str = "inference_control_plane") -> str:
    """Render snapshots (and optionally a FleetSummary) as Prometheus text.

    ``snapshots`` may be a single snapshot, a mapping of name -> snapshot, a
    sequence of snapshots / ``(name, snapshot)`` pairs / ``FleetEndpoint``,
    or a ``FleetSummary``. The endpoint label is the provided name, else
    ``snapshot.source``.
    """
    prefix = prefix.strip("_") or "inference_control_plane"
    lines: list[str] = []
    named = _named_snapshots(snapshots)
    for field, metric, help_text in _GAUGES:
        samples = [
            (name, getattr(snap, field))
            for name, snap in named
            if getattr(snap, field) is not None
        ]
        if not samples:
            continue
        lines.append(f"# HELP {prefix}_{metric} {help_text}")
        lines.append(f"# TYPE {prefix}_{metric} gauge")
        for name, value in samples:
            lines.append(
                f'{prefix}_{metric}{{endpoint="{_escape_label(name)}"}} {_format(value)}'
            )
    if named:
        lines.append(f"# HELP {prefix}_snapshot_status Snapshot collection status (1 for the reported status).")
        lines.append(f"# TYPE {prefix}_snapshot_status gauge")
        for name, snap in named:
            lines.append(
                f'{prefix}_snapshot_status{{endpoint="{_escape_label(name)}",'
                f'status="{_escape_label(snap.status)}"}} 1'
            )
    summary = snapshots if isinstance(snapshots, FleetSummary) else None
    if summary is None and not isinstance(snapshots, MetricSnapshot):
        try:
            candidate = summarize_fleet(snapshots)
        except Exception:
            candidate = None
        if candidate is not None and candidate.endpoint_count:
            summary = candidate
    if summary is not None:
        fleet_gauges = [
            ("fleet_endpoints", float(summary.endpoint_count), None),
            ("fleet_endpoints_reporting", float(summary.endpoints_reporting), None),
            ("fleet_mean_kv_cache_usage_ratio", summary.mean_kv_cache_usage_ratio, None),
            ("fleet_mean_error_rate", summary.mean_error_rate, None),
            ("fleet_worst_kv_cache_usage_ratio", summary.worst_kv_cache_usage_ratio,
             summary.worst_kv_cache_name),
            ("fleet_worst_error_rate", summary.worst_error_rate, summary.worst_error_rate_name),
            ("fleet_worst_requests_waiting", summary.worst_requests_waiting,
             summary.worst_requests_waiting_name),
            ("fleet_total_requests_running", summary.total_requests_running, None),
            ("fleet_total_requests_waiting", summary.total_requests_waiting, None),
        ]
        for metric, value, endpoint in fleet_gauges:
            if value is None:
                continue
            lines.append(f"# HELP {prefix}_{metric} Fleet aggregate gauge.")
            lines.append(f"# TYPE {prefix}_{metric} gauge")
            label = f'{{endpoint="{_escape_label(endpoint)}"}}' if endpoint else ""
            lines.append(f"{prefix}_{metric}{label} {_format(value)}")
    return "\n".join(lines) + ("\n" if lines else "")
