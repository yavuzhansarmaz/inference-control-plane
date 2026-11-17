"""Inference Control Plane — reusable inference observability."""

from importlib.metadata import PackageNotFoundError, version

from inference_control_plane.errors import ControlPlaneError, WorkloadOptimizerError
from inference_control_plane.exporter import render_prometheus
from inference_control_plane.fleet import recommend_from_fleet, summarize_fleet
from inference_control_plane.history import (
    SnapshotHistory,
    SnapshotPoller,
    rates_from_history,
)
from inference_control_plane.models import (
    ActionSuggestion,
    ComparisonRow,
    DerivedRates,
    EffectiveCost,
    EndpointCapabilities,
    EndpointConfig,
    FleetEndpoint,
    FleetSummary,
    MetricSnapshot,
    Recommendation,
)
from inference_control_plane.recommendations import (
    compare_snapshots,
    recommend_from_history,
    recommend_from_snapshot,
    suggest_actions,
)
from inference_control_plane.sglang import SglangClient, snapshot_from_sglang_metrics
from inference_control_plane.vllm import (
    VllmClient,
    derive_rates,
    estimate_effective_cost,
    parse_prometheus,
    snapshot_from_metrics,
)

try:
    __version__ = version("inference-control-plane")
except PackageNotFoundError:  # source checkout without install
    __version__ = "0.0.0+unknown"

__all__ = [
    "ActionSuggestion",
    "ControlPlaneError",
    "ComparisonRow",
    "DerivedRates",
    "EffectiveCost",
    "EndpointCapabilities",
    "EndpointConfig",
    "FleetEndpoint",
    "FleetSummary",
    "MetricSnapshot",
    "Recommendation",
    "SglangClient",
    "SnapshotHistory",
    "SnapshotPoller",
    "VllmClient",
    "WorkloadOptimizerError",
    "__version__",
    "compare_snapshots",
    "derive_rates",
    "estimate_effective_cost",
    "parse_prometheus",
    "rates_from_history",
    "recommend_from_fleet",
    "recommend_from_history",
    "recommend_from_snapshot",
    "render_prometheus",
    "snapshot_from_metrics",
    "snapshot_from_sglang_metrics",
    "suggest_actions",
    "summarize_fleet",
]
