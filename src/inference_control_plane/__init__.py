"""Inference Control Plane — reusable inference observability."""

from importlib.metadata import PackageNotFoundError, version

from inference_control_plane.models import (
    ComparisonRow,
    DerivedRates,
    EffectiveCost,
    EndpointCapabilities,
    EndpointConfig,
    MetricSnapshot,
    Recommendation,
)
from inference_control_plane.pool_status import snapshot_from_pool_status
from inference_control_plane.recommendations import (
    compare_snapshots,
    recommend_from_snapshot,
)
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
    "ComparisonRow",
    "DerivedRates",
    "EffectiveCost",
    "EndpointCapabilities",
    "EndpointConfig",
    "MetricSnapshot",
    "Recommendation",
    "VllmClient",
    "__version__",
    "compare_snapshots",
    "derive_rates",
    "estimate_effective_cost",
    "parse_prometheus",
    "recommend_from_snapshot",
    "snapshot_from_metrics",
    "snapshot_from_pool_status",
]
