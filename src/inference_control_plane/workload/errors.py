"""Workload error types (compatibility re-export).

Canonical definition lives in :mod:`inference_control_plane.errors`.
``WorkloadOptimizerError`` is a deprecated alias of ``ControlPlaneError``
kept so the 0.1.x published API keeps working.
"""

from inference_control_plane.errors import ControlPlaneError, WorkloadOptimizerError

__all__ = ["ControlPlaneError", "WorkloadOptimizerError"]
