"""Generic control-plane errors.

``ControlPlaneError`` is the canonical error type. It carries an optional
``status_code``/``code`` for hosts that map errors to HTTP responses; the
library itself is transport-agnostic and never raises HTTP exceptions.
"""

from __future__ import annotations


class ControlPlaneError(RuntimeError):
    """Generic error raised by inference-control-plane operations."""

    def __init__(
        self, message: str, *, status_code: int = 400, code: str = "control_plane_error"
    ):
        self.status_code = status_code
        self.code = code
        super().__init__(message)


# Deprecated alias kept for compatibility with the 0.1.x published API.
# Prefer ``ControlPlaneError`` in new code; this name will be removed in a
# future major release.
WorkloadOptimizerError = ControlPlaneError

__all__ = ["ControlPlaneError", "WorkloadOptimizerError"]
