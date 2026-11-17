"""Shared recommendation thresholds (single definition).

All snapshot, history and fleet recommendations use these values so the
documented cut-offs cannot drift apart.
"""

KV_SATURATED_RATIO = 0.95
KV_IDLE_RATIO = 0.25
WAITING_QUEUE_THRESHOLD = 5
ERROR_RATE_THRESHOLD = 0.05

__all__ = [
    "ERROR_RATE_THRESHOLD",
    "KV_IDLE_RATIO",
    "KV_SATURATED_RATIO",
    "WAITING_QUEUE_THRESHOLD",
]
