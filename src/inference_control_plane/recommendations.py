"""Evidence-based optimization recommendations (review-only, no auto-apply).

Facade: implementations live in ``snapshot_recommendations``,
``history_recommendations``, ``comparison``, and ``actions``; this module
keeps the historical import path working unchanged.
"""

from __future__ import annotations

from inference_control_plane.actions import suggest_actions
from inference_control_plane.comparison import compare_snapshots
from inference_control_plane.history_recommendations import recommend_from_history
from inference_control_plane.snapshot_recommendations import recommend_from_snapshot

__all__ = [
    "compare_snapshots",
    "recommend_from_history",
    "recommend_from_snapshot",
    "suggest_actions",
]
