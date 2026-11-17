"""Human-approved action suggestions mapped from recommendations.

Review-only: every suggestion has ``requires_approval=True`` and nothing
is applied.
"""

from __future__ import annotations

from typing import Any

from inference_control_plane.history_recommendations import (
    _coerce_snapshots,
    recommend_from_history,
)
from inference_control_plane.models import ActionSuggestion, MetricSnapshot
from inference_control_plane.snapshot_recommendations import recommend_from_snapshot


def suggest_actions(snapshot_or_history: Any) -> list[ActionSuggestion]:
    """Map evidence-based recommendations to human-approved action suggestions.

    Review-only: every suggestion has ``requires_approval=True`` and nothing
    is applied. A numeric ``suggested_value`` is emitted only when it can be
    computed from given values; no current mapping supports a defensible
    number (e.g. a saturated KV cache does not imply a specific new
    ``max_num_seqs``), so suggestions carry ``suggested_value=None`` with an
    explicit rationale instead of a fabricated number.
    """
    if isinstance(snapshot_or_history, MetricSnapshot):
        snapshot = snapshot_or_history
        recs = recommend_from_snapshot(snapshot)
        latest_kv = snapshot.kv_cache_usage_ratio
        latest_waiting = snapshot.requests_waiting
    else:
        try:
            items = _coerce_snapshots(snapshot_or_history)
        except Exception:
            items = []
        recs = recommend_from_history(items)
        latest_kv = items[-1].kv_cache_usage_ratio if items else None
        latest_waiting = items[-1].requests_waiting if items else None

    actions: list[ActionSuggestion] = []
    for rec in recs:
        if rec.code == "endpoint_unreachable":
            actions.append(ActionSuggestion(
                code=rec.code,
                target_parameter="observability",
                current_value="unreachable",
                suggested_value=None,
                rationale=f"{rec.evidence} Restore collection before any tuning.",
                expected_effect="Restores the measurements needed for evidence-based tuning.",
            ))
        elif rec.code == "kv_cache_saturated":
            actions.append(ActionSuggestion(
                code=rec.code,
                target_parameter="max_num_seqs",
                current_value=latest_kv,
                suggested_value=None,
                rationale=(
                    f"{rec.evidence} Raising concurrency will not relieve a saturated "
                    "KV cache; review reducing max_num_seqs or adding capacity. No "
                    "numeric value is suggested because the safe value depends on "
                    "sequence lengths not present in the snapshot."
                ),
                expected_effect="Lower KV pressure and fewer preemptions, at lower concurrency.",
            ))
        elif rec.code == "queue_backlog":
            actions.append(ActionSuggestion(
                code=rec.code,
                target_parameter="capacity",
                current_value=latest_waiting,
                suggested_value=None,
                rationale=(
                    f"{rec.evidence} Review batching/concurrency or add capacity; "
                    "the required capacity cannot be computed from queue depth alone."
                ),
                expected_effect="Shorter queueing delay once capacity or batching changes.",
            ))
        elif rec.code == "gpu_underutilized":
            actions.append(ActionSuggestion(
                code=rec.code,
                target_parameter="capacity",
                current_value=latest_kv,
                suggested_value=None,
                rationale=(
                    f"{rec.evidence} Consider a smaller GPU or consolidation; the "
                    "target size cannot be computed without workload/GPU details."
                ),
                expected_effect="Lower idle capacity cost if utilization stays low.",
            ))
        elif rec.code == "elevated_error_rate":
            actions.append(ActionSuggestion(
                code=rec.code,
                target_parameter="error_investigation",
                current_value=None,
                suggested_value=None,
                rationale=f"{rec.evidence} Inspect endpoint errors before tuning performance.",
                expected_effect="Identifies the error source; no parameter change proposed.",
            ))
        elif rec.code == "fleet_load_imbalance":
            actions.append(ActionSuggestion(
                code=rec.code,
                target_parameter="load_placement",
                current_value=None,
                suggested_value=None,
                rationale=(
                    f"{rec.evidence} Review routing/placement in the host; this "
                    "library does not route traffic."
                ),
                expected_effect="Load spread away from saturated endpoints toward idle ones.",
            ))
        # insufficient_evidence -> no action.
    return actions
