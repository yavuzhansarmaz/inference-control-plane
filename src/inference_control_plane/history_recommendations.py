"""Window-aggregated recommendations over snapshot history."""

from __future__ import annotations

from typing import Any

from inference_control_plane.models import MetricSnapshot, Recommendation
from inference_control_plane.thresholds import (
    ERROR_RATE_THRESHOLD,
    KV_IDLE_RATIO,
    KV_SATURATED_RATIO,
    WAITING_QUEUE_THRESHOLD,
)
from inference_control_plane.timeutil import parse_iso


def _coerce_snapshots(source: Any, window_seconds: float | None = None) -> list[MetricSnapshot]:
    """Accept a SnapshotHistory, a single snapshot, or a sequence of snapshots."""
    if isinstance(source, MetricSnapshot):
        return [source]
    if hasattr(source, "window") and hasattr(source, "snapshots"):
        if window_seconds is not None:
            return list(source.window(window_seconds))
        return list(source.snapshots())
    items = [s for s in list(source) if isinstance(s, MetricSnapshot)]
    if window_seconds is not None and items:
        parsed = [(parse_iso(s.collected_at), s) for s in items]
        newest = max((t for t, _ in parsed if t is not None), default=None)
        if newest is not None:
            cutoff = newest.timestamp() - window_seconds
            items = [s for t, s in parsed if t is not None and t.timestamp() >= cutoff]
    return items


def _window_confidence(count: int, fraction: float = 1.0) -> str:
    """Confidence scales with sample count: 1 -> low, sustained -> medium/high."""
    if count <= 1:
        return "low"
    if count >= 10 and fraction >= 0.8:
        return "high"
    if count >= 3 and fraction >= 0.5:
        return "medium"
    return "low"


def _span_minutes(items: list[MetricSnapshot]) -> float | None:
    times = [t for s in items if (t := parse_iso(s.collected_at)) is not None]
    if len(times) < 2:
        return None
    return max((max(times) - min(times)).total_seconds() / 60.0, 0.0)


def recommend_from_history(
    history_or_snapshots: Any, *, window_seconds: float | None = None
) -> list[Recommendation]:
    """Aggregate recommendations over a window of snapshots.

    Accepts a ``SnapshotHistory``, a sequence of snapshots, or one snapshot.
    Never raises on empty input. Emits the same codes as
    :func:`recommend_from_snapshot` where thresholds match, with evidence
    citing the sample count and window, and confidence scaled by sample count.
    """
    try:
        items = _coerce_snapshots(history_or_snapshots, window_seconds)
    except Exception:
        items = []
    if not items:
        return [
            Recommendation(
                code="insufficient_evidence",
                title="No specific change recommended from the available measurements",
                evidence="No snapshots in the selected window (0 samples).",
                assumptions=["Collect snapshots over a representative window before deciding."],
                confidence="low",
            )
        ]
    n = len(items)
    span = _span_minutes(items)
    span_txt = f" over {span:.0f} min" if span else ""
    window_txt = f"in {{hit}}/{n} snapshots{span_txt} ({n} samples)"

    def conf(hit: int) -> str:
        return _window_confidence(n, hit / n if n else 0.0)

    recs: list[Recommendation] = []
    unreachable = [s for s in items if s.status == "unreachable"]
    if len(unreachable) == n:
        return [
            Recommendation(
                code="endpoint_unreachable",
                title="Restore endpoint observability before optimizing",
                evidence=(
                    f"Collection failed in {n}/{n} snapshots{span_txt} ({n} samples): "
                    f"{'; '.join(items[-1].errors) or 'unreachable'}"
                ),
                assumptions=["No performance conclusion is drawn from failed collections."],
                confidence=_window_confidence(n, 1.0) if n > 1 else "low",
            )
        ]
    if unreachable:
        recs.append(
            Recommendation(
                code="endpoint_unreachable",
                title="Restore endpoint observability before optimizing",
                evidence=window_txt.format(hit=len(unreachable))
                + " were unreachable.",
                assumptions=["Performance conclusions use only reachable snapshots."],
                confidence=conf(len(unreachable)),
            )
        )

    reachable = [s for s in items if s.status != "unreachable"]
    kv_values = [s.kv_cache_usage_ratio for s in reachable if s.kv_cache_usage_ratio is not None]
    if kv_values:
        saturated = sum(1 for v in kv_values if v >= KV_SATURATED_RATIO)
        if saturated and saturated / len(kv_values) >= 0.5:
            recs.append(
                Recommendation(
                    code="kv_cache_saturated",
                    title="KV cache is saturated; review concurrency / max_num_seqs",
                    evidence=(
                        f"KV cache >=95% in {saturated}/{len(kv_values)} snapshots"
                        f"{span_txt} ({n} samples); mean {sum(kv_values) / len(kv_values):.0%}."
                    ),
                    assumptions=["Sustained saturation, not a single spike, should be confirmed."],
                    confidence=conf(saturated),
                )
            )
        idle = sum(
            1
            for s in reachable
            if s.kv_cache_usage_ratio is not None
            and s.kv_cache_usage_ratio <= KV_IDLE_RATIO
            and (s.requests_running or 0) <= 1
        )
        if idle and idle / len(kv_values) >= 0.5:
            recs.append(
                Recommendation(
                    code="gpu_underutilized",
                    title="Capacity appears underutilized; consider a smaller GPU or consolidation",
                    evidence=window_txt.format(hit=idle)
                    + " had KV cache <=25% with <=1 running request.",
                    assumptions=["Confirm over a representative window before consolidating."],
                    confidence=conf(idle),
                )
            )

    waiting = [s.requests_waiting for s in reachable if s.requests_waiting is not None]
    if waiting:
        queued = sum(1 for v in waiting if v >= WAITING_QUEUE_THRESHOLD)
        mean_waiting = sum(waiting) / len(waiting)
        if queued and queued / len(waiting) >= 0.5:
            recs.append(
                Recommendation(
                    code="queue_backlog",
                    title="Requests are queueing; review batching/concurrency or add capacity",
                    evidence=window_txt.format(hit=queued)
                    + f" had >=5 waiting; mean waiting {mean_waiting:.1f}.",
                    confidence=conf(queued),
                )
            )

    errors = [s.error_rate for s in reachable if s.error_rate is not None]
    if errors:
        elevated = sum(1 for v in errors if v >= ERROR_RATE_THRESHOLD)
        mean_err, max_err = sum(errors) / len(errors), max(errors)
        if (elevated and elevated / len(errors) >= 0.5) or mean_err >= ERROR_RATE_THRESHOLD:
            recs.append(
                Recommendation(
                    code="elevated_error_rate",
                    title="Elevated error rate; inspect endpoint errors before tuning performance",
                    evidence=window_txt.format(hit=elevated)
                    + f" had error rate >=5%; mean {mean_err:.1%}, max {max_err:.1%}.",
                    confidence=conf(max(elevated, 1)),
                )
            )

    if not recs:
        recs.append(
            Recommendation(
                code="insufficient_evidence",
                title="No specific change recommended from the available measurements",
                evidence=f"Available metrics in {n} snapshots{span_txt} cross no documented threshold.",
                assumptions=["Collect a longer representative window before deciding."],
                confidence="low",
            )
        )
    return recs
