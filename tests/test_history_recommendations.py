from inference_control_plane import SnapshotHistory, recommend_from_history, recommend_from_snapshot
from inference_control_plane.models import MetricSnapshot


def snap(minute, **kw):
    return MetricSnapshot(
        provider="vllm", status=kw.pop("status", "ok"),
        collected_at=f"2026-01-01T00:{minute:02d}:00+00:00", source="x", **kw,
    )


def test_empty_history_insufficient_never_raises():
    for empty in ([], SnapshotHistory(), None):
        recs = recommend_from_history(empty)
        assert [r.code for r in recs] == ["insufficient_evidence"]
        assert recs[0].confidence == "low"


def test_single_snapshot_low_confidence_matches_single_shot_codes():
    s = snap(0, kv_cache_usage_ratio=0.97, requests_waiting=7.0)
    hist_codes = {r.code for r in recommend_from_history([s])}
    single_codes = {r.code for r in recommend_from_snapshot(s)}
    assert hist_codes == single_codes
    assert all(r.confidence == "low" for r in recommend_from_history([s]))


def test_sustained_saturation_higher_confidence():
    snaps = [snap(i % 60, kv_cache_usage_ratio=0.98, requests_waiting=9.0) for i in range(20)]
    recs = recommend_from_history(snaps)
    kv = next(r for r in recs if r.code == "kv_cache_saturated")
    assert kv.confidence == "high"
    assert "20" in kv.evidence and "samples" in kv.evidence
    single = recommend_from_history(snaps[:1])
    assert next(r for r in single if r.code == "kv_cache_saturated").confidence == "low"
    assert kv.confidence != single[0].confidence


def test_window_seconds_filters():
    snaps = [snap(0, kv_cache_usage_ratio=0.1), snap(30, kv_cache_usage_ratio=0.99),
             snap(59, kv_cache_usage_ratio=0.99)]
    # Anchored at newest (00:59): 60s window keeps only the last snapshot.
    recs = recommend_from_history(snaps, window_seconds=60)
    kv = next(r for r in recs if r.code == "kv_cache_saturated")
    assert "1 samples" in kv.evidence or "(1 samples" in kv.evidence


def test_all_unreachable_and_mixed_errors():
    down = [snap(i, status="unreachable") for i in range(3)]
    assert recommend_from_history(down)[0].code == "endpoint_unreachable"
    mixed = [snap(0, error_rate=0.2), snap(1, error_rate=0.1), snap(2, error_rate=0.0)]
    recs = recommend_from_history(mixed)
    err = next(r for r in recs if r.code == "elevated_error_rate")
    assert "mean" in err.evidence and "max" in err.evidence


def test_history_object_input():
    h = SnapshotHistory()
    for i in range(5):
        h.append(snap(i, kv_cache_usage_ratio=0.5, requests_waiting=0.0, error_rate=0.0))
    recs = recommend_from_history(h)
    assert [r.code for r in recs] == ["insufficient_evidence"]
