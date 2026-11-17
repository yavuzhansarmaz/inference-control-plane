from inference_control_plane import SnapshotHistory, suggest_actions
from inference_control_plane.models import MetricSnapshot


def snap(status="ok", **kw):
    return MetricSnapshot(provider="vllm", status=status,
                          collected_at="2026-01-01T00:00:00+00:00", source="x", **kw)


def test_unreachable_single_observability_action():
    actions = suggest_actions(snap(status="unreachable", errors=["down"]))
    assert len(actions) == 1
    assert actions[0].code == "endpoint_unreachable"
    assert actions[0].requires_approval is True and actions[0].suggested_value is None


def test_saturated_action_no_fabricated_number():
    actions = suggest_actions(snap(kv_cache_usage_ratio=0.99, requests_waiting=8.0))
    kv = next(a for a in actions if a.code == "kv_cache_saturated")
    assert kv.target_parameter == "max_num_seqs"
    assert kv.current_value == 0.99 and kv.suggested_value is None
    assert kv.requires_approval is True and "max_num_seqs" in kv.rationale
    assert all(a.requires_approval for a in actions)


def test_healthy_and_empty_no_actions():
    assert suggest_actions(snap(kv_cache_usage_ratio=0.5, requests_waiting=0.0)) == []
    assert suggest_actions([]) == []
    assert suggest_actions(SnapshotHistory()) == []


def test_history_sustained_saturation_action():
    h = SnapshotHistory()
    for i in range(12):
        h.append(MetricSnapshot(
            provider="vllm", status="ok",
            collected_at=f"2026-01-01T00:{i:02d}:00+00:00", source="x",
            kv_cache_usage_ratio=0.98,
        ))
    actions = suggest_actions(h)
    assert any(a.code == "kv_cache_saturated" and a.suggested_value is None for a in actions)
