"""Facade identity and shared-helper tests for the modularization refactor."""

from inference_control_plane import parse_prometheus
from inference_control_plane.prometheus import error_rate_from_totals
from inference_control_plane.prometheus import parse_prometheus as prometheus_parse
from inference_control_plane.rates import derive_rates as rates_derive
from inference_control_plane.recommendations import (
    compare_snapshots,
    recommend_from_history,
    recommend_from_snapshot,
    suggest_actions,
)
from inference_control_plane.timeutil import parse_iso
from inference_control_plane.vllm import derive_rates as vllm_derive
from inference_control_plane.vllm import parse_prometheus as vllm_parse


def test_facades_are_the_same_objects():
    assert vllm_parse is prometheus_parse is parse_prometheus
    assert vllm_derive is rates_derive
    assert recommend_from_snapshot.__module__ == "inference_control_plane.snapshot_recommendations"
    assert recommend_from_history.__module__ == "inference_control_plane.history_recommendations"
    assert compare_snapshots.__module__ == "inference_control_plane.comparison"
    assert suggest_actions.__module__ == "inference_control_plane.actions"


def test_parse_iso_naive_is_utc_and_invalid_is_none():
    assert parse_iso("2026-01-01T00:00:00").tzinfo is not None
    assert parse_iso("2026-01-01T00:00:00Z").hour == 0
    assert parse_iso("not-a-time") is None
    assert parse_iso(None) is None  # type: ignore[arg-type]


def test_error_rate_from_totals_none_never_zero():
    assert error_rate_from_totals(None, None) is None
    assert error_rate_from_totals(0.0, 0.0) is None
    assert error_rate_from_totals(90.0, 10.0) == 0.1
    assert error_rate_from_totals(None, 5.0) == 1.0
