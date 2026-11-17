"""Prometheus exposition parsing shared by the vLLM and SGLang collectors."""

from __future__ import annotations

import re

_LINE = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{[^}]*\})?\s+([-+eE0-9.]+|NaN|\+Inf|-Inf)$")


def parse_prometheus(text: str) -> dict[str, float]:
    """Sum samples per metric name; histogram buckets keep their ``_bucket`` name."""
    totals: dict[str, float] = {}
    buckets: dict[str, list[tuple[float, float]]] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue  # bucket bounds live in the label set, handled below
        match = _LINE.match(line)
        if not match:
            continue
        name, value = match.group(1), match.group(2)
        try:
            number = float(value)
        except ValueError:
            continue
        if number != number:  # NaN
            continue
        totals[name] = totals.get(name, 0.0) + number
        if name.endswith("_bucket"):
            bound_match = re.search(r'le="([^"]+)"', line)
            if bound_match:
                try:
                    bound = float(bound_match.group(1))
                except ValueError:
                    bound = float("inf")
                buckets.setdefault(name, []).append((bound, number))
    # Store bucket-derived p95 under a synthetic key for the collector.
    for name, pairs in buckets.items():
        p95 = _histogram_p95(pairs)
        if p95 is not None:
            totals[f"{name}::p95"] = p95
    return totals


def _histogram_p95(pairs: list[tuple[float, float]]) -> float | None:
    finite = sorted((b, c) for b, c in pairs if b != float("inf"))
    if not finite:
        return None
    total = max(c for _, c in pairs)
    if total <= 0:
        return None
    target = 0.95 * total
    previous_bound, previous_count = 0.0, 0.0
    for bound, count in finite:
        if count >= target:
            if count == previous_count:
                return bound
            fraction = (target - previous_count) / (count - previous_count)
            return previous_bound + fraction * (bound - previous_bound)
        previous_bound, previous_count = bound, count
    return finite[-1][0]


def error_rate_from_totals(success: float | None, failure: float | None) -> float | None:
    """Error ratio from success/failure counters; None when neither is known."""
    if success is None and failure is None:
        return None
    total = (success or 0.0) + (failure or 0.0)
    return ((failure or 0.0) / total) if total > 0 else None
