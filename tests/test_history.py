import asyncio

import pytest

from inference_control_plane import SnapshotHistory, SnapshotPoller, rates_from_history
from inference_control_plane.models import MetricSnapshot


def snap(at, **kw):
    return MetricSnapshot(provider="vllm", status="ok", collected_at=at, source="x", **kw)


def test_history_bounded_and_ordered():
    h = SnapshotHistory(maxlen=3)
    for i in range(5):
        h.append(snap(f"2026-01-01T00:00:0{i}+00:00", generation_tokens_total=float(i)))
    assert len(h) == 3
    assert [s.generation_tokens_total for s in h.snapshots()] == [2.0, 3.0, 4.0]
    assert [s.generation_tokens_total for s in h.last(2)] == [3.0, 4.0]
    assert h.last(0) == []


def test_history_invalid_maxlen():
    with pytest.raises(ValueError):
        SnapshotHistory(maxlen=0)


def test_history_window():
    h = SnapshotHistory()
    h.append(snap("2026-01-01T00:00:00+00:00"))
    h.append(snap("2026-01-01T00:01:00+00:00"))
    h.append(snap("2026-01-01T00:10:00+00:00"))
    assert len(h.window(120)) == 1  # anchored at newest (00:10)
    assert len(h.window(3600)) == 3
    assert SnapshotHistory().window(60) == []


class StubClient:
    def __init__(self, results):
        self._results = list(results)

    async def collect(self):
        result = self._results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


@pytest.mark.asyncio
async def test_poller_records_unreachable_and_failures():
    ok = snap("2026-01-01T00:00:00+00:00")
    unreach = MetricSnapshot(
        provider="vllm", status="unreachable", collected_at="2026-01-01T00:00:15+00:00",
        source="x", errors=["boom"],
    )
    h = SnapshotHistory()
    poller = SnapshotPoller(StubClient([ok, unreach, RuntimeError("kaboom")]), h, interval_seconds=0.01)
    await poller.poll_once()
    await poller.poll_once()
    third = await poller.poll_once()  # never raises
    assert [s.status for s in h.snapshots()] == ["ok", "unreachable", "unreachable"]
    assert third.errors and "RuntimeError" in third.errors[0]


@pytest.mark.asyncio
async def test_poller_run_stops_cleanly():
    h = SnapshotHistory()
    poller = SnapshotPoller(
        StubClient([snap("2026-01-01T00:00:00+00:00")] * 100), h, interval_seconds=0.01
    )
    stop = asyncio.Event()
    task = asyncio.create_task(poller.run(stop))
    await asyncio.sleep(0.05)
    stop.set()
    await asyncio.wait_for(task, timeout=2)
    assert len(h) >= 2


@pytest.mark.asyncio
async def test_poller_cancellation_clean():
    h = SnapshotHistory()
    poller = SnapshotPoller(
        StubClient([snap("2026-01-01T00:00:00+00:00")] * 100), h, interval_seconds=60
    )
    task = asyncio.create_task(poller.run(asyncio.Event()))
    await asyncio.sleep(0.02)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


def test_rates_from_history():
    h = SnapshotHistory()
    h.append(snap("2026-01-01T00:00:00+00:00", generation_tokens_total=0.0))
    h.append(snap("2026-01-01T00:01:00+00:00", generation_tokens_total=6000.0))
    assert rates_from_history(h).tokens_per_second == 100.0
    assert rates_from_history(h, window_seconds=10).tokens_per_second is None  # only newest in window
    assert rates_from_history(SnapshotHistory()).tokens_per_second is None
    single = SnapshotHistory()
    single.append(snap("2026-01-01T00:00:00+00:00", generation_tokens_total=1.0))
    assert rates_from_history(single).generation_tokens_per_second is None
