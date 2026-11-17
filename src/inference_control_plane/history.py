"""In-memory snapshot history and asyncio polling (caller-owned, no persistence).

The history is a bounded ring buffer held in memory only. The host owns any
persistence: it may drain :class:`SnapshotHistory` and store snapshots itself.
No background threads are started; polling is an asyncio loop the caller runs.
"""

from __future__ import annotations

import asyncio
import threading
from collections import deque
from typing import Any, Protocol

from inference_control_plane.models import DerivedRates, MetricSnapshot
from inference_control_plane.rates import derive_rates
from inference_control_plane.timeutil import now_iso, parse_iso

_parse_iso = parse_iso


class SnapshotHistory:
    """Bounded, thread-safe ring buffer of :class:`MetricSnapshot` values."""

    def __init__(self, maxlen: int = 1000):
        if maxlen <= 0:
            raise ValueError("maxlen must be positive")
        self.maxlen = maxlen
        self._items: deque[MetricSnapshot] = deque(maxlen=maxlen)
        self._lock = threading.Lock()

    def append(self, snapshot: MetricSnapshot) -> None:
        with self._lock:
            self._items.append(snapshot)

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)

    def snapshots(self) -> list[MetricSnapshot]:
        """All retained snapshots, oldest first."""
        with self._lock:
            return list(self._items)

    def last(self, n: int) -> list[MetricSnapshot]:
        """The most recent ``n`` snapshots, oldest first."""
        if n <= 0:
            return []
        with self._lock:
            return list(self._items)[-n:]

    def window(self, seconds: float) -> list[MetricSnapshot]:
        """Snapshots collected within the last ``seconds`` of the newest one.

        The window is anchored at the newest snapshot's ``collected_at`` so it
        works with replayed/fake timestamps. Snapshots with unparseable
        timestamps are excluded.
        """
        items = self.snapshots()
        if not items or seconds < 0:
            return []
        newest = _parse_iso(items[-1].collected_at)
        if newest is None:
            return []
        cutoff = newest.timestamp() - seconds
        selected = []
        for item in items:
            at = _parse_iso(item.collected_at)
            if at is not None and at.timestamp() >= cutoff:
                selected.append(item)
        return selected


class _Collectable(Protocol):
    async def collect(self) -> MetricSnapshot: ...


class SnapshotPoller:
    """Poll any object with ``async collect()`` into a :class:`SnapshotHistory`.

    Collection failures never propagate: a client that itself returns an
    ``unreachable`` snapshot (as ``VllmClient``/``SglangClient`` do) is recorded
    as-is, and an unexpected exception is converted into an ``unreachable``
    snapshot so the failure is visible in the history rather than lost.
    """

    def __init__(
        self,
        client: _Collectable,
        history: SnapshotHistory,
        *,
        interval_seconds: float = 15.0,
    ):
        if interval_seconds <= 0:
            raise ValueError("interval_seconds must be positive")
        self.client = client
        self.history = history
        self.interval_seconds = interval_seconds

    async def poll_once(self) -> MetricSnapshot:
        try:
            snapshot = await self.client.collect()
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # record, never raise
            snapshot = self._failure_snapshot(exc)
        self.history.append(snapshot)
        return snapshot

    def _failure_snapshot(self, exc: Exception) -> MetricSnapshot:
        config = getattr(self.client, "config", None)
        provider = str(getattr(config, "provider", None) or "unknown")
        source = str(
            getattr(config, "metrics_url", None)
            or getattr(config, "base_url", None)
            or "unknown"
        )
        return MetricSnapshot(
            provider=provider,
            status="unreachable",
            collected_at=now_iso(),
            source=source,
            errors=[f"{type(exc).__name__}: collection failed"],
        )

    async def run(self, stop_event: asyncio.Event) -> None:
        """Poll until ``stop_event`` is set or the task is cancelled."""
        while not stop_event.is_set():
            await self.poll_once()
            try:
                await asyncio.wait_for(
                    stop_event.wait(), timeout=self.interval_seconds
                )
            except TimeoutError:
                continue


def rates_from_history(
    history: SnapshotHistory | list[MetricSnapshot] | Any,
    *,
    window_seconds: float | None = None,
) -> DerivedRates:
    """Derive rates from the first/last snapshots of a history window.

    Unknown rates stay ``None`` (fewer than two snapshots, missing counters,
    rewound counters, or an unusable window all yield ``None`` rates).
    """
    if isinstance(history, SnapshotHistory):
        items = (
            history.window(window_seconds)
            if window_seconds is not None
            else history.snapshots()
        )
    else:
        items = list(history)  # type: ignore[arg-type]
        if window_seconds is not None and items:
            newest = _parse_iso(items[-1].collected_at)
            if newest is not None:
                cutoff = newest.timestamp() - window_seconds
                items = [
                    s
                    for s in items
                    if (t := _parse_iso(s.collected_at)) is not None
                    and t.timestamp() >= cutoff
                ]
    if len(items) < 2:
        return DerivedRates()
    return derive_rates(items[0], items[-1])
