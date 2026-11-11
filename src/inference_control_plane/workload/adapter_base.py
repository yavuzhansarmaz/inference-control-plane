"""Shared contracts and normalization helpers for observability provider adapters."""

from __future__ import annotations

import asyncio
import ipaddress
import re
import socket
from abc import ABC, abstractmethod
from datetime import datetime, timedelta, timezone
from typing import Any
from urllib.parse import urlparse

import httpx

MAX_PROVIDER_EVENTS = 10_000
PAGE_SIZE = 100


class ProviderAdapterError(RuntimeError):
    """Safe error returned when a provider cannot be reached or parsed."""


def _number(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
        return parsed if parsed == parsed else default
    except (TypeError, ValueError):
        return default


def _integer(value: Any, default: int = 0) -> int:
    return max(0, int(_number(value, default)))


def _first(mapping: dict[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        value = mapping.get(key)
        if value is not None:
            return value
    return default


def _safe_name(value: Any, fallback: str) -> str:
    name = str(value or "").strip()
    return name[:160] if name else fallback


def _workload_key(name: str, model: str) -> str:
    raw = f"{name}-{model}".lower()
    slug = re.sub(r"[^a-z0-9]+", "-", raw).strip("-")
    return (slug or "unclassified")[:180]


def _quality_score(value: Any) -> float | None:
    scores: list[float] = []
    if isinstance(value, dict):
        for item in value.values():
            if isinstance(item, dict):
                item = _first(item, "value", "score")
            number = _number(item, -1)
            if number >= 0:
                scores.append(number)
    elif isinstance(value, (int, float)):
        scores.append(float(value))
    if not scores:
        return None
    mean = sum(scores) / len(scores)
    if 1 < mean <= 100:
        mean /= 100
    return min(1.0, max(0.0, mean))


async def validate_provider_url(
    raw_url: str,
    *,
    allow_private_endpoints: bool = False,
    resolve_dns: bool = True,
) -> str:
    """Reject malformed and private-network URLs unless explicitly allowed."""
    value = raw_url.strip().rstrip("/")
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ProviderAdapterError("Provider URL must be an absolute http(s) URL")
    if parsed.username or parsed.password:
        raise ProviderAdapterError("Provider URL must not contain credentials")
    if parsed.scheme != "https" and parsed.hostname not in {"localhost", "127.0.0.1", "::1"}:
        raise ProviderAdapterError("Provider URL must use HTTPS")
    if allow_private_endpoints:
        return value

    try:
        literal = ipaddress.ip_address(parsed.hostname)
    except ValueError:
        literal = None
    if literal is not None:
        if not literal.is_global:
            raise ProviderAdapterError(
                "Private observability endpoints are disabled by server policy"
            )
        return value
    if not resolve_dns:  # injected-transport test seam; literal checks above still apply
        return value

    try:
        resolved = await asyncio.to_thread(socket.getaddrinfo, parsed.hostname, parsed.port or 443)
    except socket.gaierror as exc:
        raise ProviderAdapterError("Provider hostname could not be resolved") from exc
    for entry in resolved:
        address = ipaddress.ip_address(entry[4][0])
        if not address.is_global:
            raise ProviderAdapterError(
                "Private observability endpoints are disabled by server policy"
            )
    return value


class ProviderAdapter(ABC):
    def __init__(
        self,
        base_url: str,
        credentials: dict[str, Any],
        *,
        allow_private_endpoints: bool = False,
    ):
        self.base_url = base_url
        self.credentials = credentials
        self.allow_private_endpoints = allow_private_endpoints

    @abstractmethod
    async def pull(self, start_iso: str, end_iso: str, *, limit: int) -> list[dict[str, Any]]: ...

    async def test(self) -> None:
        now = datetime.now(timezone.utc)
        events = await self.pull((now - timedelta(hours=1)).isoformat(), now.isoformat(), limit=1)
        if not isinstance(events, list):
            raise ProviderAdapterError("Provider returned an invalid response")

    @staticmethod
    def client() -> httpx.AsyncClient:
        return httpx.AsyncClient(
            timeout=httpx.Timeout(20.0, connect=8.0),
            follow_redirects=False,
            limits=httpx.Limits(max_connections=5, max_keepalive_connections=2),
        )
