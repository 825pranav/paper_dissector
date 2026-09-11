"""Tiny disk cache for external API responses.

Keeps development iterations fast and keeps us well under free-tier rate limits
by never re-fetching the same query twice. Falls back to an in-memory dict if
diskcache is unavailable or the cache directory cannot be created.
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any

from paper_dissector.config import CACHE_DIR, CACHE_ENABLED, CACHE_TTL_SECONDS

log = logging.getLogger(__name__)

_MISS = object()
_memory: dict[str, Any] = {}
_disk = None

if CACHE_ENABLED:
    try:
        import diskcache

        _disk = diskcache.Cache(CACHE_DIR)
    except Exception as exc:  # pragma: no cover - environment dependent
        log.warning("disk cache unavailable (%s); using in-memory cache", exc)


def make_key(namespace: str, payload: Any) -> str:
    """Stable cache key from a namespace plus any JSON-serialisable payload."""
    blob = json.dumps(payload, sort_keys=True, default=str)
    digest = hashlib.sha256(blob.encode("utf-8")).hexdigest()[:32]
    return f"{namespace}:{digest}"


def get(key: str, default: Any = None) -> Any:
    """Read a cached value, or ``default`` on miss."""
    if not CACHE_ENABLED:
        return default
    if _disk is not None:
        try:
            value = _disk.get(key, default=_MISS)
            if value is not _MISS:
                return value
        except Exception as exc:  # pragma: no cover
            log.debug("cache read failed for %s: %s", key, exc)
    return _memory.get(key, default)


def has(key: str) -> bool:
    return get(key, _MISS) is not _MISS


def set(key: str, value: Any, ttl: int | None = None) -> None:
    """Write a value to the cache. Failures are non-fatal."""
    if not CACHE_ENABLED:
        return
    if _disk is not None:
        try:
            _disk.set(key, value, expire=ttl if ttl is not None else CACHE_TTL_SECONDS)
            return
        except Exception as exc:  # pragma: no cover
            log.debug("cache write failed for %s: %s", key, exc)
    _memory[key] = value


def clear() -> None:
    """Drop everything — useful between benchmark runs."""
    _memory.clear()
    if _disk is not None:
        try:
            _disk.clear()
        except Exception as exc:  # pragma: no cover
            log.debug("cache clear failed: %s", exc)
