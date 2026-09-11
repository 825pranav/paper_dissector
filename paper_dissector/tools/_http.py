"""Shared HTTP layer for external literature APIs.

Every scientific API we talk to is free-tier and rate limited, so they all need
the same four behaviours: throttling, retry with backoff, disk caching, and a
circuit breaker that stops hammering a service that is clearly refusing us.
This keeps that logic in one place instead of once per provider.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from paper_dissector import cache

log = logging.getLogger(__name__)

_MISS = object()


class Retryable(RuntimeError):
    """A transient upstream failure (429 / 5xx / network)."""


class ResilientJSONClient:
    """
    A throttled, retrying, caching JSON GET client for one upstream API.

    ``get()`` never raises: on persistent failure it logs and returns the
    caller's ``default`` so a rate-limited lookup degrades the enclosing stage
    rather than aborting the pipeline.
    """

    def __init__(
        self,
        name: str,
        base_url: str,
        headers: dict[str, str] | None = None,
        min_interval: float = 1.0,
        timeout: float = 30.0,
        max_attempts: int = 3,
        failure_threshold: int = 3,
        cooldown: float = 120.0,
        negative_ttl: int = 300,
    ) -> None:
        self.name = name
        self.base_url = base_url.rstrip("/")
        self.headers = headers or {}
        self.min_interval = min_interval
        self.timeout = timeout
        self.max_attempts = max_attempts
        self.failure_threshold = failure_threshold
        self.cooldown = cooldown
        self.negative_ttl = negative_ttl

        self._rate_lock = threading.Lock()
        self._last_call = 0.0
        self._breaker_lock = threading.Lock()
        self._consecutive_failures = 0
        self._open_until = 0.0

    # ── Rate limiting ────────────────────────────────────────────

    def _throttle(self) -> None:
        with self._rate_lock:
            elapsed = time.monotonic() - self._last_call
            if elapsed < self.min_interval:
                time.sleep(self.min_interval - elapsed)
            self._last_call = time.monotonic()

    # ── Circuit breaker ──────────────────────────────────────────

    def circuit_is_open(self) -> bool:
        with self._breaker_lock:
            return time.monotonic() < self._open_until

    def _record_success(self) -> None:
        with self._breaker_lock:
            self._consecutive_failures = 0
            self._open_until = 0.0

    def _record_failure(self) -> None:
        with self._breaker_lock:
            self._consecutive_failures += 1
            if self._consecutive_failures >= self.failure_threshold:
                self._open_until = time.monotonic() + self.cooldown
                log.warning(
                    "%s unreachable %d times in a row; pausing calls for %.0fs",
                    self.name, self._consecutive_failures, self.cooldown,
                )

    def reset(self) -> None:
        """Clear breaker state. Intended for tests and between benchmark runs."""
        with self._breaker_lock:
            self._consecutive_failures = 0
            self._open_until = 0.0

    # ── Request ──────────────────────────────────────────────────

    def _request(self, path: str, params: dict) -> Any:
        @retry(
            retry=retry_if_exception_type(Retryable),
            wait=wait_exponential(multiplier=2, min=1, max=15),
            stop=stop_after_attempt(self.max_attempts),
            reraise=True,
        )
        def _attempt() -> Any:
            self._throttle()
            try:
                resp = httpx.get(
                    f"{self.base_url}{path}",
                    params=params,
                    headers=self.headers,
                    timeout=self.timeout,
                )
            except httpx.HTTPError as exc:
                raise Retryable(f"network error: {exc}") from exc

            if resp.status_code == 429:
                retry_after = resp.headers.get("Retry-After")
                if retry_after:
                    try:
                        time.sleep(min(float(retry_after), 30))
                    except ValueError:
                        pass
                raise Retryable("rate limited (429)")
            if resp.status_code >= 500:
                raise Retryable(f"server error ({resp.status_code})")
            if resp.status_code == 404:
                return None

            resp.raise_for_status()
            return resp.json()

        return _attempt()

    def get(self, path: str, params: dict, default: Any) -> Any:
        """Cached, throttled GET that degrades to ``default`` instead of raising."""
        key = cache.make_key(self.name, [path, params])
        cached = cache.get(key, _MISS)
        if cached is not _MISS:
            return cached

        # Short-lived negative cache: don't re-retry a query that just failed.
        fail_key = f"{key}:failed"
        if cache.get(fail_key) is not None:
            return default

        if self.circuit_is_open():
            log.debug("%s circuit open; skipping %s", self.name, path)
            return default

        try:
            data = self._request(path, params)
        except Exception as exc:
            log.warning("%s %s failed, degrading gracefully: %s", self.name, path, exc)
            self._record_failure()
            cache.set(fail_key, True, ttl=self.negative_ttl)
            return default

        self._record_success()
        if data is None:  # 404
            data = default
        cache.set(key, data)
        return data
