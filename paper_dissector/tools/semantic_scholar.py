"""Semantic Scholar API wrapper for evidence retrieval."""

from __future__ import annotations
import httpx
import logging
import os
import threading
import time
from typing import Any

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from paper_dissector import cache
from paper_dissector.config import SEMANTIC_SCHOLAR_API_KEY, SEMANTIC_SCHOLAR_BASE

log = logging.getLogger(__name__)

_FIELDS = "paperId,title,year,authors,abstract,citationCount,url,externalIds"
_HEADERS = {"x-api-key": SEMANTIC_SCHOLAR_API_KEY} if SEMANTIC_SCHOLAR_API_KEY else {}

# The unauthenticated pool is shared across all users and 429s aggressively;
# an API key raises the ceiling but does not remove it.
_MIN_INTERVAL = float(os.getenv("S2_MIN_INTERVAL", "0.4" if SEMANTIC_SCHOLAR_API_KEY else "1.2"))
_TIMEOUT = float(os.getenv("S2_TIMEOUT", "30"))
_MISS = object()

_rate_lock = threading.Lock()
_last_call = 0.0

# Circuit breaker: the unauthenticated search pool 429s persistently, and five
# exponential retries per query would add minutes to a run. After a few
# consecutive failures we stop calling out entirely for a cooldown window.
_FAILURE_THRESHOLD = int(os.getenv("S2_FAILURE_THRESHOLD", "3"))
_COOLDOWN_SECONDS = float(os.getenv("S2_COOLDOWN", "120"))
_NEGATIVE_TTL = int(os.getenv("S2_NEGATIVE_TTL", "300"))

_breaker_lock = threading.Lock()
_consecutive_failures = 0
_circuit_open_until = 0.0


def _circuit_is_open() -> bool:
    with _breaker_lock:
        return time.monotonic() < _circuit_open_until


def _record_success() -> None:
    global _consecutive_failures, _circuit_open_until
    with _breaker_lock:
        _consecutive_failures = 0
        _circuit_open_until = 0.0


def _record_failure() -> None:
    global _consecutive_failures, _circuit_open_until
    with _breaker_lock:
        _consecutive_failures += 1
        if _consecutive_failures >= _FAILURE_THRESHOLD:
            _circuit_open_until = time.monotonic() + _COOLDOWN_SECONDS
            log.warning(
                "Semantic Scholar unreachable %d times in a row; pausing calls for %.0fs. "
                "Set SEMANTIC_SCHOLAR_API_KEY to raise the rate limit.",
                _consecutive_failures, _COOLDOWN_SECONDS,
            )


class _Retryable(RuntimeError):
    """Transient Semantic Scholar failure (429 / 5xx / network)."""


def _throttle() -> None:
    """Space out requests so we stay under the shared rate limit."""
    global _last_call
    with _rate_lock:
        elapsed = time.monotonic() - _last_call
        if elapsed < _MIN_INTERVAL:
            time.sleep(_MIN_INTERVAL - elapsed)
        _last_call = time.monotonic()


@retry(
    retry=retry_if_exception_type(_Retryable),
    wait=wait_exponential(multiplier=2, min=1, max=15),
    stop=stop_after_attempt(int(os.getenv("S2_MAX_ATTEMPTS", "3"))),
    reraise=True,
)
def _request(path: str, params: dict) -> Any:
    """Single throttled GET with retry on rate limits and server errors."""
    _throttle()
    try:
        resp = httpx.get(
            f"{SEMANTIC_SCHOLAR_BASE}{path}",
            params=params,
            headers=_HEADERS,
            timeout=_TIMEOUT,
        )
    except httpx.HTTPError as exc:
        raise _Retryable(f"network error: {exc}") from exc

    if resp.status_code == 429:
        retry_after = resp.headers.get("Retry-After")
        if retry_after:
            try:
                time.sleep(min(float(retry_after), 30))
            except ValueError:
                pass
        raise _Retryable("rate limited (429)")
    if resp.status_code >= 500:
        raise _Retryable(f"server error ({resp.status_code})")
    if resp.status_code == 404:
        return None

    resp.raise_for_status()
    return resp.json()


def _get(path: str, params: dict, default: Any) -> Any:
    """
    Cached, throttled GET that degrades gracefully.

    On persistent failure we log and return ``default`` rather than raising, so a
    rate-limited literature search never takes the whole pipeline down.
    """
    key = cache.make_key("s2", [path, params])
    cached = cache.get(key, _MISS)
    if cached is not _MISS:
        return cached

    # Short-lived negative cache: don't re-retry a query that just failed.
    fail_key = f"{key}:failed"
    if cache.get(fail_key) is not None:
        return default

    if _circuit_is_open():
        log.debug("Semantic Scholar circuit open; skipping %s", path)
        return default

    try:
        data = _request(path, params)
    except Exception as exc:
        log.warning("Semantic Scholar %s failed, degrading gracefully: %s", path, exc)
        _record_failure()
        cache.set(fail_key, True, ttl=_NEGATIVE_TTL)
        return default

    _record_success()
    if data is None:  # 404
        data = default
    cache.set(key, data)
    return data


def search_papers(query: str, limit: int = 10, year_range: str | None = None) -> list[dict]:
    """
    Search Semantic Scholar for papers matching a query.
    
    Args:
        query: search string
        limit: max results (1-100)
        year_range: e.g. "2020-2025" or "2022-" for open-ended
    
    Returns:
        List of paper dicts with fields: paperId, title, year, authors, abstract, etc.
    """
    params = {
        "query": query,
        "limit": min(limit, 100),
        "fields": _FIELDS,
    }
    if year_range:
        params["year"] = year_range

    payload = _get("/paper/search", params, default={})
    return payload.get("data", []) if isinstance(payload, dict) else []


def get_paper_details(paper_id: str) -> dict | None:
    """Get full details for a single paper by Semantic Scholar ID or DOI."""
    return _get(
        f"/paper/{paper_id}",
        {"fields": _FIELDS + ",references,citations,tldr"},
        default=None,
    )


def get_citations(paper_id: str, limit: int = 20) -> list[dict]:
    """Get papers that cite the given paper (forward citations)."""
    payload = _get(
        f"/paper/{paper_id}/citations",
        {"fields": "paperId,title,year,abstract,citationCount", "limit": limit},
        default={},
    )
    rows = payload.get("data", []) if isinstance(payload, dict) else []
    return [c["citingPaper"] for c in rows if c.get("citingPaper")]


def get_references(paper_id: str, limit: int = 50) -> list[dict]:
    """Get papers cited by the given paper (backward references)."""
    payload = _get(
        f"/paper/{paper_id}/references",
        {"fields": "paperId,title,year,abstract,citationCount", "limit": limit},
        default={},
    )
    rows = payload.get("data", []) if isinstance(payload, dict) else []
    return [r["citedPaper"] for r in rows if r.get("citedPaper")]


def search_sota_for_task(task: str, metric: str, before_year: int, limit: int = 5) -> list[dict]:
    """
    Search for SOTA results on a task+metric, filtered to papers published before a cutoff.
    Used by the staleness checker to find baselines the paper should have compared against.
    """
    query = f"state of the art {task} {metric}"
    year_range = f"-{before_year}"
    return search_papers(query, limit=limit, year_range=year_range)


def lookup_paper_by_title(title: str) -> dict | None:
    """
    Find the Semantic Scholar record whose title best matches ``title``.

    Used as the last-resort source for a paper's publication year when neither
    Docling metadata nor the text itself yields one.
    """
    title = (title or "").strip()
    if len(title) < 8:
        return None

    payload = _get(
        "/paper/search/match",
        {"query": title, "fields": "paperId,title,year,authors,externalIds"},
        default=None,
    )
    if isinstance(payload, dict) and payload.get("data"):
        return payload["data"][0]

    # /match is strict; fall back to ordinary search and take the top hit.
    results = search_papers(title, limit=1)
    return results[0] if results else None
