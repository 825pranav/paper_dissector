"""Provider-agnostic literature search.

The pipeline talks to this module rather than to a specific backend. Which
backend answers is decided by ``LITERATURE_PROVIDER``:

- ``auto`` (default): Semantic Scholar when ``SEMANTIC_SCHOLAR_API_KEY`` is set,
  OpenAlex otherwise. The unauthenticated S2 search pool 429s on essentially
  every call, so a keyless setup would otherwise find nothing.
- ``semantic_scholar`` / ``openalex``: force one backend.

Both backends return records in the same shape (the Semantic Scholar one), so
callers do not need to know which is active.
"""

from __future__ import annotations

import logging

from paper_dissector.config import resolve_literature_provider
from paper_dissector.tools import openalex, semantic_scholar

log = logging.getLogger(__name__)

_BACKENDS = {
    "semantic_scholar": semantic_scholar,
    "openalex": openalex,
}


def active_provider() -> str:
    """Name of the backend currently serving requests."""
    return resolve_literature_provider()


def _backend():
    return _BACKENDS[active_provider()]


def _fallback_backend():
    """The other backend, used when the primary returns nothing usable."""
    return _BACKENDS["openalex" if active_provider() == "semantic_scholar" else "semantic_scholar"]


def search_papers(query: str, limit: int = 10, year_range: str | None = None) -> list[dict]:
    """Search the active literature backend, falling back to the other one."""
    results = _backend().search_papers(query, limit=limit, year_range=year_range)
    if results:
        return results

    # Only worth a second attempt when the fallback needs no credentials.
    fallback = _fallback_backend()
    if fallback is openalex:
        log.debug("primary literature backend returned nothing for %r; trying OpenAlex", query)
        return fallback.search_papers(query, limit=limit, year_range=year_range)
    return results


def search_sota_for_task(task: str, metric: str, before_year: int, limit: int = 5) -> list[dict]:
    """Find work on a task+metric published before a cutoff, for staleness checks."""
    results = _backend().search_sota_for_task(task, metric, before_year, limit=limit)
    if results:
        return results

    fallback = _fallback_backend()
    if fallback is openalex:
        return fallback.search_sota_for_task(task, metric, before_year, limit=limit)
    return results


def lookup_paper_by_title(title: str) -> dict | None:
    """Resolve a paper record from its title, for publication-year lookup."""
    record = _backend().lookup_paper_by_title(title)
    if record:
        return record

    fallback = _fallback_backend()
    if fallback is openalex:
        return fallback.lookup_paper_by_title(title)
    return record
