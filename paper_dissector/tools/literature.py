"""Provider-agnostic literature search.

The pipeline talks to this module rather than to a specific backend. Which
backend answers is decided by ``LITERATURE_PROVIDER``:

- ``auto`` (default): Semantic Scholar when ``SEMANTIC_SCHOLAR_API_KEY`` is set,
  OpenAlex otherwise.
- ``openalex`` / ``arxiv`` / ``semantic_scholar``: force one backend first.

Whatever the preference, the remaining keyless backends are tried in turn when
the preferred one comes back empty. Each of these sources fails in a different
way and none is reliably up on its own:

- Semantic Scholar: unauthenticated search 429s on essentially every call.
- OpenAlex: periodically pauses *anonymous* search under load (HTTP 503,
  "Anonymous search is paused while the search cluster recovers"). A free
  self-serve API key is exempt.
- arXiv: reliable and keyless, but preprints only and no citation counts.

All backends return records in the Semantic Scholar shape, so callers never
need to know which one answered.
"""

from __future__ import annotations

import logging

from paper_dissector.config import literature_chain, resolve_literature_provider
from paper_dissector.tools import arxiv, openalex, semantic_scholar

log = logging.getLogger(__name__)

_BACKENDS = {
    "semantic_scholar": semantic_scholar,
    "openalex": openalex,
    "arxiv": arxiv,
}


def active_provider() -> str:
    """Name of the preferred backend."""
    return resolve_literature_provider()


def _try_chain(operation: str, call, empty):
    """
    Run ``call`` against each backend in turn, returning the first real result.

    ``call`` takes the backend module; ``empty`` is the value meaning "nothing
    found" for this operation.
    """
    chain = literature_chain()
    for i, name in enumerate(chain):
        backend = _BACKENDS.get(name)
        if backend is None:
            continue
        try:
            result = call(backend)
        except Exception as exc:
            log.warning("%s via %s raised: %s", operation, name, exc)
            continue

        if result:
            if i:
                log.info("%s served by fallback backend %s", operation, name)
            return result

    return empty


def search_papers(query: str, limit: int = 10, year_range: str | None = None) -> list[dict]:
    """Search for papers, trying each available backend until one returns results."""
    return _try_chain(
        f"search({query[:48]!r})",
        lambda b: b.search_papers(query, limit=limit, year_range=year_range),
        [],
    )


def search_sota_for_task(task: str, metric: str, before_year: int, limit: int = 5) -> list[dict]:
    """Find work on a task+metric published before a cutoff, for staleness checks."""
    return _try_chain(
        "sota_search",
        lambda b: b.search_sota_for_task(task, metric, before_year, limit=limit),
        [],
    )


def lookup_paper_by_title(title: str) -> dict | None:
    """Resolve a paper record from its title, for publication-year lookup."""
    return _try_chain(
        f"title_lookup({title[:48]!r})",
        lambda b: b.lookup_paper_by_title(title),
        None,
    )
